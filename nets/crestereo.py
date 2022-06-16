from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .update import BasicUpdateBlock
from .extractor import BasicEncoder
from .corr import corr_iter, corr_att_offset
from .utils import coords_grid

from .attention import position_encoding_sine, LocalFeatureTransformer

#Ref: https://github.com/princeton-vl/RAFT/blob/master/core/raft.py
class CREStereo(nn.Module):
    def __init__(self, max_disp: int = 192, mixed_precision: bool = False, test_mode: bool = False, iters: int = 10):
        super(CREStereo, self).__init__()

        self.max_flow = max_disp
        self.mixed_precision = mixed_precision
        self.test_mode = test_mode

        self.hidden_dim = 128
        self.context_dim = 128
        self.dropout = 0

        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=self.dropout)  
        self.update_block = BasicUpdateBlock(hidden_dim=self.hidden_dim, cor_planes=4 * 9, mask_size=4)

        # loftr
        self.self_att_fn = LocalFeatureTransformer(
            d_model=256, nhead=8, layer_name="self", attention="linear"
        )
        self.cross_att_fn = LocalFeatureTransformer(
            d_model=256, nhead=8, layer_name="cross", attention="linear"
        )

        # adaptive search
        self.search_num = 9
        self.conv_offset_16 = nn.Conv2d(
            256, self.search_num * 2, kernel_size=3, stride=1, padding=1
        )
        self.conv_offset_8 = nn.Conv2d(
            256, self.search_num * 2, kernel_size=3, stride=1, padding=1
        )
        self.range_16 = 1
        self.range_8 = 1

        self.iters = iters
        self.test_mode = test_mode

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def convex_upsample(self, flow, mask, rate: int = 4):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        # print(flow.shape, mask.shape, rate)
        mask = mask.view(N, 1, 9, rate, rate, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(rate * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, rate*H, rate*W)

    def zero_init(self, fmap):
        N, C, H, W = fmap.shape
        _x = torch.zeros([N, 1, H, W], dtype=torch.float32)
        _y = torch.zeros([N, 1, H, W], dtype=torch.float32)
        zero_flow = torch.cat((_x, _y), dim=1).to(fmap.device)
        return zero_flow

    def forward(self, image1, image2, flow_init: Optional[torch.Tensor]) -> torch.Tensor:
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim

        # run the feature network
        fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        # 1/4 -> 1/8
        # feature
        fmap1_dw8 = F.avg_pool2d(fmap1, 2, stride=2)
        fmap2_dw8 = F.avg_pool2d(fmap2, 2, stride=2)

        # offset
        offset_dw8 = self.conv_offset_8(fmap1_dw8)
        offset_dw8 = self.range_8 * (torch.sigmoid(offset_dw8) - 0.5) * 2.0

        # context
        net, inp = torch.split(fmap1, [hdim,hdim], dim=1)
        net = torch.tanh(net)
        inp = F.relu(inp)
        net_dw8 = F.avg_pool2d(net, 2, stride=2)
        inp_dw8 = F.avg_pool2d(inp, 2, stride=2)

        # 1/4 -> 1/16
        # feature
        fmap1_dw16 = F.avg_pool2d(fmap1, 4, stride=4)
        fmap2_dw16 = F.avg_pool2d(fmap2, 4, stride=4)
        offset_dw16 = self.conv_offset_16(fmap1_dw16)
        offset_dw16 = self.range_16 * (torch.sigmoid(offset_dw16) - 0.5) * 2.0

        # context
        net_dw16 = F.avg_pool2d(net, 4, stride=4)
        inp_dw16 = F.avg_pool2d(inp, 4, stride=4)

        # positional encoding and self-attention
        # 'n c h w -> n (h w) c'
        x_tmp = position_encoding_sine(fmap1_dw16, d_model=256, max_shape=(image1.shape[2] // 16, image1.shape[3] // 16))
        fmap1_dw16 = x_tmp.permute(0, 2, 3, 1).reshape(x_tmp.shape[0], x_tmp.shape[2] * x_tmp.shape[3], x_tmp.shape[1])
        # 'n c h w -> n (h w) c'
        x_tmp = position_encoding_sine(fmap2_dw16, d_model=256, max_shape=(image1.shape[2] // 16, image1.shape[3] // 16))
        fmap2_dw16 = x_tmp.permute(0, 2, 3, 1).reshape(x_tmp.shape[0], x_tmp.shape[2] * x_tmp.shape[3], x_tmp.shape[1])

        fmap1_dw16, fmap2_dw16 = self.self_att_fn(fmap1_dw16, fmap2_dw16)
        fmap1_dw16, fmap2_dw16 = [
            x.reshape(x.shape[0], image1.shape[2] // 16, -1, x.shape[2]).permute(0, 3, 1, 2)
            for x in [fmap1_dw16, fmap2_dw16]
        ]

        # Cascaded refinement (1/16 + 1/8 + 1/4)
        predictions = []
        flow: Optional[torch.Tensor] = None
        flow_up: Optional[torch.Tensor] = None
        if flow_init is not None:
            scale = fmap1.shape[2] / flow_init.shape[2]
            flow = -scale * F.interpolate(
                flow_init,
                size=(fmap1.shape[2], fmap1.shape[3]),
                mode="bilinear",
                align_corners=True,
                )
        else:
            # zero initialization
            flow_dw16 = self.zero_init(fmap1_dw16)

            # Recurrent Update Module
            # RUM: 1/16
            coords = coords_grid(fmap1_dw16.shape[0], fmap1_dw16.shape[2], fmap1_dw16.shape[3], fmap1_dw16.device)
            for itr in range(self.iters // 2):
                if itr % 2 == 0:
                    small_patch = False
                else:
                    small_patch = True

                flow_dw16 = flow_dw16.detach()
                out_corrs = corr_att_offset(
                    coords, fmap1_dw16, fmap2_dw16, flow_dw16, offset_dw16, small_patch=small_patch, att=self.cross_att_fn
                    )

                net_dw16, up_mask, delta_flow = self.update_block(
                    net_dw16, inp_dw16, out_corrs, flow_dw16
                )

                flow_dw16 = flow_dw16 + delta_flow
                flow = self.convex_upsample(flow_dw16, up_mask, rate=4)
                flow_up = -4 * F.interpolate(
                    flow,
                    size=(4 * flow.shape[2], 4 * flow.shape[3]),
                    mode="bilinear",
                    align_corners=True,
                )
                predictions.append(flow_up)

            if flow is not None:
                scale = fmap1_dw8.shape[2] / flow.shape[2]
                flow_dw8 = -scale * F.interpolate(
                    flow,
                    size=(fmap1_dw8.shape[2], fmap1_dw8.shape[3]),
                    mode="bilinear",
                    align_corners=True,
                )
            else:
                raise RuntimeError("flow is unexpectedly None")

            # RUM: 1/8
            coords = coords_grid(fmap1_dw8.shape[0], fmap1_dw8.shape[2], fmap1_dw8.shape[3], fmap1_dw8.device)
            for itr in range(self.iters // 2):
                if itr % 2 == 0:
                    small_patch = False
                else:
                    small_patch = True

                flow_dw8 = flow_dw8.detach()
                out_corrs = corr_att_offset(coords, fmap1_dw8, fmap2_dw8, flow_dw8, offset_dw8, small_patch=small_patch)

                net_dw8, up_mask, delta_flow = self.update_block(
                    net_dw8, inp_dw8, out_corrs, flow_dw8
                )

                flow_dw8 = flow_dw8 + delta_flow
                flow = self.convex_upsample(flow_dw8, up_mask, rate=4)
                flow_up = -2 * F.interpolate(
                    flow,
                    size=(2 * flow.shape[2], 2 * flow.shape[3]),
                    mode="bilinear",
                    align_corners=True,
                )
                predictions.append(flow_up)

            scale = fmap1.shape[2] / flow.shape[2]
            flow = -scale * F.interpolate(
                flow,
                size=(fmap1.shape[2], fmap1.shape[3]),
                mode="bilinear",
                align_corners=True,
            )

        # RUM: 1/4
        coords = coords_grid(fmap1.shape[0], fmap1.shape[2], fmap1.shape[3], fmap1.device)
        for itr in range(self.iters):
            if itr % 2 == 0:
                small_patch = False
            else:
                small_patch = True

            flow = flow.detach()
            out_corrs = corr_iter(coords, fmap1, fmap2, flow, small_patch=small_patch)

            net, up_mask, delta_flow = self.update_block(net, inp, out_corrs, flow)

            flow = flow + delta_flow
            flow_up = -self.convex_upsample(flow, up_mask, rate=4)
            predictions.append(flow_up)

        # if self.test_mode:
        #     return flow_up

#         return predictions
        if flow_up is None:
            raise RuntimeError("flow_up is unexpectedly None")
        else:
            return flow_up
