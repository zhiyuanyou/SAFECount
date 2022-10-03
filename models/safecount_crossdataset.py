import copy

import torch
import torch.nn.functional as F
from torch import nn

from models.utils import (
    build_backbone,
    build_regressor,
    crop_roi_feat,
    get_activation,
    get_clones,
)
from utils.init_helper import initialize_from_cfg


class SAFECount(nn.Module):
    def __init__(
        self,
        block,
        backbone,
        pool,
        embed_dim,
        mid_dim,
        head,
        dropout,
        activation,
        exemplar_scales=[],
        initializer=None,
    ):
        super().__init__()
        assert pool.size[0] % 2 == 1 and pool.size[1] % 2 == 1
        assert pool.type in ["max", "avg"]
        self.pool = pool
        self.exemplar_scales = exemplar_scales
        if 1 in self.exemplar_scales:
            self.exemplar_scales.remove(1)
        self.backbone = build_backbone(**backbone)
        self.out_stride = backbone.out_stride
        self.in_conv = nn.Conv2d(
            self.backbone.out_dim, embed_dim, kernel_size=1, stride=1
        )
        self.safecount = SAFECountMultiBlock(
            block=block,
            out_stride=backbone.out_stride,
            embed_dim=embed_dim,
            mid_dim=mid_dim,
            head=head,
            dropout=dropout,
            activation=activation,
        )
        self.count_regressor = build_regressor(in_dim=embed_dim, activation=activation)
        for module in [self.in_conv, self.safecount, self.count_regressor]:
            initialize_from_cfg(module, initializer)

    def forward(self, input):
        image = input["image"]  # [1,c,h,w]
        assert image.shape[0] == 1, "Batch size must be 1!"
        boxes = input["boxes"].squeeze(0)  # [1,m,4] -> [m,4]
        feat = self.backbone(image)
        # single-scale exemplars
        feat_boxes_list = []
        feat_boxes = crop_roi_feat(
            feat, boxes, self.out_stride
        )  # list of [1,c,h,w], len=m
        for feat_box in feat_boxes:
            if self.pool.type == "max":
                feat_box = F.adaptive_max_pool2d(
                    feat_box, self.pool.size, return_indices=False
                )  # [1,c,h,w]
            else:
                feat_box = F.adaptive_avg_pool2d(feat_box, self.pool.size)  # [1,c,h,w]
            feat_boxes_list.append(feat_box)
        # multi-scale exemplars
        _, _, h, w = image.shape
        for scale in self.exemplar_scales:  # len=n
            h_rsz = int(h * scale) // 16 * 16
            w_rsz = int(w * scale) // 16 * 16
            image_scale = F.interpolate(image, size=(w_rsz, h_rsz), mode="bilinear")
            scale_h = h_rsz / h
            scale_w = w_rsz / w
            boxes_scale = copy.deepcopy(boxes)
            boxes_scale[:, 0] *= scale_h
            boxes_scale[:, 1] *= scale_w
            boxes_scale[:, 2] *= scale_h
            boxes_scale[:, 3] *= scale_w
            feat_scale = self.backbone(image_scale)
            feat_boxes = crop_roi_feat(
                feat_scale, boxes_scale, self.out_stride
            )  # list of [1,c,h,w], len=m
            for feat_box in feat_boxes:
                if self.pool.type == "max":
                    feat_box = F.adaptive_max_pool2d(
                        feat_box, self.pool.size, return_indices=False
                    )  # [1,c,h,w]
                else:
                    feat_box = F.adaptive_avg_pool2d(
                        feat_box, self.pool.size
                    )  # [1,c,h,w]
                feat_boxes_list.append(feat_box)
        feat_boxes = torch.cat(feat_boxes_list, dim=0)  # [m(n+1),c,h,w]
        feat_boxes = self.in_conv(feat_boxes)
        # image
        feat = self.in_conv(feat)
        # count
        output = self.safecount(feat=feat, feat_boxes=feat_boxes)
        density_pred = self.count_regressor(output)
        input.update({"density_pred": density_pred})
        return input


class SAFECountMultiBlock(nn.Module):
    def __init__(
        self,
        block,
        out_stride,
        embed_dim,
        mid_dim,
        head,
        dropout,
        activation,
    ):
        super().__init__()
        self.out_stride = out_stride
        safecount_block = SAFECountBlock(
            embed_dim=embed_dim,
            mid_dim=mid_dim,
            head=head,
            dropout=dropout,
            activation=activation,
        )
        self.blocks = get_clones(safecount_block, block)

    def forward(self, feat, feat_boxes):
        output = feat
        for block in self.blocks:
            output = block(output, feat_boxes)
        return output


class SAFECountBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        mid_dim,
        head,
        dropout,
        activation,
    ):
        super().__init__()
        self.aggt = SimilarityWeightedAggregation(embed_dim, head, dropout)
        self.conv1 = nn.Conv2d(embed_dim, mid_dim, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(mid_dim, embed_dim, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)()

    def forward(self, tgt, src):
        tgt2 = self.aggt(query=tgt, keys=src, values=src)
        ##################################################################################
        # fuse feature
        ##################################################################################
        tgt = tgt + self.dropout1(tgt2)
        tgt = tgt.permute(0, 2, 3, 1).contiguous()
        tgt = self.norm1(tgt).permute(0, 3, 1, 2).contiguous()
        tgt2 = self.conv2(self.dropout(self.activation(self.conv1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = tgt.permute(0, 2, 3, 1).contiguous()
        tgt = self.norm2(tgt).permute(0, 3, 1, 2).contiguous()
        return tgt


class SimilarityWeightedAggregation(nn.Module):
    """
    Implement the multi-head attention with convolution to keep the spatial structure.
    """

    def __init__(self, embed_dim, head, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.head = head
        self.dropout = nn.Dropout(dropout)
        self.head_dim = embed_dim // head
        assert self.head_dim * head == self.embed_dim
        self.norm = nn.LayerNorm(embed_dim)
        self.in_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1)
        self.out_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1)

    def forward(self, query, keys, values):
        """
        query: 1 x C x H x W
        keys: MN x C x H x W
        values: MN x C x H x W
        """
        _, _, h_p, w_p = keys.shape
        pad = (w_p // 2, w_p // 2, h_p // 2, h_p // 2)
        _, _, h_q, w_q = query.shape

        ##################################################################################
        # calculate similarity (attention)
        ##################################################################################
        query = self.in_conv(query)
        query = query.permute(0, 2, 3, 1).contiguous()
        query = self.norm(query).permute(0, 3, 1, 2).contiguous()
        query = query.contiguous().view(
            self.head, self.head_dim, h_q, w_q
        )  # [head,c,h,w]
        attns_list = []
        for key in keys:
            key = key.unsqueeze(0)
            key = self.in_conv(key)
            key = key.permute(0, 2, 3, 1).contiguous()
            key = self.norm(key).permute(0, 3, 1, 2).contiguous()
            key = key.contiguous().view(
                self.head, self.head_dim, h_p, w_p
            )  # [head,c,h,w]
            attn_list = []
            for q, k in zip(query, key):
                attn = F.conv2d(F.pad(q.unsqueeze(0), pad), k.unsqueeze(0))  # [1,1,h,w]
                attn_list.append(attn)
            attn = torch.cat(attn_list, dim=0)  # [head,1,h,w]
            attns_list.append(attn)
        attns = torch.cat(attns_list, dim=1)  # [head,n,h,w]
        assert list(attns.size()) == [self.head, keys.shape[0], h_q, w_q]

        ##################################################################################
        # score normalization
        ##################################################################################
        attns = attns * float(self.embed_dim * h_p * w_p) ** -0.5  # scaling
        attns = torch.exp(attns)  # [head,n,h,w]
        attns_sn = (
            attns / (attns.max(dim=2, keepdim=True)[0]).max(dim=3, keepdim=True)[0]
        )
        attns_en = attns / attns.sum(dim=1, keepdim=True)
        attns = self.dropout(attns_sn * attns_en)

        ##################################################################################
        # similarity weighted aggregation
        ##################################################################################
        feats = 0
        for idx, value in enumerate(values):
            attn = attns[:, idx, :, :].unsqueeze(1)  # [head,1,h,w]
            value = value.unsqueeze(0)
            value = self.in_conv(value)
            value = value.contiguous().view(
                self.head, self.head_dim, h_p, w_p
            )  # [head,c,h,w]
            feat_list = []
            for w, v in zip(attn, value):
                feat = F.conv2d(
                    F.pad(w.unsqueeze(0), pad), v.unsqueeze(1).flip(2, 3)
                )  # [1,c,h,w]
                feat_list.append(feat)
            feat = torch.cat(feat_list, dim=0)  # [head,c,h,w]
            feats += feat
        assert list(feats.size()) == [self.head, self.head_dim, h_q, w_q]
        feats = feats.contiguous().view(1, self.embed_dim, h_q, w_q)  # [1,c,h,w]
        feats = self.out_conv(feats)
        return feats


def build_network(**kwargs):
    return SAFECount(**kwargs)
