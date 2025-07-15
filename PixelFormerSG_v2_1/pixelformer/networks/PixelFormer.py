import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_transformer import SwinTransformer
from .PQI import PSP
from .SAM import SAM
from torch_geometric.nn import GATConv
from torchvision.ops import box_iou
########################################################################################################################

# def match_box_indices(sub_boxes, pred_boxes, iou_threshold=0.9):
#     """Return indices of pred_boxes with highest IoU to each sub_box."""
    
#     matched_indices = []
#     for sb in sub_boxes:
#         ious = box_iou(sb.unsqueeze(0), pred_boxes).squeeze(0)  # [N]
#         max_iou, idx = ious.max(dim=0)
#         matched_indices.append(idx if max_iou > iou_threshold else -1)
#     return matched_indices

def match_box_indices(boxes, reference_boxes, iou_threshold=0.6):
    """
    boxes: Tensor [T, 4] — each row is (x1, y1, x2, y2)
    reference_boxes: Tensor [N, 4]
    Returns a list of matched indices into reference_boxes for each box.
    """
    matched_indices = []
    for b in boxes:
        if reference_boxes.size(1) != 4:
            raise ValueError(f"Expected reference_boxes of shape [N, 4], got {reference_boxes.shape}")
        ious = box_iou(b.unsqueeze(0), reference_boxes).squeeze(0)  # [N]
        max_iou, idx = ious.max(dim=0)
        matched_indices.append(idx if max_iou > iou_threshold else -1)
    return matched_indices

def extract_scene_graph_edges(sub_boxes, obj_boxes, rel_logits, pred_boxes, iou_threshold=0.6):
    edge_index_list = []
    edge_type_list = []
    B, T, R = rel_logits.shape
    for b in range(B):
        edge_index = []
        edge_type = []
        rel_probs = F.softmax(rel_logits[b], dim=-1)
        pred_rel = torch.argmax(rel_probs, dim=-1)

        sub_idxs = match_box_indices(sub_boxes[b], pred_boxes[b], iou_threshold)
        obj_idxs = match_box_indices(obj_boxes[b], pred_boxes[b], iou_threshold)

        for t in range(T):
            si = sub_idxs[t]
            oi = obj_idxs[t]
            if si >= 0 and oi >= 0:
                edge_index.append([si, oi])
                edge_type.append(pred_rel[t])
        if len(edge_index) == 0:  # avoid crash
            edge_index = [[0, 0]]
            edge_type = [torch.tensor(0, device=rel_logits.device)]
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=rel_logits.device).T
        edge_type = torch.stack(edge_type)
        edge_index_list.append(edge_index)
        edge_type_list.append(edge_type)
    return edge_index_list, edge_type_list

# class SceneGraphEncoder(nn.Module):
#     def __init__(self, enc_dim, node_dim, out_dim, rel_classes=51, feat_size=(7, 7)):
#         super().__init__()
#         self.roi_size = feat_size
#         self.gat = GATConv(node_dim, out_dim, edge_dim=node_dim)
#         self.relation_embed = nn.Embedding(rel_classes, node_dim)
#         self.project = nn.Sequential(
#             nn.Linear(enc_dim, node_dim),  # assuming enc_feat has 1024 channels
#             nn.GELU()
#         )

#     def extract_roi_feats(self, feat_map, boxes, image_size):
#         """ROIAlign wrapper"""
#         B, T, _ = boxes.shape
#         _, _, H, W = feat_map.shape
#         all_rois = []

#         for b in range(B):
#             cx, cy, w, h = boxes[b].unbind(-1)
#             x1 = (cx - w / 2) * image_size[1]
#             y1 = (cy - h / 2) * image_size[0]
#             x2 = (cx + w / 2) * image_size[1]
#             y2 = (cy + h / 2) * image_size[0]
#             coords = torch.stack([x1, y1, x2, y2], dim=-1)  # [T, 4]
#             img_inds = torch.full((T, 1), b, device=boxes.device)
#             rois = torch.cat([img_inds, coords], dim=-1)  # [T, 5]
#             all_rois.append(rois)

#         rois = torch.cat(all_rois, dim=0)  # [B*T, 5]
#         roi_feats = roi_align(
#             feat_map, rois, output_size=self.roi_size,
#             spatial_scale=float(W) / image_size[1],
#             aligned=True
#         )  # [B*T, C, H, W]
#         pooled = roi_feats.mean(dim=[2, 3])  # [B*T, C]
#         return pooled.view(B, T, -1)  # [B, T, C]

#     def forward(self, enc_feat, sub_boxes, obj_boxes, rel_logits):
#         B, T, _ = rel_logits.shape
#         H_img, W_img =  enc_feat.shape[-2] * 4, enc_feat.shape[-1] * 4  # assume 1/4 scale from image
#         device = enc_feat.device

#         # Step 1: Extract RoI features for sub & obj boxes
#         sub_feat = self.extract_roi_feats(enc_feat, sub_boxes, (H_img, W_img))  # [B, T, C]
#         obj_feat = self.extract_roi_feats(enc_feat, obj_boxes, (H_img, W_img))  # [B, T, C]

#         # Step 2: Project to node_dim
#         sub_proj = self.project(sub_feat)  # [B, T, D]
#         obj_proj = self.project(obj_feat)  # [B, T, D]
#         node_feat = torch.cat([sub_proj, obj_proj], dim=1)  # [B, 2T, D]

#         # Step 3: Prepare edge_index and edge_attr
#         outputs = []
#         for b in range(B):
#             edge_index = torch.stack([
#                 torch.arange(0, T, device=device),
#                 torch.arange(T, 2 * T, device=device)
#             ], dim=0)  # [2, T]

#             rel_cls = torch.argmax(F.softmax(rel_logits[b, :, :-1], dim=-1), dim=-1)  # [T]
#             edge_attr = self.relation_embed(rel_cls)  # [T, D]

#             # Run GAT
#             out = self.gat(node_feat[b], edge_index, edge_attr)  # [2T, out_dim]
#             outputs.append(out)

#         return torch.stack(outputs, dim=0)  # [B, 2T, out_dim]


class BCP(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, max_depth, min_depth, in_features=512, hidden_features=512*4, out_features=256, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.min_depth = min_depth
        self.max_depth = max_depth

    def forward(self, x):
        x = torch.mean(x.flatten(start_dim=2), dim = 2)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        bins = torch.softmax(x, dim=1)
        bins = bins / bins.sum(dim=1, keepdim=True)
        bin_widths = (self.max_depth - self.min_depth) * bins
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_depth)
        bin_edges = torch.cumsum(bin_widths, dim=1)
        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.contiguous().view(n, dout, 1, 1)
        return centers

import torch
import torch.nn.functional as F
from torchvision.ops import roi_align

class CrossAttnBlock(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)

    def forward(self, q_feat, sg_feat):
        """
        q_feat: [B, C, H, W]
        sg_feat: [B, N, C] (node features from scene graph encoder)
        """
        B, C, H, W = q_feat.shape
        q_flat = q_feat.flatten(2).transpose(1, 2)  # [B, HW, C]
        q_flat = self.norm_q(q_flat)
        kv_feat = self.norm_kv(sg_feat)  # [B, N, C]

        # Project to Q, K, V
        Q = self.q_proj(q_flat)   # [B, HW, C]
        K = self.k_proj(kv_feat)  # [B, N, C]
        V = self.v_proj(kv_feat)  # [B, N, C]

        # Cross Attention
        attn_out, _ = self.attn(Q, K, V)  # [B, HW, C]

        # Output
        out = self.out_proj(attn_out).transpose(1, 2).view(B, C, H, W)
        return q_feat + out  # residual connection


import torch
import torch.nn as nn
from torchvision.ops import box_iou, roi_align
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv


class SceneGraphBuilder(nn.Module):
    def __init__(self, num_rel_classes, topk=10, conf_thresh=0.3, iou_thresh=0.6):
        super().__init__()
        self.topk = topk
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.num_rel_classes = num_rel_classes
    
    def xywh_to_xyxy(self, boxes):
        # Convert [x_center, y_center, w, h] -> [x1, y1, x2, y2]
        x_c, y_c, w, h = boxes.unbind(dim=1)
        x1 = x_c - 0.5 * w
        y1 = y_c - 0.5 * h
        x2 = x_c + 0.5 * w
        y2 = y_c + 0.5 * h
        return torch.stack([x1, y1, x2, y2], dim=1)


    def forward(self, scene_graph):
        """
        scene_graph: dict with keys: sub_boxes, obj_boxes, sub_logits, obj_logits, rel_logits
        Returns:
            node_boxes [N, 4], node_labels [N], edge_index [2, K], edge_attr [K, R+1]
        """
        results = []
        B = scene_graph['rel_logits'].size(0)

        for b in range(B):
            rel_logits = scene_graph['rel_logits'][b]        # [N, R+1]
            sub_logits = scene_graph['sub_logits'][b]        # [N, C+1]
            obj_logits = scene_graph['obj_logits'][b]
    
            probas_rel = rel_logits.softmax(-1)[:, :-1]
            probas_sub = sub_logits.softmax(-1)[:, :-1]
            probas_obj = obj_logits.softmax(-1)[:, :-1]
    
            keep = (probas_rel.max(-1).values > self.conf_thresh) & \
                   (probas_sub.max(-1).values > self.conf_thresh) & \
                   (probas_obj.max(-1).values > self.conf_thresh)
    
            keep_queries = torch.nonzero(keep, as_tuple=True)[0]
            scores = probas_rel[keep_queries].max(-1)[0] * \
                     probas_sub[keep_queries].max(-1)[0] * \
                     probas_obj[keep_queries].max(-1)[0]
    
            top_indices = torch.argsort(-scores)[:self.topk]
            keep_queries = keep_queries[top_indices]
    
            sub_boxes = scene_graph['sub_boxes'][b][keep_queries]  # [K, 4]
            obj_boxes = scene_graph['obj_boxes'][b][keep_queries]  # [K, 4]
            sub_labels = sub_logits[keep_queries, :-1].argmax(-1)  # [K]
            obj_labels = obj_logits[keep_queries, :-1].argmax(-1)  # [K]
            rel_feats = rel_logits[keep_queries]                   # [K, R+1]
    
            num_rels = sub_boxes.size(0)
    
            # === Merge subject and object boxes ===
            boxes = torch.cat([sub_boxes, obj_boxes], dim=0)       # [2K, 4]
            labels = torch.cat([sub_labels, obj_labels], dim=0)    # [2K]
    
            # === IoU-based deduplication with label agreement ===
            boxes_xyxy = self.xywh_to_xyxy(boxes)
    
            # === IoU-based deduplication with label agreement ===
            iou_matrix = box_iou(boxes_xyxy, boxes_xyxy)
            
            node_map = torch.arange(len(boxes))
            for i in range(len(boxes)):
                for j in range(i):
                    if labels[i] == labels[j] and iou_matrix[i, j] >= self.iou_thresh:
                        node_map[i] = node_map[j]
    
            _, unique_indices = torch.unique(node_map, return_inverse=True)
            dedup_boxes = []
            dedup_labels = []
            seen = {}
            for i, new_idx in enumerate(unique_indices):
                if new_idx.item() not in seen:
                    dedup_boxes.append(boxes[i])
                    dedup_labels.append(labels[i])
                    seen[new_idx.item()] = True
    
            node_boxes = torch.stack(dedup_boxes) if dedup_boxes else boxes
            node_labels = torch.stack(dedup_labels) if dedup_labels else labels
    
            subj_ids = unique_indices[:num_rels]
            obj_ids = unique_indices[num_rels:]
            edge_index = torch.stack([subj_ids, obj_ids], dim=0)  # [2, K]
            
            results.append({
                'node_boxes': node_boxes,        # [N, 4] in xywh
                'node_labels': node_labels,      # [N] - class indices 0-151
                'edge_index': edge_index,        # [2, K]
                'edge_attr': rel_feats           # [K, R+1]
            })
        return results

class SceneGraphEncoder(nn.Module):
    def __init__(self, feat_dim, node_dim, out_dim, rel_classes=51, feat_size=(7, 7)):
        super().__init__()
        self.node_proj = nn.Sequential(
            nn.Linear(feat_dim, node_dim),
            nn.ReLU(inplace=True),
        )

        self.roi_size = feat_size
        self.relation_embed = nn.Embedding(rel_classes, node_dim)
        self.edge_proj = nn.Sequential(
                                nn.Linear(3 * node_dim, node_dim),
                                nn.ReLU(inplace=True)
                            )

        self.gnn = GATConv(node_dim, out_dim)

    def xywh_to_xyxy(self, boxes):
        """
        Args:
            boxes: Tensor of shape [B, T, 4] in (cx, cy, w, h)
        Returns:
            Tensor of shape [B, T, 4] in (x1, y1, x2, y2)
        """
        x_c = boxes[..., 0]
        y_c = boxes[..., 1]
        w = boxes[..., 2]
        h = boxes[..., 3]
        
        x1 = x_c - 0.5 * w
        y1 = y_c - 0.5 * h
        x2 = x_c + 0.5 * w
        y2 = y_c + 0.5 * h
        
        return torch.stack([x1, y1, x2, y2], dim=-1)  # [B, T, 4]
        

    def forward(self, feat_map, sg_data_list):
        B, C, H, W = feat_map.shape
        outputs = []
        for b in range(B):
            rois = self.xywh_to_xyxy(sg_data_list[b]["node_boxes"])  # <- fix here
            rois[:, 0::2] *= W
            rois[:, 1::2] *= H

            num_boxes = rois.size(0)
            batch_idx = torch.zeros((num_boxes, 1), device=feat_map.device) 
            roi_boxes = torch.cat([batch_idx, rois], dim=1)  # [N, 5]
            
            roi_feats = roi_align(feat_map[b].unsqueeze(0), roi_boxes, output_size=self.roi_size, spatial_scale=1.0, aligned=True)
            roi_feats = roi_feats.view(roi_feats.size(0), -1)

            node_feats = self.node_proj(roi_feats)

            # Extract edge features
            subj_feat = node_feats[sg_data_list[b]["edge_index"][0]]
            obj_feat  = node_feats[sg_data_list[b]["edge_index"][1]]
            _, edge_rel_type = sg_data_list[b]["edge_attr"].softmax(-1)[:,:-1].max(-1)
            rel_embed = self.relation_embed(edge_rel_type)

            edge_attr_input = torch.cat([subj_feat, obj_feat, rel_embed], dim=-1)
            edge_attr = self.edge_proj(edge_attr_input)  # [E, D]

            node_feats = self.gnn(node_feats, sg_data_list[b].edge_index, edge_attr)
            outputs.append(node_feats)
        outputs = torch.stack(outputs, dim = 0)
        print(outputs.shape)
        return outputs


class PixelFormerSG(nn.Module):

    def __init__(self, version=None, inv_depth=False, pretrained=None, 
                    frozen_stages=-1, min_depth=0.1, max_depth=100.0, combine_option = "plus", **kwargs):
        super().__init__()

        self.inv_depth = inv_depth
        self.with_auxiliary_head = False
        self.with_neck = False
        assert combine_option in ["plus", "cross-attn"]
        self.combine_option = combine_option
        norm_cfg = dict(type='BN', requires_grad=True)
        # norm_cfg = dict(type='GN', requires_grad=True, num_groups=8)

        window_size = int(version[-2:])

        if version[:-2] == 'base':
            embed_dim = 128
            depths = [2, 2, 18, 2]
            num_heads = [4, 8, 16, 32]
            in_channels = [128, 256, 512, 1024]
        elif version[:-2] == 'large':
            embed_dim = 192
            depths = [2, 2, 18, 2]
            num_heads = [6, 12, 24, 48]
            in_channels = [192, 384, 768, 1536]
        elif version[:-2] == 'tiny':
            embed_dim = 96
            depths = [2, 2, 6, 2]
            num_heads = [3, 6, 12, 24]
            in_channels = [96, 192, 384, 768]

        backbone_cfg = dict(
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            ape=False,
            drop_path_rate=0.3,
            patch_norm=True,
            use_checkpoint=False,
            frozen_stages=frozen_stages
        )

        embed_dim = 512
        decoder_cfg = dict(
            in_channels=in_channels,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=embed_dim,
            dropout_ratio=0.0,
            num_classes=32,
            norm_cfg=norm_cfg,
            align_corners=False
        )

        self.backbone = SwinTransformer(**backbone_cfg)
        v_dim = decoder_cfg['num_classes']*4
        win = 7
        sam_dims = [128, 256, 512, 1024]
        v_dims = [64, 128, 256, embed_dim]
        self.sam4 = SAM(input_dim=in_channels[3], embed_dim=sam_dims[3], window_size=win, v_dim=v_dims[3], num_heads=32)
        self.sam3 = SAM(input_dim=in_channels[2], embed_dim=sam_dims[2], window_size=win, v_dim=v_dims[2], num_heads=16)
        self.sam2 = SAM(input_dim=in_channels[1], embed_dim=sam_dims[1], window_size=win, v_dim=v_dims[1], num_heads=8)
        self.sam1 = SAM(input_dim=in_channels[0], embed_dim=sam_dims[0], window_size=win, v_dim=v_dims[0], num_heads=4)

        self.decoder = PSP(**decoder_cfg)
        self.disp_head1 = DispHead(input_dim=sam_dims[0])

        self.bcp = BCP(max_depth=max_depth, min_depth=min_depth)
        
        # scene graph encoder
        self.sg_encoder_q4 = SceneGraphEncoder(feat_dim=in_channels[3], node_dim=in_channels[3], out_dim=v_dims[3])
        self.sg_builder = SceneGraphBuilder(num_rel_classes = 51, iou_thresh=0.6)
        # self.sg_encoder_q3 = SceneGraphEncoder(node_dim=in_channels[2], out_dim=v_dims[2])
        # self.sg_encoder_q2 = SceneGraphEncoder(node_dim=in_channels[1], out_dim=v_dims[1])
        # self.sg_encoder_q1 = SceneGraphEncoder(node_dim=in_channels[0], out_dim=v_dims[0])
        
        # cross attention blocks
        # self.cross_attn_q4 = CrossAttnBlock(dim=v_dims[3], num_heads=8)
        # self.cross_attn_q3 = CrossAttnBlock(dim=v_dims[2], num_heads=8)
        # self.cross_attn_q2 = CrossAttnBlock(dim=v_dims[1], num_heads=4)
        # self.cross_attn_q1 = CrossAttnBlock(dim=v_dims[0], num_heads=4)

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        print(f'== Load encoder backbone from: {pretrained}')
        self.backbone.init_weights(pretrained=pretrained)
        self.decoder.init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()
    
    def forward(self, imgs, scene_graph):

        enc_feats = self.backbone(imgs)
        if self.with_neck:
            enc_feats = self.neck(enc_feats)


        # 1. Scene graph features cho từng tầng
        graph_data_list = self.sg_builder(scene_graph)
        sg_feat_q4 = self.sg_encoder_q4(enc_feats[3], graph_data_list)
        # sg_feat_q3 = self.sg_encoder_q3(enc_feats[2], scene_graph['sub_boxes'], scene_graph['obj_boxes'], scene_graph['rel_logits'])
        # sg_feat_q2 = self.sg_encoder_q2(enc_feats[1], scene_graph['sub_boxes'], scene_graph['obj_boxes'], scene_graph['rel_logits'])
        # sg_feat_q1 = self.sg_encoder_q1(enc_feats[0], scene_graph['sub_boxes'], scene_graph['obj_boxes'], scene_graph['rel_logits'])

        # 2. Kết hợp với qX qua cross-attention (giả sử bạn có CrossAttnBlock sẵn)
        q4 = self.decoder(enc_feats)
        
        # if self.combine_option == "plus":
        sg_feat_q4 = sg_feat_q4.mean(dim=1).unsqueeze(-1).unsqueeze(-1)  # [B, D, 1, 1]
        q4 = q4 + sg_feat_q4
        # elif self.combine_option == "cross-attn":
        #     q4 = self.cross_attn_q4(q4, sg_feat_q4)

        q3 = self.sam4(enc_feats[3], q4)
        q3 = nn.PixelShuffle(2)(q3)
        # if self.combine_option == "plus":
        #     sg_feat_q3 = sg_feat_q3.mean(dim=1).unsqueeze(-1).unsqueeze(-1)  # [B, D, 1, 1]
        #     q3 = q3 + sg_feat_q3
        # elif self.combine_option == "cross-attn":
        #     q3 = self.cross_attn_q3(q3, sg_feat_q3)

        q2 = self.sam3(enc_feats[2], q3)
        q2 = nn.PixelShuffle(2)(q2)
        # if self.combine_option == "plus":
        #     sg_feat_q2 = sg_feat_q2.mean(dim=1).unsqueeze(-1).unsqueeze(-1)  # [B, D, 1, 1]
        #     q2 = q2 + sg_feat_q2
        # elif self.combine_option == "cross-attn":
        #     q2 = self.cross_attn_q2(q2, sg_feat_q2)

        q1 = self.sam2(enc_feats[1], q2)
        q1 = nn.PixelShuffle(2)(q1)
        # if self.combine_option == "plus":
        #     sg_feat_q1 = sg_feat_q1.mean(dim=1).unsqueeze(-1).unsqueeze(-1)  # [B, D, 1, 1]
        #     q1 = q1 + sg_feat_q1
        # elif self.combine_option == "cross-attn":
        #     q1 = self.cross_attn_q1(q1, sg_feat_q1)

        q0 = self.sam1(enc_feats[0], q1)
        bin_centers = self.bcp(q4)
        f = self.disp_head1(q0, bin_centers, 4)

        return f


class DispHead(nn.Module):
    def __init__(self, input_dim=100):
        super(DispHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 256, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, centers, scale):
        x = self.conv1(x)
        x = x.softmax(dim=1)
        x = torch.sum(x * centers, dim=1, keepdim=True)
        if scale > 1:
            x = upsample(x, scale_factor=scale)
        return x


def upsample(x, scale_factor=2, mode="bilinear", align_corners=False):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=align_corners)



def build_obj_tokens(imgs, enc_feats, obj_logits, obj_boxes, obj_projector, top_k=16, output_size=(7,7)):
    """
    Extract top-k object tokens from encoder feature maps and object predictions.
    Args:
        imgs: input image batch (B,C,H,W)
        enc_feats: encoder feature maps, assumed as a single tensor [B, C, H', W']
        obj_logits: object class logits [B, num_queries, num_classes]
        obj_boxes: normalized center-size boxes [B, num_queries, 4] (cx, cy, w, h)
        obj_projector: nn.Module to project pooled features
        top_k: number of top objects to use
        output_size: spatial output size for ROIAlign
    Returns:
        obj_embeddings: projected object features [B, top_k, D_proj]
    """
    B, num_queries, num_classes = obj_logits.shape
    device = obj_logits.device
    H, W = imgs.shape[-2:]
    obj_embeddings = []
    for b in range(B):
        # Get top-k indices per image
        probs = F.softmax(obj_logits[b, :, :-1], dim=-1).max(dim=-1).values  # ignore no-object class
        vals, inds = torch.topk(probs, k=top_k, dim=0)

        # Get boxes
        cx, cy, w, h = obj_boxes[b, inds].unbind(dim=1)
        x1 = (cx - w/2) * W
        y1 = (cy - h/2) * H
        x2 = (cx + w/2) * W
        y2 = (cy + h/2) * H
        rois = torch.stack([x1, y1, x2, y2], dim=1)

        # RoiAlign expects (image_idx, x1, y1, x2, y2)
        img_inds = torch.full((rois.size(0),), b, device=device, dtype=torch.float32).unsqueeze(1)
        rois_full = torch.cat([img_inds, rois], dim=1)

        # Align features
        pooled = roi_align(
            enc_feats, rois_full, output_size=output_size, spatial_scale=enc_feats.size(-1)/float(H), aligned=True
        )  # [top_k, C, output_size[0], output_size[1]]

        # Average pool
        pooled_mean = pooled.mean(dim=[2, 3])  # [top_k, C]

        # Project
        projected = obj_projector(pooled_mean)  # [top_k, D_proj]
        obj_embeddings.append(projected)

    return torch.stack(obj_embeddings, dim=0)  # [B, top_k, D_proj]

