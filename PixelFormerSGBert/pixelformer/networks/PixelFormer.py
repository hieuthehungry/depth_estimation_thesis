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

def match_box_indices(boxes, reference_boxes, iou_threshold=0.9):
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

def extract_scene_graph_edges(sub_boxes, obj_boxes, rel_logits, pred_boxes, iou_threshold=0.9):
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

from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch
from .obj_and_rel import REL_CLASSES 

class BertRelationEncoder(nn.Module):
    def __init__(self, rel_classes: list, pretrained_model='bert-base-uncased', out_dim=256):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.linear = nn.Linear(self.bert.config.hidden_size, out_dim)
        self.rel_tokens = rel_classes  # list of relation strings
        # print(self.rel_tokens)
        self.rel_embeddings = self._encode_relations(self.rel_tokens)

    def to(self, *args, **kwargs):
        """Ensure BERT and buffers move with the module"""
        super().to(*args, **kwargs)
        device = kwargs.get('device', args[0] if args else None)
        if device is not None:
            self.bert = self.bert.to(device)
            if self.rel_embeddings is not None:
                self.rel_embeddings = self.rel_embeddings.to(device)
        return self

    def _encode_relations(self, rel_texts):
        with torch.no_grad():
            tokens = self.tokenizer(rel_texts, padding=True, return_tensors='pt')
            outputs = self.bert(**tokens)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        return self.linear(cls_embeddings)  # [num_rel, out_dim]

    def forward(self, rel_type_ids):
        # rel_type_ids: (num_edges,) integer tensor
        rel_type_ids = rel_type_ids.to(self.rel_embeddings.device)
        return self.rel_embeddings[rel_type_ids]


class SceneGraphEncoder(nn.Module):
    def __init__(self, node_dim, out_dim, rel_classes=REL_CLASSES):
        super().__init__()
        self.embed_rel = BertRelationEncoder(rel_classes, out_dim=out_dim)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embed_rel.to(device)
        self.gat = GATConv(node_dim, out_dim, edge_dim=node_dim)

    def forward(self, pred_logits, rel_logits, sub_boxes, obj_boxes, pred_boxes):
        B, N, C = pred_logits.shape
        x = pred_logits.softmax(dim=-1)
        edge_indices, edge_types = extract_scene_graph_edges(sub_boxes, obj_boxes, rel_logits, pred_boxes)

        outputs = []
        # print(self.embed_rel.num_embeddings)
        for b in range(B):
            
            assert torch.all((edge_types[b] >= 0) & (edge_types[b] < self.embed_rel.rel_embeddings.shape[0])), f"Invalid rel class ID: {edge_types[b]}"
            rel_embed = self.embed_rel(edge_types[b])
            print("============================")
            print(x[b].shape)
            print(edge_indices[b].shape)
            print(rel_embed.shape)
            print("============================")
            out = self.gat(x[b], edge_indices[b], rel_embed)
            outputs.append(out)
        return torch.stack(outputs, dim=0)




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

class ObjTokenProjector(nn.Module):
    def __init__(self, num_tokens, token_dim=256, embed_dim=512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(token_dim, embed_dim),
            nn.GELU(),
        )
    def forward(self, obj_tokens):
        # obj_tokens: [B, num_tokens, token_dim]
        return self.proj(obj_tokens)  # [B, num_tokens, embed_dim]


import torch
import torch.nn.functional as F
from torchvision.ops import roi_align

class ObjTokenProjector(nn.Module):
    def __init__(self, num_tokens, token_dim=256, embed_dim=512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(token_dim, embed_dim),
            nn.GELU(),
        )
    def forward(self, obj_tokens):
        # obj_tokens: [B, num_tokens, token_dim]
        return self.proj(obj_tokens)  # [B, num_tokens, embed_dim]


class PixelFormerSG(nn.Module):

    def __init__(self, version=None, inv_depth=False, pretrained=None, 
                    frozen_stages=-1, min_depth=0.1, max_depth=100.0, node_dim=152, out_dim=512, use_roi_align=False, **kwargs):
        super().__init__()

        self.inv_depth = inv_depth
        self.with_auxiliary_head = False
        self.with_neck = False
        self.use_roi_align=use_roi_align

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
        self.obj_projector = ObjTokenProjector(num_tokens = 100, token_dim=in_channels[-1], embed_dim=embed_dim)
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
        self.sg_encoder = SceneGraphEncoder(node_dim=node_dim, out_dim=out_dim, rel_classes=REL_CLASSES)
        # print(self.sg_encoder.device)
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
    
    def build_obj_tokens(self, imgs, enc_feats, obj_logits, obj_boxes, top_k=8, output_size=(7,7)):
        B, num_queries, num_classes = obj_logits.shape
        device = obj_logits.device
        H, W = imgs.shape[-2:]
        obj_embeddings = []
        for b in range(B):
            # Get top-k indices by max object class score
            probs = F.softmax(obj_logits[b, :, :-1], dim=-1).max(dim=-1).values
            vals, inds = torch.topk(probs, k=top_k, dim=0)

            # Convert normalized boxes to pixel coords
            cx, cy, w, h = obj_boxes[b, inds].unbind(dim=1)
            x1 = (cx - w/2) * W
            y1 = (cy - h/2) * H
            x2 = (cx + w/2) * W
            y2 = (cy + h/2) * H
            rois = torch.stack([x1, y1, x2, y2], dim=1)

            # Prepend image index
            img_inds = torch.full((rois.size(0), 1), b, device=device, dtype=torch.float32)
            rois_full = torch.cat([img_inds, rois], dim=1)

            # ROI Align
            scale = enc_feats[-1].size(-1) / float(H)
            pooled = roi_align(
                enc_feats[-1], rois_full, output_size=output_size,
                spatial_scale=scale, aligned=True
            )  # [top_k, C, output_size[0], output_size[1]]

            # Average spatial dims
            pooled_mean = pooled.mean(dim=[2, 3])  # [top_k, C]

            # Project to token space
            projected = self.obj_projector(pooled_mean)  # [top_k, D_proj]
            obj_embeddings.append(projected)

        return torch.stack(obj_embeddings, dim=0)  # [B, top_k, D_proj]
    
    def forward(self, imgs, scene_graph):

        enc_feats = self.backbone(imgs)
        if self.with_neck:
            enc_feats = self.neck(enc_feats)

        q4 = self.decoder(enc_feats)

        # NEW: condition q4 with object tokens
        # 2. Conditionally incorporate object tokens

        if self.use_roi_align and 'obj_logits' in scene_graph and 'obj_boxes' in scene_graph:
            obj_tokens = self.build_obj_tokens(imgs, enc_feats, scene_graph['obj_logits'], scene_graph['obj_boxes'],
                                                p_k=8, output_size=(7,7))  # [B, top_k, D_proj]
            obj_mean = obj_tokens.mean(dim=1)[:, :, None, None]  # [B, D_proj, 1, 1]
            q4 = q4 + obj_mean

        elif not self.use_roi_align and all(k in scene_graph for k in ['pred_logits', 'rel_logits', 'sub_boxes', 'obj_boxes', 'pred_boxes']):
            """pred_logits, rel_logits, sub_boxes, obj_boxes, pred_boxes"""
            sg_global = self.sg_encoder(scene_graph['pred_logits'], scene_graph['rel_logits'],
                                        scene_graph['sub_boxes'], scene_graph['obj_boxes'],
                                        scene_graph['pred_boxes'])
            sg_global = sg_global.mean(dim=1).unsqueeze(-1).unsqueeze(-1)
            q4 = q4 + sg_global

        q3 = self.sam4(enc_feats[3], q4)
        q3 = nn.PixelShuffle(2)(q3)
        q2 = self.sam3(enc_feats[2], q3)
        q2 = nn.PixelShuffle(2)(q2)
        q1 = self.sam2(enc_feats[1], q2)
        q1 = nn.PixelShuffle(2)(q1)
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

