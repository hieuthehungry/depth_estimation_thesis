# full_depth_scenegraph_pipeline.py
# Single-file pipeline:
#  - MobileNetV3 encoder (timm)
#  - FastDepth-style decoder -> init_depth
#  - YOLO-Nano detector wrapper (Ultralytics placeholder)
#  - ROIAlign on encoder features + init_depth for per-object features
#  - SceneGraphBuilder with GATConv (torch_geometric)
#  - Perceiver-style cross-attention fusion into decoder features
#
# Requirements:
#   pip install torch torchvision timm ultralytics
#   torch_geometric (see docs for correct wheel for your setup)
#
# Usage: run as script to smoke-test.

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision.ops import roi_align
import numpy as np

# GAT from PyG
try:
    from torch_geometric.nn import GATConv
except Exception as e:
    GATConv = None
    print("Warning: torch_geometric.GATConv not available. Install torch_geometric to use GAT.")

# Optional Ultralytics YOLO wrapper (replace if you use another detector)
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None

# ---------------------------
# Config
# ---------------------------
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# Blocks: Depthwise separable conv
# ---------------------------
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1, bias=False):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, k, padding=p, groups=in_ch, bias=bias)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)

# ---------------------------
# FastDepth-style Decoder
# ---------------------------
class FastDepthDecoder(nn.Module):
    def __init__(self, enc_channels, decoder_channels=256):
        super().__init__()
        self.enc_channels = enc_channels  # list low->high
        self.in_ch = enc_channels[-1]
        self.up_blocks = nn.ModuleList()
        for skip_ch in reversed(enc_channels[:-1]):
            block = nn.Sequential(
                nn.Conv2d(self.in_ch, decoder_channels, 1, bias=False),
                nn.BatchNorm2d(decoder_channels),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='nearest'),
                DepthwiseSeparableConv(decoder_channels, decoder_channels),
                DepthwiseSeparableConv(decoder_channels, decoder_channels),
                nn.Conv2d(decoder_channels, skip_ch, 1, bias=False),
                nn.BatchNorm2d(skip_ch),
                nn.ReLU(inplace=True),
            )
            self.up_blocks.append(block)
            self.in_ch = skip_ch
        self.smooth = nn.Sequential(
            DepthwiseSeparableConv(self.in_ch, self.in_ch),
            nn.Conv2d(self.in_ch, self.in_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        # final depth conv
        self.final_conv = nn.Conv2d(self.in_ch, 1, 3, padding=1)

    def forward(self, features, out_size=None):
        x = features[-1]
        for block, skip in zip(self.up_blocks, reversed(features[:-1])):
            x = block(x)
            x = x + skip
        x = self.smooth(x)
        depth_lowres = self.final_conv(x)  # (B,1,Hq,Wq)
        if out_size is not None:
            depth = F.interpolate(depth_lowres, size=out_size, mode='bilinear', align_corners=False)
            return depth, x  # return upsampled depth and decoder spatial features (before final conv smoothing)
        return depth_lowres, x

# ---------------------------
# YOLO-Nano wrapper (simple Ultralytics-based). Replace as needed.
# ---------------------------
class YOLONanoWrapper:
    def __init__(self, weights_path=None, device="cuda", conf=0.25, topk=32):
        self.device = device
        self.conf = conf
        self.topk = topk
        self.device = device
        if YOLO is None or weights_path is None:
            self.model = None
            print("YOLO not initialized: pass weights_path or install ultralytics.")
        else:
            self.model = YOLO(weights_path)
            self.model.to(self.device)

    def detect(self, imgs):
        """
        imgs: torch.Tensor (B,3,H,W) float 0..1 or list of numpy HWC uint8
        returns boxes_list, labels_list, scores_list
        """
        if self.model is None:
            # return empty lists
            if isinstance(imgs, torch.Tensor):
                B = imgs.shape[0]
            else:
                B = len(imgs)
            return [torch.zeros((0,4), device=self.device) for _ in range(B)], [torch.zeros((0,), dtype=torch.long, device=device) for _ in range(B)], [torch.zeros((0,), device=device) for _ in range(B)]
        # prepare numpy images
        if isinstance(imgs, torch.Tensor):
            arr = imgs.detach().cpu().clone()
            if arr.max() > 2:
                arr = arr / 255.0
            arr = arr.clamp(0,1)
            list_imgs = []
            for i in range(arr.shape[0]):
                im = (arr[i].permute(1,2,0).numpy()*255.0).astype(np.uint8)
                list_imgs.append(im)
        else:
            list_imgs = imgs
        results = self.model.predict(source=list_imgs, conf=self.conf, device=self.device, verbose=False)
        boxes_list, labels_list, scores_list = [], [], []
        for res in results:
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                boxes_list.append(torch.zeros((0,4), device=self.device))
                labels_list.append(torch.zeros((0,), dtype=torch.long, device=self.device))
                scores_list.append(torch.zeros((0,), device=self.device))
                continue
            xyxy = boxes.xyxy.cpu().float()
            confs = boxes.conf.cpu().float()
            cls = boxes.cls.cpu().long()
            # topk filtering
            if xyxy.shape[0] > self.topk:
                idx = torch.argsort(confs, descending=True)[:self.topk]
                xyxy = xyxy[idx].to(self.device)
                confs = confs[idx].to(self.device)
                cls = cls[idx].to(self.device)
            boxes_list.append(xyxy.to(self.device))
            labels_list.append(cls.to(self.device))
            scores_list.append(confs.to(self.device))
        return boxes_list, labels_list, scores_list

# ---------------------------
# SceneGraphBuilder with ROI depth & GAT update
# ---------------------------
from torch_geometric.nn import GATv2Conv

class SceneGraphBuilderGAT(nn.Module):
    def __init__(self, num_classes=80, roi_feat_dim=256, node_dim=256,
                 relation_dim=128, max_objects=16, gat_heads=4):
        super().__init__()
        self.num_classes = num_classes
        self.node_dim = node_dim
        self.max_objects = max_objects

        # Node feature submodules
        self.class_embed = nn.Embedding(num_classes, node_dim // 2)
        self.bbox_mlp = nn.Sequential(
            nn.Linear(4, node_dim // 4),
            nn.ReLU(),
            nn.Linear(node_dim // 4, node_dim // 4),
            nn.ReLU(),
        )
        self.roi_proj = nn.Sequential(
            nn.Linear(roi_feat_dim, node_dim // 4),
            nn.ReLU(),
        )
        self.depth_proj = nn.Sequential(
            nn.Linear(3, node_dim // 4),
            nn.ReLU(),
        )
        self.node_fuse = nn.Sequential(
            nn.Linear(node_dim // 2 + node_dim // 4 + node_dim // 4 + node_dim // 4, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim),
            nn.ReLU(),
        )

        # Edge feature projector (node_i, node_j, geometry, depth diff)
        self.edge_proj = nn.Sequential(
            nn.Linear(node_dim*2 + 4 + 1, relation_dim),  # +4 bbox diff, +1 depth diff
            nn.ReLU(),
            nn.Linear(relation_dim, relation_dim),
            nn.ReLU(),
        )

        # GATv2Conv with edge attributes
        self.gat = GATv2Conv(in_channels=node_dim, out_channels=node_dim // gat_heads,
                             heads=gat_heads, edge_dim=relation_dim, concat=True, add_self_loops=False)

    def forward(self, boxes_list, labels_list, roi_feats, depth_maps, image_sizes):
        device = roi_feats.device
        B = len(boxes_list)
        ptr = 0
        tokens = []
        valid_counts = []
        depth_stats_batch = []
        bbox_norm_batch = []

        for b in range(B):
            boxes = boxes_list[b].to(device)
            labels = labels_list[b].to(device)
            n = boxes.shape[0]
            v = min(n, self.max_objects)
            valid_counts.append(v)

            if n == 0:
                tokens.append(torch.zeros((self.max_objects, self.node_dim), device=device))
                depth_stats_batch.append(torch.zeros((self.max_objects, 3), device=device))
                bbox_norm_batch.append(torch.zeros((self.max_objects, 4), device=device))
                continue

            feats = roi_feats[ptr:ptr+n].to(device)
            ptr += n

            # ROI depth stats
            depth_map = depth_maps[b:b+1]  # (1,1,H,W)
            boxes_clamped = boxes.clone()
            Himg, Wimg = image_sizes[b]
            boxes_clamped[:, [0,2]] = boxes_clamped[:, [0,2]].clamp(0, Wimg-1)
            boxes_clamped[:, [1,3]] = boxes_clamped[:, [1,3]].clamp(0, Himg-1)
            pooled_depth = roi_align(depth_map, [boxes_clamped], output_size=ROI_POOL, spatial_scale=1.0, aligned=True)
            if pooled_depth.numel() == 0:
                depth_stats = torch.zeros((n,3), device=device)
            else:
                flat = pooled_depth.view(n, -1)
                depth_stats = torch.cat([
                    flat.mean(dim=1, keepdim=True),
                    flat.var(dim=1, keepdim=True),
                    flat.median(dim=1).values.unsqueeze(1)
                ], dim=1)

            cx = ((boxes[:,0] + boxes[:,2]) / 2.0) / Wimg
            cy = ((boxes[:,1] + boxes[:,3]) / 2.0) / Himg
            bw = ((boxes[:,2] - boxes[:,0]) / Wimg).clamp(min=1e-6)
            bh = ((boxes[:,3] - boxes[:,1]) / Himg).clamp(min=1e-6)
            bbox_norm = torch.stack([cx, cy, bw, bh], dim=1)

            cls_emb = self.class_embed(labels[:self.max_objects])
            bbox_emb = self.bbox_mlp(bbox_norm[:self.max_objects])
            roi_emb = self.roi_proj(feats[:self.max_objects])
            depth_emb = self.depth_proj(depth_stats[:self.max_objects])

            node_in = torch.cat([cls_emb, bbox_emb, roi_emb, depth_emb], dim=1)
            nodes = self.node_fuse(node_in)

            k = nodes.shape[0]
            if k < self.max_objects:
                pad_nodes = torch.zeros((self.max_objects - k, self.node_dim), device=device)
                pad_bbox = torch.zeros((self.max_objects - k, 4), device=device)
                pad_depth = torch.zeros((self.max_objects - k, 3), device=device)
                nodes = torch.cat([nodes, pad_nodes], dim=0)
                bbox_norm = torch.cat([bbox_norm, pad_bbox], dim=0)
                depth_stats = torch.cat([depth_stats, pad_depth], dim=0)

            tokens.append(nodes)
            bbox_norm_batch.append(bbox_norm)
            depth_stats_batch.append(depth_stats)

        scene_tokens = torch.stack(tokens, dim=0)
        valid_counts = torch.tensor(valid_counts, device=device)
        bbox_norm_batch = torch.stack(bbox_norm_batch, dim=0)
        depth_stats_batch = torch.stack(depth_stats_batch, dim=0)

        updated_tokens = []
        for b in range(B):
            v = valid_counts[b].item()
            if v <= 1:
                updated_tokens.append(scene_tokens[b])
                continue

            nodes = scene_tokens[b, :v]
            bbox_norm = bbox_norm_batch[b, :v]
            depth_means = depth_stats_batch[b, :v, 0]

            # Build edges fully-connected (no self-loop)
            src, dst = [], []
            edge_attr = []
            for i in range(v):
                for j in range(v):
                    if i == j:
                        continue
                    src.append(i)
                    dst.append(j)
                    # Edge features: node_i, node_j, bbox_diff, depth_diff
                    bbox_diff = bbox_norm[j] - bbox_norm[i]
                    depth_diff = (depth_means[j] - depth_means[i]).unsqueeze(0)
                    feat_ij = torch.cat([nodes[i], nodes[j], bbox_diff, depth_diff], dim=0)
                    edge_attr.append(feat_ij)
            edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
            edge_attr = torch.stack(edge_attr, dim=0)
            edge_emb = self.edge_proj(edge_attr)

            nodes_updated = self.gat(nodes, edge_index, edge_attr=edge_emb)
            padded = torch.zeros((self.max_objects, self.node_dim), device=device)
            padded[:v] = nodes_updated
            updated_tokens.append(padded)

        scene_tokens = torch.stack(updated_tokens, dim=0)
        return scene_tokens, valid_counts

# ---------------------------
# Perceiver-style cross-attention fusion (queries = spatial tokens from decoder, keys/vals = scene tokens)
# ---------------------------
class PerceiverFusion(nn.Module):
    def __init__(self, query_dim, token_dim, attn_dim=None, num_heads=8):
        super().__init__()
        self.attn_dim = attn_dim if attn_dim is not None else query_dim
        self.query_proj = nn.Linear(query_dim, self.attn_dim)
        self.k_proj = nn.Linear(token_dim, self.attn_dim)
        self.v_proj = nn.Linear(token_dim, self.attn_dim)
        self.mha = nn.MultiheadAttention(self.attn_dim, num_heads, batch_first=True)
        self.out = nn.Linear(self.attn_dim, query_dim)
        self.norm = nn.LayerNorm(query_dim)

    def forward(self, spatial_feats, scene_tokens):
        # spatial_feats: (B, Cq, H, W)
        B, Cq, H, W = spatial_feats.shape
        q = spatial_feats.flatten(2).permute(0,2,1)  # (B, HW, Cq)
        q_proj = self.query_proj(q)  # (B, HW, D)
        k = self.k_proj(scene_tokens)  # (B, M, D)
        v = self.v_proj(scene_tokens)  # (B, M, D)
        attn_out, _ = self.mha(q_proj, k, v)  # (B, HW, D)
        attn_out = self.out(attn_out)
        q = q + attn_out
        q = self.norm(q)
        q = q.permute(0,2,1).reshape(B, Cq, H, W)
        return q

# ---------------------------
# MLP-attention alternative (simple concat & MLP per spatial token)
# ---------------------------
class MLPFusion(nn.Module):
    def __init__(self, query_dim, token_dim, hidden=512):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)  # optional global pooling over tokens for cheap fuse
        self.mlp = nn.Sequential(
            nn.Linear(query_dim + token_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, query_dim),
        )
        self.norm = nn.LayerNorm(query_dim)

    def forward(self, spatial_feats, scene_tokens):
        B, Cq, H, W = spatial_feats.shape
        # collapse spatial to per-image avg feature and tile -> cheap fusion
        q_global = spatial_feats.mean(dim=(2,3))  # (B, Cq)
        # pool scene tokens to global token
        s_global = scene_tokens.mean(dim=1)  # (B, token_dim)
        cat = torch.cat([q_global, s_global], dim=1)  # (B, Cq+token_dim)
        delta = self.mlp(cat)  # (B, Cq)
        # broadcast to spatial
        delta_spatial = delta.view(B, Cq, 1, 1).expand(-1, -1, H, W)
        out = spatial_feats + delta_spatial
        # layernorm over channel dim per spatial location (apply per spatial vector)
        out_flat = out.flatten(2).permute(0,2,1)
        out_norm = self.norm(out_flat).permute(0,2,1).reshape(B, Cq, H, W)
        return out_norm

# ---------------------------
# Full model that ties everything
# ---------------------------
class DepthSceneGATModel(nn.Module):
    def __init__(self, encoder = 'mobilenetv3_small_100', yolo_weights=None, fusion_method= "perceiver", decoder_channels=256, roi_size = (7,7), num_classes = 80, node_dim = 256, relation_dim = 128, max_objects = 16, device="cuda"):
        super().__init__()
        self.device = device
        # Encoder (mobilenetv4)
        self.encoder = timm.create_model(encoder, pretrained=True, features_only=True).to(device)
        enc_ch = self.encoder.feature_info.channels()
        # FastDepth decoder
        self.decoder = FastDepthDecoder(enc_ch, decoder_channels=decoder_channels).to(device)
        # YOLO wrapper
        self.detector = YOLONanoWrapper(weights_path=yolo_weights, device=device)
        # roi pooling proj dims for deepest encoder feature
        deepest_ch = enc_ch[-1]
        self.roi_pool_size = roi_size
        roi_flat = deepest_ch * roi_size[0] * roi_size[1]
        self.roi_proj = nn.Sequential(nn.Linear(roi_flat, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU()).to(device)
        # scene graph builder with GAT
        self.scene_builder = SceneGraphBuilderGAT(num_classes=num_classes, roi_feat_dim=256, node_dim=node_dim, relation_dim=relation_dim, max_objects=max_objects).to(device)
        # fusion
        query_dim = self.decoder.in_ch  # decoder's current in_ch after last up -> matches last skip channel
        if fusion_method == "perceiver":
            self.fuser = PerceiverFusion(query_dim=query_dim, token_dim=node_dim, num_heads=8).to(device)
        else:
            self.fuser = MLPFusion(query_dim=query_dim, token_dim=node_dim).to(device)
        # final refine conv to depth
        self.final_conv = nn.Conv2d(query_dim, 1, 3, padding=1).to(device)
        self.fusion_method = fusion_method

    def extract_roi_feats(self, deepest_feat, boxes_list, image_sizes):
        """
        deepest_feat: (B, C, Hf, Wf) deepest encoder map
        boxes_list: list of (n_i,4) in image pixel coords
        returns: roi_proj_feats (sum_n, 256)
        """
        B, C, Hf, Wf = deepest_feat.shape
        device = deepest_feat.device
        # make boxes_for_roi list scaled to feature map coords
        all_boxes = []
        for b in range(len(boxes_list)):
            bboxes = boxes_list[b]
            if bboxes.numel() == 0:
                all_boxes.append(torch.zeros((0,4), device=device))
                continue
            # scale boxes from image to feature map
            Himg, Wimg = image_sizes[b]
            scale_x = Wf / Wimg
            scale_y = Hf / Himg
            bscaled = bboxes.clone()
            bscaled[:, [0,2]] = bscaled[:, [0,2]] * scale_x
            bscaled[:, [1,3]] = bscaled[:, [1,3]] * scale_y
            all_boxes.append(bscaled.to(device))
        rois = roi_align(deepest_feat, all_boxes, output_size=self.roi_pool_size, spatial_scale=1.0, aligned=True)  # (sum_n, C, ph, pw)
        if rois.numel() == 0:
            return torch.zeros((0, self.roi_proj[0].in_features), device=device)
        rois_flat = rois.view(rois.shape[0], -1)
        roi_proj = self.roi_proj(rois_flat)
        return roi_proj

    def forward(self, images):
        """
        images: torch.Tensor (B,3,H,W) float 0..1 or uint8 0..255
        returns refined_depth (B,1,H,W), init_depth (B,1,H,W), meta
        """
        # normalize images for encoder
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        B, C, H, W = images.shape
        mean = torch.tensor([0.485,0.456,0.406], device=self.device).view(1,3,1,1)
        std  = torch.tensor([0.229,0.224,0.225], device=self.device).view(1,3,1,1)
        inp = (images.to(self.device) - mean) / std

        # encoder features
        features = self.encoder(inp)  # list low->high
        deepest = features[-1]  # (B, C, Hf, Wf)

        # initial depth via decoder (upsampled to input size), also get spatial decoder feature map
        init_depth, decoder_spatial = self.decoder(features, out_size=(H,W))  # init_depth: (B,1,H,W), decoder_spatial: (B, Cq, Hq, Wq)

        # detect objects
        boxes_list, labels_list, scores_list = self.detector.detect(images)
        boxes_list = [b.to(self.device) for b in boxes_list]
        labels_list = [l.to(self.device) for l in labels_list]

        # roi features from deepest map
        roi_feats = self.extract_roi_feats(deepest, boxes_list, [(H,W)]*B)  # (sum_n, 256)
        # scene graph builder: includes roi depth pooling internally
        scene_tokens, valid_counts = self.scene_builder(boxes_list, labels_list, roi_feats, init_depth.detach(), [(H,W)]*B)

        # fuse scene tokens into decoder spatial features
        if self.fusion_method == "perceiver":
            fused_spatial = self.fuser(decoder_spatial, scene_tokens)  # (B, Cq, Hq, Wq)
        else:
            fused_spatial = self.fuser(decoder_spatial, scene_tokens)

        refined_lowres = self.final_conv(fused_spatial)  # (B,1,Hq,Wq)
        refined_depth = F.interpolate(refined_lowres, size=(H,W), mode='bilinear', align_corners=False)
        refined_depth = torch.sigmoid(refined_depth)  # optional activation (scale to 0..1)
        return refined_depth, init_depth, {'boxes': boxes_list, 'labels': labels_list, 'scores': scores_list, 'valid_counts': valid_counts}

# ---------------------------
# Smoke test
# ---------------------------
if __name__ == "__main__":
    device = "cuda"
    model = DepthSceneGATModel(yolo_weights="yolo11n.pt", fusion_method="perceiver", device=device).to(device)
    model.eval()
    B = 2; H = 320; W = 960
    dummy = torch.randint(0, 255, (B,3,H,W), dtype=torch.uint8).to(device)
    with torch.no_grad():
        refined, init_d, meta = model(dummy)
    print("init depth:", init_d.shape, "refined depth:", refined.shape)
    print("meta keys:", meta.keys())