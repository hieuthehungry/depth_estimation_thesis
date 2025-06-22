import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_transformer import SwinTransformer
from .PQI import PSP
from .SAM import SAM
########################################################################################################################

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

class PixelFormerSG(nn.Module):

    def __init__(self, version=None, inv_depth=False, pretrained=None, 
                    frozen_stages=-1, min_depth=0.1, max_depth=100.0, **kwargs):
        super().__init__()

        self.inv_depth = inv_depth
        self.with_auxiliary_head = False
        self.with_neck = False

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
            scale = enc_feats.size(-1) / float(H)
            pooled = roi_align(
                enc_feats, rois_full, output_size=output_size,
                spatial_scale=scale, aligned=True
            )  # [top_k, C, output_size[0], output_size[1]]

            # Average spatial dims
            pooled_mean = pooled.mean(dim=[2, 3])  # [top_k, C]

            # Project to token space
            projected = self.obj_projector(pooled_mean)  # [top_k, D_proj]
            obj_embeddings.append(projected)

        return torch.stack(obj_embeddings, dim=0)  # [B, top_k, D_proj]
    
    def forward(self, imgs, obj_logits=None, obj_boxes=None):

        enc_feats = self.backbone(imgs)
        if self.with_neck:
            enc_feats = self.neck(enc_feats)

        q4 = self.decoder(enc_feats)

        # NEW: condition q4 with object tokens
        # 2. Conditionally incorporate object tokens
        if obj_logits is not None and obj_boxes is not None:
        # Build object tokens
            obj_tokens = self.build_obj_tokens(
            imgs, enc_feats, obj_logits, obj_boxes, top_k=8, output_size=(7,7)
        )                                     # [B, top_k, D_proj]

        # Mean pooling across tokens
            obj_mean = obj_tokens.mean(dim=1)[:, :, None, None]  # [B, D_proj, 1, 1]
            q4 = q4 + obj_mean

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

