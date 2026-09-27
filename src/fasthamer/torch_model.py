"""Self-contained PyTorch implementation of the HaMeR network for the torch backend.

ViT-H backbone (ViTPose flavour) + MANO transformer-decoder head + MANO linear blend
skinning, with the same module names as the reference implementation so the official
`hamer.ckpt` state dict loads directly. No timm / einops / smplx / hamer imports: torch
and numpy only.

The MANO buffers (template, blend shapes, joint regressor, skinning weights, faces) are
read from the checkpoint itself — the released HaMeR checkpoint embeds them — so no
MANO pickle or chumpy is needed at conversion time. MANO is license-gated
(https://mano.is.tue.mpg.de); see assets.py for the acknowledgment flow.

Backbone reference: ViTPose (Apache-2.0), as vendored in HaMeR (MIT).
"""
from functools import partial
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- constants
BACKBONE_INPUT = (256, 192)   # (H, W) the backbone sees: the 256x256 crop minus 32 px each side
PATCH_SIZE = 16
EMBED_DIM = 1280
DEPTH = 32
NUM_HEADS = 16
PATCH_PADDING = 2             # HaMeR: 4 + 2 * (ratio // 2 - 1) with ratio = 1
HEAD_DIM = 1024
HEAD_DEPTH = 6
HEAD_HEADS = 8
HEAD_DIM_HEAD = 64
HEAD_MLP_DIM = 1024
NUM_HAND_JOINTS = 15
NPOSE = 6 * (NUM_HAND_JOINTS + 1)
NUM_BETAS = 10
BUNDLE_FORMAT = 1

# state-dict keys of the checkpoint that the runtime model needs
MANO_BUFFERS = ("v_template", "shapedirs", "posedirs", "J_regressor", "parents",
                "lbs_weights", "extra_joints_idxs", "joint_map", "faces_tensor")


# --------------------------------------------------------------------------- ViT backbone
class Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x = F.scaled_dot_product_attention(q, k, v)  # == softmax(q k^T * scale) v
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, norm_layer):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class PatchEmbed(nn.Module):
    def __init__(self, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=PATCH_SIZE, stride=PATCH_SIZE, padding=PATCH_PADDING)

    def forward(self, x):
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        return x.flatten(2).transpose(1, 2), (Hp, Wp)


def _patch_grid(hw: Tuple[int, int]) -> Tuple[int, int]:
    h, w = hw
    return ((h + 2 * PATCH_PADDING - PATCH_SIZE) // PATCH_SIZE + 1,
            (w + 2 * PATCH_PADDING - PATCH_SIZE) // PATCH_SIZE + 1)


class ViT(nn.Module):
    """ViTPose-style ViT-H with a learned absolute position embedding (cls slot + patches)."""

    def __init__(self, embed_dim=EMBED_DIM, depth=DEPTH, num_heads=NUM_HEADS, mlp_ratio=4.0):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.patch_embed = PatchEmbed(3, embed_dim)
        self.grid = _patch_grid(BACKBONE_INPUT)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.grid[0] * self.grid[1] + 1, embed_dim))
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, norm_layer) for _ in range(depth)])
        self.last_norm = norm_layer(embed_dim)

    def pos_embed_for(self, hp: int, wp: int) -> torch.Tensor:
        """Position embedding for an hp x wp token grid; bicubic-resampled when the input
        resolution differs from the training one (same as HaMeR's get_abs_pos)."""
        cls, pos = self.pos_embed[:, :1], self.pos_embed[:, 1:]
        if (hp, wp) != self.grid:
            pos = pos.reshape(1, self.grid[0], self.grid[1], -1).permute(0, 3, 1, 2)
            pos = F.interpolate(pos.float(), size=(hp, wp), mode="bicubic", align_corners=False).to(self.pos_embed.dtype)
            pos = pos.permute(0, 2, 3, 1).reshape(1, hp * wp, -1)
        return pos + cls

    def forward(self, x):
        B = x.shape[0]
        x, (hp, wp) = self.patch_embed(x)
        x = x + self.pos_embed_for(hp, wp)
        for blk in self.blocks:
            x = blk(x)
        x = self.last_norm(x)
        return x.permute(0, 2, 1).reshape(B, -1, hp, wp)


# --------------------------------------------------------------------------- MANO head
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kw):
        return self.fn(self.norm(x), **kw)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(0.0), nn.Linear(hidden, dim), nn.Dropout(0.0))

    def forward(self, x):
        return self.net(x)


def _split_heads(t, heads):
    b, n, hd = t.shape
    return t.reshape(b, n, heads, hd // heads).transpose(1, 2)


def _merge_heads(t):
    b, h, n, d = t.shape
    return t.transpose(1, 2).reshape(b, n, h * d)


class SelfAttention(nn.Module):
    def __init__(self, dim, heads, dim_head):
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(0.0))

    def forward(self, x):
        q, k, v = (_split_heads(t, self.heads) for t in self.to_qkv(x).chunk(3, dim=-1))
        attn = (q @ k.transpose(-1, -2) * self.scale).softmax(dim=-1)
        return self.to_out(_merge_heads(attn @ v))


class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim, heads, dim_head):
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_kv = nn.Linear(context_dim, inner * 2, bias=False)
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(0.0))

    def forward(self, x, context=None):
        k, v = (_split_heads(t, self.heads) for t in self.to_kv(context).chunk(2, dim=-1))
        q = _split_heads(self.to_q(x), self.heads)
        attn = (q @ k.transpose(-1, -2) * self.scale).softmax(dim=-1)
        return self.to_out(_merge_heads(attn @ v))


class TransformerCrossAttn(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, context_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([PreNorm(dim, SelfAttention(dim, heads, dim_head)),
                           PreNorm(dim, CrossAttention(dim, context_dim, heads, dim_head)),
                           PreNorm(dim, FeedForward(dim, mlp_dim))])
            for _ in range(depth)])

    def forward(self, x, context):
        for sa, ca, ff in self.layers:
            x = sa(x) + x
            x = ca(x, context=context) + x
            x = ff(x) + x
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, num_tokens, token_dim, dim, depth, heads, dim_head, mlp_dim, context_dim):
        super().__init__()
        self.to_token_embedding = nn.Linear(token_dim, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, dim))
        self.transformer = TransformerCrossAttn(dim, depth, heads, dim_head, mlp_dim, context_dim)

    def forward(self, inp, context):
        x = self.to_token_embedding(inp)
        x = x + self.pos_embedding[:, :x.shape[1]]
        return self.transformer(x, context)


def rot6d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    """(B, 6) -> (B, 3, 3), Zhou et al. continuous 6D representation (HaMeR convention)."""
    x = x.reshape(-1, 2, 3).permute(0, 2, 1).contiguous()
    a1, a2 = x[:, :, 0], x[:, :, 1]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


class MANOTransformerDecoderHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = TransformerDecoder(num_tokens=1, token_dim=1, dim=HEAD_DIM, depth=HEAD_DEPTH, heads=HEAD_HEADS,
                                              dim_head=HEAD_DIM_HEAD, mlp_dim=HEAD_MLP_DIM, context_dim=EMBED_DIM)
        self.decpose = nn.Linear(HEAD_DIM, NPOSE)
        self.decshape = nn.Linear(HEAD_DIM, NUM_BETAS)
        self.deccam = nn.Linear(HEAD_DIM, 3)
        self.register_buffer("init_hand_pose", torch.zeros(1, NPOSE))
        self.register_buffer("init_betas", torch.zeros(1, NUM_BETAS))
        self.register_buffer("init_cam", torch.zeros(1, 3))

    def forward(self, feats):
        B = feats.shape[0]
        context = feats.flatten(2).transpose(1, 2)              # b c h w -> b (h w) c
        token = torch.zeros(B, 1, 1, device=feats.device, dtype=feats.dtype)
        out = self.transformer(token, context).squeeze(1)      # single IEF iteration
        pose6d = self.decpose(out) + self.init_hand_pose
        betas = self.decshape(out) + self.init_betas
        cam = self.deccam(out) + self.init_cam
        rotmats = rot6d_to_rotmat(pose6d.float()).view(B, NUM_HAND_JOINTS + 1, 3, 3)
        return rotmats, betas.float(), cam.float()


# --------------------------------------------------------------------------- MANO skinning
def _transform_mat(R, t):
    return torch.cat([F.pad(R, [0, 0, 0, 1]), F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(rot_mats, joints, parents):
    joints = joints.unsqueeze(-1)
    rel = joints.clone()
    rel[:, 1:] -= joints[:, parents[1:]]
    T = _transform_mat(rot_mats.reshape(-1, 3, 3), rel.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)
    chain = [T[:, 0]]
    for i in range(1, parents.shape[0]):
        chain.append(chain[parents[i]] @ T[:, i])
    transforms = torch.stack(chain, dim=1)
    posed = transforms[:, :, :3, 3]
    jh = F.pad(joints, [0, 0, 0, 1])
    rel_transforms = transforms - F.pad(transforms @ jh, [3, 0, 0, 0, 0, 0, 0, 0])
    return posed, rel_transforms


class MANOLayer(nn.Module):
    """MANO right hand, rotation-matrix pose input (smplx MANOLayer semantics, pose2rot=False),
    21 output joints in the HaMeR/OpenPose order (wrist, thumb..pinky, 4 each)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("v_template", torch.zeros(778, 3))
        self.register_buffer("shapedirs", torch.zeros(778, 3, NUM_BETAS))
        self.register_buffer("posedirs", torch.zeros(9 * NUM_HAND_JOINTS, 778 * 3))
        self.register_buffer("J_regressor", torch.zeros(16, 778))
        self.register_buffer("parents", torch.zeros(16, dtype=torch.long))
        self.register_buffer("lbs_weights", torch.zeros(778, 16))
        self.register_buffer("extra_joints_idxs", torch.zeros(5, dtype=torch.long))
        self.register_buffer("joint_map", torch.zeros(21, dtype=torch.long))
        self.register_buffer("faces_tensor", torch.zeros(1538, 3, dtype=torch.long))

    def forward(self, betas, pose_rotmats):
        """betas (B, 10); pose_rotmats (B, 16, 3, 3) = global orient + 15 joints. Returns (verts, joints)."""
        B = betas.shape[0]
        v_shaped = self.v_template + torch.einsum("bl,mkl->bmk", betas, self.shapedirs)
        J = torch.einsum("bik,ji->bjk", v_shaped, self.J_regressor)
        ident = torch.eye(3, dtype=betas.dtype, device=betas.device)
        pose_feature = (pose_rotmats[:, 1:] - ident).reshape(B, -1)
        v_posed = v_shaped + (pose_feature @ self.posedirs).view(B, -1, 3)
        J_posed, A = batch_rigid_transform(pose_rotmats, J, self.parents)
        T = (self.lbs_weights.unsqueeze(0).expand(B, -1, -1) @ A.view(B, 16, 16)).view(B, -1, 4, 4)
        v_h = torch.cat([v_posed, torch.ones(B, v_posed.shape[1], 1, dtype=betas.dtype, device=betas.device)], dim=2)
        verts = (T @ v_h.unsqueeze(-1))[:, :, :3, 0]
        joints = torch.cat([J_posed, verts[:, self.extra_joints_idxs]], dim=1)[:, self.joint_map]
        return verts, joints


# --------------------------------------------------------------------------- full model
class HamerTorch(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ViT()
        self.mano_head = MANOTransformerDecoderHead()
        self.mano = MANOLayer()

    @torch.no_grad()
    def forward(self, x: torch.Tensor, return_feats: bool = False) -> Dict[str, torch.Tensor]:
        """x: (B, 3, H, W) normalized backbone input (H, W = 256, 192 for the reference model).
        Returns cam (B,3), global_orient (B,3,3), hand_pose (B,15,3,3), betas (B,10), vertices (B,778,3),
        keypoints3d (B,21,3) — all float32, hand-centered (add the camera translation downstream)."""
        feats = self.backbone(x.to(self.compute_dtype)).float()
        rotmats, betas, cam = self.mano_head(feats)
        verts, joints = self.mano(betas, rotmats)
        out = {"cam": cam, "global_orient": rotmats[:, 0], "hand_pose": rotmats[:, 1:], "betas": betas,
                "vertices": verts, "keypoints3d": joints}
        if return_feats:
            out["feats"] = feats       # (B, 1280, 16, 12) ViT token map, for heads on the backbone
        return out

    def set_compute_dtype(self, dtype: torch.dtype):
        """Run the ViT-H backbone in `dtype` (float16 on CUDA halves memory and doubles speed at
        fp16 noise, like the CoreML/ANE build). The small MANO head and the skinning always run
        in float32: it costs nothing and keeps the fp16 error at the ~1 mm level."""
        self.backbone.to(dtype)
        self.mano_head.float()
        self.mano.float()
        self._compute_dtype = dtype
        return self

    @property
    def compute_dtype(self) -> torch.dtype:
        return getattr(self, "_compute_dtype", torch.float32)

    @property
    def faces(self) -> np.ndarray:
        return self.mano.faces_tensor.cpu().numpy().astype(np.int32)


# --------------------------------------------------------------------------- weights
def _runtime_keys(model: nn.Module):
    return set(model.state_dict().keys())


def state_dict_from_hamer_checkpoint(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Keep the backbone, MANO head and MANO buffers of a hamer.ckpt state dict; drop the GAN
    discriminator and training-only buffers."""
    keep = {}
    wanted = _runtime_keys(HamerTorch())
    for k, v in sd.items():
        if k in wanted:
            keep[k] = v
    missing = wanted - set(keep)
    if missing:
        raise ValueError(f"checkpoint is missing {len(missing)} expected tensors, e.g. {sorted(missing)[:5]}")
    return keep


def load_hamer_checkpoint(path: str) -> Dict[str, torch.Tensor]:
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    return state_dict_from_hamer_checkpoint(sd)


def convert_hamer_checkpoint(ckpt_path: str, out_dir: str, dtype: str = "float16") -> str:
    """Build a fasthamer torch bundle (hamer_torch.pt + mano_faces.npy) from the official hamer.ckpt.
    Network weights are stored in `dtype` (float16 by default: 1.3 GB instead of 2.7 GB); the MANO
    buffers stay float32. Returns the bundle path."""
    import os
    from .assets import TORCH_MODEL_NAME, FACES_NAME

    sd = load_hamer_checkpoint(ckpt_path)
    store = getattr(torch, dtype)
    out = {}
    for k, v in sd.items():
        if k.startswith("mano.") or not v.is_floating_point():
            out[k] = v.clone()
        else:
            out[k] = v.to(store)
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"format": BUNDLE_FORMAT, "source": os.path.basename(ckpt_path), "state_dict": out,
                "backbone_input": list(BACKBONE_INPUT)}, os.path.join(out_dir, TORCH_MODEL_NAME))
    np.save(os.path.join(out_dir, FACES_NAME), sd["mano.faces_tensor"].numpy().astype(np.int32))
    return out_dir


def build_model(bundle_or_ckpt: str, device: Optional[str] = None, dtype: Optional[str] = None) -> HamerTorch:
    """Instantiate HamerTorch from a torch bundle directory / hamer_torch.pt file / hamer.ckpt."""
    import os
    from .assets import TORCH_MODEL_NAME

    path = bundle_or_ckpt
    if os.path.isdir(path):
        path = os.path.join(path, TORCH_MODEL_NAME)
    if os.path.basename(path) == TORCH_MODEL_NAME:
        blob = torch.load(path, map_location="cpu", weights_only=True)
        if blob.get("format") != BUNDLE_FORMAT:
            raise ValueError(f"unsupported fasthamer torch bundle format {blob.get('format')}")
        sd = blob["state_dict"]
    else:
        sd = load_hamer_checkpoint(path)
    model = HamerTorch()
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()}, strict=True)
    model.eval()
    if device is None or device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    if dtype is None or dtype == "auto":
        dtype = default_dtype(device)
    model.set_compute_dtype(getattr(torch, dtype))
    return model


def default_dtype(device) -> str:
    """float16 on CUDA GPUs with tensor cores (compute capability >= 7.0: Volta and newer), where
    it halves memory and roughly doubles throughput at fp16 noise; float32 on older GPUs (Pascal
    fp16 is slower than fp32) and on CPU/MPS."""
    dev = str(device)
    if dev.startswith("cuda") and torch.cuda.is_available():
        idx = int(dev.split(":")[1]) if ":" in dev else torch.cuda.current_device()
        major, _ = torch.cuda.get_device_capability(idx)
        return "float16" if major >= 7 else "float32"
    return "float32"
