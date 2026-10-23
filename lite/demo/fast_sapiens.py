"""General fast inference for Sapiens-Lite dense heads (exp 007, attempt/general): body-part segmentation (seg) checkpoints
0.3b / 0.6b / 1b / 2b from the upstream torchscript files; any batch size, any input size (multiple of 16; positional
embedding resized bicubic like mmpretrain for sizes other than 1024x768), nothing compiled or tuned per size.

    from fast_sapiens import FastSapiensSeg
    m = FastSapiensSeg("sapiens_1b_goliath_..._torchscript.pt2", attn="sdpa", gacc=False)   # friendly
    logits = m(x)            # x [B,3,H,W] normalised like the lite demo (mean 123.5/116.5/103.5, std 58.5/57/57.5)

Changes vs the upstream torchscript (fp32; TF32 convs): network math unchanged, precision / kernels only:
- weights in fp16; ViT blocks with fused SDPA flash (attn="sdpa", exportable) or fastbb's prebuilt head_dim-64 flash
  kernel (attn="fh": fp32 QK^T, fp16 PV); LayerNorm statistics in fp32 (inside F.layer_norm)
- optional fp16-accumulation GEMMs (`gacc`) and convs / deconvs (`hacc`, fastbb.cudnn_conv cuDNN graphs, precompiled plans)
- decoder (3 deconv + 3 1x1 conv, InstanceNorm + SiLU) in fp16 channels-last; InstanceNorm statistics in fp32;
  final 1x1 classifier output in fp32
"""
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F


@contextlib.contextmanager
def fp16_accumulate(enabled):
    old = torch.backends.cuda.matmul.allow_fp16_accumulation
    torch.backends.cuda.matmul.allow_fp16_accumulation = bool(enabled)
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_fp16_accumulation = old


def _inorm_silu(x):
    """InstanceNorm2d (no affine, eps 1e-5) + SiLU, in place on the fresh conv output x (fp16, channels-last kept);
    statistics via var_mean (fp32 accumulation inside), no fp32 copy of the map"""
    v, m = torch.var_mean(x, dim=(2, 3), keepdim=True, unbiased=False)
    a = (v.float() + 1e-5).rsqrt()
    b = (-m.float() * a).half()
    return F.silu(torch.addcmul(b, x, a.half(), out=x), inplace=True)  # (x - m) * a in one pass, then SiLU


class FastSapiensSeg(nn.Module):
    def __init__(self, ts_path, attn="sdpa", gacc=False, hacc=False, device="cuda"):
        super().__init__()
        sd = {k: v for k, v in torch.jit.load(ts_path, map_location="cpu").state_dict().items()}
        h = lambda t: t.to(device, torch.float16)  # noqa: E731
        self.attn, self.gacc, self.hacc = attn, gacc, hacc
        if hacc:  # fp16-accumulating cuDNN conv / deconv graphs (fastbb.cudnn_conv: precompiled plans, no per-size tuning)
            from fastbb.cudnn_conv import conv_half_acc, deconv_half_acc
            self._conv = lambda x, w, b, stride=1: conv_half_acc(x, w, b, [0, 0], [stride, stride])
            self._deconv = lambda x, w: deconv_half_acc(x, w, [1, 1], [2, 2])
        else:
            self._conv = lambda x, w, b, stride=1: F.conv2d(x, w, b, stride=stride)
            self._deconv = lambda x, w: F.conv_transpose2d(x, w, stride=2, padding=1)
        self.pos = sd["backbone.pos_embed"].to(device)  # [1, P, C] fp32
        self.C = C = self.pos.shape[-1]
        self.nh = C // 64
        self.pw, self.pb = h(sd["backbone.patch_embed.projection.weight"]), h(sd["backbone.patch_embed.projection.bias"])
        L = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("backbone.layers."))
        self.blocks = []
        for i in range(L):
            p = f"backbone.layers.{i}."
            self.blocks.append({n: h(sd[p + k]) for n, k in [
                ("ln1w", "ln1.weight"), ("ln1b", "ln1.bias"), ("qkvw", "attn.qkv.weight"), ("qkvb", "attn.qkv.bias"),
                ("projw", "attn.proj.weight"), ("projb", "attn.proj.bias"), ("ln2w", "ln2.weight"), ("ln2b", "ln2.bias"),
                ("fc1w", "ffn.layers.0.0.weight"), ("fc1b", "ffn.layers.0.0.bias"),
                ("fc2w", "ffn.layers.1.weight"), ("fc2b", "ffn.layers.1.bias")]})
        self.lnw, self.lnb = h(sd["backbone.ln1.weight"]), h(sd["backbone.ln1.bias"])
        cl = lambda t: h(t).contiguous(memory_format=torch.channels_last)  # noqa: E731
        self.deconv = [cl(sd[f"decode_head.deconv_layers.{j}.weight"]) for j in (0, 3, 6)]
        self.convs = [(cl(sd[f"decode_head.conv_layers.{j}.weight"]), h(sd[f"decode_head.conv_layers.{j}.bias"]))
                      for j in (0, 3, 6)]
        self.segw, self.segb = cl(sd["decode_head.conv_seg.weight"]), h(sd["decode_head.conv_seg.bias"])
        self.grid0 = (64, 48)  # pos-embed grid of the checkpoints (1024x768 / 16)
        assert self.pos.shape[1] == self.grid0[0] * self.grid0[1]
        self._pos_cache = {}
        if attn != "sdpa":
            from fastbb.attention import attention as fb_attention
            self._fb = fb_attention
        self.settings = f"fp16 weights, attn={attn}, gemm_acc16={gacc}, conv_acc16={hacc}, decoder fp16 channels-last, " \
                        f"fp32 InstanceNorm stats"

    def pos_embed(self, gh, gw):
        if (gh, gw) not in self._pos_cache:
            p = self.pos
            if (gh, gw) != self.grid0:
                p = p.view(1, *self.grid0, self.C).permute(0, 3, 1, 2)
                p = F.interpolate(p, (gh, gw), mode="bicubic", align_corners=False).flatten(2).transpose(1, 2)
            self._pos_cache[(gh, gw)] = p.half().contiguous()
        return self._pos_cache[(gh, gw)]

    def block(self, b, x):
        B, N, C = x.shape
        nh = self.nh
        qkv = F.linear(F.layer_norm(x, (C,), b["ln1w"], b["ln1b"], 1e-6), b["qkvw"], b["qkvb"]).view(B, N, 3, nh, 64)
        if self.attn == "sdpa":
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C)
        else:
            q, k, v = qkv.unbind(2)
            o = self._fb(q, k, v, variant=self.attn).view(B, N, C)
        x = x + F.linear(o, b["projw"], b["projb"])
        y = F.gelu(F.linear(F.layer_norm(x, (C,), b["ln2w"], b["ln2b"], 1e-6), b["fc1w"], b["fc1b"]))
        return x.add_(F.linear(y, b["fc2w"], b["fc2b"]))

    @torch.no_grad()
    def forward(self, x):
        with fp16_accumulate(self.gacc):
            B, _, H, W = x.shape
            x = x.half().contiguous(memory_format=torch.channels_last)
            t = self._conv(x, self.pw, self.pb, 16)  # [B, C, gh, gw]
            gh, gw = t.shape[-2:]
            t = t.flatten(2).transpose(1, 2) + self.pos_embed(gh, gw)
            for b in self.blocks:
                t = self.block(b, t)
            t = F.layer_norm(t, (self.C,), self.lnw, self.lnb, 1e-6)
            f = t.view(B, gh, gw, self.C).permute(0, 3, 1, 2)  # channels-last view [B, C, gh, gw]
            for w in self.deconv:
                f = _inorm_silu(self._deconv(f, w))
            for w, bb in self.convs:
                f = _inorm_silu(self._conv(f, w, bb))
            return self._conv(f, self.segw, self.segb).float()
