"""Fused NVFP4 hyper-connection kernels for small-batch decode (MBX_HC_NVFP4_FUSED=1 with the turbo HC patch).

With engine/llm_patches/hc_fp8.py (LLM_HC_FP8=all, LLM_HC_FMT=nvfp4) each of the target's 96 HC mixes runs five kernels:
Marlin down+inject [M,10240]x[10240,336], a split copy, hc_silu, Marlin up [M,320]x[320,10240] and hc_gate_mix.
Here they are two:

  hc_down_silu_nvfp4   split-K over K=10240; every program writes an fp32 partial, the last program of each column tile
                       (atomic counter, reset by that program, so no zeroing kernel) sums them and writes
                       silu(bf16(d)/HC) for the 320 low-rank columns and bf16(d) for the 4 injection columns.
  hc_up_mix_nvfp4      one program per tile of the 2560 output columns reads the matching rows of all 4 streams of the up
                       weight, so the sigmoid gate and the mean over streams run in the epilogue: out = mean_s sigmoid(g_s)*xn_s.

Weights come from the raw NVFP4 quantization (packed e2m1 [N, K/2], element 2j in the low nibble, e4m3 scales [N, K/16],
one fp32 global scale) and are re-tiled at load time so that every program reads one contiguous block per K step.
Dequantization uses the hardware e2m1x2 -> f16x2 conversion; e2m1 * e4m3 is exact in f16, activations go to f16 for the
MMA (exact for |x| < 65504) and accumulate in fp32. Rounding points follow the unfused path (down and gate outputs to bf16).
"""
import os

import torch
import triton
import triton.language as tl

HC = 4
H = 2560
K_DOWN = HC * H          # 10240
N_DOWN = 336             # 320 low-rank + 4 injection + 12 pad
LORA = 320
MAX_M = 32               # covers the whole first compile range (1..32), so the size test never splits a graph
ENABLED = os.environ.get("MBX_HC_NVFP4_FUSED", "1") != "0"   # weights are attached by the turbo HC patch only

# Layout and launch configs (the layout is fixed when the weights are packed).
DOWN_BN, DOWN_SPLIT, DOWN_BK, DOWN_WARPS, DOWN_STAGES = 16, 8, 256, 2, 3   # sweep_hc_nvfp4.py, M=7
UP_BC, UP_BK, UP_WARPS, UP_STAGES = 16, 32, 8, 1


@triton.jit
def _e2m1x2_f16x2(pk):
    return tl.inline_asm_elementwise(
        "{ .reg .b8 t; cvt.u8.u32 t, $1; cvt.rn.f16x2.e2m1x2 $0, t; }",
        "=r,r", [pk.to(tl.uint32)], dtype=tl.uint32, is_pure=True, pack=1)


@triton.jit
def _dequant(pk, sc, BN: tl.constexpr, BK: tl.constexpr):
    """pk u8 [BN, BK/2], sc u8 (e4m3 bits) [BN, BK/16] -> f16 [BN, BK] = e2m1 * scale."""
    h = _e2m1x2_f16x2(pk)
    lo = (h & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
    hi = (h >> 16).to(tl.uint16).to(tl.float16, bitcast=True)
    w = tl.join(lo, hi).reshape(BN, BK)
    s = sc.to(tl.float8e4nv, bitcast=True).to(tl.float16)
    return (w.reshape(BN, BK // 16, 16) * s[:, :, None]).reshape(BN, BK)


@triton.jit
def _down_silu_kernel(x_ptr, pk_ptr, sc_ptr, part_ptr, lock_ptr, lora_ptr, inj_ptr, gscale, M, stride_x,
                      stride_lora, stride_inj,
                      K: tl.constexpr, N: tl.constexpr, NL: tl.constexpr, NHC: tl.constexpr,
                      BM: tl.constexpr, BN: tl.constexpr, SPLIT: tl.constexpr, BK: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    KS: tl.constexpr = K // SPLIT
    KC: tl.constexpr = KS // BK
    offs_m = tl.arange(0, BM)
    mask_m = offs_m < M
    offs_r = tl.arange(0, BN)
    blk = (pid_n * SPLIT + pid_k) * KC
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for i in tl.range(0, KC):
        k0 = pid_k * KS + i * BK
        x = tl.load(x_ptr + offs_m[:, None] * stride_x + (k0 + tl.arange(0, BK))[None, :], mask=mask_m[:, None],
                    other=0.0).to(tl.float16)
        pk = tl.load(pk_ptr + (blk + i) * (BN * BK // 2) + offs_r[:, None] * (BK // 2) + tl.arange(0, BK // 2)[None, :])
        sc = tl.load(sc_ptr + (blk + i) * (BN * BK // 16) + offs_r[:, None] * (BK // 16)
                     + tl.arange(0, BK // 16)[None, :])
        acc = tl.dot(x, tl.trans(_dequant(pk, sc, BN, BK)), acc)
    offs_n = pid_n * BN + offs_r
    tl.store(part_ptr + pid_k * (BM * N) + offs_m[:, None] * N + offs_n[None, :], acc)
    tl.debug_barrier()
    done = tl.atomic_add(lock_ptr + pid_n, 1, sem="acq_rel")
    if done == SPLIT - 1:
        tot = tl.zeros((BM, BN), dtype=tl.float32)
        for s in tl.static_range(SPLIT):
            tot += tl.load(part_ptr + s * (BM * N) + offs_m[:, None] * N + offs_n[None, :], cache_modifier=".cg")
        d = (tot * gscale).to(tl.bfloat16).to(tl.float32)
        y = d / NHC
        y = y * tl.sigmoid(y)
        is_lora = offs_n < NL
        tl.store(lora_ptr + offs_m[:, None] * stride_lora + offs_n[None, :], y.to(tl.bfloat16),
                 mask=mask_m[:, None] & is_lora[None, :])
        is_inj = (offs_n >= NL) & (offs_n < NL + NHC)
        tl.store(inj_ptr + offs_m[:, None] * stride_inj + (offs_n - NL)[None, :], d.to(tl.bfloat16),
                 mask=mask_m[:, None] & is_inj[None, :])
        tl.atomic_xchg(lock_ptr + pid_n, 0)


@triton.jit
def _up_mix_kernel(lora_ptr, xn_ptr, pk_ptr, sc_ptr, out_ptr, gscale, M, stride_lora, stride_xn, stride_out,
                   K: tl.constexpr, HD: tl.constexpr, NHC: tl.constexpr,
                   BM: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    R: tl.constexpr = NHC * BC          # weight rows of one program: BC columns of each stream, stream-major
    KC: tl.constexpr = K // BK
    offs_m = tl.arange(0, BM)
    mask_m = offs_m < M
    offs_r = tl.arange(0, R)
    acc = tl.zeros((BM, R), dtype=tl.float32)
    for i in tl.static_range(KC):
        a = tl.load(lora_ptr + offs_m[:, None] * stride_lora + (i * BK + tl.arange(0, BK))[None, :],
                    mask=mask_m[:, None], other=0.0).to(tl.float16)
        pk = tl.load(pk_ptr + (pid * KC + i) * (R * BK // 2) + offs_r[:, None] * (BK // 2)
                     + tl.arange(0, BK // 2)[None, :])
        sc = tl.load(sc_ptr + (pid * KC + i) * (R * BK // 16) + offs_r[:, None] * (BK // 16)
                     + tl.arange(0, BK // 16)[None, :])
        acc = tl.dot(a, tl.trans(_dequant(pk, sc, R, BK)), acc)
    g = (acc * gscale).to(tl.bfloat16).to(tl.float32)
    cols = (offs_r // BC) * HD + pid * BC + offs_r % BC
    x = tl.load(xn_ptr + offs_m[:, None] * stride_xn + cols[None, :], mask=mask_m[:, None], other=0.0)
    mix = tl.sum((tl.sigmoid(g) * x.to(tl.float32)).reshape(BM, NHC, BC), axis=1) / NHC
    offs_c = pid * BC + tl.arange(0, BC)
    tl.store(out_ptr + offs_m[:, None] * stride_out + offs_c[None, :], mix.to(tl.bfloat16), mask=mask_m[:, None])


def _tile(t, groups, bk_bytes):
    """[G*R, Kb] -> contiguous [G, Kb/bk_bytes, R, bk_bytes] (one block per program and K step)."""
    rows, kb = t.shape
    r = rows // groups
    return t.reshape(groups, r, kb // bk_bytes, bk_bytes).permute(0, 2, 1, 3).contiguous()


class HcNvfp4Weights:
    """Re-tiled NVFP4 copies of one HC module's down+inject [336, 10240] and up [10240, 320] weights.

    down/up: (packed u8 [N, K/2], e4m3 scales as u8 [N, K/16], global scale float) from the raw quantization."""

    def __init__(self, down, up):
        d_pk, d_sc, self.d_g = down
        u_pk, u_sc, self.u_g = up
        assert tuple(d_pk.shape) == (N_DOWN, K_DOWN // 2) and tuple(u_pk.shape) == (K_DOWN, LORA // 2)
        self.d_g, self.u_g = float(self.d_g), float(self.u_g)
        # down: rows in tiles of DOWN_BN; K split into DOWN_SPLIT slices of K steps -> [NT, SPLIT*KC, BN, BK/2]
        nt = N_DOWN // DOWN_BN
        self.d_pk = _tile(d_pk, nt, DOWN_BK // 2)
        self.d_sc = _tile(d_sc, nt, DOWN_BK // 16)
        # up: program p owns rows {s*2560 + p*BC + r}; reorder rows to [p, s, r] first
        np_ = H // UP_BC
        perm = (torch.arange(HC).view(1, HC, 1) * H + torch.arange(np_).view(np_, 1, 1) * UP_BC
                + torch.arange(UP_BC).view(1, 1, UP_BC)).reshape(-1).to(u_pk.device)
        self.u_pk = _tile(u_pk[perm], np_, UP_BK // 2)
        self.u_sc = _tile(u_sc[perm], np_, UP_BK // 16)
        # one counter per down column tile, reset by the program that finishes the tile
        self.lock = torch.zeros(nt, dtype=torch.int32, device=d_pk.device)


_part = {}


def _partials(device):
    t = _part.get(device)
    if t is None:
        t = _part[device] = torch.empty((DOWN_SPLIT, MAX_M, N_DOWN), dtype=torch.float32, device=device)
    return t


def hc_down_silu_nvfp4(xn: torch.Tensor, w: HcNvfp4Weights):
    """xn [M, 10240] bf16 -> (silu(down/HC) [M, 320] bf16, injection [M, 4] bf16)."""
    M = xn.shape[0]
    lora = xn.new_empty((M, LORA))
    inj = xn.new_empty((M, HC))
    _down_silu_kernel[(N_DOWN // DOWN_BN, DOWN_SPLIT)](
        xn, w.d_pk, w.d_sc, _partials(xn.device), w.lock, lora, inj, w.d_g, M, xn.stride(0), lora.stride(0),
        inj.stride(0), K=K_DOWN, N=N_DOWN, NL=LORA, NHC=HC, BM=16 if M <= 16 else 32, BN=DOWN_BN, SPLIT=DOWN_SPLIT, BK=DOWN_BK,
        num_warps=DOWN_WARPS, num_stages=DOWN_STAGES)
    return lora, inj


def hc_up_mix_nvfp4(xn: torch.Tensor, lora: torch.Tensor, w: HcNvfp4Weights) -> torch.Tensor:
    """mean over the 4 streams of sigmoid(lora @ up^T) * xn -> [M, 2560] bf16."""
    M = xn.shape[0]
    out = xn.new_empty((M, H))
    _up_mix_kernel[(H // UP_BC,)](
        lora, xn, w.u_pk, w.u_sc, out, w.u_g, M, lora.stride(0), xn.stride(0), out.stride(0),
        K=LORA, HD=H, NHC=HC, BM=16 if M <= 16 else 32, BC=UP_BC, BK=UP_BK, num_warps=UP_WARPS, num_stages=UP_STAGES)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Model integration: one custom op per HC mix, weights as non-persistent buffers of the GatedResidual module.
# ---------------------------------------------------------------------------------------------------------------------
_BUFFERS = ("hc4_d_pk", "hc4_d_sc", "hc4_lock", "hc4_u_pk", "hc4_u_sc")


def _mix(xn: torch.Tensor, d_pk: torch.Tensor, d_sc: torch.Tensor, lock: torch.Tensor, u_pk: torch.Tensor,
         u_sc: torch.Tensor, d_g: float, u_g: float) -> tuple[torch.Tensor, torch.Tensor]:
    w = HcNvfp4Weights.__new__(HcNvfp4Weights)
    w.d_pk, w.d_sc, w.lock, w.u_pk, w.u_sc, w.d_g, w.u_g = d_pk, d_sc, lock, u_pk, u_sc, d_g, u_g
    lora, inj = hc_down_silu_nvfp4(xn, w)
    return hc_up_mix_nvfp4(xn, lora, w), inj


def _mix_fake(xn, d_pk, d_sc, lock, u_pk, u_sc, d_g, u_g):
    return xn.new_empty((xn.shape[0], H)), xn.new_empty((xn.shape[0], HC))


_registered = False


def _register():
    global _registered
    if not _registered:
        from vllm.utils.torch_utils import direct_register_custom_op
        direct_register_custom_op(op_name="rtx_spark_hc_nvfp4_mix", op_func=_mix, fake_impl=_mix_fake)
        _registered = True


def attach(module, down, up) -> None:
    """Give a GatedResidual (use_combine=True) module the fused path. down/up: raw NVFP4 (packed, scales u8, global)."""
    _register()
    w = HcNvfp4Weights(down, up)
    for name, t in zip(_BUFFERS, (w.d_pk, w.d_sc, w.lock, w.u_pk, w.u_sc)):
        module.register_buffer(name, t, persistent=False)
    module.hc4_scales = (w.d_g, w.u_g)


def eligible(module, xn: torch.Tensor) -> bool:
    return (ENABLED and getattr(module, "hc4_scales", None) is not None and 1 <= xn.shape[0] <= MAX_M
            and xn.dtype == torch.bfloat16 and xn.stride(-1) == 1)


def mix(module, xn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(block_input [M, 2560], injection [M, 4]) for one HC mix of `module`."""
    d_g, u_g = module.hc4_scales
    return torch.ops.vllm.rtx_spark_hc_nvfp4_mix(xn, module.hc4_d_pk, module.hc4_d_sc, module.hc4_lock,
                                                module.hc4_u_pk, module.hc4_u_sc, d_g, u_g)
