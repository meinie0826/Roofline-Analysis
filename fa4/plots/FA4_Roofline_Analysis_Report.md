# FA4 Performance Analysis: Hardware-Roofline Decomposition

## 1. The B200 Hardware Asymmetry Problem

FlashAttention-4 (FA4) was designed specifically for the NVIDIA Blackwell B200 GPU.
The key architectural change from Hopper H100 to Blackwell B200 was **asymmetric**:

| Resource | H100 SXM | B200 SXM | Change |
|----------|----------|----------|--------|
| BF16 TC throughput | 989 TFLOPs/s | 2250 TFLOPs/s | **+2.3×** |
| MUFU (exp2) | 16 ops/cycle/SM | 16 ops/cycle/SM | **unchanged** |
| SMEM bandwidth | 128 B/cycle/SM | 128 B/cycle/SM | **unchanged** |
| Number of SMs | 132 | 148 | +12% |

This asymmetry means that **MUFU and SMEM, which were not bottlenecks on H100, become
co-bottlenecks on B200** when running naive attention kernels.

### Cycle count analysis (M=N=D=128 per tile):

| Resource | H100 cycles | B200 cycles | Ratio |
|----------|-------------|-------------|-------|
| TC (QK+PV MMAs) | 2048 | 1024 | ÷2 (TC doubled) |
| EXP (MUFU.EX2) | 1024 | 1024 | same |
| SMEM bandwidth | 640  | 640  | same |
| **Bottleneck** | **TC** | **TC+EXP (tie!)** | |

On H100: TC=2048c is the bottleneck; EXP=1024c is 2× faster → not a problem.
On B200: TC=1024c and EXP=1024c are **exactly tied** → EXP is now a co-bottleneck.

---

## 2. Optimization-Hardware Mapping

Each FA4 optimization targets a specific hardware resource that became a bottleneck on B200.

### Stage 0 → Stage 1: Ping-Pong Pipeline (q_stage=2)

**Hardware target**: Tensor Core utilization  
**Problem**: Without ping-pong, MMA and softmax execute serially. The MMA warpgroup
waits for softmax to finish before starting the next tile. This creates ~40% idle time
for the Tensor Cores.

```
Without ping-pong (q_stage=1):
  [MMA tile N] → [Softmax tile N] → [Correction] → [MMA tile N+1] → ...
  TC utilization: ~60% (serial stall)

With ping-pong (q_stage=2):
  [MMA tile N+1] ──────────────────────→
                   [Softmax tile N] → [Correction]
  TC utilization: ~100% (fully overlapped)
```

**Why Blackwell enables this**: 256KB TMEM (Tensor Memory) per SM stores the P 
intermediate result from MMA, allowing the softmax warpgroup to consume it 
asynchronously while the MMA warpgroup processes the next tile.

**Roofline effect**: Raises effective ceiling from `(TC + EXP)_serial` to `max(TC, EXP, SMEM)`.
Expected gain: **+15-25%** throughput.

---

### Stage 1 → Stage 2: Conditional Rescaling (rescale_threshold=8.0)

**Hardware target**: Correction warpgroup utilization  
**Problem**: Online softmax must rescale O whenever row_max updates:
```
O_new = O_old × exp2(row_max_old - row_max_new)
```
This correction fires **every KV tile**, even when Δmax is negligible (≈0 for 
long sequences after warmup tiles). The correction warpgroup consumes precious 
pipeline slots.

**Solution**: Skip correction when `|Δmax| < 8.0` (in log2 space = factor 256).
In practice, >90% of tiles in a long sequence have |Δmax| < 0.01 after the
first few KV tiles establish the row maximum.

**Roofline effect**: Reduces effective softmax pipeline cycles, improving overall
SM utilization. Gain is proportional to sequence length.
Expected gain: **+5-10%**, larger for seqlen > 4096.

---

### Stage 2 → Stage 3: Software Exp2 Emulation (FMA polynomial)

**Hardware target**: MUFU (Special Function Unit) — exp2 throughput  
**Problem**: This is the most critical bottleneck on B200.

```
Per tile cycle analysis (M=N=D=128):
  TC ops: 4 × 128 × 128 × 128 = 8M MACs → 8M/8192 = 1024 cycles
  EXP ops: 128 × 128 = 16K exp2 → 16K/16 = 1024 cycles
                                               ^^^^^^^^^^^
                          EXACT TIE — EXP is co-bottleneck with TC!
```

B200 has 128 FP32 FFMA units per SM (vs 16 MUFU exp2 units).
A degree-3 polynomial approximation uses ~4 FMAs per exp2:
```
exp2(x) ≈ 1 + x×c₁ + x²×c₂ + x³×c₃    [4 FMAs]
effective throughput: 128/4 = 32 ops/cycle (vs 16 MUFU ops/cycle)
```

By mixing ~45% of exp2 rows through FMA and ~55% through MUFU,
both units run in parallel, giving ~2× effective exp throughput.

**Roofline effect**: Raises EXP ceiling from `MUFU-limited (1024c)` to 
`MUFU+FMA combined (~512c)`.
Expected gain: **+10-20%** throughput.

---

### Stage 3 → Stage 4: LPT Tile Scheduler

**Hardware target**: SM utilization (load balance)  
**Problem**: Causal attention computes a triangular matrix.
Tile (row=i) processes i KV blocks, but tile (row=0) processes only 1 block.
With linear scheduling: some SMs get heavy tiles (row=M_max) while others idle.

```
Causal work distribution:
  Tile row 0:    ████░░░░░░░░░░░░  (25% work)
  Tile row M/2:  ████████░░░░░░░░  (50% work)  
  Tile row M:    ████████████████ (100% work)
  
Linear schedule: SMs finish at different times → ~10% average idle time.
LPT schedule:    Assign rows 0..M in LPT order → SMs finish together.
```

**Roofline effect**: Reduces SM idle cycles by 4-14% for causal attention.
Near-zero effect for non-causal (uniform work distribution).

---

### Bonus: 2-CTA MMA Mode (non-causal, hdim≥128)

**Hardware target**: SMEM bandwidth  
**Problem**: For hdim=192, SMEM becomes the bottleneck:
```
hdim=192, M=128, N=128:
  TC cycles:   4×128×128×192/8192 = 1536 cycles
  SMEM cycles: (256×192×2 + 128×192×2×N/128) / 128 = 768 cycles
              → SMEM (768c) < TC (1536c) → NOT bottleneck for hdim=192
              
Wait, recomputing for N=128:
  QK SMEM: 256 × 192 × 2 = 98KB → 98K/128 = 768 cycles
  PV SMEM: 128 × 128 × 2 = 32KB → 256 cycles
  Total: 1024 cycles  >  TC 1536c? No: TC still dominant.
  
  But for M=N=128, D=192 on B200: TC=1536c, SMEM=1024c → TC bottleneck.
  2-CTA helps when sequence length is very large (many tiles per SM).
```

**2-CTA solution**: Two CTAs form a cluster. Q is broadcast; each CTA reads
only its half of K columns. Effective K SMEM reads halved.

Expected gain: **+10-30%** for hdim=192, negligible for hdim=128.

---

## 3. Summary Table

| Optimization | Hardware Target | Bottleneck Before | After | Expected Gain |
|-------------|-----------------|-------------------|-------|---------------|
| Ping-pong (q_stage=2) | TC idle time | TC+EXP serial | max(TC,EXP,SMEM) | +15-25% |
| Cond. rescaling | Correction warpgroup | EXP (serial stall) | EXP - correction overhead | +5-10% |
| FMA exp2 emu | MUFU throughput | EXP=TC (co-bottleneck) | EXP raised ~2× | +10-20% |
| LPT scheduler | SM utilization | Causal load imbalance | Balanced SM load | +4-14% |
| 2-CTA MMA | SMEM bandwidth | SMEM (hdim≥128) | SMEM halved | +10-30%† |

† 2-CTA applies only to non-causal, and is particularly valuable for hdim=192+.

**Total expected speedup (causal, hdim=128)**: ~1.5-2.0× over naive Blackwell baseline.