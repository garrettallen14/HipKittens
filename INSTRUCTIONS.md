# HipKittens MHA Reproduction & Extension Instructions

These instructions are for running the benchmarks on your AMD MI300X machine.

## 1. Setup

Ensure you are on the `feature/mha-repro-extension` branch.

```bash
git checkout feature/mha-repro-extension
```

## 2. Reproduction (MHA Forward, Head Dim 128)

This will build the standard kernel and run the benchmark to reproduce the ~893 TFLOPs result.

```bash
cd benchmarks/attention
make benchmark
```

Expected output:
- HipKittens Performance: ~893 TFLOPs
- Should beat PyTorch SDPA and match/beat AITER.

## 3. Extension (MHA Forward, Head Dim 64)

### Standard Extension (KV_BLOCK_SIZE=64)
This uses the existing D=64 implementation (which has stricter synchronization).

```bash
cd benchmarks/attention
make benchmark_d64
```

### Optimized Extension (KV_BLOCK_SIZE=128)
This uses the new optimized kernel with larger KV tiles to improve occupancy.

```bash
cd benchmarks/attention
make benchmark_d64_opt
```

## 4. Troubleshooting

If you see import errors, ensure you have built the kernel:

```bash
cd kernels/attn/gqa_causal
make clean
make ATTN_D=128  # For reproduction
# OR
make ATTN_D=64   # For extension
```

Then run the python script manually:

```bash
python3 ../../../benchmarks/attention/bench_mha.py --head_dim 128  # or 64
```
