import torch
import sys
import os
import random
import time

# Add kernel directory to path to import tk_kernel
kernel_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../kernels/attn/gqa_causal"))
sys.path.append(kernel_dir)

try:
    import tk_kernel
except ImportError:
    print(f"Error: Could not import tk_kernel from {kernel_dir}")
    print("Please build the kernel first: cd kernels/attn/gqa_causal && make")
    sys.exit(1)

try:
    import aiter
    HAS_AITER = True
except ImportError:
    print("Warning: aiter not found. Skipping baseline comparison.")
    HAS_AITER = False

def flops(batch, seqlen, nheads, headdim, causal):
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f

def efficiency(flop, time_ms):
    return (flop / 1e12) / (time_ms / 1e3)

def run_benchmark(B=16, N=4096, H=16, D=128, causal=True):
    # MHA implies H_KV = H
    H_KV = H
    
    print(f"Benchmarking MHA: B={B}, N={N}, H={H}, D={D}, causal={causal}")
    
    dtype = torch.bfloat16
    device = 'cuda'
    
    torch.manual_seed(0)
    random.seed(0)
    
    # Warmup and Benchmark HipKittens
    print("\n--- HipKittens ---")
    
    # Allocate buffers
    out = torch.zeros(B, N, H, D, dtype=dtype, device=device)
    lse = torch.zeros(B, H, 1, N, dtype=torch.float32, device=device)
    
    # Create events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    num_warmup = 100
    num_iters = 100
    
    # Warmup
    for _ in range(num_warmup):
        q = torch.randn(B, N, H, D, dtype=dtype, device=device)
        k = torch.randn(B, N, H_KV, D, dtype=dtype, device=device)
        v = torch.randn(B, N, H_KV, D, dtype=dtype, device=device)
        tk_kernel.dispatch_micro(q, k, v, out, lse)
        
    timings = []
    
    # Benchmark
    for _ in range(num_iters):
        q = torch.randn(B, N, H, D, dtype=dtype, device=device)
        k = torch.randn(B, N, H_KV, D, dtype=dtype, device=device)
        v = torch.randn(B, N, H_KV, D, dtype=dtype, device=device)
        
        torch.cuda.synchronize()
        start_event.record()
        tk_kernel.dispatch_micro(q, k, v, out, lse)
        end_event.record()
        torch.cuda.synchronize()
        
        timings.append(start_event.elapsed_time(end_event))
        
    avg_time = sum(timings) / len(timings)
    total_flops = flops(B, N, H, D, causal)
    tflops = efficiency(total_flops, avg_time)
    
    print(f"Average time: {avg_time:.4f} ms")
    print(f"Performance: {tflops:.2f} TFLOPS")
    
    # Baseline (AITER)
    if HAS_AITER:
        print("\n--- AITER Baseline ---")
        timings_ref = []
        for _ in range(num_iters):
            q = torch.randn(B, N, H, D, dtype=dtype, device=device)
            k = torch.randn(B, N, H_KV, D, dtype=dtype, device=device)
            v = torch.randn(B, N, H_KV, D, dtype=dtype, device=device)
            
            torch.cuda.synchronize()
            start_event.record()
            aiter.flash_attn_func(q, k, v, causal=causal, return_lse=True)
            end_event.record()
            torch.cuda.synchronize()
            
            timings_ref.append(start_event.elapsed_time(end_event))
            
        avg_time_ref = sum(timings_ref) / len(timings_ref)
        tflops_ref = efficiency(total_flops, avg_time_ref)
        print(f"Average time: {avg_time_ref:.4f} ms")
        print(f"Performance: {tflops_ref:.2f} TFLOPS")
        print(f"Speedup vs AITER: {avg_time_ref / avg_time:.2f}x")

    # Baseline (PyTorch SDPA)
    print("\n--- PyTorch SDPA Baseline ---")
    timings_pt = []
    for _ in range(num_iters):
        q = torch.randn(B, N, H, D, dtype=dtype, device=device)
        k = torch.randn(B, N, H_KV, D, dtype=dtype, device=device)
        v = torch.randn(B, N, H_KV, D, dtype=dtype, device=device)
        
        torch.cuda.synchronize()
        start_event.record()
        torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
        end_event.record()
        torch.cuda.synchronize()
        
        timings_pt.append(start_event.elapsed_time(end_event))
        
    avg_time_pt = sum(timings_pt) / len(timings_pt)
    tflops_pt = efficiency(total_flops, avg_time_pt)
    print(f"Average time: {avg_time_pt:.4f} ms")
    print(f"Performance: {tflops_pt:.2f} TFLOPS")
    print(f"Speedup vs PyTorch: {avg_time_pt / avg_time:.2f}x")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', '-b', type=int, default=16)
    parser.add_argument('--seq_len', '-n', type=int, default=4096)
    parser.add_argument('--heads', '-h_q', type=int, default=16)
    parser.add_argument('--head_dim', '-d', type=int, default=128)
    parser.add_argument('--causal', action='store_true', default=True)
    args = parser.parse_args()
    
    run_benchmark(B=args.batch, N=args.seq_len, H=args.heads, D=args.head_dim, causal=args.causal)
