.PHONY: all clean benchmark benchmark_d64

KERNEL_DIR = ../../kernels/attn/gqa_causal

all:
	$(MAKE) -C $(KERNEL_DIR)

clean:
	$(MAKE) -C $(KERNEL_DIR) clean

benchmark: all
	python3 bench_mha.py --batch 16 --seq_len 4096 --heads 16 --head_dim 128 --causal

benchmark_d64:
	# Rebuild for D=64
	$(MAKE) -C $(KERNEL_DIR) clean
	$(MAKE) -C $(KERNEL_DIR) ATTN_D=64
	python3 bench_mha.py --batch 16 --seq_len 4096 --heads 16 --head_dim 64 --causal

benchmark_d64_opt:
	# Rebuild for D=64 Optimized
	$(MAKE) -C $(KERNEL_DIR) clean
	$(MAKE) -C $(KERNEL_DIR) ATTN_D=64 OPT=1
	python3 bench_mha.py --batch 16 --seq_len 4096 --heads 16 --head_dim 64 --causal
