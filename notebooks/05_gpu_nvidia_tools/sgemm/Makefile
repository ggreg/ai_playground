# Build the SGEMM tutorial kernels and the benchmark harness.
#
# Default arch is SM_80 (Ampere: A100, A10, RTX 30xx series). Override on the command line
# for other GPUs:
#
#   make SM_ARCH=sm_86       # RTX 3090 / A40
#   make SM_ARCH=sm_89       # RTX 4090 / L40
#   make SM_ARCH=sm_90       # H100
#
# Then run:
#   ./bench --size 4096 --iters 50 --json results.json

NVCC      ?= nvcc
SM_ARCH   ?= sm_80
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(SM_ARCH) --use_fast_math -lineinfo
LIBS      ?= -lcublas

SRCS := 01_naive.cu 02_smem.cu 03_block_tile.cu 04_vectorized.cu 05_async_pipeline.cu
OBJS := $(SRCS:.cu=.o)

.PHONY: all clean run

all: bench

%.o: %.cu common.cuh
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

bench: $(OBJS) bench.cu common.cuh
	$(NVCC) $(NVCCFLAGS) bench.cu $(OBJS) $(LIBS) -o bench

run: bench
	./bench --size 4096 --iters 50 --json results.json

clean:
	rm -f $(OBJS) bench results.json
