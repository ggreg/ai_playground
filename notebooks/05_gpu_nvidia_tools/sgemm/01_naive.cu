// Kernel 1: naive SGEMM. One thread = one output element of C.
//
// What this teaches: the baseline. Every thread reads K floats from A and K from B
// straight out of HBM, with zero reuse. Arithmetic intensity is O(1) FLOPs/byte —
// this kernel is pinned to the bandwidth roofline regardless of how many SMs the GPU has.
//
// Expected throughput on RTX 3090 (M=N=K=4096): ~250-300 GFLOPS, vs ~20 TFLOPS peak.
// That's ~1-2% of peak — a useful number to remember next time you write a "first pass" kernel.

#include "common.cuh"

namespace {

// 32 x 32 = 1024 threads/block, the maximum allowed.
constexpr int BLOCK = 32;

__global__ void sgemm_naive_kernel(int M, int N, int K, float alpha,
                                    const float* __restrict__ A,
                                    const float* __restrict__ B, float beta,
                                    float* __restrict__ C) {
  // threadIdx.x varies across a warp -> use it for the column so consecutive threads
  // in a warp access consecutive columns of C and B (coalesced 128-byte loads).
  const int col = blockIdx.x * BLOCK + threadIdx.x;
  const int row = blockIdx.y * BLOCK + threadIdx.y;
  if (row >= M || col >= N) return;

  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    acc += A[row * K + k] * B[k * N + col];
  }
  C[row * N + col] = alpha * acc + beta * C[row * N + col];
}

}  // namespace

extern "C" void launch_sgemm_naive(int M, int N, int K, float alpha, const float* A,
                                   const float* B, float beta, float* C,
                                   cudaStream_t stream) {
  dim3 block(BLOCK, BLOCK);
  dim3 grid(CEIL_DIV(N, BLOCK), CEIL_DIV(M, BLOCK));
  sgemm_naive_kernel<<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
}
