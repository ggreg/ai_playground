// Kernel 2: shared-memory tiling. One thread still computes one output, but each
// block first stages a BM x BK tile of A and a BK x BN tile of B into shared memory,
// then all threads in the block read from shared (fast) instead of HBM (slow).
//
// What this teaches: the central GEMM optimization. Each tile of A is loaded into
// shared once and reused by BN threads; each tile of B is reused by BM threads.
// HBM traffic per output element drops from O(K) to O(K / BLOCK) — for BLOCK=32,
// that's 32x less bandwidth pressure.
//
// Expected throughput on RTX 3090 (M=N=K=4096): ~3-5 TFLOPS (~15-25% of peak).
// We're now compute-bound on FMA throughput, but each thread still issues many
// shared-memory loads per FMA -- kernel 3 fixes that with register accumulators.

#include "common.cuh"

namespace {

constexpr int BM = 32;
constexpr int BN = 32;
constexpr int BK = 32;

__global__ void sgemm_smem_kernel(int M, int N, int K, float alpha,
                                   const float* __restrict__ A,
                                   const float* __restrict__ B, float beta,
                                   float* __restrict__ C) {
  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];

  const int tx = threadIdx.x;  // 0..BN-1, varies within a warp -> coalesces N
  const int ty = threadIdx.y;  // 0..BM-1
  const int row = blockIdx.y * BM + ty;
  const int col = blockIdx.x * BN + tx;

  float acc = 0.0f;
  for (int k0 = 0; k0 < K; k0 += BK) {
    // Each thread loads exactly one element of As and one of Bs.
    // Threads in a warp share `ty` and vary `tx` -> they load consecutive columns
    // of A's tile and consecutive columns of B's tile (coalesced).
    As[ty][tx] = (row < M && (k0 + tx) < K) ? A[row * K + (k0 + tx)] : 0.0f;
    Bs[ty][tx] = ((k0 + ty) < K && col < N) ? B[(k0 + ty) * N + col] : 0.0f;
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BK; ++k) {
      acc += As[ty][k] * Bs[k][tx];
    }
    __syncthreads();
  }
  if (row < M && col < N) {
    C[row * N + col] = alpha * acc + beta * C[row * N + col];
  }
}

}  // namespace

extern "C" void launch_sgemm_smem(int M, int N, int K, float alpha, const float* A,
                                  const float* B, float beta, float* C,
                                  cudaStream_t stream) {
  dim3 block(BN, BM);
  dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  sgemm_smem_kernel<<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
}
