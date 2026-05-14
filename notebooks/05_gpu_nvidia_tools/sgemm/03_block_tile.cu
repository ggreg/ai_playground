// Kernel 3: 2D thread tile + register accumulators.
//
// What this teaches: shared memory is fast (~10 TB/s effective on Ampere), but
// registers are ~10x faster again. Kernel 2 issued one shared-memory load per FMA;
// here each thread holds an 8x8 = 64-element accumulator in registers and reuses
// each loaded value 8 times. Per FMA we now read ~0.25 floats from shared memory.
//
// Block tile: BM x BN x BK = 128 x 128 x 8.
//   threads/block = (BM*BN) / (TM*TN) = 16384 / 64 = 256.
//   threads laid out as 16 x 16 (threadCol = tid % 16, threadRow = tid / 16).
// Per block we compute one 128x128 output tile. Per thread we compute 8x8 outputs.
//
// Each main loop iteration:
//   - cooperatively load 128x8 tile of A and 8x128 tile of B into shared memory
//     (256 threads, 1024 elements each -> 4 elements/thread, all coalesced)
//   - for k in 0..7:
//       load 8 floats of A and 8 floats of B from shared into registers
//       perform 64 FMAs into the 8x8 accumulator
// That's 64 FMAs for 16 register reads -> 4 FLOPs/byte from shared memory.
//
// Expected throughput on RTX 3090 (M=N=K=4096): ~13-15 TFLOPS (~65-75% of peak).
// Big jump over kernel 2. Kernel 4 will close the rest of the gap with vectorized loads.

#include "common.cuh"

namespace {

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;
constexpr int TM = 8;
constexpr int TN = 8;
constexpr int THREADS = (BM * BN) / (TM * TN);  // 256
static_assert(THREADS == 256, "thread count must match the layout below");

__global__ void sgemm_block_tile_kernel(int M, int N, int K, float alpha,
                                         const float* __restrict__ A,
                                         const float* __restrict__ B, float beta,
                                         float* __restrict__ C) {
  __shared__ float As[BM * BK];   // row-major:  As[m*BK + k]
  __shared__ float Bs[BK * BN];   // row-major:  Bs[k*BN + n]

  const int tid = threadIdx.x;
  // 16 x 16 thread layout over the 128 x 128 output tile (each thread does 8 x 8).
  const int threadCol = tid % (BN / TN);  // 0..15
  const int threadRow = tid / (BN / TN);  // 0..15

  // Move pointers to this block's strip of A, B, C.
  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;

  // Cooperative-load layout for As (BM x BK = 128 x 8 = 1024 floats / 256 threads = 4 each):
  // threads index along K within a warp (varies fastest) -> coalesced read of A's row.
  const int innerRowA = tid / BK;          // 0..31
  const int innerColA = tid % BK;          // 0..7
  constexpr int strideA = THREADS / BK;     // 32  (rows of A loaded per pass)

  // Cooperative-load layout for Bs (BK x BN = 8 x 128 = 1024 / 256 = 4 each):
  // threads vary innerColB within a warp -> coalesced read of B's row.
  const int innerRowB = tid / BN;          // 0..1
  const int innerColB = tid % BN;          // 0..127
  constexpr int strideB = THREADS / BN;     // 2  (rows of B loaded per pass)

  float threadResults[TM * TN] = {0.0f};
  float regM[TM];
  float regN[TN];

  for (int k0 = 0; k0 < K; k0 += BK) {
#pragma unroll
    for (int o = 0; o < BM; o += strideA) {
      As[(innerRowA + o) * BK + innerColA] = A[(innerRowA + o) * K + innerColA];
    }
#pragma unroll
    for (int o = 0; o < BK; o += strideB) {
      Bs[(innerRowB + o) * BN + innerColB] = B[(innerRowB + o) * N + innerColB];
    }
    __syncthreads();

    A += BK;
    B += BK * N;

#pragma unroll
    for (int k = 0; k < BK; ++k) {
#pragma unroll
      for (int i = 0; i < TM; ++i) regM[i] = As[(threadRow * TM + i) * BK + k];
#pragma unroll
      for (int j = 0; j < TN; ++j) regN[j] = Bs[k * BN + threadCol * TN + j];
#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          threadResults[i * TN + j] += regM[i] * regN[j];
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int r = threadRow * TM + i;
      const int c = threadCol * TN + j;
      C[r * N + c] = alpha * threadResults[i * TN + j] + beta * C[r * N + c];
    }
  }
}

}  // namespace

extern "C" void launch_sgemm_block_tile(int M, int N, int K, float alpha, const float* A,
                                        const float* B, float beta, float* C,
                                        cudaStream_t stream) {
  dim3 block(THREADS);
  dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  sgemm_block_tile_kernel<<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
}
