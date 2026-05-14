// Kernel 4: vectorized 128-bit memory traffic + transposed A in shared memory.
//
// What this teaches: two changes layered on kernel 3.
//
//   (1) float4 (LDG.128 / STG.128) for A, B, and C. A single 128-bit load amortizes
//       address calculation and instruction-issue overhead over four floats. The HBM
//       bandwidth utilization improves and SASS issues fewer LD/ST instructions per FMA.
//
//   (2) A is *transposed during the load*: shared memory layout becomes As[k][m] instead
//       of As[m][k]. This means the 8 elements of regM that one thread needs are now
//       contiguous in shared memory and can be read as 2 x float4 instead of 8 scalar
//       loads. Without the transpose, those 8 elements are strided by BK and you'd hit
//       4-way bank conflicts in the inner loop.
//
// Expected throughput on RTX 3090 (M=N=K=4096): ~16-18 TFLOPS (~80-90% of peak FP32).
// At this point we're close to cuBLAS for FP32 FMA. The remaining gap is software
// pipelining of memory transfers, which kernel 5 adds via cp.async.

#include "common.cuh"

namespace {

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;
constexpr int TM = 8;
constexpr int TN = 8;
constexpr int THREADS = (BM * BN) / (TM * TN);  // 256

__global__ void sgemm_vectorized_kernel(int M, int N, int K, float alpha,
                                         const float* __restrict__ A,
                                         const float* __restrict__ B, float beta,
                                         float* __restrict__ C) {
  __shared__ float As[BK * BM];  // transposed: index as As[k*BM + m]
  __shared__ float Bs[BK * BN];  // straight:    index as Bs[k*BN + n]

  const int tid = threadIdx.x;
  const int threadCol = tid % (BN / TN);  // 0..15
  const int threadRow = tid / (BN / TN);  // 0..15

  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;

  // Each thread cooperatively loads exactly one float4 of A (BM*BK/4 = 256 vecs / 256 threads).
  const int innerRowA = tid / (BK / 4);  // 0..127
  const int innerColA = tid % (BK / 4);  // 0..1     (each "col" is a 4-float chunk)
  // Each thread cooperatively loads exactly one float4 of B (BK*BN/4 = 256 vecs / 256 threads).
  const int innerRowB = tid / (BN / 4);  // 0..7
  const int innerColB = tid % (BN / 4);  // 0..31

  float threadResults[TM * TN] = {0.0f};
  float regM[TM];
  float regN[TN];

  for (int k0 = 0; k0 < K; k0 += BK) {
    // Vectorized load of A's tile, then transpose-store into As.
    float4 a = *reinterpret_cast<const float4*>(&A[innerRowA * K + innerColA * 4]);
    As[(innerColA * 4 + 0) * BM + innerRowA] = a.x;
    As[(innerColA * 4 + 1) * BM + innerRowA] = a.y;
    As[(innerColA * 4 + 2) * BM + innerRowA] = a.z;
    As[(innerColA * 4 + 3) * BM + innerRowA] = a.w;

    // Vectorized load of B's tile, store as-is.
    *reinterpret_cast<float4*>(&Bs[innerRowB * BN + innerColB * 4]) =
        *reinterpret_cast<const float4*>(&B[innerRowB * N + innerColB * 4]);

    __syncthreads();

    A += BK;
    B += BK * N;

#pragma unroll
    for (int k = 0; k < BK; ++k) {
      // Two float4 reads grab the 8 contiguous A elements this thread needs.
      *reinterpret_cast<float4*>(&regM[0]) =
          *reinterpret_cast<float4*>(&As[k * BM + threadRow * TM + 0]);
      *reinterpret_cast<float4*>(&regM[4]) =
          *reinterpret_cast<float4*>(&As[k * BM + threadRow * TM + 4]);
      *reinterpret_cast<float4*>(&regN[0]) =
          *reinterpret_cast<float4*>(&Bs[k * BN + threadCol * TN + 0]);
      *reinterpret_cast<float4*>(&regN[4]) =
          *reinterpret_cast<float4*>(&Bs[k * BN + threadCol * TN + 4]);

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

  // Vectorized C update: each thread writes 2 x float4 per output row.
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int r = threadRow * TM + i;
#pragma unroll
    for (int j = 0; j < TN; j += 4) {
      const int c = threadCol * TN + j;
      float4 c_old = *reinterpret_cast<float4*>(&C[r * N + c]);
      float4 c_new;
      c_new.x = alpha * threadResults[i * TN + j + 0] + beta * c_old.x;
      c_new.y = alpha * threadResults[i * TN + j + 1] + beta * c_old.y;
      c_new.z = alpha * threadResults[i * TN + j + 2] + beta * c_old.z;
      c_new.w = alpha * threadResults[i * TN + j + 3] + beta * c_old.w;
      *reinterpret_cast<float4*>(&C[r * N + c]) = c_new;
    }
  }
}

}  // namespace

extern "C" void launch_sgemm_vectorized(int M, int N, int K, float alpha, const float* A,
                                        const float* B, float beta, float* C,
                                        cudaStream_t stream) {
  dim3 block(THREADS);
  dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  sgemm_vectorized_kernel<<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
}
