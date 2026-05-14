// Kernel 5: cp.async + double-buffered software pipeline (Ampere+, SM_80+).
//
// What this teaches: even with vectorized loads, kernel 4 still serializes "load tile,
// then compute tile, then load next tile, then compute". The SM's FMA units stall while
// HBM serves each tile. cp.async (PTX `cp.async.ca.shared.global`) issues a global -> shared
// transfer that does NOT pass through registers and does NOT block the issuing thread.
// Combined with two shared-memory stages, we can:
//
//   compute tile t          ┐
//                           ├─── overlap, the SM stays busy
//   prefetch tile t+1 (HBM) ┘
//
// The key PTX instructions:
//   cp.async.ca.shared.global [smem], [gmem], 16;   // copy 16 bytes async
//   cp.async.commit_group;                           // mark a group of pending copies
//   cp.async.wait_all;                               // wait for all groups
//
// Caveat: cp.async preserves the global-memory layout. We still want A in TRANSPOSED
// form in shared memory (so the inner loop can read it as float4). So we cp.async into a
// row-major scratchpad `A_raw`, then have each thread transpose-store it into `As`. The
// transpose is small (one shared-memory pass) and pipelines well with the next async load.
//
// Salykova's article (https://salykova.github.io/sgemm-gpu) eliminates the scratchpad with
// inline-PTX 4-byte cp.async into transposed positions; we keep the scratchpad here for
// readability. See PAPERS.md "GPU Kernels & Performance".
//
// Expected throughput on RTX 3090 (M=N=K=4096): ~18-20 TFLOPS, matching cuBLAS for FP32.

#include "common.cuh"
#include <cstdint>

namespace {

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;
constexpr int TM = 8;
constexpr int TN = 8;
constexpr int THREADS = (BM * BN) / (TM * TN);  // 256
constexpr int STAGES = 2;

__device__ __forceinline__ void cp_async_16(uint32_t smem_int_addr, const void* gmem) {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
               :
               : "r"(smem_int_addr), "l"(gmem));
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

__device__ __forceinline__ void cp_async_wait_all() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_all;\n" ::);
#endif
}

__global__ void sgemm_async_kernel(int M, int N, int K, float alpha,
                                    const float* __restrict__ A,
                                    const float* __restrict__ B, float beta,
                                    float* __restrict__ C) {
  // Three shared buffers per stage: row-major scratchpad for A, transposed A, row-major B.
  __shared__ float A_raw[STAGES][BM][BK];   // 2 * 128 * 8 * 4 = 8 KB
  __shared__ float As[STAGES][BK][BM];      // 2 * 8 * 128 * 4 = 8 KB (transposed for fast inner loop)
  __shared__ float Bs[STAGES][BK][BN];      // 2 * 8 * 128 * 4 = 8 KB

  const int tid = threadIdx.x;
  const int threadCol = tid % (BN / TN);  // 0..15
  const int threadRow = tid / (BN / TN);  // 0..15

  // Per-thread load coordinates: each thread handles one float4 of A and one of B.
  const int innerRowA = tid / (BK / 4);   // 0..127
  const int innerColA = tid % (BK / 4);   // 0..1
  const int innerRowB = tid / (BN / 4);   // 0..7
  const int innerColB = tid % (BN / 4);   // 0..31

  // Move pointers to this block's strip.
  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;

  const int num_tiles = K / BK;

  auto issue_load = [&](int s, int tile_idx) {
    const float* tileA = A + tile_idx * BK;
    const float* tileB = B + tile_idx * BK * N;
    cp_async_16(static_cast<uint32_t>(
                    __cvta_generic_to_shared(&A_raw[s][innerRowA][innerColA * 4])),
                &tileA[innerRowA * K + innerColA * 4]);
    cp_async_16(static_cast<uint32_t>(
                    __cvta_generic_to_shared(&Bs[s][innerRowB][innerColB * 4])),
                &tileB[innerRowB * N + innerColB * 4]);
  };

  auto transpose_A = [&](int s) {
    // Each thread reads its 4 floats from the row-major scratchpad and writes them
    // into the transposed As. Bank-conflict free because the m-dimension matches the
    // thread-id stride for innerRowA.
    float a0 = A_raw[s][innerRowA][innerColA * 4 + 0];
    float a1 = A_raw[s][innerRowA][innerColA * 4 + 1];
    float a2 = A_raw[s][innerRowA][innerColA * 4 + 2];
    float a3 = A_raw[s][innerRowA][innerColA * 4 + 3];
    As[s][innerColA * 4 + 0][innerRowA] = a0;
    As[s][innerColA * 4 + 1][innerRowA] = a1;
    As[s][innerColA * 4 + 2][innerRowA] = a2;
    As[s][innerColA * 4 + 3][innerRowA] = a3;
  };

  float threadResults[TM * TN] = {0.0f};
  float regM[TM];
  float regN[TN];

  auto compute_stage = [&](int s) {
#pragma unroll
    for (int k = 0; k < BK; ++k) {
      *reinterpret_cast<float4*>(&regM[0]) =
          *reinterpret_cast<float4*>(&As[s][k][threadRow * TM + 0]);
      *reinterpret_cast<float4*>(&regM[4]) =
          *reinterpret_cast<float4*>(&As[s][k][threadRow * TM + 4]);
      *reinterpret_cast<float4*>(&regN[0]) =
          *reinterpret_cast<float4*>(&Bs[s][k][threadCol * TN + 0]);
      *reinterpret_cast<float4*>(&regN[4]) =
          *reinterpret_cast<float4*>(&Bs[s][k][threadCol * TN + 4]);
#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          threadResults[i * TN + j] += regM[i] * regN[j];
        }
      }
    }
  };

  // ---- Prologue: load tile 0, transpose, kick off tile 1's load. ----
  issue_load(0, 0);
  cp_async_commit();
  cp_async_wait_all();
  __syncthreads();
  transpose_A(0);
  __syncthreads();

  if (num_tiles > 1) {
    issue_load(1, 1);
    cp_async_commit();
  }

  // ---- Main loop: overlap compute(t) with the in-flight load of tile t+1. ----
  for (int t = 0; t < num_tiles - 1; ++t) {
    const int s_compute = t % STAGES;
    const int s_just_loaded = (t + 1) % STAGES;
    const int next_tile = t + 2;

    compute_stage(s_compute);

    cp_async_wait_all();
    __syncthreads();
    transpose_A(s_just_loaded);
    __syncthreads();

    if (next_tile < num_tiles) {
      issue_load(s_compute, next_tile);  // reuse the now-free stage
      cp_async_commit();
    }
  }

  // ---- Epilogue: compute the last tile. ----
  compute_stage((num_tiles - 1) % STAGES);

  // Vectorized C write-back (same as kernel 4).
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

extern "C" void launch_sgemm_async(int M, int N, int K, float alpha, const float* A,
                                   const float* B, float beta, float* C,
                                   cudaStream_t stream) {
  dim3 block(THREADS);
  dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  sgemm_async_kernel<<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
}
