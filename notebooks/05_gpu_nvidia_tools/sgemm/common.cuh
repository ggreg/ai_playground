// Shared utilities for the SGEMM tutorial kernels.
//
// All kernels compute  C = alpha * A @ B + beta * C  with row-major matrices,
//   A : M x K
//   B : K x N
//   C : M x N
// matching the convention of cuBLAS column-major B^T @ A^T = C^T (handled in bench.cu).

#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

#define CUDA_CHECK(call)                                                                  \
  do {                                                                                    \
    cudaError_t _err = (call);                                                            \
    if (_err != cudaSuccess) {                                                            \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_err),          \
                   __FILE__, __LINE__);                                                   \
      std::exit(1);                                                                       \
    }                                                                                     \
  } while (0)

// Each kernel file defines one of these.
extern "C" {
void launch_sgemm_naive(int M, int N, int K, float alpha, const float* A, const float* B,
                        float beta, float* C, cudaStream_t stream);
void launch_sgemm_smem(int M, int N, int K, float alpha, const float* A, const float* B,
                       float beta, float* C, cudaStream_t stream);
void launch_sgemm_block_tile(int M, int N, int K, float alpha, const float* A,
                             const float* B, float beta, float* C, cudaStream_t stream);
void launch_sgemm_vectorized(int M, int N, int K, float alpha, const float* A,
                             const float* B, float beta, float* C, cudaStream_t stream);
void launch_sgemm_async(int M, int N, int K, float alpha, const float* A, const float* B,
                        float beta, float* C, cudaStream_t stream);
}
