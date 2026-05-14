// Benchmark harness for the SGEMM tutorial kernels.
//
// Validates every kernel against cuBLAS SGEMM (max abs error must be small) and then
// times each one with L2 flushed between replays. Prints a summary table and optionally
// writes results.json for the companion notebook to load.
//
// Build: see Makefile in this directory.
// Run:   ./bench --size 4096 --iters 50 --json results.json
//
// Notes on cuBLAS row-major:
//   cuBLAS is column-major. A row-major M x K matrix shares its bytes with a column-major
//   K x M matrix (= A^T in the cuBLAS view). To compute C = A @ B in row-major, we ask
//   cuBLAS to compute C^T = B^T @ A^T in column-major, which is the SAME bytes as C
//   row-major. Concretely: cublasSgemm(N, N, n, m, k, &a, B, n, A, k, &b, C, n).

#include "common.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cublas_v2.h>
#include <random>
#include <string>
#include <vector>

#define CUBLAS_CHECK(call)                                                                \
  do {                                                                                    \
    cublasStatus_t _s = (call);                                                           \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                                    \
      std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)_s, __FILE__, __LINE__);    \
      std::exit(1);                                                                       \
    }                                                                                     \
  } while (0)

struct KernelEntry {
  const char* name;
  void (*launch)(int, int, int, float, const float*, const float*, float, float*,
                 cudaStream_t);
};

static const KernelEntry kKernels[] = {
    {"01_naive", launch_sgemm_naive},
    {"02_smem", launch_sgemm_smem},
    {"03_block_tile", launch_sgemm_block_tile},
    {"04_vectorized", launch_sgemm_vectorized},
    {"05_async_pipeline", launch_sgemm_async},
};

static void cublas_sgemm_rowmajor(cublasHandle_t handle, int M, int N, int K, float alpha,
                                   const float* A, const float* B, float beta, float* C) {
  // Swap A,B and dims to get a row-major result out of column-major cuBLAS.
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K,
                            &beta, C, N));
}

static double max_abs_error(const std::vector<float>& a, const std::vector<float>& b) {
  double m = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    double d = std::fabs((double)a[i] - (double)b[i]);
    if (d > m) m = d;
  }
  return m;
}

int main(int argc, char** argv) {
  int size = 4096;
  int iters = 50;
  int warmup = 5;
  bool validate = true;
  std::string json_path;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--size" && i + 1 < argc) size = std::atoi(argv[++i]);
    else if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
    else if (a == "--warmup" && i + 1 < argc) warmup = std::atoi(argv[++i]);
    else if (a == "--no-validate") validate = false;
    else if (a == "--json" && i + 1 < argc) json_path = argv[++i];
    else if (a == "--help") {
      std::printf("usage: %s [--size N] [--iters N] [--warmup N] [--no-validate] "
                  "[--json PATH]\n", argv[0]);
      return 0;
    }
  }

  const int M = size, N = size, K = size;
  const float alpha = 1.0f, beta = 0.0f;
  const size_t bytesA = (size_t)M * K * sizeof(float);
  const size_t bytesB = (size_t)K * N * sizeof(float);
  const size_t bytesC = (size_t)M * N * sizeof(float);
  const double flops = 2.0 * (double)M * N * K;  // M*N outputs, each 2K FLOPs

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("GPU: %s  (SM %d.%d, %d SMs)\n", prop.name, prop.major, prop.minor,
              prop.multiProcessorCount);
  std::printf("Problem: M=N=K=%d  (%.2f GFLOPs/call, %.1f MB working set)\n", size,
              flops * 1e-9, (bytesA + bytesB + bytesC) / (1024.0 * 1024.0));
  std::printf("Iterations: %d (warmup %d), validate=%s\n\n", iters, warmup,
              validate ? "yes" : "no");

  // Host buffers + RNG.
  std::vector<float> hA((size_t)M * K), hB((size_t)K * N), hC_ref((size_t)M * N),
      hC((size_t)M * N);
  std::mt19937 rng(0);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& x : hA) x = dist(rng);
  for (auto& x : hB) x = dist(rng);

  // Device buffers.
  float *dA, *dB, *dC;
  CUDA_CHECK(cudaMalloc(&dA, bytesA));
  CUDA_CHECK(cudaMalloc(&dB, bytesB));
  CUDA_CHECK(cudaMalloc(&dC, bytesC));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));

  // L2 flush buffer: bigger than L2, written between replays to evict cache lines.
  size_t l2_size = 0;
  cudaDeviceGetAttribute((int*)&l2_size, cudaDevAttrL2CacheSize, 0);
  size_t flush_bytes = std::max<size_t>(2 * l2_size, 64 * 1024 * 1024);
  void* dFlush = nullptr;
  CUDA_CHECK(cudaMalloc(&dFlush, flush_bytes));

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  // ---- Reference: cuBLAS SGEMM (run twice, take the second to avoid first-call setup). ----
  CUDA_CHECK(cudaMemsetAsync(dC, 0, bytesC));
  cublas_sgemm_rowmajor(handle, M, N, K, alpha, dA, dB, beta, dC);
  CUDA_CHECK(cudaMemcpy(hC_ref.data(), dC, bytesC, cudaMemcpyDeviceToHost));

  // ---- Time cuBLAS for reference. ----
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  for (int i = 0; i < warmup; ++i) cublas_sgemm_rowmajor(handle, M, N, K, alpha, dA, dB, beta, dC);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> times_ms(iters);
  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(cudaMemsetAsync(dFlush, 0, flush_bytes));
    CUDA_CHECK(cudaEventRecord(start));
    cublas_sgemm_rowmajor(handle, M, N, K, alpha, dA, dB, beta, dC);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&times_ms[i], start, stop));
  }
  std::sort(times_ms.begin(), times_ms.end());
  float cublas_med = times_ms[iters / 2];
  double cublas_tflops = flops * 1e-12 / (cublas_med * 1e-3);

  std::printf("%-22s %12s %12s %10s %12s\n", "kernel", "median (ms)", "TFLOPS",
              "% cuBLAS", "max |err|");
  std::printf("%-22s %12.3f %12.2f %10s %12s\n", "cublasSgemm (ref)", cublas_med,
              cublas_tflops, "100.0%", "0");

  // ---- For each kernel: validate then time. ----
  struct Result {
    std::string name;
    float median_ms;
    double tflops;
    double pct_cublas;
    double max_err;
  };
  std::vector<Result> results;
  results.push_back({"cublas", cublas_med, cublas_tflops, 100.0, 0.0});

  for (const auto& kernel : kKernels) {
    double err = 0.0;
    if (validate) {
      CUDA_CHECK(cudaMemsetAsync(dC, 0, bytesC));
      kernel.launch(M, N, K, alpha, dA, dB, beta, dC, 0);
      CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytesC, cudaMemcpyDeviceToHost));
      err = max_abs_error(hC, hC_ref);
    }

    for (int i = 0; i < warmup; ++i) kernel.launch(M, N, K, alpha, dA, dB, beta, dC, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < iters; ++i) {
      CUDA_CHECK(cudaMemsetAsync(dFlush, 0, flush_bytes));
      CUDA_CHECK(cudaEventRecord(start));
      kernel.launch(M, N, K, alpha, dA, dB, beta, dC, 0);
      CUDA_CHECK(cudaEventRecord(stop));
      CUDA_CHECK(cudaEventSynchronize(stop));
      CUDA_CHECK(cudaEventElapsedTime(&times_ms[i], start, stop));
    }
    std::sort(times_ms.begin(), times_ms.end());
    float median = times_ms[iters / 2];
    double tflops = flops * 1e-12 / (median * 1e-3);
    double pct = 100.0 * tflops / cublas_tflops;
    std::printf("%-22s %12.3f %12.2f %9.1f%% %12.2g\n", kernel.name, median, tflops, pct,
                err);
    results.push_back({kernel.name, median, tflops, pct, err});
  }

  // ---- Optional JSON output. ----
  if (!json_path.empty()) {
    FILE* f = std::fopen(json_path.c_str(), "w");
    if (!f) {
      std::fprintf(stderr, "Failed to open %s for writing.\n", json_path.c_str());
    } else {
      std::fprintf(f, "{\n  \"gpu\": \"%s\",\n  \"sm\": \"%d.%d\",\n  \"size\": %d,\n",
                   prop.name, prop.major, prop.minor, size);
      std::fprintf(f, "  \"iters\": %d,\n  \"results\": [\n", iters);
      for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        std::fprintf(f,
                     "    {\"kernel\": \"%s\", \"median_ms\": %.6f, \"tflops\": %.4f, "
                     "\"pct_cublas\": %.4f, \"max_err\": %.6g}%s\n",
                     r.name.c_str(), r.median_ms, r.tflops, r.pct_cublas, r.max_err,
                     i + 1 == results.size() ? "" : ",");
      }
      std::fprintf(f, "  ]\n}\n");
      std::fclose(f);
      std::printf("\nWrote %s\n", json_path.c_str());
    }
  }

  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  CUDA_CHECK(cudaFree(dFlush));
  cublasDestroy(handle);
  return 0;
}
