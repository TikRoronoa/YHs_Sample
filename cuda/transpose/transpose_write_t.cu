#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

// 对照 transpose.cu：新增「写 smem 时转置」版本
// - tiled+pad：原样写入 smem，读出时转置（与 transpose.cu 相同）
// - tiled_wT：写入 smem 时转置，读出按行（全局写仍合并）

constexpr int TILE = 32;
constexpr int BLOCK_ROWS = 8;

__global__ void transpose_naive(const float *__restrict__ in, float *__restrict__ out,
                                int M, int N) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < N && y < M) {
    out[x * M + y] = in[y * N + x];
  }
}

// ---------- 原版：读入不转置，读出时转置 ----------
__global__ void transpose_tiled(const float *__restrict__ in, float *__restrict__ out,
                                int M, int N) {
  __shared__ float tile[TILE][TILE + 1];

  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;

#pragma unroll
  for (int j = 0; j < TILE; j += BLOCK_ROWS) {
    if (x < N && (y + j) < M) {
      tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * N + x];
    }
  }
  __syncthreads();

  x = blockIdx.y * TILE + threadIdx.x;
  y = blockIdx.x * TILE + threadIdx.y;

#pragma unroll
  for (int j = 0; j < TILE; j += BLOCK_ROWS) {
    if (x < M && (y + j) < N) {
      out[(y + j) * M + x] = tile[threadIdx.x][threadIdx.y + j];
    }
  }
}

// ---------- 新版：写入 smem 时转置，读出按行 ----------
__global__ void transpose_tiled_write_T(const float *__restrict__ in,
                                        float *__restrict__ out, int M, int N) {
  __shared__ float tile[TILE][TILE + 1];

  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;

  // 全局合并读 → smem 按转置下标写（tile[col][row]）
#pragma unroll
  for (int j = 0; j < TILE; j += BLOCK_ROWS) {
    if (x < N && (y + j) < M) {
      tile[threadIdx.x][threadIdx.y + j] = in[(y + j) * N + x];
    }
  }
  __syncthreads();

  x = blockIdx.y * TILE + threadIdx.x;
  y = blockIdx.x * TILE + threadIdx.y;

  // smem 按行读 → 全局合并写
#pragma unroll
  for (int j = 0; j < TILE; j += BLOCK_ROWS) {
    if (x < M && (y + j) < N) {
      out[(y + j) * M + x] = tile[threadIdx.y + j][threadIdx.x];
    }
  }
}

__global__ void copy_kernel(const float *__restrict__ in, float *__restrict__ out,
                            int M, int N) {
  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;
#pragma unroll
  for (int j = 0; j < TILE; j += BLOCK_ROWS) {
    if (x < N && (y + j) < M) {
      out[(y + j) * N + x] = in[(y + j) * N + x];
    }
  }
}

static void check(cudaError_t e, const char *msg) {
  if (e != cudaSuccess) {
    fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(e));
    exit(1);
  }
}

static bool verify(const float *in, const float *out, int M, int N) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float expect = in[i * N + j];
      float got = out[j * M + i];
      if (fabsf(expect - got) > 1e-5f) {
        printf("mismatch at (%d,%d): %f vs %f\n", i, j, expect, got);
        return false;
      }
    }
  }
  return true;
}

template <typename Kernel>
static float bench(Kernel k, dim3 grid, dim3 block, const float *d_in, float *d_out,
                   int M, int N, int repeat, const char *name) {
  k<<<grid, block>>>(d_in, d_out, M, N);
  check(cudaDeviceSynchronize(), "warmup");

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int i = 0; i < repeat; ++i) {
    k<<<grid, block>>>(d_in, d_out, M, N);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0.f;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  double bytes = 2.0 * (double)M * (double)N * sizeof(float) * repeat;
  double gbps = bytes / (ms / 1000.0) / 1e9;
  printf("%-16s  %.3f ms (avg %.4f)  %.1f GB/s\n", name, ms, ms / repeat, gbps);
  return ms;
}

int main() {
  const int M = 4096;
  const int N = 4096;
  const int repeat = 100;

  size_t bytes = (size_t)M * N * sizeof(float);
  float *h_in = new float[M * N];
  float *h_out = new float[M * N];
  for (int i = 0; i < M * N; ++i) {
    h_in[i] = static_cast<float>(i % 997) * 0.001f;
  }

  float *d_in = nullptr, *d_out = nullptr;
  check(cudaMalloc(&d_in, bytes), "cudaMalloc in");
  check(cudaMalloc(&d_out, bytes), "cudaMalloc out");
  check(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice), "H2D");

  dim3 block_naive(32, 8);
  dim3 grid_naive((N + block_naive.x - 1) / block_naive.x,
                  (M + block_naive.y - 1) / block_naive.y);

  dim3 block_tiled(TILE, BLOCK_ROWS);
  dim3 grid_tiled((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

  printf("transpose_write_t %d x %d, repeat=%d\n", M, N, repeat);

  bench(copy_kernel, grid_tiled, block_tiled, d_in, d_out, M, N, repeat, "copy");
  bench(transpose_naive, grid_naive, block_naive, d_in, d_out, M, N, repeat, "naive");
  bench(transpose_tiled, grid_tiled, block_tiled, d_in, d_out, M, N, repeat,
        "tiled+pad");
  bench(transpose_tiled_write_T, grid_tiled, block_tiled, d_in, d_out, M, N,
        repeat, "tiled_wT");

  check(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost), "D2H");
  printf("verify: %s\n", verify(h_in, h_out, M, N) ? "OK" : "FAIL");

  cudaFree(d_in);
  cudaFree(d_out);
  delete[] h_in;
  delete[] h_out;
  return 0;
}
