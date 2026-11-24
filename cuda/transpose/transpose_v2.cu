#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

// 相对 transpose.cu 的新版本对照文件
// 新增：copy_kernel_vec4、transpose_tiled_vec4_load（读 float4，写仍标量）
// block vec4 均为 (8,32)：tx=0..7, ty=0..31

constexpr int TILE = 32;
constexpr int BLOCK_ROWS = 8;  // 每线程负责 TILE/BLOCK_ROWS 行，提高 ILP
constexpr int VEC = 4;
constexpr int VEC_BLOCK_X = TILE / VEC;  // 8
constexpr int VEC_BLOCK_Y = TILE;        // 32

// ---------- naive：全局直写，写路径不合并 ----------
__global__ void transpose_naive(const float *__restrict__ in, float *__restrict__ out,
                                int M, int N) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;  // col of in
  int y = blockIdx.y * blockDim.y + threadIdx.y;  // row of in
  if (x < N && y < M) {
    out[x * M + y] = in[y * N + x];  // in(y,x) -> out(x,y)
  }
}

// ---------- tiled：smem 中转，[TILE][TILE+1] 消 bank conflict ----------
__global__ void transpose_tiled(const float *__restrict__ in, float *__restrict__ out,
                                int M, int N) {
  __shared__ float tile[TILE][TILE + 1];

  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;

  // 合并读：同一 warp 读连续地址，写入 smem（带 padding）
#pragma unroll
  for (int j = 0; j < TILE; j += BLOCK_ROWS) {
    if (x < N && (y + j) < M) {
      tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * N + x];
    }
  }
  __syncthreads();

  // 转置后的全局坐标
  x = blockIdx.y * TILE + threadIdx.x;
  y = blockIdx.x * TILE + threadIdx.y;

  // 合并写：从 smem 按转置下标读出
#pragma unroll
  for (int j = 0; j < TILE; j += BLOCK_ROWS) {
    if (x < M && (y + j) < N) {
      out[(y + j) * M + x] = tile[threadIdx.x][threadIdx.y + j];
    }
  }
}

// ---------- tiled：阶段1 float4 宽读+宽写 smem，阶段2 标量写全局 ----------
// block=(8,32)；每线程负责一行内连续 4 列
// smem 行距用 TILE+4（不是 +1）：保证 &tile[ty][tx*4] 16B 对齐，可安全 st.v4
__global__ void transpose_tiled_vec4_load(const float *__restrict__ in,
                                          float *__restrict__ out, int M,
                                          int N) {
  __shared__ float tile[TILE][TILE + VEC];

  const int tx = threadIdx.x;  // 0..7
  const int ty = threadIdx.y;  // 0..31

  const int x = blockIdx.x * TILE + tx * VEC;
  const int y = blockIdx.y * TILE + ty;

  if (x + VEC - 1 < N && y < M) {
    float4 v = *reinterpret_cast<const float4 *>(&in[y * N + x]);
    *reinterpret_cast<float4 *>(&tile[ty][tx * VEC]) = v;
  }
  __syncthreads();

  // 转置后写出：每线程写连续 4 列（标量 store，合并）
  const int out_x = blockIdx.y * TILE + tx * VEC;
  const int out_y = blockIdx.x * TILE + ty;

  if (out_x + VEC - 1 < M && out_y < N) {
    out[out_y * M + out_x + 0] = tile[tx * VEC + 0][ty];
    out[out_y * M + out_x + 1] = tile[tx * VEC + 1][ty];
    out[out_y * M + out_x + 2] = tile[tx * VEC + 2][ty];
    out[out_y * M + out_x + 3] = tile[tx * VEC + 3][ty];
  }
}

// ---------- 原版 copy：标量 load/store ----------
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

// ---------- 新版 copy：vector load/store ----------
// tile 覆盖仍 32x32；block=(8,32)：tx=0..7, ty=0..31；每线程连续 4 列
__global__ void copy_kernel_vec4(const float *__restrict__ in, float *__restrict__ out,
                                 int M, int N) {
  const int x = blockIdx.x * TILE + threadIdx.x * VEC;
  const int y = blockIdx.y * TILE + threadIdx.y;
  if (x + VEC - 1 < N && y < M) {
    float4 v = *reinterpret_cast<const float4 *>(&in[y * N + x]);
    *reinterpret_cast<float4 *>(&out[y * N + x]) = v;
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
  // warmup
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

  // 读 + 写各一遍
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

  dim3 block_tiled(TILE, BLOCK_ROWS);              // 32 x 8
  dim3 block_vec4(VEC_BLOCK_X, VEC_BLOCK_Y);       // 8 x 32
  dim3 grid_tiled((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

  printf("transpose_v2 %d x %d (float), repeat=%d\n", M, N, repeat);
  printf("scalar block=(%d,%d), vec4 block=(%d,%d)\n", TILE, BLOCK_ROWS,
         VEC_BLOCK_X, VEC_BLOCK_Y);

  bench(copy_kernel, grid_tiled, block_tiled, d_in, d_out, M, N, repeat, "copy");
  bench(copy_kernel_vec4, grid_tiled, block_vec4, d_in, d_out, M, N, repeat,
        "copy_vec4");
  bench(transpose_naive, grid_naive, block_naive, d_in, d_out, M, N, repeat, "naive");
  bench(transpose_tiled, grid_tiled, block_tiled, d_in, d_out, M, N, repeat,
        "tiled+pad");
  bench(transpose_tiled_vec4_load, grid_tiled, block_vec4, d_in, d_out, M, N,
        repeat, "tiled_v4load");

  check(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost), "D2H");
  printf("verify: %s\n", verify(h_in, h_out, M, N) ? "OK" : "FAIL");

  cudaFree(d_in);
  cudaFree(d_out);
  delete[] h_in;
  delete[] h_out;
  return 0;
}
