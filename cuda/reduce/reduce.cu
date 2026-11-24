#include <cstdio>
#include <cuda_runtime.h>

// ---------- warp reduce ----------
__device__ __forceinline__ float warp_reduce_sum(float v) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, offset);
  }
  return v;
}

// ---------- stage1: grid 读全局，每 block 出一个 partial ----------
template <int BLOCK>
__global__ void reduce_stage1(const float* __restrict__ input,
                              float* __restrict__ partial,
                              int N) {
  __shared__ float warp_sums[BLOCK / 32];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  // float4 向量读；grid-stride 拉满带宽 + ILP
  float sum = 0.f;
  const float4* input4 = reinterpret_cast<const float4*>(input);
  const int n4 = N >> 2;  // N 建议是 4 的倍数

  for (int i = blockIdx.x * BLOCK + tid; i < n4; i += gridDim.x * BLOCK) {
    float4 v = input4[i];
    sum += v.x + v.y + v.z + v.w;
  }

  // 处理尾部（N 不是 4 对齐时）
  for (int i = (n4 << 2) + blockIdx.x * BLOCK + tid; i < N; i += gridDim.x * BLOCK) {
    sum += input[i];
  }

  sum = warp_reduce_sum(sum);
  if (lane == 0) {
    warp_sums[warp] = sum;
  }
  __syncthreads();

  if (warp == 0) {
    sum = (lane < (BLOCK / 32)) ? warp_sums[lane] : 0.f;
    sum = warp_reduce_sum(sum);
    if (lane == 0) {
      partial[blockIdx.x] = sum;
    }
  }
}

// ---------- stage2: 归约 partial ----------
template <int BLOCK>
__global__ void reduce_stage2(const float* __restrict__ partial,
                              float* __restrict__ output,
                              int n_partial) {
  __shared__ float warp_sums[BLOCK / 32];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  float sum = 0.f;
  for (int i = tid; i < n_partial; i += BLOCK) {
    sum += partial[i];
  }

  sum = warp_reduce_sum(sum);
  if (lane == 0) {
    warp_sums[warp] = sum;
  }
  __syncthreads();

  if (warp == 0) {
    sum = (lane < (BLOCK / 32)) ? warp_sums[lane] : 0.f;
    sum = warp_reduce_sum(sum);
    if (lane == 0) {
      output[0] = sum;
    }
  }
}

int main() {
  const int N = 1 << 24;          // 更大一点更能看出带宽优势；可改回 1<<20
  const int BLOCK = 256;
  // 经验：让 grid 别太小也不要远超 SM 数很多倍；这里按元素粗算
  int grid = (N / 4 + BLOCK - 1) / BLOCK;
  if (grid > 2048) grid = 2048;   // 限制上限，避免 partial 过大
  if (grid < 1) grid = 1;

  float* h_input = new float[N];
  for (int i = 0; i < N; ++i) h_input[i] = 1.f;

  float *d_input = nullptr, *d_partial = nullptr, *d_output = nullptr;
  cudaMalloc(&d_input, N * sizeof(float));
  cudaMalloc(&d_partial, grid * sizeof(float));
  cudaMalloc(&d_output, sizeof(float));
  cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // warmup
  reduce_stage1<BLOCK><<<grid, BLOCK>>>(d_input, d_partial, N);
  reduce_stage2<BLOCK><<<1, BLOCK>>>(d_partial, d_output, grid);
  cudaDeviceSynchronize();

  const int REPEAT = 100;
  cudaEventRecord(start);
  for (int i = 0; i < REPEAT; ++i) {
    reduce_stage1<BLOCK><<<grid, BLOCK>>>(d_input, d_partial, N);
    reduce_stage2<BLOCK><<<1, BLOCK>>>(d_partial, d_output, grid);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0.f;
  cudaEventElapsedTime(&ms, start, stop);

  float h_output = 0.f;
  cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

  double bytes = (double)N * sizeof(float) * REPEAT;  // 主要是读
  double gbps = bytes / (ms / 1000.0) / 1e9;

  printf("sum = %f (expected %f)\n", h_output, (float)N);
  printf("time = %f ms (avg %f ms), ~%.1f GB/s (read)\n",
         ms, ms / REPEAT, gbps);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_input);
  cudaFree(d_partial);
  cudaFree(d_output);
  delete[] h_input;
  return 0;
}