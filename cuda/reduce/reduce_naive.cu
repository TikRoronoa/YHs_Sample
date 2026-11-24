#include <cstdio>
#include <cuda_runtime.h>

__global__ void reduce_sum(const float *__restrict__ input, float *output,
                           int N) {
  __shared__ float warp_sums[32];

  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp = tid >> 5;

  float sum = 0.0f;

  // 每个线程处理多个元素
  for (int i = tid; i < N; i += blockDim.x) {
    sum += input[i];
  }

  // Warp-level reduction
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }

  // 每个 Warp 写一个结果
  if (lane == 0) {
    warp_sums[warp] = sum;
  }

  __syncthreads();

  // 第一个 Warp 完成最终 Reduce
  if (warp == 0) {
    int num_warps = (blockDim.x + 31) / 32;

    sum = (lane < num_warps) ? warp_sums[lane] : 0.0f;

    for (int offset = 16; offset > 0; offset >>= 1) {
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
      atomicAdd(output, sum);
    }
  }
}

int main() {
    const int N = 1 << 20;  // 1024*1024
    const int threads = 256;
  
    float *h_input = new float[N];
    for (int i = 0; i < N; ++i) {
      h_input[i] = 1.0f;  // 期望 sum = N
    }
  
    float *d_input = nullptr;
    float *d_output = nullptr;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));
  
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, sizeof(float));
  
    reduce_sum<<<1, threads>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
  
    float h_output = 0.0f;
    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
  
    printf("gpu sum = %f, expected = %f\n", h_output, (float)N);
  
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    return 0;
  }