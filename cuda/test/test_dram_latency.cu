#include <cstdint>
#include <cuda_runtime.h>
#include <stdio.h>

const int THREADS = 32;
const int STRIDE = 256;
const int ROUND = 100;

const int DATA_SIZE = (1 << 20) * 128; // 单位是bytes

const int MEM_TOTAL = (ROUND + 1) * STRIDE; // 这里的1用于warpup，填充TLB
template <int REPEAT>
__global__ void dram_latency_kernel(int *ptr, uint64_t *clocks, int *output) {
  int tid = threadIdx.x;
  // 先warpup一次，刷一下TLB
  int value_first = __ldcg(ptr + tid);
  ptr += value_first;

  __syncthreads();
  int value;
  auto t_start = clock64();
#pragma unroll
  for (int i = 0; i < REPEAT; ++i) {
    value = __ldcg(ptr + tid);
    ptr += value;
  }

  __syncthreads();
  auto t_end = clock64();
  clocks[tid] = t_end - t_start;
  output[tid] = value;
}

template <int BLOCKSIZE> __global__ void flushL2(int *x, int *y) {
  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;
  int index = blockIdx.x * BLOCKSIZE + warp_id * 32 + lane_id;
  int value = __ldcg(x + index);
  if (threadIdx.x == 0 && blockIdx.x == 0)
    *y = value;
}
void flush() {

  int *x;
//   int *y;
  cudaMalloc(&x, DATA_SIZE);
//   cudaMalloc(&y, sizeof(int));
  cudaMemset(x, 0, DATA_SIZE);//这样就可以flush掉L2，不需要再读一遍了

//   int n = DATA_SIZE / sizeof(int);
//   flushL2<128><<<n / 128, 128>>>(x, y);

  cudaFree(x);
//   cudaFree(y);
}

int main() {

  int *ptr;

  uint64_t *clocks;

  // 分配host内存
  cudaMallocHost(&ptr, MEM_TOTAL * sizeof(int));
  cudaMallocHost(&clocks, THREADS * sizeof(uint64_t));
  // 分配device内存
  int *ptr_device;
  int *output;
  uint64_t *clocks_device;
  cudaMalloc(&ptr_device, MEM_TOTAL * sizeof(int));
  cudaMalloc(&clocks_device, THREADS * sizeof(uint64_t));
  cudaMalloc(&output, THREADS*sizeof(int));
  // 构造输入
  for (int i = 0; i < MEM_TOTAL; ++i) {
    ptr[i] = STRIDE;
  }
  // copy input from host to device
  cudaMemcpy(ptr_device, ptr, MEM_TOTAL * sizeof(int), cudaMemcpyHostToDevice);

  // 先跑一遍，填充TLB icache， dcache会被后面的flush冲掉
  dram_latency_kernel<ROUND><<<1, THREADS>>>(ptr_device, clocks_device, output);
  // flush L2
  flush();
  // call kernel
  dram_latency_kernel<ROUND><<<1, THREADS>>>(ptr_device, clocks_device, output);

  // copy result
  cudaMemcpy(clocks, clocks_device, THREADS * sizeof(uint64_t),
             cudaMemcpyDeviceToHost);

  // 统计clocks中的数据
  printf("DRAM latency %llu cycles\n",
    (unsigned long long)(clocks[0] / ROUND));
  // free
  // free host mem
  cudaFreeHost(ptr);
  cudaFreeHost(clocks);
  // free device mem
  cudaFree(ptr_device);
  cudaFree(clocks_device);
  cudaFree(output);

  return 0;
}
