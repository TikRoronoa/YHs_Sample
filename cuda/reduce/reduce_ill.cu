#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>



/*
main 函数创建一个填充随机数的长度为 2^24 的一维数组，存入 GPU 全局内存，交由 kernel 求总和：
  reduce_ill<<<gridSize, blockSize, shmemSize>>>(input, output, n);
其中
  - blockSize 取 256，作者习惯用这个值，设计每个 thread 负责 8 个数；
  - gridSize 取 8192。已设计每个 block 要负责 2048 个数
    （256 线程 × 每线程 8 个），故取 2^24 / 2048 = 8192，正好把整个
    数组不重不漏地分完；
  - shmemSize 取 8192。设计 kernel 里的每一步都建立在
    “先把 2048 个数整个搬进 smem 再慢慢折腾”的前提上，所以 smem
    需要放得下 2048 个 float，故取 2048 × sizeof(float) = 8192，即 8 KB。
  - n 取 1 << 24 即 2^24。待求和数组总长度。
  - input 取 n × sizeof(float)。
  - output 取 gridSize × sizeof(float)。每个 block 把自己负责的 2048
    个数求成一个部分和，main 函数最后再把 8192 个部分和加起来，得到最终结果。
*/
__global__ void reduce_ill(float* input, float* output, int n) {

    extern __shared__ float smem[];

    // 【病态 1】非合并的全局读
    // 循环 8 轮，每轮 block 里的 256 个 thread 各搬 1 个数到共享内存，
    // 一轮 256 个，八轮搬够 2048 个。
    for (int i = 0; i < 8; ++i) {
        // 搬到哪：故意把 smem 看成一个 8 行 256 列的矩阵，各 thread 每轮
        // 合作填满矩阵的一整行，八轮即可填满整个矩阵
        const int smem_idx = blockDim.x * i + threadIdx.x;
        // 从哪搬：又故意把 input 看成一个 2048 行 8192 列的矩阵，第 blockIdx.x 列
        // 整列归本 block 管，每轮各 thread 从这一列取 256 个数，
        // 八轮取完矩阵整个第 blockIdx.x 列
        const int globalIdx = gridDim.x * smem_idx + blockIdx.x;
        // 开始搬，对于加法 reduce，越界的元素填 0
        smem[smem_idx] = (globalIdx < n) ? input[globalIdx] : 0.0f;
    }
    __syncthreads();

    // 【病态 2】共享内存读冲突
    // 又故意把 smem 看成一个 256 行 8 列的矩阵，各 thread 都在循环中负责把
    // 第 threadIdx.x 行的 8 个数加起来
    float threadSum = 0.0f;
    for (int i = 0; i < 8; ++i) {
        threadSum += smem[8 * threadIdx.x + i];
    }
    __syncthreads();
    // 各 thread 求和完成后，再把结果存为长为 256 的一维数组排放在共享内存的低位
    smem[threadIdx.x] = threadSum;
    __syncthreads();

    // 【病态 3】朴素树形归并
    // 这里是很常见的朴素树形归并，随着步数的翻倍，各 thread 不停检查现在需不需要自己干活，
    // 有活就干没活闲着，直到 256 个数完成求和
    for (int step = 1; step < blockDim.x; step *= 2) {
        if (threadIdx.x % (2 * step) == 0) {
            smem[threadIdx.x] += smem[threadIdx.x + step];
        }
        __syncthreads();
    }

    // 【病态 4】稀疏写回
    // 终于，本 block 负责的 2048 个数的和，存入结果数组，待 main 读回CPU侧后求最终和
    if (threadIdx.x == 0) {
        output[blockIdx.x] = smem[0];
    }
}



int main() {
    const int n = 1 << 24;
    const int blockSize = 256;
    const int gridSize = 8192;                          // n / (256*8)
    const size_t shmemSize = 2048 * sizeof(float);      // 8192 bytes
    float *h_input = (float *)malloc(n * sizeof(float));
    float *h_output = (float *)malloc(gridSize * sizeof(float));
    for (int i = 0; i < n; ++i) {
      h_input[i] = (float)rand() / RAND_MAX;
    }
    float *d_input = nullptr;
    float *d_output = nullptr;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, gridSize * sizeof(float));
    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);
    reduce_ill<<<gridSize, blockSize, shmemSize>>>(d_input, d_output, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
    double gpu_sum = 0.0;
    for (int i = 0; i < gridSize; ++i) {
      gpu_sum += h_output[i];
    }
    double cpu_sum = 0.0;
    for (int i = 0; i < n; ++i) {
      cpu_sum += h_input[i];
    }
    printf("gpu sum = %.6f\ncpu sum = %.6f\n", gpu_sum, cpu_sum);
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    return 0;
  }
  