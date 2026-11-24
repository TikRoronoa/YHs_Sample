#include <cstdio>
#include <cuda_runtime.h>


__global__ void reduce_stage1(
    const float* __restrict__ input,
    float* partial,
    int N
) {
    __shared__ float smem[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 4 + tid;

    float sum = 0.0f;

    // 每个线程连续处理 4 个元素
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        int i = idx + k * blockDim.x;
        if (i < N) {
            sum += input[i];
        }
    }

    smem[tid] = sum;
    __syncthreads();

    // Block reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial[blockIdx.x] = smem[0];
    }
}

__global__ void reduce_stage2(
    const float* partial,
    float* output,
    int num_blocks
) {
    __shared__ float smem[256];

    int tid = threadIdx.x;
    float sum = 0.0f;

    for (int i = tid; i < num_blocks; i += blockDim.x) {
        sum += partial[i];
    }

    smem[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[0] = smem[0];
    }
}

int main() {
    const int N = 1 << 20;
    const int threads = 256;
    const int elements_per_block = threads * 4;
    const int num_blocks = (N + elements_per_block - 1) / elements_per_block;
    float *h_input = new float[N];
    for (int i = 0; i < N; ++i) {
      h_input[i] = 1.0f;  // 期望 sum = N
    }
    float *d_input = nullptr;
    float *d_output = nullptr;
    float *d_partial = nullptr;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));
    cudaMalloc(&d_partial, num_blocks * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, sizeof(float));
    reduce_stage1<<<num_blocks, threads>>>(d_input, d_partial, N);
    reduce_stage2<<<1, threads>>>(d_partial, d_output, num_blocks);
    cudaDeviceSynchronize();
    float h_output = 0.0f;
    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    printf("gpu sum = %f, expected = %f\n", h_output, (float)N);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_partial);
    delete[] h_input;
    return 0;
  }
  