#include <cuda_runtime.h>
#include <stdio.h>

const int ThreadNum = 32;
const int Repeat = 100;
const int LoopNum = 10;
#define ffma_8(a, b, c)                                                        \
  a = __fmaf_rn(a, b, c);                                                      \
  a = __fmaf_rn(a, b, c);                                                      \
  a = __fmaf_rn(a, b, c);                                                      \
  a = __fmaf_rn(a, b, c);                                                      \
  a = __fmaf_rn(a, b, c);                                                      \
  a = __fmaf_rn(a, b, c);                                                      \
  a = __fmaf_rn(a, b, c);                                                      \
  a = __fmaf_rn(a, b, c);

#define ffma_64(a, b, c)                                                       \
  ffma_8(a, b, c) ffma_8(a, b, c) ffma_8(a, b, c) ffma_8(a, b, c)              \
      ffma_8(a, b, c) ffma_8(a, b, c) ffma_8(a, b, c) ffma_8(a, b, c)

#define ffma_512(a, b, c)                                                      \
  ffma_64(a, b, c) ffma_64(a, b, c) ffma_64(a, b, c) ffma_64(a, b, c)          \
      ffma_64(a, b, c) ffma_64(a, b, c) ffma_64(a, b, c) ffma_64(a, b, c)

template <int Repeat>
__global__ void test_ffma_latency(float a, float b, float c, float *output,
                                  long long *cycles_out) {
  int tid = threadIdx.x;
  a = tid;
  long long t0 = clock64();
  for (int i = 0; i < Repeat; i++) {
    ffma_512(a, b, c);
    ffma_512(a, b, c);
    ffma_512(a, b, c);
    ffma_512(a, b, c);

  }
  long long t1 = clock64();

  output[tid] = a;
  if (tid == 0)
    cycles_out[0] = t1 - t0;
}

__global__ void empty_kernel(float *output) {
  output[threadIdx.x] = (float)threadIdx.x;
}

int main() {
  float *output;
  // 分配device内存
  cudaMalloc(&output, ThreadNum * sizeof(float));
  long long *d_cycles;
  cudaMalloc(&d_cycles, sizeof(long long));
  // warmup
  test_ffma_latency<Repeat><<<1, ThreadNum>>>(1.f, 1.f, 1.f, output, d_cycles);

  // 计时
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  // 测 empty
  float t_empty = 0.f;
  long long sum = 0;
  empty_kernel<<<1, ThreadNum>>>(output);
  for (int i = 0; i < LoopNum; i++) {
    cudaEventRecord(start);
    empty_kernel<<<1, ThreadNum>>>(output);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, end);
    t_empty += ms;

  }
  t_empty /= LoopNum;
  printf("Empty latency:%f cycles\n", t_empty * 1.905 * 1e6);

  float duration = 0.f;
  for (int i = 0; i < LoopNum; i++) {
    cudaEventRecord(start);
    test_ffma_latency<Repeat>
        <<<1, ThreadNum>>>(1.f, 1.f, 1.f, output, d_cycles);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float duration_sigle = 0.f;
    cudaEventElapsedTime(&duration_sigle, start, end);
    duration += duration_sigle;

    long long c = 0;
    cudaMemcpy(&c, d_cycles, sizeof(long long), cudaMemcpyDeviceToHost);
    sum += c;

  }

  cudaEventDestroy(start);
  cudaEventDestroy(end);

  cudaFree(output);
  float cycles =
      ((duration / LoopNum - t_empty) * 1.905 * 1e6) / (2048  * Repeat);
  printf("FFMA latency:%f cycles\n", cycles);

  const int FfmaPerRepeat = 4 * 512; // 改成和核里一致
  float latency = (float)sum / LoopNum / (FfmaPerRepeat * Repeat);
  printf("Clock FFMA latency: %f cycles\n", latency);
  return 0;
}