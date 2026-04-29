#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// 用宏展开依赖链，避免循环引入额外指令
// 每组 8 条 FFMA，展开 128 组 = 1024 条 FFMA
#define FFMA_8(r, a, b) \
    r = __fmaf_rn(r, a, b); \
    r = __fmaf_rn(r, a, b); \
    r = __fmaf_rn(r, a, b); \
    r = __fmaf_rn(r, a, b); \
    r = __fmaf_rn(r, a, b); \
    r = __fmaf_rn(r, a, b); \
    r = __fmaf_rn(r, a, b); \
    r = __fmaf_rn(r, a, b);

#define FFMA_64(r, a, b)  \
    FFMA_8(r, a, b)  FFMA_8(r, a, b)  FFMA_8(r, a, b)  FFMA_8(r, a, b) \
    FFMA_8(r, a, b)  FFMA_8(r, a, b)  FFMA_8(r, a, b)  FFMA_8(r, a, b)

#define FFMA_512(r, a, b) \
    FFMA_64(r, a, b) FFMA_64(r, a, b) FFMA_64(r, a, b) FFMA_64(r, a, b) \
    FFMA_64(r, a, b) FFMA_64(r, a, b) FFMA_64(r, a, b) FFMA_64(r, a, b)

__global__ void ffma_latency_kernel(float* out, float a, float b, int repeat)
{
    float r = (float)(threadIdx.x + 1);  // 非零初始值，防止编译器优化

    // 外层循环让总指令数足够多，摊薄测量误差
    for (int i = 0; i < repeat; i++) {
        FFMA_512(r, a, b)  // 每次循环 512 条 FFMA
        FFMA_512(r, a, b)  // 共 1024 条
    }

    // 写出结果，防止编译器把整个计算优化掉
    if (r != 0) out[threadIdx.x] = r;
}

__global__ void clock_overhead_kernel(float* out, float a, float b, int repeat)
{
    // 空循环，测量 clock64() 本身的开销
    float r = (float)(threadIdx.x + 1);
    for (int i = 0; i < repeat; i++) { }
    if (r != 0) out[threadIdx.x] = r;
}

int main()
{
    const int THREADS   = 32;   // 1 个 warp，避免 SM 调度干扰
    const int BLOCKS    = 1;
    const int REPEAT    = 100;
    const int FFMA_PER_REPEAT = 1024;
    const int TOTAL_FFMA = REPEAT * FFMA_PER_REPEAT;

    float *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, THREADS * sizeof(float)));

    // warmup
    ffma_latency_kernel<<<BLOCKS, THREADS>>>(d_out, 1.0001f, 0.0001f, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 用 cuda event 计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 多次测量取平均
    const int RUNS = 10;
    float total_ms = 0.f;

    for (int r = 0; r < RUNS; r++) {
        cudaEventRecord(start);
        ffma_latency_kernel<<<BLOCKS, THREADS>>>(d_out, 1.0001f, 0.0001f, REPEAT);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }

    float avg_ms = total_ms / RUNS;

    // 换算成 cycle
    // cycle = time_ms × frequency_GHz × 1e6
    // RTX 3080/3090 boost clock ≈ 1.7~1.9 GHz，先用 1.8GHz 估算
    // 更准确的做法：用 nvml 读实时频率
    float freq_ghz = 1.785f;  // 替换成你的卡的实际 boost clock
    float total_cycles = avg_ms * freq_ghz * 1e6f;
    float latency_per_ffma = total_cycles / TOTAL_FFMA;

    std::cout << "Total FFMA:        " << TOTAL_FFMA << std::endl;
    std::cout << "Avg time:          " << avg_ms << " ms" << std::endl;
    std::cout << "Freq assumption:   " << freq_ghz << " GHz" << std::endl;
    std::cout << "Total cycles:      " << total_cycles << std::endl;
    std::cout << "Latency per FFMA:  " << latency_per_ffma << " cycles" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_out);
    return 0;
}