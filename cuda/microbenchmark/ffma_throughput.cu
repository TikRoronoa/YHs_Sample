#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>

// 纯寄存器 FFMA 吞吐：kernel 内 clock64() 计时
// 注意：FLOP 只能按「同时驻留」的 block 计；launch 过多会分波，
// 若仍用单 block 的 cycles 去除以全部 block 的 FLOP，会虚高到 >100% peak。

constexpr int N_ACC = 64;
constexpr int UNROLL = 64;
constexpr int THREADS = 256;
constexpr int REPEAT_OUTER = 2048;
constexpr int N_TIMED_RUNS = 3;

// 限制每 SM 驻留 2 个 block，与 launch 数量一致，避免分波虚高
__global__ __launch_bounds__(256, 2)
void ffma_throughput_kernel(float *out, int64_t *block_cycles, int loops) {
    float acc[N_ACC];
#pragma unroll
    for (int i = 0; i < N_ACC; ++i) {
        acc[i] = float(threadIdx.x + i + 1);
    }

    float a = 1.0000001f + 0.0000001f * float(threadIdx.x & 7);
    float b = 1.0000001f;

    __syncthreads();
    const int64_t t0 = clock64();

    for (int t = 0; t < loops; ++t) {
#pragma unroll 1
        for (int u = 0; u < UNROLL; ++u) {
#pragma unroll
            for (int i = 0; i < N_ACC; ++i) {
                acc[i] = __fmaf_rn(acc[i], a, b);
            }
        }
    }

    __syncthreads();
    const int64_t t1 = clock64();

    if (threadIdx.x == 0) {
        block_cycles[blockIdx.x] = t1 - t0;
    }

    float s = 0.f;
#pragma unroll
    for (int i = 0; i < N_ACC; ++i) {
        s += acc[i];
    }
    if (s == 0.f) {
        out[blockIdx.x * blockDim.x + threadIdx.x] = s;
    }
}

static void check(cudaError_t e, const char *msg) {
    if (e != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(e));
        exit(1);
    }
}

int main() {
    int device = 0;
    check(cudaSetDevice(device), "setDevice");
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, device), "getProp");

    const int sms = prop.multiProcessorCount;
    // 与 launch_bounds(256,2) 一致：每 SM 恰好 2 block，一轮跑完
    const int blocks_per_sm = 2;
    const int blocks = sms * blocks_per_sm;

    const int cores_per_sm = (prop.major >= 8) ? 128 : 64;
    const double freq_ghz = prop.clockRate / 1e6;
    const double freq_hz = prop.clockRate * 1000.0;
    const double peak_tflops =
        double(sms) * cores_per_sm * freq_ghz * 2.0 / 1e3;

    float *d_out = nullptr;
    int64_t *d_cycles = nullptr;
    check(cudaMalloc(&d_out, size_t(blocks) * THREADS * sizeof(float)), "malloc out");
    check(cudaMalloc(&d_cycles, size_t(blocks) * sizeof(int64_t)), "malloc cycles");

    ffma_throughput_kernel<<<blocks, THREADS>>>(d_out, d_cycles, 64);
    check(cudaDeviceSynchronize(), "warmup");

    const double fma_per_thread =
        double(REPEAT_OUTER) * double(UNROLL) * double(N_ACC);
    // 仅统计本轮并发执行的 block（= grid 全部，因已按驻留数 launch）
    const double total_flops =
        double(blocks) * double(THREADS) * fma_per_thread * 2.0;

    double tflops_clk_sum = 0.0;
    double tflops_evt_sum = 0.0;
    int64_t cycles_max_sum = 0;

    std::vector<int64_t> h_cycles(blocks);

    cudaEvent_t ev0, ev1;
    check(cudaEventCreate(&ev0), "ev");
    check(cudaEventCreate(&ev1), "ev");

    for (int run = 0; run < N_TIMED_RUNS; ++run) {
        check(cudaEventRecord(ev0), "record");
        ffma_throughput_kernel<<<blocks, THREADS>>>(d_out, d_cycles, REPEAT_OUTER);
        check(cudaEventRecord(ev1), "record");
        check(cudaEventSynchronize(ev1), "sync");

        float ms_evt = 0.f;
        check(cudaEventElapsedTime(&ms_evt, ev0, ev1), "elapsed");
        tflops_evt_sum += (total_flops / 1e12) / (double(ms_evt) / 1e3);

        check(cudaMemcpy(h_cycles.data(), d_cycles, size_t(blocks) * sizeof(int64_t),
                         cudaMemcpyDeviceToHost),
              "D2H cycles");

        int64_t cmax = 0;
        int64_t cmin = h_cycles[0];
        for (int i = 0; i < blocks; ++i) {
            cmax = std::max(cmax, h_cycles[i]);
            cmin = std::min(cmin, h_cycles[i]);
        }
        cycles_max_sum += cmax;

        const double seconds = double(cmax) / freq_hz;
        tflops_clk_sum += (total_flops / 1e12) / seconds;

        if (run == 0) {
            printf("block cycles: min=%lld max=%lld (spread check)\n",
                   (long long)cmin, (long long)cmax);
        }
    }

    const double tflops_clk = tflops_clk_sum / N_TIMED_RUNS;
    const double tflops_evt = tflops_evt_sum / N_TIMED_RUNS;
    const double cycles_avg = double(cycles_max_sum) / N_TIMED_RUNS;
    const double ms_clk = cycles_avg / freq_hz * 1e3;

    const double fma_total = total_flops / 2.0;
    const double pct_by_cycles =
        100.0 * fma_total / (cycles_avg * double(sms) * double(cores_per_sm));

    printf("Device: %s  (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    printf("SMs: %d  blocks: %d (%d/SM, launch_bounds minBlocks=2)\n",
           sms, blocks, blocks_per_sm);
    printf("clockRate (cudaDeviceProp): %.3f GHz\n", freq_ghz);
    printf("Theoretical FP32 peak: %.2f TFLOPS\n", peak_tflops);
    printf("\n");
    printf("REPEAT_OUTER=%d  runs=%d\n", REPEAT_OUTER, N_TIMED_RUNS);
    printf("max block cycles (avg): %.0f  -> %.3f ms\n", cycles_avg, ms_clk);
    printf("Achieved (clock64):     %.2f TFLOPS  (%.1f%% peak, cycle-model %.1f%%)\n",
           tflops_clk, 100.0 * tflops_clk / peak_tflops, pct_by_cycles);
    printf("Achieved (cudaEvent):   %.2f TFLOPS  (%.1f%% peak)\n",
           tflops_evt, 100.0 * tflops_evt / peak_tflops);
    printf("vs 35.6 TFLOPS:         clock %.1f%% / event %.1f%%\n",
           100.0 * tflops_clk / 35.6, 100.0 * tflops_evt / 35.6);

    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaFree(d_out);
    cudaFree(d_cycles);
    return 0;
}
