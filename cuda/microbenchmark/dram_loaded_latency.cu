#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

// 制造拥堵的参数：1024 * 256 = 262,144 个线程同时在途
const int GRID = 1024;    
const int BLOCK = 256;    
const int UNROLL = 8;     
const int STRIDE = 2048;  // 大步长确保不命中 L2 Cache

__global__ void loaded_latency_kernel(const uint32_t *x, uint32_t *clk_out, uint32_t *dummy_out) {
    // 计算每个线程的唯一偏移
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t* ptr = x + (tid * STRIDE / sizeof(uint32_t));
    
    uint32_t regs[UNROLL];
    uint32_t start, stop;

    // 探测线程：选择 Grid 中间的一个线程进行精准计时
    bool is_probe = (blockIdx.x == GRID / 2 && threadIdx.x == 0);

    // 全局同步（块内），尽量让所有线程同时开始访存
    __syncthreads();

    if (is_probe) {
        asm volatile("mov.u32 %0, %%clock;" : "=r"(start));
    }

    // 核心：无依赖地发射 UNROLL 条指令，强行塞满显存控制器队列
    #pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        // 使用 .cg 绕过 L1 以直接压迫 DRAM
        asm volatile("ld.global.cg.b32 %0, [%1];" : "=r"(regs[i]) : "l"(ptr + i * 32) : "memory");
    }

    // 建立数据依赖，确保 stop 计时在所有数据返回之后
    uint32_t final_sum = 0;
    #pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        final_sum += regs[i];
    }

    if (is_probe) {
        asm volatile("mov.u32 %0, %%clock;" : "=r"(stop));
        // 计算这一组并行请求的平均周期
        // 在拥堵状态下，这个数值会包含物理延迟 + 排队延迟
        clk_out[0] = (stop - start); 
    }

    // 防止编译器优化
    if (final_sum == 0xdeadbeef) *dummy_out = final_sum;
}

int main() {
    // 分配约 4GB 显存，确保压测范围足够广
    size_t size = (size_t)GRID * BLOCK * STRIDE * 4; 
    uint32_t *d_x, *d_clk, *d_dummy;
    
    cudaMalloc(&d_x, size);
    cudaMalloc(&d_clk, sizeof(uint32_t));
    cudaMalloc(&d_dummy, sizeof(uint32_t));
    cudaMemset(d_x, 1, size);

    printf("Starting Loaded Latency Test on RTX 3080...\n");
    printf("Config: GRID=%d, BLOCK=%d, UNROLL=%d\n", GRID, BLOCK, UNROLL);

    // 1. 预热
    loaded_latency_kernel<<<GRID, BLOCK>>>(d_x, d_clk, d_dummy);
    cudaDeviceSynchronize();
    
    // 2. 正式测量
    loaded_latency_kernel<<<GRID, BLOCK>>>(d_x, d_clk, d_dummy);
    cudaDeviceSynchronize();

    uint32_t h_clk;
    cudaMemcpy(&h_clk, d_clk, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // 注意：这里除以 UNROLL 得到的是单条指令的摊薄延迟
    // 在满载情况下，这个值会反映出排队后的真实体感延迟
    printf("Measured Loaded Latency: %u cycles (per %d instructions)\n", h_clk, UNROLL);
    printf("Average Latency per LDG: %u cycles\n", h_clk / UNROLL);

    cudaFree(d_x);
    cudaFree(d_clk);
    cudaFree(d_dummy);
    return 0;
}