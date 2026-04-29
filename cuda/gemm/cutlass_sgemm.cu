/*
 * CUTLASS SIMT SGEMM for Ampere (sm_86)
 * 对应 cutlass_simt_sgemm_128x128_8x2_nn_align1
 *
 * 使用方式：
 *   nvcc -O3 -arch=sm_86 -I/path/to/cutlass/include cutlass_sgemm.cu -o a.out
 * 
 * nvcc -O3 -arch=sm_86 -I/data/zeta/conan/vlmcpp/20251020.3.0/_/_/package/b0eee1ba0ea3c927813825e2570d24644a2ef715/include cutlass_sgemm.cu -o a.out
 *
 * CUTLASS include 路径通常在：
 *   /usr/local/cutlass/include  或
 *   ~/cutlass/include           或
 *   通过 cmake 安装后在 build/_deps/cutlass-src/include
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>

// CUTLASS headers
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

// ── 错误检查 ────────────────────────────────────────────────────────
#define CUTLASS_CHECK(status)                                               \
    do {                                                                    \
        if (status != cutlass::Status::kSuccess) {                         \
            printf("CUTLASS error at %s:%d: %s\n", __FILE__, __LINE__,    \
                   cutlass::cutlassGetStatusString(status));               \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(err));                                \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

// ── CUTLASS SIMT SGEMM 类型定义 ─────────────────────────────────────
// 对应 cutlass_simt_sgemm_128x128_8x2_nn_align1
// block tile: 128x128x8
// stages: 2（double buffer）
// warps: 4x2
// op class: SIMT（CUDA core）
using CutlassSgemm = cutlass::gemm::device::Gemm<
    float,                                      // A 元素类型
    cutlass::layout::RowMajor,                  // A layout（row-major = N）
    float,                                      // B 元素类型
    cutlass::layout::RowMajor,                  // B layout（row-major = N）
    float,                                      // C 元素类型
    cutlass::layout::RowMajor,                  // C layout
    float,                                      // 累加器类型
    cutlass::arch::OpClassSimt,                 // CUDA core（SIMT）
    cutlass::arch::Sm80,                        // Ampere
    cutlass::gemm::GemmShape<128, 256, 8>,      // block tile
    cutlass::gemm::GemmShape<64, 64, 8>,        // warp tile
    cutlass::gemm::GemmShape<1, 1, 1>,          // instruction shape（SIMT = 1x1x1）
    cutlass::epilogue::thread::LinearCombination<
        float, 1, float, float>,                // epilogue: alpha*AB + beta*C
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3                                           // stages（double buffer）
>;

// ── 辅助函数 ─────────────────────────────────────────────────────────
void random_init(float *data, size_t size) {
    for (size_t i = 0; i < size; ++i)
        data[i] = float(rand()) / RAND_MAX;
}

bool check(const float *A, const float *B, const float *C, int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.f;
            for (int p = 0; p < k; ++p)
                sum += A[i * k + p] * B[p * n + j];
            if (std::fabs(sum - C[i * n + j]) / std::fabs(sum) > 1e-5f) {
                printf("C[%d][%d] not match, %f vs %f\n", i, j, sum, C[i * n + j]);
                return false;
            }
        }
    }
    return true;
}

int main() {
    int m = 5120;
    int n = 4096;
    int k = 4096;
    int n_iter = 10;

    float alpha = 1.0f, beta = 0.0f;

    float *h_A, *h_B, *h_C;
    CUDA_CHECK(cudaMallocHost(&h_A, m * k * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_B, k * n * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_C, m * n * sizeof(float)));
    random_init(h_A, m * k);
    random_init(h_B, k * n);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, m * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, k * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, m * n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyDefault));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyDefault));

    // 构造 CUTLASS GEMM 参数
    CutlassSgemm gemm_op;
    CutlassSgemm::Arguments args(
        {m, n, k},          // problem size
        {d_A, k},           // A（行主序，lda=k）
        {d_B, n},           // B（行主序，ldb=n）
        {d_C, n},           // C（行主序，ldc=n）
        {d_C, n},           // D = alpha*A*B + beta*C
        {alpha, beta}       // epilogue 参数
    );

    // 初始化并检查是否支持当前配置
    CUTLASS_CHECK(gemm_op.initialize(args));

    // warmup
    CUTLASS_CHECK(gemm_op.run());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < n_iter; ++i)
        CUTLASS_CHECK(gemm_op.run());
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));

    long workload = n_iter * long(m) * n * k * 2;
    double gflops = (double(workload) / 1e9) / (double(ms) / 1e3);
    printf("Performance: %fGFLOPS\n", gflops);

    CUDA_CHECK(cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDefault));
    bool chk = check(h_A, h_B, h_C, m, n, k);
    printf("Matrix_C check: %s\n", chk ? "OK" : "Failed");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFreeHost(h_A));
    CUDA_CHECK(cudaFreeHost(h_B));
    CUDA_CHECK(cudaFreeHost(h_C));
    return 0;
}
