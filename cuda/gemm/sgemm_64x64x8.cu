#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>

void random_init(float *data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = float(rand()) / RAND_MAX;
    }
}

bool check(const float *A, const float *B, const float *C, int m, int n, int k) {
    // Full 5k×4k CPU verify is too slow; check a tile window + trailing edges.
    const int cm = m < 256 ? m : 256;
    const int cn = n < 256 ? n : 256;

    auto verify = [&](int i, int j) -> bool {
        float sum = 0.f;
        for (int p = 0; p < k; ++p) {
            sum += A[i * k + p] * B[j + p * n];
        }
        if (std::fabs(sum - C[i * n + j]) / std::fabs(sum) > 1e-5f) {
            printf("C[%d][%d] not match, %f vs %f
", i, j, sum, C[i * n + j]);
            return false;
        }
        return true;
    };

    for (int i = 0; i < cm; ++i) {
        for (int j = 0; j < cn; ++j) {
            if (!verify(i, j)) return false;
        }
    }
    if (m > cm) {
        for (int j = 0; j < cn; ++j) {
            if (!verify(m - 1, j)) return false;
        }
    }
    if (n > cn) {
        for (int i = 0; i < cm; ++i) {
            if (!verify(i, n - 1)) return false;
        }
    }
    if (m > cm && n > cn) {
        if (!verify(m - 1, n - 1)) return false;
    }
    return true;
}


__device__ __forceinline__
uint32_t smem_u32addr(const void *smem_ptr) {
    uint32_t addr;
    asm ("{.reg .u64 u64addr;\n"
         " cvta.to.shared.u64 u64addr, %1;\n"
         " cvt.u32.u64 %0, u64addr;}\n"
         : "=r"(addr)
         : "l"(smem_ptr)
    );

    return addr;
}

__device__ __forceinline__
void ldg32_nc(float &reg, const void *ptr, bool guard) {
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750
        " @p ld.global.nc.L2::128B.f32 %0, [%1];}\n"
#else
        " @p ld.global.nc.f32 %0, [%1];}\n"
#endif
        : "=f"(reg)
        : "l"(ptr), "r"((int)guard)
    );
}

__device__ __forceinline__
void ldg32_nc_0(float &reg, const void *ptr, bool guard) {
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
        " @!p mov.b32 %0, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750
        " @p ld.global.nc.L2::128B.f32 %0, [%1];}\n"
#else
        " @p ld.global.nc.f32 %0, [%1];}\n"
#endif
        : "=f"(reg)
        : "l"(ptr), "r"((int)guard)
    );
}

__device__ __forceinline__
void stg32(const float &reg, void *ptr, bool guard) {
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
        " @p st.global.f32 [%0], %1;}\n"
        : : "l"(ptr), "f"(reg), "r"((int)guard)
    );
}

__device__ __forceinline__
void lds128(float &reg0, float &reg1,
            float &reg2, float &reg3,
            const uint32_t &addr) {
    asm volatile (
        "ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n"
        : "=f"(reg0), "=f"(reg1), "=f"(reg2), "=f"(reg3)
        : "r"(addr)
    );
}

__device__ __forceinline__
void sts32(const float &reg, const uint32_t &addr) {
    asm volatile (
        "st.shared.f32 [%0], %1;\n"
        : : "r"(addr), "f"(reg)
    );
}

__device__ __forceinline__
void sts128(const float &reg0, const float &reg1,
            const float &reg2, const float &reg3,
            const uint32_t &addr) {
    asm volatile (
        "st.shared.v4.f32 [%0], {%1, %2, %3, %4};\n"
        : : "r"(addr), "f"(reg0), "f"(reg1), "f"(reg2), "f"(reg3)
    );
}

struct StgFrag {
    float data[4][4];

    __device__ __forceinline__
    StgFrag(const float (&C_frag)[8][8], int tile_x, int tile_y) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                data[i][j] = C_frag[tile_y * 4 + i][tile_x * 4 + j];
            }
        }
    }
};

__device__ __noinline__
void C_tile_wb(StgFrag C_frag,
               float *C_stg_ptr,
               const float *C_lds_ptr,
               uint32_t C_sts_addr,
               uint32_t m,
               uint32_t n,
               uint32_t m_idx,
               uint32_t n_idx) {
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        sts128(C_frag.data[i][0],
               C_frag.data[i][1],
               C_frag.data[i][2],
               C_frag.data[i][3],
               C_sts_addr + i * 8 * sizeof(float4));
    }

    __syncthreads();

    uint32_t m_guard = m < m_idx ? 0 : m - m_idx;

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        stg32(C_lds_ptr[i * 32],
              C_stg_ptr + i * n,
              i < m_guard && n_idx < n);
    }
}

/*
 * matrix A, B and C: row-major
 *
 * scaled down from sgemm_128x128x8:
 * thread block tile: m64n64k8
 * warps per block: 2 (64 threads)
 * warp tile: m32n64k8
 * thread tile: m8n8k8
 * thread fragment:
 *     matrixA: 8x1 FP32
 *     matrixB: 1x8 FP32
 *
 * ----------------------------------------------------------------
 * thread block tile map:
 *
 *                                64
 *                    --|---------------------|
 *             B_tile  8|                     |
 *                    --|---------------------|
 *
 *  A_tile   | 8 |      |         64          |
 *         --|---|    --|---------------------|
 *           |   |    32|       warp_0        |
 *        64 |   |    --|---------------------|
 *           |   |    32|       warp_1        |
 *         --|---|      |---------------------|
 *
 * warp tile / thread map same as sgemm_128x128x8 (z-map, m8n8).
 */
__global__ void sgemm_64x64x8_kernel(const float *A,
                          const float *B,
                          float *C,
                          uint32_t m,
                          uint32_t n,
                          uint32_t k,
                          uint32_t A_ldg_step,    // k * sizeof(float)
                          uint32_t B_ldg_step) {  // n * sizeof(float) * 8
    /*
     * A_smem: [K=8][M=68] * 4B * double buffer, each buf 4KB, xor 0x1000
     * B_smem: [K=8][N=64] * 4B * double buffer, each buf 2KB, xor 0x800
     * layout: A at 0, B at 8KB; total 16KB
     */
    __shared__ __align__(8 * 1024) char smem[16 * 1024];
    float *A_smem = reinterpret_cast<float *>(smem);
    float *B_smem = reinterpret_cast<float *>(smem + 8 * 1024);

    float A_frag[2][8];
    float B_frag[2][8];
    float C_frag[8][8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            C_frag[i][j] = 0;
        }
    }

    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t warp_id = threadIdx.x / 32;

    // 4x8 threads each warp for FFMA (same z-map as 128x128)
    const uint32_t mma_tid_x = (lane_id / 2) % 8;
    const uint32_t mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);

    // A: 64 threads, each loads 8 floats (2x float4) → 64x8 tile
    //    k = tid%8, base_row = (tid/8)*4; second float4 at +32
    const char *A_ldg_ptr = (const char *)(
        A + (blockIdx.y * 64 + threadIdx.x / 8 * 4) * k + threadIdx.x % 8);
    // B: k = tid/8 (0..7), n = tid%8; each loads 8 floats stride 8 → 8x64
    const char *B_ldg_ptr = (const char *)(
        B + (threadIdx.x / 8) * n + blockIdx.x * 64 + threadIdx.x % 8);

    // A laid out [k][m_pad68]; B laid out [k][n64]
    uint32_t A_sts_addr = smem_u32addr(
        A_smem + (threadIdx.x % 8) * 68 + (threadIdx.x / 8) * 4);
    uint32_t B_sts_addr = smem_u32addr(
        B_smem + (threadIdx.x / 8) * 64 + (threadIdx.x % 8));

    // 2 warps stacked on M: warp0 → m[0:32), warp1 → m[32:64); both own full N
    uint32_t A_lds_addr = smem_u32addr(
        A_smem + warp_id * 32 + mma_tid_y * 4);
    uint32_t B_lds_addr = smem_u32addr(
        B_smem + mma_tid_x * 4);

    uint32_t A_ldg_guard = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int m_idx = blockIdx.y * 64 + threadIdx.x / 8 * 4 + i;
        if (m_idx < m) {
            A_ldg_guard |= (1u << i);
        }
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int m_idx = blockIdx.y * 64 + threadIdx.x / 8 * 4 + 32 + i;
        if (m_idx < m) {
            A_ldg_guard |= (1u << (4 + i));
        }
    }

    uint32_t B_ldg_guard = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int n_idx = blockIdx.x * 64 + threadIdx.x % 8 + i * 8;
        if (n_idx < n) {
            B_ldg_guard |= (1u << i);
        }
    }

    float A_ldg_reg[8];
    float B_ldg_reg[8];

    uint32_t k_tiles = (k + 7) / 8 - 1;

    // load 1'st tile to shared memory
    {
        uint32_t first_k_tile = k - k_tiles * 8;

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            bool guard = (A_ldg_guard & (1u << i)) != 0 &&
                         threadIdx.x % 8 < first_k_tile;
            ldg32_nc_0(A_ldg_reg[i],
                       A_ldg_ptr + i * A_ldg_step,
                       guard);
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            bool guard = (A_ldg_guard & (1u << (4 + i))) != 0 &&
                         threadIdx.x % 8 < first_k_tile;
            ldg32_nc_0(A_ldg_reg[4 + i],
                       A_ldg_ptr + (32 + i) * A_ldg_step,
                       guard);
        }

        sts128(A_ldg_reg[0], A_ldg_reg[1], A_ldg_reg[2], A_ldg_reg[3],
               A_sts_addr);
        sts128(A_ldg_reg[4], A_ldg_reg[5], A_ldg_reg[6], A_ldg_reg[7],
               A_sts_addr + 32 * sizeof(float));

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            bool guard = (B_ldg_guard & (1u << i)) != 0 &&
                         threadIdx.x / 8 < first_k_tile;
            ldg32_nc_0(B_ldg_reg[i],
                       B_ldg_ptr + i * 8 * sizeof(float),
                       guard);
        }

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            sts32(B_ldg_reg[i], B_sts_addr + i * 8 * sizeof(float));
        }

        __syncthreads();

        A_sts_addr ^= 0x1000;
        B_sts_addr ^= 0x800;

        A_ldg_ptr += first_k_tile * sizeof(float);
        B_ldg_ptr += n * first_k_tile * sizeof(float);
    }

    // load 1'st fragment
    lds128(A_frag[0][0], A_frag[0][1], A_frag[0][2], A_frag[0][3],
           A_lds_addr);
    lds128(A_frag[0][4], A_frag[0][5], A_frag[0][6], A_frag[0][7],
           A_lds_addr + 16 * sizeof(float));
    lds128(B_frag[0][0], B_frag[0][1], B_frag[0][2], B_frag[0][3],
           B_lds_addr);
    lds128(B_frag[0][4], B_frag[0][5], B_frag[0][6], B_frag[0][7],
           B_lds_addr + 32 * sizeof(float));

    for (; k_tiles > 0; --k_tiles) {
        #pragma unroll
        for (int k_frag = 0; k_frag < 8; ++k_frag) {
            if (k_frag == 7) {
                sts128(A_ldg_reg[0], A_ldg_reg[1], A_ldg_reg[2], A_ldg_reg[3],
                       A_sts_addr);
                sts128(A_ldg_reg[4], A_ldg_reg[5], A_ldg_reg[6], A_ldg_reg[7],
                       A_sts_addr + 32 * sizeof(float));
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    sts32(B_ldg_reg[i], B_sts_addr + i * 8 * sizeof(float));
                }

                __syncthreads();

                A_lds_addr ^= 0x1000;
                B_lds_addr ^= 0x800;
                A_sts_addr ^= 0x1000;
                B_sts_addr ^= 0x800;

                A_ldg_ptr += 8 * sizeof(float);
                B_ldg_ptr += B_ldg_step;
            }

            lds128(A_frag[(k_frag + 1) % 2][0],
                   A_frag[(k_frag + 1) % 2][1],
                   A_frag[(k_frag + 1) % 2][2],
                   A_frag[(k_frag + 1) % 2][3],
                   A_lds_addr + (k_frag + 1) % 8 * 68 * sizeof(float));
            lds128(A_frag[(k_frag + 1) % 2][4],
                   A_frag[(k_frag + 1) % 2][5],
                   A_frag[(k_frag + 1) % 2][6],
                   A_frag[(k_frag + 1) % 2][7],
                   A_lds_addr + ((k_frag + 1) % 8 * 68 + 16) * sizeof(float));
            lds128(B_frag[(k_frag + 1) % 2][0],
                   B_frag[(k_frag + 1) % 2][1],
                   B_frag[(k_frag + 1) % 2][2],
                   B_frag[(k_frag + 1) % 2][3],
                   B_lds_addr + (k_frag + 1) % 8 * 64 * sizeof(float));
            lds128(B_frag[(k_frag + 1) % 2][4],
                   B_frag[(k_frag + 1) % 2][5],
                   B_frag[(k_frag + 1) % 2][6],
                   B_frag[(k_frag + 1) % 2][7],
                   B_lds_addr + ((k_frag + 1) % 8 * 64 + 32) * sizeof(float));

            if (k_frag == 0) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    ldg32_nc(A_ldg_reg[i],
                             A_ldg_ptr + i * A_ldg_step,
                             (A_ldg_guard & (1u << i)) != 0);
                }
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    ldg32_nc(A_ldg_reg[4 + i],
                             A_ldg_ptr + (32 + i) * A_ldg_step,
                             (A_ldg_guard & (1u << (4 + i))) != 0);
                }

                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    ldg32_nc(B_ldg_reg[i],
                             B_ldg_ptr + i * 8 * sizeof(float),
                             (B_ldg_guard & (1u << i)) != 0);
                }
            }

            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    C_frag[i][j] += A_frag[k_frag % 2][i] *
                                    B_frag[k_frag % 2][j];
                }
            }
        }
    }

    // FFMA for the last tile
    #pragma unroll
    for (int k_frag = 0; k_frag < 8; ++k_frag) {
        if (k_frag < 7) {
            lds128(A_frag[(k_frag + 1) % 2][0],
                   A_frag[(k_frag + 1) % 2][1],
                   A_frag[(k_frag + 1) % 2][2],
                   A_frag[(k_frag + 1) % 2][3],
                   A_lds_addr + (k_frag + 1) % 8 * 68 * sizeof(float));
            lds128(A_frag[(k_frag + 1) % 2][4],
                   A_frag[(k_frag + 1) % 2][5],
                   A_frag[(k_frag + 1) % 2][6],
                   A_frag[(k_frag + 1) % 2][7],
                   A_lds_addr + ((k_frag + 1) % 8 * 68 + 16) * sizeof(float));
            lds128(B_frag[(k_frag + 1) % 2][0],
                   B_frag[(k_frag + 1) % 2][1],
                   B_frag[(k_frag + 1) % 2][2],
                   B_frag[(k_frag + 1) % 2][3],
                   B_lds_addr + (k_frag + 1) % 8 * 64 * sizeof(float));
            lds128(B_frag[(k_frag + 1) % 2][4],
                   B_frag[(k_frag + 1) % 2][5],
                   B_frag[(k_frag + 1) % 2][6],
                   B_frag[(k_frag + 1) % 2][7],
                   B_lds_addr + ((k_frag + 1) % 8 * 64 + 32) * sizeof(float));
        }

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                C_frag[i][j] += A_frag[k_frag % 2][i] *
                                B_frag[k_frag % 2][j];
            }
        }
    }

    // C write-back: warp tile still 32x64
    uint32_t C_sts_addr = smem_u32addr((float4 *)(smem + warp_id * 2048) +
                                       mma_tid_y * 4 * 8 + mma_tid_x);
    const float *C_lds_ptr = (float *)(smem + warp_id * 2048) + lane_id;

    uint32_t m_idx = blockIdx.y * 64 + warp_id * 32;
    uint32_t n_idx = blockIdx.x * 64 + lane_id;

    float *C_stg_ptr = C + m_idx * n + n_idx;

    if (m_idx >= m) {
        return;
    } else if (m_idx + 32 <= m) {
        uint32_t n_guard = n < n_idx ? 0 : n - n_idx;

        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                __syncthreads();

                #pragma unroll
                for (int p = 0; p < 4; ++p) {
                    sts128(C_frag[i * 4 + p][j * 4],
                           C_frag[i * 4 + p][j * 4 + 1],
                           C_frag[i * 4 + p][j * 4 + 2],
                           C_frag[i * 4 + p][j * 4 + 3],
                           C_sts_addr + p * 8 * sizeof(float4));
                }

                __syncthreads();

                #pragma unroll
                for (int p = 0; p < 16; ++p) {
                    stg32(C_lds_ptr[p * 32],
                          C_stg_ptr + (i * 16 + p) * n + j * 32,
                          j * 32 < n_guard);
                }
            }
        }
    } else {
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                StgFrag stg_frag(C_frag, j, i);

                C_tile_wb(stg_frag,
                          C_stg_ptr + i * 16 * n + j * 32,
                          C_lds_ptr,
                          C_sts_addr,
                          m,
                          n,
                          m_idx + i * 16,
                          n_idx + j * 32);
            }
        }
    }
}

int main() {
    int m = 5120;
    int n = 4096;
    int k = 4096;
    int n_iter = 10;

    float *h_A, *h_B, *h_C;
    cudaMallocHost(&h_A, m * k * sizeof(float));
    cudaMallocHost(&h_B, k * n * sizeof(float));
    cudaMallocHost(&h_C, m * n * sizeof(float));
    random_init(h_A, m * k);
    random_init(h_B, k * n);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * k * sizeof(float));
    cudaMalloc(&d_B, k * n * sizeof(float));
    cudaMalloc(&d_C, m * n * sizeof(float));

    cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyDefault);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    dim3 grid((n + 63) / 64, (m + 63) / 64);

    // warmup
    sgemm_64x64x8_kernel<<<grid, 64>>>(
        d_A, d_B, d_C, m, n, k, k * sizeof(float), n * sizeof(float) * 8);

    cudaEventRecord(start);
    for (int i = 0; i < n_iter; ++i) {
        sgemm_64x64x8_kernel<<<grid, 64>>>(
            d_A, d_B, d_C, m, n, k, k * sizeof(float), n * sizeof(float) * 8);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float ms;
    cudaEventElapsedTime(&ms, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    long workload = n_iter * long(m) * n * k * 2;
    double gflops = (double(workload) / 1e9) / (double(ms) / 1e3);
    printf("Performance: %fGFLOPS\n", gflops);

    cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDefault);
    bool chk = check(h_A, h_B, h_C, m, n, k);
    printf("Matrix_C check: %s\n", chk ? "OK" : "Failed");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
}
