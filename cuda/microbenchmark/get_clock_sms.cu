#include <cuda_runtime.h>
#include <cstdio>

int main() {
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("==== CUDA Device Info ====\n");
    printf("Device name            : %s\n", prop.name);
    printf("Compute capability     : %d.%d\n",
           prop.major, prop.minor);

    printf("\n---- Core Architecture ----\n");
    printf("SM count               : %d\n", prop.multiProcessorCount);
    printf("CUDA cores / SM        : %d\n",
           (prop.major == 8 ? 128 : -1)); // Ampere SM

    printf("\n---- Clock ----\n");
    printf("SM clock (base)        : %.2f GHz\n",
           prop.clockRate / 1e6);
    printf("Memory clock           : %.2f GHz\n",
           prop.memoryClockRate / 1e6);

    printf("\n---- Memory ----\n");
    printf("Global memory          : %.2f GB\n",
           prop.totalGlobalMem / (1024.0 * 1024 * 1024));
    printf("Memory bus width       : %d bits\n",
           prop.memoryBusWidth);
    printf("L2 cache size          : %.2f MB\n",
           prop.l2CacheSize / (1024.0 * 1024));

    printf("\n---- SM Resources ----\n");
    printf("Shared memory / SM     : %.0f KB\n",
           prop.sharedMemPerMultiprocessor / 1024.0);
    printf("Registers / SM         : %d\n",
           prop.regsPerMultiprocessor);
    printf("Max warps / SM         : %d\n",
           prop.maxThreadsPerMultiProcessor / 32);

    printf("\n---- Bandwidth (theoretical) ----\n");
    double mem_bw =
        2.0 * prop.memoryClockRate * 1e3 *
        (prop.memoryBusWidth / 8.0) / 1e9;
    printf("DRAM bandwidth         : %.1f GB/s\n", mem_bw);

    printf("\n===========================\n");
    return 0;
}