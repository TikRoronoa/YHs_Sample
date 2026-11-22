#include <nvml.h>
#include <stdio.h>

int main() {
    nvmlInit();
    nvmlDevice_t dev;
    nvmlDeviceGetHandleByIndex(0, &dev);
    unsigned int sm_mhz = 0;
    nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &sm_mhz);
    printf("SM = %u MHz\n", sm_mhz);
  return 0;
}