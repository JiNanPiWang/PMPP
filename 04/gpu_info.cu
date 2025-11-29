#include <cuda_runtime.h>
#include <stdio.h>

// 函数前向声明
int _ConvertSMVer2Cores(int major, int minor);
const char* GetArchName(int major, int minor);

int main() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        printf("cudaGetDeviceCount failed: %s\n", cudaGetErrorString(error));
        return -1;
    }
    
    if (deviceCount == 0) {
        printf("No CUDA-capable devices found!\n");
        return -1;
    }
    
    printf("========================================\n");
    printf("Found %d CUDA device(s)\n", deviceCount);
    printf("========================================\n\n");
    
    // 遍历所有显卡
    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        
        printf("╔════════════════════════════════════════╗\n");
        printf("║          Device %d: %s           \n", dev, prop.name);
        printf("╚════════════════════════════════════════╝\n\n");
        
        // === 核心架构信息 ===
        printf("【核心架构】\n");
        printf("  计算能力(Compute Capability):        %d.%d\n", 
               prop.major, prop.minor);
        printf("  SM 数量(Multiprocessor Count):       %d\n", 
               prop.multiProcessorCount);
        printf("  每个 SM 的 CUDA 核心数:               %d (估算)\n", 
               _ConvertSMVer2Cores(prop.major, prop.minor));
        printf("  总 CUDA 核心数(估算):                 %d\n\n", 
               prop.multiProcessorCount * _ConvertSMVer2Cores(prop.major, prop.minor));
        
        // === 时钟频率 ===
        printf("【时钟频率】\n");
        printf("  GPU 时钟频率:                         %.2f MHz\n", 
               prop.clockRate / 1000.0f);
        printf("  内存时钟频率:                         %.2f MHz\n\n", 
               prop.memoryClockRate / 1000.0f);
        
        // === 内存信息 ===
        printf("【内存信息】\n");
        printf("  全局内存(Global Memory):              %.2f GB\n", 
               prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  共享内存/SM(Shared Memory per SM):    %zu KB\n", 
               prop.sharedMemPerMultiprocessor / 1024);
        printf("  共享内存/Block(Shared Memory per Block): %zu KB\n", 
               prop.sharedMemPerBlock / 1024);
        printf("  L2 缓存大小:                          %.2f MB\n", 
               prop.l2CacheSize / (1024.0 * 1024.0));
        printf("  内存总线位宽:                         %d-bit\n", 
               prop.memoryBusWidth);
        printf("  内存带宽(峰值):                       %.2f GB/s\n\n", 
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        
        // === 寄存器信息 ===
        printf("【寄存器】\n");
        printf("  寄存器/SM(Registers per SM):          %d\n", 
               prop.regsPerMultiprocessor);
        printf("  寄存器/Block(Registers per Block):    %d\n\n", 
               prop.regsPerBlock);
        
        // === 线程/Warp 信息 ===
        printf("【线程与 Warp】\n");
        printf("  Warp 大小:                            %d\n", 
               prop.warpSize);
        printf("  最大线程数/Block:                     %d\n", 
               prop.maxThreadsPerBlock);
        printf("  最大线程数/SM:                        %d\n", 
               prop.maxThreadsPerMultiProcessor);
        printf("  最大 Block 维度:                      (%d, %d, %d)\n", 
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  最大 Grid 维度:                       (%d, %d, %d)\n\n", 
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        
        // === 常量内存与纹理 ===
        printf("【其他内存】\n");
        printf("  常量内存(Constant Memory):            %zu KB\n", 
               prop.totalConstMem / 1024);
        printf("  纹理对齐(Texture Alignment):          %zu bytes\n\n", 
               prop.textureAlignment);
        
        // === 特性支持 ===
        printf("【特性支持】\n");
        printf("  支持并发 Kernel 执行:                 %s\n", 
               prop.concurrentKernels ? "是" : "否");
        printf("  支持 ECC 内存:                        %s\n", 
               prop.ECCEnabled ? "是" : "否");
        printf("  支持统一寻址(Unified Addressing):     %s\n", 
               prop.unifiedAddressing ? "是" : "否");
        printf("  支持托管内存(Managed Memory):         %s\n", 
               prop.managedMemory ? "是" : "否");
        printf("  支持并发内存拷贝:                     %s\n", 
               prop.asyncEngineCount > 0 ? "是" : "否");
        printf("  支持 Compute Preemption:              %s\n", 
               prop.computePreemptionSupported ? "是" : "否");
        
        // === 架构代号 ===
        printf("\n【架构代号】\n");
        const char* archName = GetArchName(prop.major, prop.minor);
        printf("  架构:                                 %s\n", archName);
        
        if (dev < deviceCount - 1) {
            printf("\n========================================\n\n");
        }
    }
    
    return 0;
}

// 辅助函数：根据计算能力估算每个 SM 的 CUDA 核心数
int _ConvertSMVer2Cores(int major, int minor) {
    typedef struct {
        int SM; // 0xMm (M = major, m = minor)
        int Cores;
    } sSMtoCores;
    
    sSMtoCores nGpuArchCoresPerSM[] = {
        {0x30, 192}, // Kepler
        {0x32, 192},
        {0x35, 192},
        {0x37, 192},
        {0x50, 128}, // Maxwell
        {0x52, 128},
        {0x53, 128},
        {0x60,  64}, // Pascal
        {0x61, 128},
        {0x62, 128},
        {0x70,  64}, // Volta
        {0x72,  64},
        {0x75,  64}, // Turing
        {0x80,  64}, // Ampere
        {0x86, 128},
        {0x87, 128},
        {0x89, 128}, // Ada Lovelace
        {0x90, 128}, // Hopper
        {-1, -1}
    };
    
    int index = 0;
    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchCoresPerSM[index].Cores;
        }
        index++;
    }
    
    printf("  警告: 未知的计算能力 %d.%d, 无法估算核心数\n", major, minor);
    return 0;
}

// 辅助函数：获取架构代号
const char* GetArchName(int major, int minor) {
    switch (major) {
        case 3: return "Kepler";
        case 5: return "Maxwell";
        case 6: return "Pascal";
        case 7: 
            if (minor == 0) return "Volta";
            if (minor == 5) return "Turing";
            return "Unknown 7.x";
        case 8:
            if (minor == 0) return "Ampere (A100)";
            if (minor == 6) return "Ampere (RTX 30/40 series)";
            if (minor == 9) return "Ada Lovelace";
            return "Unknown 8.x";
        case 9:
            if (minor == 0) return "Hopper";
            return "Unknown 9.x";
        default:
            return "Unknown";
    }
}