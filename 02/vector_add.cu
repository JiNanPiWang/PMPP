#include <stdio.h>
#include <cuda_runtime.h>

// 错误检查宏（Pro Tip: 以后你的每个 CUDA 程序都要有这个）
#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

// --------------------------------------------------------
// TODO 1: 编写 Kernel 函数
// 目标：计算 C[i] = A[i] + B[i]
// 要求：
// 1. 计算全局索引 i
// 2. 必须进行边界检查 (防止 i 越界)
// --------------------------------------------------------
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements)
        C[idx] = A[idx] + B[idx];
}

int main(void)
{
    // 1. 设置向量大小 (50000 个元素)
    int numElements = 50000;
    size_t size = numElements * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements);

    // 2. 分配 Host 内存
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // 初始化输入数据
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // --------------------------------------------------------
    // TODO 2: 分配 Device 内存 (d_A, d_B, d_C)
    // 提示：使用 cudaMalloc
    // --------------------------------------------------------
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    
    // 请在此处填入 cudaMalloc 代码
    // cudaMalloc原地修改了传入的那个指针变量d_A，所以要使用&d_A
    CHECK(cudaMalloc((void**) &d_A, size));
    CHECK(cudaMalloc((void**) &d_B, size));
    CHECK(cudaMalloc((void**) &d_C, size));

    // --------------------------------------------------------
    // TODO 3: 将数据从 Host 拷贝到 Device
    // 提示：使用 cudaMemcpy，方向是 HostToDevice
    // --------------------------------------------------------
    // 请在此处填入 cudaMemcpy 代码
    CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // --------------------------------------------------------
    // TODO 4: 设置 Kernel 启动参数
    // 要求：block 大小设为 256
    // grid 大小需要根据 numElements 动态计算 (向上取整)
    // --------------------------------------------------------
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock; // 修改这里

    printf("CUDA Kernel Launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    // --------------------------------------------------------
    // TODO 5: 启动 Kernel
    // --------------------------------------------------------
    // 请在此处填入 Kernel 启动代码
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    // 检查 Kernel 是否出错 (同步检查)
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // --------------------------------------------------------
    // TODO 6: 将计算结果从 Device 拷回 Host
    // 提示：使用 cudaMemcpy，方向是 DeviceToHost
    // --------------------------------------------------------
    // 请在此处填入 cudaMemcpy 代码
    CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // 7. 验证结果 (CPU 验证)
    // 这一步非常重要，用来确保 GPU 没算错
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("TEST PASSED\n");

    // --------------------------------------------------------
    // TODO 7: 释放 Device 内存
    // 提示：使用 cudaFree
    // --------------------------------------------------------
    // 请在此处填入 cudaFree 代码
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    // 释放 Host 内存
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}