#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

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
// TODO 1: 编写矩阵乘法 Kernel
// 计算 C(M, N) = A(M, K) * B(K, N)
// --------------------------------------------------------
__global__ void matrixMulKernel(const float *A, const float *B, float *C, int M, int N, int K)
{
    // 1. 计算 2D 索引 (row, col)
    // row 对应矩阵 C 的行 (0 .. M-1)
    // col 对应矩阵 C 的列 (0 .. N-1)
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 修改这里
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 修改这里

    // 2. 边界检查
    if (row < M && col < N)
    {
        float value = 0.0f;

        // --------------------------------------------------------
        // TODO 2: 实现点积 (Dot Product) 循环
        // 提示：遍历 k (0 .. K-1)
        // A 的索引是 [row, k] -> 线性化为 ?
        // B 的索引是 [k, col] -> 线性化为 ?
        // C[row, col] = A[row, 0~K-1] * B[0~K-1, col]
        // --------------------------------------------------------
        for (int i = 0; i < K; ++i)
            value += A[row * K + i] * B[i * N + col];
        
        // 3. 写回结果
        // C 的索引是 [row, col]
        C[row * N + col] = value;
    }
}

int main(int argc, char **argv)
{
    // 设定矩阵大小
    // 为了测试 robust，我们故意用非 2 的幂次大小，且非正方形
    int M = 512;
    int N = 512;
    int K = 256;

    printf("Matrix Multiplication: A(%d x %d) * B(%d x %d) = C(%d x %d)\n", M, K, K, N, M, N);

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // 分配 Host 内存
    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    float *h_C = (float *)malloc(size_C);
    float *h_C_ref = (float *)malloc(size_C); // 用于 CPU 验证结果

    // 初始化矩阵 (随机数)
    for (int i = 0; i < M * K; ++i) h_A[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = rand() / (float)RAND_MAX;

    // 分配 Device 内存
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((void **)&d_A, size_A));
    CHECK(cudaMalloc((void **)&d_B, size_B));
    CHECK(cudaMalloc((void **)&d_C, size_C));

    // 数据拷贝 H2D
    CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    // --------------------------------------------------------
    // TODO 3: 配置 Kernel 启动参数 (关键！)
    // 使用 2D 的 Block 和 Grid
    // Block 大小建议设为 16x16
    // Grid 大小要覆盖整个 C 矩阵 (M x N)
    // --------------------------------------------------------
    dim3 blockDim(16, 16); // 修改这里
    // 我们通常希望 col（水平方向）对应矩阵的宽度 N。
    // 我们通常希望 row（垂直方向）对应矩阵的高度 M。
    dim3 gridDim((N + 15) / 16, (M + 15) / 16);  // 修改这里

    printf("Launching kernel with Grid(%d, %d), Block(%d, %d)\n", 
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    // 启动 Kernel
    matrixMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // 数据拷贝 D2H
    CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

    // --------------------------------------------------------
    // CPU 验证 (极其慢，但为了验证正确性是必须的)
    // --------------------------------------------------------
    printf("Verifying result on CPU...\n");
    int error_count = 0;
    for (int r = 0; r < M; ++r)
    {
        for (int c = 0; c < N; ++c)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                sum += h_A[r * K + k] * h_B[k * N + c];
            }
            // 允许一点点浮点误差
            if (fabs(h_C[r * N + c] - sum) > 1e-3)
            {
                if (error_count < 5) 
                    printf("Error at (%d, %d): GPU=%.4f, CPU=%.4f\n", r, c, h_C[r * N + c], sum);
                error_count++;
            }
        }
    }

    if (error_count == 0)
    {
        printf("TEST PASSED! \n");
    }
    else
    {
        printf("TEST FAILED with %d errors!\n", error_count);
    }

    // 释放内存
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}