#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

// 设定 Tile 大小 (必须是编译时常量，除非用动态共享内存)
#define TILE_WIDTH 16

#define CHECK(call)                                                            \
    {                                                                          \
        const cudaError_t error = call;                                        \
        if (error != cudaSuccess)                                              \
        {                                                                      \
            printf("Error: %s:%d, ", __FILE__, __LINE__);                      \
            printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
            exit(1);                                                           \
        }                                                                      \
    }

// --------------------------------------------------------
// TODO: 编写 Tiled 矩阵乘法 Kernel
// 计算 C = A * B
// M: A的行, C的行
// N: B的列, C的列
// K: A的列, B的行
// --------------------------------------------------------
__global__ void matrixMulTiled(const float *A, const float *B, float *C, int M, int N, int K)
{
    // 1. 定义 Shared Memory
    // 需要两个 2D 数组，大小为 [TILE_WIDTH][TILE_WIDTH]
    // --------------------------------------------------------
    // TODO 1: 声明 As 和 Bs
    // --------------------------------------------------------
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    // 2. 计算线程和 Block 的索引
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 3. 计算当前线程负责计算 C 中哪个元素的坐标 (row, col)
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    // 累加器
    float Pvalue = 0.0f;

    // 4. 循环遍历所有的 Tile (Phases)
    // 核心逻辑：把长长的 K 维度切成一段一段，每段长度 TILE_WIDTH
    // phase (ph) 从 0 到 ceil(K / TILE_WIDTH)
    for (int ph = 0; ph < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++ph)
    {
        // --------------------------------------------------------
        // TODO 2: 协作加载 A 的 Tile 到 As
        // 目标：加载 A[row][ph * TILE_WIDTH + tx]
        // 关键：必须进行边界检查！如果索引超出了 (row < M && current_col < K)，则补 0.0
        // --------------------------------------------------------
        // int A_col = ...?
        // if (...) As[ty][tx] = ...; else As[ty][tx] = 0.0f;
        if (row < M && tx + ph * TILE_WIDTH < K)
            As[ty][tx] = A[row * K + tx + ph * TILE_WIDTH];
        else
            As[ty][tx] = 0;

        // --------------------------------------------------------
        // TODO 3: 协作加载 B 的 Tile 到 Bs
        // 目标：加载 B[ph * TILE_WIDTH + ty][col]
        // 关键：必须进行边界检查！如果索引超出了 (current_row < K && col < N)，则补 0.0
        // --------------------------------------------------------
        // int B_row = ...?
        // if (...) Bs[ty][tx] = ...; else Bs[ty][tx] = 0.0f;
        if (ty + ph * TILE_WIDTH < K && col < N)
            Bs[ty][tx] = B[(ty + ph * TILE_WIDTH) * N + col];
        else
            Bs[ty][tx] = 0;

        // --------------------------------------------------------
        // TODO 4: 第一次同步
        // 确保大家把 shared memory 填满了，才能开始算
        // --------------------------------------------------------
        __syncthreads();

        // --------------------------------------------------------
        // TODO 5: 计算 Partial Dot Product
        // 在 Shared Memory 上进行矩阵乘法
        // 这里的 k 对应 Tile 内部的维度 (0 到 TILE_WIDTH-1)
        // --------------------------------------------------------
        // for (int k = 0; k < TILE_WIDTH; ++k) ...
        for (int i = 0; i < TILE_WIDTH; ++i)
            Pvalue += As[ty][i] * Bs[i][tx];

        // --------------------------------------------------------
        // TODO 6: 第二次同步
        // 确保大家都算完了当前 Tile，才能进入下一轮循环覆盖 Shared Memory
        // --------------------------------------------------------
        __syncthreads();
    }

    // 5. 写回结果到 Global Memory
    // 只有在矩阵范围内的线程才写回
    if (row < M && col < N)
    {
        C[row * N + col] = Pvalue;
    }
}

int main(int argc, char **argv)
{
    // 设定矩阵大小
    // 故意设定为非 TILE_WIDTH 整数倍，测试你的边界检查逻辑
    int M = 512;
    int N = 512;
    int K = 512;

    printf("Tiled Matrix Multiplication: A(%d x %d) * B(%d x %d) = C(%d x %d)\n", M, K, K, N, M, N);

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // Host 内存分配
    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    float *h_C = (float *)malloc(size_C);
    float *h_C_ref = (float *)malloc(size_C);

    // 初始化
    for (int i = 0; i < M * K; ++i)
        h_A[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < K * N; ++i)
        h_B[i] = rand() / (float)RAND_MAX;

    // Device 内存分配
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((void **)&d_A, size_A));
    CHECK(cudaMalloc((void **)&d_B, size_B));
    CHECK(cudaMalloc((void **)&d_C, size_C));

    // H2D
    CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    // 配置 Kernel
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    // Grid 覆盖 M 和 N
    dim3 gridDim((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    printf("Launching kernel with Grid(%d, %d), Block(%d, %d)\n",
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    // 启动 Kernel
    matrixMulTiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // D2H
    CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

    // CPU 验证
    printf("Verifying result on CPU (this might take a while)...\n");
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
            if (fabs(h_C[r * N + c] - sum) > 1e-2) // 稍微放宽误差容限，因为浮点累加顺序不同
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

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}