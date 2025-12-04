#include <stdio.h>
#include <cuda_runtime.h>

// 设定 Tile 大小 (32 对应 warp size，是常用选择)
#define TILE_DIM 32

// 错误检查宏
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

// =========================================================
// Kernel 1: 朴素版转置 (Naive)
// 目标：直接读，直接写。
// 预期问题：写入 Global Memory 时也是非合并访问 (Strided Access)。
// =========================================================
__global__ void transposeNaive(float *out, const float *in, int width, int height)
{
    // 计算当前线程对应的全局坐标 (x: 列, y: 行)
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // 边界检查
    if (col < width && row < height)
    {
        // --------------------------------------------------------
        // TODO 1: 计算输入和输出的线性索引
        // in 矩阵大小: [height][width]
        // out 矩阵大小: [width][height] (注意宽高互换)
        // --------------------------------------------------------

        int idx_in = row * width + col;
        int idx_out = col * height + row;

        // 执行拷贝
        out[idx_out] = in[idx_in];
    }
}

// =========================================================
// Kernel 2: 优化版转置 (Coalesced)
// 目标：利用 Shared Memory 作为一个缓冲区，
// 保证 "读 Global" 和 "写 Global" 两个动作都是合并的 (Coalesced)。
// =========================================================
__global__ void transposeCoalesced(float *out, const float *in, int width, int height)
{
    // --------------------------------------------------------
    // TODO 2: 声明 Shared Memory
    // 提示：为了避免 Bank Conflict，列数通常需要 +1 (Padding)
    // 大小应该是 [TILE_DIM][TILE_DIM + 1]
    // --------------------------------------------------------
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    // 1. 读取阶段：从 Global 读到 Shared
    // 逻辑：线程 (x, y) 读取原矩阵的 (x, y)
    // blockDim.x：Block的x方向的维度，这里是5，即每行5个线程。
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height)
    {
        // --------------------------------------------------------
        // TODO 3: 将数据加载到 Shared Memory
        // 按照 (y, x) 的顺序读取 Global Memory (这是合并的)
        // 存入 tile 的哪个位置？通常是 tile[threadIdx.y][threadIdx.x]
        // --------------------------------------------------------

        int idx_in = row * width + col;
        tile[threadIdx.y][threadIdx.x] = in[idx_in];
    }

    // --------------------------------------------------------
    // TODO 4: 线程同步
    // 必须确保 Block 内所有人都读完了，才能进行下一步
    // --------------------------------------------------------
    __syncthreads();

    // 2. 写入阶段：从 Shared 写到 Global
    // 逻辑：我们要改变线程的职能。
    // 我们需要计算一个新的坐标 (new_x, new_y) 对应输出矩阵。
    // 关键：为了保证写入是合并的，输出矩阵的 "列索引(new_x)" 必须随 threadIdx.x 变化。

    // 重新计算 Block 偏移：交换 blockIdx.x 和 blockIdx.y
    // 原来的 blockIdx.y 现在控制输出的 x 轴 (列)
    // 原来的 blockIdx.x 现在控制输出的 y 轴 (行)
    int new_row = blockIdx.y * blockDim.x + threadIdx.x;
    int new_col = blockIdx.x * blockDim.y + threadIdx.y;

    // 注意：输出矩阵的宽是 height，高是 width
    if (new_row < height && new_col < width)
    {
        // --------------------------------------------------------
        // TODO 5: 计算输出索引并写回
        // 1. 计算 idx_out (基于 new_y 和 new_x)
        // 2. 从 tile 取数据。
        //    这是最容易错的地方！我们现在的 threadIdx.x 对应的是输出矩阵的列。
        //    而在读取阶段，原矩阵的行数据是存在 tile 的 threadIdx.y 里的。
        //    所以这里需要交换下标： tile[threadIdx.x][threadIdx.y]
        // --------------------------------------------------------

        int idx_out = new_col * height + new_row;
        out[idx_out] = tile[threadIdx.x][threadIdx.y];
    }
}

int main()
{
    const int N = 2048; // 矩阵大小 2048 x 2048
    const int MEM_SIZE = N * N * sizeof(float);

    float *h_in = (float *)malloc(MEM_SIZE);
    float *h_out = (float *)malloc(MEM_SIZE);
    float *d_in, *d_out;

    // 初始化数据
    for (int i = 0; i < N * N; i++)
        h_in[i] = (float)i;

    CHECK(cudaMalloc(&d_in, MEM_SIZE));
    CHECK(cudaMalloc(&d_out, MEM_SIZE));
    CHECK(cudaMemcpy(d_in, h_in, MEM_SIZE, cudaMemcpyHostToDevice));

    dim3 dimGrid(N / TILE_DIM, N / TILE_DIM);
    dim3 dimBlock(TILE_DIM, TILE_DIM);

    // 创建计时事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms_naive, ms_opt;

    // --- 运行 Naive Kernel ---
    CHECK(cudaMemset(d_out, 0, MEM_SIZE));
    cudaEventRecord(start);
    transposeNaive<<<dimGrid, dimBlock>>>(d_out, d_in, N, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_naive, start, stop);

    printf("Naive Kernel: %f ms\n", ms_naive);

    // --- 运行 Optimized Kernel ---
    CHECK(cudaMemset(d_out, 0, MEM_SIZE));
    cudaEventRecord(start);
    transposeCoalesced<<<dimGrid, dimBlock>>>(d_out, d_in, N, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_opt, start, stop);

    printf("Coalesced Kernel: %f ms\n", ms_opt);
    printf("Speedup: %.2fx\n", ms_naive / ms_opt);

    // 简单验证
    CHECK(cudaMemcpy(h_out, d_out, MEM_SIZE, cudaMemcpyDeviceToHost));
    bool correct = true;
    for (int i = 0; i < N * N; i++)
    {
        // 验证逻辑：Out[row][col] 应该等于 In[col][row]
        int row = i / N;
        int col = i % N;
        if (h_out[i] != h_in[col * N + row])
        {
            printf("Error at index %d: Expected %f, got %f\n", i, h_in[col * N + row], h_out[i]);
            correct = false;
            break;
        }
    }
    if (correct)
        printf("Test PASSED!\n");

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    return 0;
}