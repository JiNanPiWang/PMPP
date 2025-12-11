#include <stdio.h>
#include <cuda_runtime.h>

// --- 预定义参数 ---
#define FILTER_RADIUS 2
#define IN_TILE_DIM 16
#define OUT_TILE_DIM (IN_TILE_DIM - 2 * FILTER_RADIUS)

// 滤波器宽度
#define FILTER_WIDTH (2 * FILTER_RADIUS + 1)
#define MAX_FILTER_SIZE (FILTER_WIDTH * FILTER_WIDTH)

// 常量内存 (上一节的成果)
__constant__ float F_c[MAX_FILTER_SIZE];

// ==================================================
// 🟢 你的战场：Tiled Convolution Kernel
// ==================================================
__global__ void convolution_tiled_kernel(float *N, float *P, int width, int height)
{

    // 1. 准备工作：计算线程在 Block 内的局部坐标
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // TODO 1: 声明共享内存 (Shared Memory)
    // 名字叫 N_s (N_shared的意思)
    // 大小应该是 [IN_TILE_DIM][IN_TILE_DIM]
    // 记得加上 __shared__ 关键字
    
    // 2. 计算对应的 Global Memory 输入坐标
    // 逻辑：
    // 当前 Block 负责的输出区域左上角是 (blockIdx.x * OUT_TILE_DIM, blockIdx.y * OUT_TILE_DIM)
    // 但我们需要读取的 Input 区域要往左上角“外扩” r 个单位。
    // 所以 Input Tile 的左上角是 (blockIdx.x * OUT_TILE_DIM - r, ...)
    // 加上线程偏移 (tx, ty)，就是当前线程要搬运的那个像素。
    int srcCol = blockIdx.x * OUT_TILE_DIM + tx - FILTER_RADIUS;
    int srcRow = blockIdx.y * OUT_TILE_DIM + ty - FILTER_RADIUS;

    // TODO 2: 加载数据到 Shared Memory (处理 Ghost Cells)
    // 逻辑：如果 srcRow 和 srcCol 在图像范围内 (0 <= x < width...)
    //      则 N_s[ty][tx] = N[...]; (注意 N 的 1D 索引计算)
    //      否则 N_s[ty][tx] = 0.0f;

    // TODO 3: 线程同步
    // 必须确保所有人把数据搬完了，大家才能开始下一步计算

    // 3. 计算阶段 (Computing)
    // 只有“内部”线程需要计算输出。边缘线程只是为了搬运 Halo 数据，现在可以休息了。

    // TODO 4: 确定 Active Thread 并计算
    // 逻辑：
    // 有效的 tx 范围是 [FILTER_RADIUS, IN_TILE_DIM - FILTER_RADIUS)
    // 有效的 ty 范围同理
    // 如果是 Active Thread:
    //    1. 初始化 Pvalue = 0
    //    2. 遍历 Filter (0 到 2*r+1):
    //       读取 N_s[ty - r + fRow][tx - r + fCol] (注意这里是在 SharedMem 里找邻居)
    //       乘以 F_c[...]
    //    3. 计算全局输出坐标 (outRow, outCol) 并写回 P
    //       注意：输出坐标 = srcRow + FILTER_RADIUS? 不对，看上面 srcRow 的公式反推
    //       更简单的算法：outCol = blockIdx.x * OUT_TILE_DIM + (tx - FILTER_RADIUS)
    //       记得检查 outCol < width && outRow < height 防止越界写
    if (tx >= FILTER_RADIUS && tx < IN_TILE_DIM - FILTER_RADIUS &&
        ty >= FILTER_RADIUS && ty < IN_TILE_DIM - FILTER_RADIUS)
    {

        float Pvalue = 0.0f;

        // --- 在这里写循环代码 ---

        // --- 写回 Global Memory P ---
        int outCol = blockIdx.x * OUT_TILE_DIM + (tx - FILTER_RADIUS);
        int outRow = blockIdx.y * OUT_TILE_DIM + (ty - FILTER_RADIUS);

        if (outCol < width && outRow < height)
        {
            P[outRow * width + outCol] = Pvalue;
        }
    }
}

// --- 辅助代码 (Host 端) ---
int main()
{
    int width = 64;
    int height = 64;
    int size = width * height * sizeof(float);
    int fSize = MAX_FILTER_SIZE * sizeof(float);

    float *h_N = (float *)malloc(size);
    float *h_F = (float *)malloc(fSize);
    float *h_P = (float *)malloc(size);
    // 初始化
    for (int i = 0; i < width * height; i++)
        h_N[i] = 1.0f;
    for (int i = 0; i < MAX_FILTER_SIZE; i++)
        h_F[i] = 1.0f;

    float *d_N, *d_P;
    cudaMalloc(&d_N, size);
    cudaMalloc(&d_P, size);
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(F_c, h_F, fSize);

    // --- 关键点：Grid 的计算发生了变化 ---
    // 因为每个 Block 产出的有效像素变少了 (只有中间那块)
    // 所以我们需要更多的 Block 来覆盖整个图像
    dim3 dimBlock(IN_TILE_DIM, IN_TILE_DIM);
    dim3 dimGrid((width + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                 (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

    printf("Block Dim: %d (Input Tile Size)\n", IN_TILE_DIM);
    printf("Output Tile Dim: %d\n", OUT_TILE_DIM);
    printf("Grid Size: %d x %d\n", dimGrid.x, dimGrid.y);

    convolution_tiled_kernel<<<dimGrid, dimBlock>>>(d_N, d_P, width, height);

    cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);

    // 简单验证: 中心点应该是 25.0
    printf("Center Check: %f (Expected 25.0)\n", h_P[32 * 64 + 32]);

    free(h_N);
    free(h_F);
    free(h_P);
    cudaFree(d_N);
    cudaFree(d_P);
    return 0;
}