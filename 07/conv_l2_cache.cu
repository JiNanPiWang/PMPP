#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

// --- 预定义参数 ---
#define FILTER_RADIUS 2
// 注意：在这里，TILE_DIM 既是 Input Tile 大小，也是 Output Tile 大小
// 我们不再区分 IN_TILE_DIM 和 OUT_TILE_DIM
#define TILE_DIM 16

#define FILTER_WIDTH (2 * FILTER_RADIUS + 1)
#define MAX_FILTER_SIZE (FILTER_WIDTH * FILTER_WIDTH)

// 常量内存
__constant__ float F_c[MAX_FILTER_SIZE];

// ==================================================
// 🟢 7.5 练习：利用 L2 Cache 处理 Halo 的 Tiled 卷积
// ==================================================
__global__ void convolution_cached_tiled_kernel(float *N, float *P, int width, int height)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 计算全局坐标 (变得非常简单，一一对应)
    int outCol = blockIdx.x * TILE_DIM + tx;
    int outRow = blockIdx.y * TILE_DIM + ty;

    // ==================================================
    // 🟢 TODO 1: 声明共享内存
    // 大小只需 [TILE_DIM][TILE_DIM]
    // 因为我们只把 Block 内部的数据搬进去，边缘 Halo 留给 L2 Cache
    // ==================================================
    __shared__ float N_s[TILE_DIM][TILE_DIM];

    // ==================================================
    // 🟢 TODO 2: 加载数据到 Shared Memory
    // 逻辑比上一节简单得多：
    // 只要当前线程对应的像素在图像范围内 (outCol < width && outRow < height)
    // 就把它搬进 N_s[ty][tx]
    // 否则填 0，因为可能不是整除大小
    // ==================================================
    if (outCol < width && outRow < height)
        N_s[ty][tx] = N[outRow * width + outCol];

    // 线程同步：确保大家都搬完了
    __syncthreads();

    // 计算阶段
    // 只有在图像范围内的线程才计算
    if (outRow < height && outCol < width)
    {
        float Pvalue = 0.0f;

        // 遍历滤波器
        // #define FILTER_WIDTH (2 * FILTER_RADIUS + 1)
        for (int fRow = 0; fRow < FILTER_WIDTH; ++fRow)
        {
            for (int fCol = 0; fCol < FILTER_WIDTH; ++fCol)
            {
                // 计算我们需要的“邻居”在 Shared Memory 里的坐标
                // 注意：这里是有可能算出一个负数，或者超过 TILE_DIM 的数的
                int sRow = ty - FILTER_RADIUS + fRow;
                int sCol = tx - FILTER_RADIUS + fCol;

                // ==================================================
                // 🟢 TODO 3: 混合读取逻辑 (核心难点)
                // ==================================================
                // 逻辑：
                // 1. 判断 sRow 和 sCol 是否在 Shared Memory 范围内 (0 <= s < TILE_DIM)
                //    如果是 -> 读 N_s[sRow][sCol]
                // 2. 如果越界了 (说明落在了 Halo 区域) -> 去读 Global Memory N[...]
                //    注意：去 Global Memory 读的时候，要算出全局坐标 haloRow/haloCol
                //    全局坐标 = outRow - RADIUS + fRow ...
                //    并且要检查全局坐标是否越界 (Ghost Cells)，越界视为 0

                // 超出shared memory界
                if (sRow < 0 || sCol < 0 || sRow >= TILE_DIM || sCol >= TILE_DIM)
                {
                    int gRow = outRow - FILTER_RADIUS + fRow;
                    int gCol = outCol - FILTER_RADIUS + fCol;
                    if (gRow < 0 || gCol < 0 || gRow >= height || gCol >= width)
                        continue;
                    else
                        Pvalue += N[gRow * width + gCol] * F_c[fRow * FILTER_WIDTH + fCol];
                }
                else
                    Pvalue += N_s[sRow][sCol] * F_c[fRow * FILTER_WIDTH + fCol];

                // 写在这里...
            }
        }
        P[outRow * width + outCol] = Pvalue;
    }
}

// ==================================================
// 🟡 CPU 参考实现 (用于验证)
// ==================================================
void convolution_cpu(float *N, float *F, float *P, int width, int height)
{
    for (int outRow = 0; outRow < height; outRow++)
    {
        for (int outCol = 0; outCol < width; outCol++)
        {
            float Pvalue = 0.0f;
            for (int fRow = 0; fRow < FILTER_WIDTH; fRow++)
            {
                for (int fCol = 0; fCol < FILTER_WIDTH; fCol++)
                {
                    int inRow = outRow - FILTER_RADIUS + fRow;
                    int inCol = outCol - FILTER_RADIUS + fCol;
                    if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width)
                    {
                        Pvalue += N[inRow * width + inCol] * F[fRow * FILTER_WIDTH + fCol];
                    }
                }
            }
            P[outRow * width + outCol] = Pvalue;
        }
    }
}

int main()
{
    // 设置图像大小
    int width = 1024;
    int height = 1024;
    int size = width * height * sizeof(float);
    int fSize = MAX_FILTER_SIZE * sizeof(float);

    float *h_N = (float *)malloc(size);
    float *h_F = (float *)malloc(fSize);
    float *h_P_gpu = (float *)malloc(size);
    float *h_P_cpu = (float *)malloc(size);

    // 初始化
    for (int i = 0; i < width * height; i++)
        h_N[i] = (float)(i % 10);
    for (int i = 0; i < MAX_FILTER_SIZE; i++)
        h_F[i] = 1.0f;

    float *d_N, *d_P;
    cudaMalloc(&d_N, size);
    cudaMalloc(&d_P, size);
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(F_c, h_F, fSize);

    // --- Grid 设置 ---
    // 这一节 Grid 变得很简单，完全按照 Block 铺满图像即可
    dim3 dimBlock(TILE_DIM, TILE_DIM);
    dim3 dimGrid((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);

    printf("Image: %dx%d\n", width, height);
    printf("Grid: %dx%d, Block: %dx%d\n", dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);

    convolution_cached_tiled_kernel<<<dimGrid, dimBlock>>>(d_N, d_P, width, height);

    cudaMemcpy(h_P_gpu, d_P, size, cudaMemcpyDeviceToHost);

    // 验证
    printf("Running CPU verification...\n");
    convolution_cpu(h_N, h_F, h_P_cpu, width, height);

    bool passed = true;
    int error_count = 0;
    for (int i = 0; i < width * height; i++)
    {
        if (fabs(h_P_gpu[i] - h_P_cpu[i]) > 1e-3)
        {
            if (error_count < 5)
                printf("Error at %d: GPU=%.2f CPU=%.2f\n", i, h_P_gpu[i], h_P_cpu[i]);
            passed = false;
            error_count++;
        }
    }

    if (passed)
        printf("✅ Test Passed!\n");
    else
        printf("❌ Test Failed with %d errors.\n", error_count);

    free(h_N);
    free(h_F);
    free(h_P_gpu);
    free(h_P_cpu);
    cudaFree(d_N);
    cudaFree(d_P);
    return 0;
}