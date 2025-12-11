#include <stdio.h>
#include <math.h> // 为了使用 abs()
#include <cuda_runtime.h>

// --- 预定义参数 ---
#define FILTER_RADIUS 2
#define IN_TILE_DIM 16
#define OUT_TILE_DIM (IN_TILE_DIM - 2 * FILTER_RADIUS)
#define FILTER_WIDTH (2 * FILTER_RADIUS + 1)
#define MAX_FILTER_SIZE (FILTER_WIDTH * FILTER_WIDTH)

// 常量内存
__constant__ float F_c[MAX_FILTER_SIZE];

// ==================================================
// 🟢 GPU Kernel (保持你刚才写的代码不变)
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
    __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];

    // 2. 计算对应的 Global Memory 输入坐标
    // 逻辑：
    // 当前 Block 负责的输出区域左上角是 (blockIdx.x * OUT_TILE_DIM, blockIdx.y * OUT_TILE_DIM)
    // 但我们需要读取的 Input 区域要往左上角“外扩” r 个单位。
    // 所以 Input Tile 的左上角是 (blockIdx.x * OUT_TILE_DIM - r, ...)
    // 加上线程偏移 (tx, ty)，就是当前线程要搬运的那个像素。

    // 我们按16x16分块，但是我们一次只算12x12的结果，我们分16x16的块，
    // 但是整个块都向左上移两格，我们只算中间那12x12的结果，我们每次步进的长度是OUT_TILE_DIM，
    // 我们启动block的数量也是和OUT_TILE_DIM一致的
    int srcCol = blockIdx.x * OUT_TILE_DIM + tx - FILTER_RADIUS;
    int srcRow = blockIdx.y * OUT_TILE_DIM + ty - FILTER_RADIUS;

    // TODO 2: 加载数据到 Shared Memory (处理 Ghost Cells)
    // 逻辑：如果 srcRow 和 srcCol 在图像范围内 (0 <= x < width...)
    //      则 N_s[ty][tx] = N[...]; (注意 N 的 1D 索引计算)
    //      否则 N_s[ty][tx] = 0.0f;
    if (srcRow >= 0 && srcRow < height && srcCol >= 0 && srcCol < width)
        N_s[ty][tx] = N[srcRow * width + srcCol];
    else
        N_s[ty][tx] = 0;

    // TODO 3: 线程同步
    // 必须确保所有人把数据搬完了，大家才能开始下一步计算
    __syncthreads();

    // 3. 计算阶段 (Computing)
    // 只有“内部”线程需要计算输出。边缘线程只是为了搬运 Halo 数据，现在可以休息了。

    // TODO 4: 确定 Active Thread 并计算
    // 逻辑：
    // 因为我们的块向左上角移了两格，所以我们计算的idx应该是从(2, 2)开始
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
        // F_c[ty][tx] * N_s[ty][tx]
        for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; ++fRow)
        {
            for (int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; ++fCol)
            {
                Pvalue += F_c[fRow * (2 * FILTER_RADIUS + 1) + fCol] *
                          N_s[ty + fRow - FILTER_RADIUS][tx + fCol - FILTER_RADIUS];
            }
        }

        // --- 写回 Global Memory P ---
        int outCol = blockIdx.x * OUT_TILE_DIM + (tx - FILTER_RADIUS);
        int outRow = blockIdx.y * OUT_TILE_DIM + (ty - FILTER_RADIUS);

        if (outCol < width && outRow < height)
        {
            P[outRow * width + outCol] = Pvalue;
        }
    }
}

// ==================================================
// 🟡 新增：CPU 参考实现 (Golden Reference)
// ==================================================
// 这是一个标准的 3层循环卷积实现，用于生成标准答案
void convolution_cpu(float *N, float *F, float *P, int width, int height)
{
    for (int outRow = 0; outRow < height; outRow++)
    {
        for (int outCol = 0; outCol < width; outCol++)
        {
            float Pvalue = 0.0f;

            // 遍历滤波器
            for (int fRow = 0; fRow < FILTER_WIDTH; fRow++)
            {
                for (int fCol = 0; fCol < FILTER_WIDTH; fCol++)
                {
                    // 计算对应的输入坐标
                    int inRow = outRow - FILTER_RADIUS + fRow;
                    int inCol = outCol - FILTER_RADIUS + fCol;

                    // 边界检查 (和 GPU 逻辑一致，越界视为 0)
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

// ==================================================
// 🔵 Main 函数
// ==================================================
int main()
{
    // 为了让测试更有意义，我们可以稍微加大一点尺寸
    // 比如不是 64x64，而是非对齐的大小，比如 70x70，测试边界情况
    int width = 1024;
    int height = 1024;

    int size = width * height * sizeof(float);
    int fSize = MAX_FILTER_SIZE * sizeof(float);

    printf("Image Size: %d x %d\n", width, height);

    // 1. Host 内存分配
    float *h_N = (float *)malloc(size);
    float *h_F = (float *)malloc(fSize);
    float *h_P_gpu = (float *)malloc(size); // 存放 GPU 结果
    float *h_P_cpu = (float *)malloc(size); // 存放 CPU 结果

    // 2. 初始化数据
    // 让输入数据有一些随机性，而不仅是全1，这样能测出索引错误
    for (int i = 0; i < width * height; i++)
        h_N[i] = (float)(i % 10);
    for (int i = 0; i < MAX_FILTER_SIZE; i++)
        h_F[i] = 1.0f; // 简单起见 Filter 还是全1

    // 3. Device 内存分配
    float *d_N, *d_P;
    cudaMalloc(&d_N, size);
    cudaMalloc(&d_P, size);

    // 4. 数据拷贝
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(F_c, h_F, fSize);

    // 5. 启动 Kernel
    dim3 dimBlock(IN_TILE_DIM, IN_TILE_DIM);
    dim3 dimGrid((width + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                 (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

    printf("Grid: %d x %d, Block: %d x %d\n", dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);
    convolution_tiled_kernel<<<dimGrid, dimBlock>>>(d_N, d_P, width, height);

    // 6. 拷贝 GPU 结果回 Host
    cudaMemcpy(h_P_gpu, d_P, size, cudaMemcpyDeviceToHost);

    // 7. 运行 CPU 参考版本 (这步会比较慢，是正常的)
    printf("Running CPU verification...\n");
    convolution_cpu(h_N, h_F, h_P_cpu, width, height);

    // 8. 🔍 全量对比验证
    bool passed = true;
    int error_count = 0;
    // 允许一点点浮点误差
    float epsilon = 1e-4;

    for (int i = 0; i < width * height; i++)
    {
        float diff = fabs(h_P_gpu[i] - h_P_cpu[i]);
        if (diff > epsilon)
        {
            passed = false;
            error_count++;
            // 只打印前 10 个错误，避免刷屏
            if (error_count <= 10)
            {
                int y = i / width;
                int x = i % width;
                printf("❌ Error at (%d, %d): GPU=%.4f, CPU=%.4f, Diff=%.4f\n",
                       x, y, h_P_gpu[i], h_P_cpu[i], diff);
            }
        }
    }

    if (passed)
    {
        printf("✅ Test Passed! All pixels match CPU result.\n");
    }
    else
    {
        printf("❌ Test Failed with %d errors.\n", error_count);
    }

    // 清理
    free(h_N);
    free(h_F);
    free(h_P_gpu);
    free(h_P_cpu);
    cudaFree(d_N);
    cudaFree(d_P);

    return 0;
}