#include <stdio.h>
#include <cuda_runtime.h>

#define FILTER_RADIUS 2
#define FILTER_WIDTH (2 * FILTER_RADIUS + 1)

// ==================================================
// 🟢 你的任务：补全这个 Kernel
// ==================================================
__global__ void convolution_2D_basic_kernel(float *N, float *F, float *P,
                                            int r, int width, int height)
{
    // TODO 1: 计算当前线程负责的输出像素坐标 (outCol, outRow)
    int outCol = blockIdx.x * blockDim.x + threadIdx.x; // 修改这里
    int outRow = blockIdx.y * blockDim.y + threadIdx.y; // 修改这里

    // 检查是否在有效图像范围内
    if (outCol < width && outRow < height)
    {
        float Pvalue = 0.0f;

        // 遍历滤波器 (Filter)
        for (int fRow = 0; fRow < 2 * r + 1; fRow++)
        {
            for (int fCol = 0; fCol < 2 * r + 1; fCol++)
            {

                // TODO 2: 计算对应的输入像素坐标 (inRow, inCol)
                // 提示：输入坐标 = 输出坐标 - 半径 + 滤波器偏移
                int inRow = outRow - r + fRow; // 修改这里`
                int inCol = outCol - r + fCol; // 修改这里

                // TODO 3: 边界检查 (Ghost Cells) 并累加
                // 如果 inRow 和 inCol 在有效范围内 (0 到 height-1, 0 到 width-1)
                // 则：Pvalue += F[...] * N[...]
                // 注意：这里需要把 2D 坐标转换为 1D 索引
                // F 的 1D 索引是: fRow * (2*r+1) + fCol
                // N 的 1D 索引是: inRow * width + inCol
                if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width)
                    Pvalue += N[inRow * width + inCol] * F[fRow * (2*r+1) + fCol];
            }
        }

        // 写回结果
        P[outRow * width + outCol] = Pvalue;
    }
}

// ==================================================
// 🟡 下面是辅助代码 (Host端)，你不需要修改，但可以看看它是怎么调用的
// ==================================================

void convolution_cpu(float *N, float *F, float *P, int r, int width, int height)
{
    for (int outRow = 0; outRow < height; outRow++)
    {
        for (int outCol = 0; outCol < width; outCol++)
        {
            float Pvalue = 0.0f;
            for (int fRow = 0; fRow < 2 * r + 1; fRow++)
            {
                for (int fCol = 0; fCol < 2 * r + 1; fCol++)
                {
                    int inRow = outRow - r + fRow;
                    int inCol = outCol - r + fCol;
                    if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width)
                    {
                        Pvalue += F[fRow * (2 * r + 1) + fCol] * N[inRow * width + inCol];
                    }
                }
            }
            P[outRow * width + outCol] = Pvalue;
        }
    }
}

int main()
{
    int width = 64;  // 图像宽
    int height = 64; // 图像高
    int r = FILTER_RADIUS;
    int size = width * height * sizeof(float);
    int fSize = FILTER_WIDTH * FILTER_WIDTH * sizeof(float);

    // 1. 分配 Host 内存
    float *h_N = (float *)malloc(size);
    float *h_F = (float *)malloc(fSize);
    float *h_P = (float *)malloc(size);
    float *h_P_ref = (float *)malloc(size); // CPU参考结果

    // 2. 初始化数据 (全1，滤波器也是全1，方便肉眼验证)
    for (int i = 0; i < width * height; i++)
        h_N[i] = 1.0f;
    for (int i = 0; i < FILTER_WIDTH * FILTER_WIDTH; i++)
        h_F[i] = 1.0f;

    // 3. 分配 Device 内存
    float *d_N, *d_F, *d_P;
    cudaMalloc(&d_N, size);
    cudaMalloc(&d_F, fSize);
    cudaMalloc(&d_P, size);

    // 4. 数据拷贝 Host -> Device
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_F, h_F, fSize, cudaMemcpyHostToDevice);

    // 5. 定义 Grid 和 Block
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);

    printf("Running Kernel with Grid(%d, %d), Block(%d, %d)...\n",
           dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);

    // 6. 启动 Kernel
    convolution_2D_basic_kernel<<<dimGrid, dimBlock>>>(d_N, d_F, d_P, r, width, height);

    // 7. 拷贝回结果
    cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);

    // 8. 验证结果 (对比 CPU 计算)
    convolution_cpu(h_N, h_F, h_P_ref, r, width, height);

    bool correct = true;
    for (int i = 0; i < width * height; i++)
    {
        if (abs(h_P[i] - h_P_ref[i]) > 1e-5)
        {
            printf("Error at index %d: GPU=%f, CPU=%f\n", i, h_P[i], h_P_ref[i]);
            correct = false;
            break;
        }
    }

    if (correct)
        printf("✅ Test Passed! Computation is correct.\n");
    else
        printf("❌ Test Failed!\n");

    // 清理
    free(h_N);
    free(h_F);
    free(h_P);
    free(h_P_ref);
    cudaFree(d_N);
    cudaFree(d_F);
    cudaFree(d_P);
    return 0;
}