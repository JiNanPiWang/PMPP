#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// 我们把数据量加大到 3300 万，让 GPU 真正跑起来
#define N (32 * 1024 * 1024)
#define THREADS_PER_BLOCK 256

// ---------------------------------------------------------
// 你的任务：完成 10.8 节 "线程粗化 / 网格跨步循环" Kernel
// ---------------------------------------------------------
__global__ void threadCoarseningReduction(float *input, float *output)
{
    __shared__ float sdata[THREADS_PER_BLOCK];

    // 1. 初始化寄存器 sum
    // 我们不再直接把 input 搬到 shared memory。
    // 而是先在寄存器里累加一堆数据，最后把这个寄存器的值给 shared memory。
    float sum = 0.0f;

    unsigned int tid = threadIdx.x;

    // [TODO 1]: 计算初始的全局索引 i
    // 和 10.7 一样: blockIdx.x * blockDim.x + threadIdx.x
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // [TODO 2]: 计算“跨步” (Stride)
    // 既然我们只有有限的 Blocks，当线程处理完当前 i 的数据后，
    // 它需要跳过整个 Grid 的宽度，去处理下一批数据。
    // Stride = Grid总宽度 = gridDim.x * blockDim.x
    unsigned int stride = gridDim.x * blockDim.x;

    // [TODO 3]: 实现 "网格跨步循环" (Grid-Stride Loop)
    // 这是线程粗化的核心。
    // 使用 while 循环：只要 i 小于 N，就累加 input[i] 到 sum 中
    // 每次迭代，i 增加 stride

    while (i < N)
    {
        sum += input[i];
        i += stride;
    }

    // [TODO 4]: 将寄存器累加结果存入 Shared Memory
    // 此时，sum 里面已经包含了该线程负责的所有数据的总和（可能加了成百上千次）。
    // 现在我们把它存入 sdata，准备进行块内规约。
    sdata[tid] = sum;

    __syncthreads();

    // [TODO 5]: 块内规约 (直接复用之前的代码)
    // 注意：这里的 stride 是 reduction 的步长，不要和上面的 grid stride 搞混了
    for (int s = blockDim.x / 2; s > 0; s /= 2)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // [TODO 6]: 原子汇总 (和 10.7 一样)
    if (tid == 0)
    {
        atomicAdd(output, sdata[0]);
    }
}

// ---------------------------------------------------------
// Host 代码
// ---------------------------------------------------------
int main()
{
    float *h_input, *h_output_gpu;
    float *d_input, *d_output;
    float expected_result = N * 1.0f;

    // 分配大内存
    size_t size = N * sizeof(float);
    h_input = (float *)malloc(size);
    h_output_gpu = (float *)malloc(sizeof(float));

    // 初始化
    for (int i = 0; i < N; i++)
        h_input[i] = 1.0f;

    cudaMalloc((void **)&d_input, size);
    cudaMalloc((void **)&d_output, sizeof(float));

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, sizeof(float));

    // [重点]: 限制 Grid 大小
    // 在 10.7 中，我们会计算需要多少个 Block 才能覆盖 N。
    // 在 10.8 中，我们要限制 Block 的数量，比如只用 256 个 Block。
    // 这意味着 GPU 不需要频繁地进行 Block 调度，减少了 Overhead。
    // 剩下的工作由 Kernel 内部的 while 循环完成（软件串行化）。
    int numBlocks = 256;

    printf("Launching Thread Coarsening Kernel...\n");
    printf("Data Size: %d\n", N);
    printf("Grid Size: %d Blocks (Fixed)\n", numBlocks);
    printf("Threads per Block: %d\n", THREADS_PER_BLOCK);

    // 粗化因子计算 (仅供参考)
    // 每个线程平均要处理多少个数据？
    int totalThreads = numBlocks * THREADS_PER_BLOCK;
    printf("Coarsening Factor (Avg elements per thread): %.1f\n", (float)N / totalThreads);

    threadCoarseningReduction<<<numBlocks, THREADS_PER_BLOCK>>>(d_input, d_output);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    cudaMemcpy(h_output_gpu, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Expected Sum: %.2f\n", expected_result);
    printf("GPU Result:   %.2f\n", *h_output_gpu);

    // 允许一定的浮点误差
    if (fabs(*h_output_gpu - expected_result) < N * 1e-5)
    {
        printf("✅ TEST PASSED! 线程粗化成功！\n");
    }
    else
    {
        printf("❌ TEST FAILED. 结果不匹配。\n");
    }

    free(h_input);
    free(h_output_gpu);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}