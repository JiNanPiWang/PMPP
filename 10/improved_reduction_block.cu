#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define N (1024 * 1024)       // 100万数据
#define THREADS_PER_BLOCK 512 // 每个 Block 512 个线程

// ---------------------------------------------------------
// 你的任务：完成 10.7 节的多 Block 规约 Kernel
// ---------------------------------------------------------
__global__ void multiBlockReduction(float *input, float *output)
{
    __shared__ float sdata[THREADS_PER_BLOCK];

    unsigned int tid = threadIdx.x;

    // [TODO 1]: 计算全局索引 i
    // 每个 Block 处理的数据量是 2 * blockDim.x
    // block 0 从 0 开始，block 1 从 1024 开始...
    // 公式参考导读中的 "A. 全局索引计算"

    unsigned int i = blockIdx.x * (2 * blockDim.x) + tid;

    // [TODO 2]: Load & Add (从 Global 到 Shared)
    // 注意：这里的 i 是全局索引。
    // 我们要确保不要越界 (虽然在本例中 N 是完美的倍数，但加上边界检查 i < N 是好习惯)
    // sdata[tid] = input[i] + input[i + blockDim.x];
    // 提示：你现在的 i 已经定位到了这一段数据的开头。
    // input[i] 是前一半，input[i + blockDim.x] 是这一段的后一半。

    // 先初始化为 0 (以防万一)
    sdata[tid] = 0.0f; 

    // 只有当第一个数在范围内时才读取
    if (i < N) {
        sdata[tid] = input[i];
    }

    // 只有当第二个数也在范围内时才累加
    if (i + blockDim.x < N) {
        sdata[tid] += input[i + blockDim.x];
    }

    __syncthreads();

    // [TODO 3]: 块内规约 (这部分和 10.6 完全一样，直接复用)
    // for (...) { ... }
    for (int stride = blockDim.x / 2; stride >= 1; stride /= 2)
    {
        if (tid < stride)
        {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // [TODO 4]: 原子汇总
    // 当规约结束后，sdata[0] 存放的是当前 Block 的总和。
    // 我们需要把它加到全局的 output 上。
    // 只有线程 0 做这件事。
    // 关键字：atomicAdd(address, val)

    if (tid == 0)
        atomicAdd(output, sdata[0]);
}

// ---------------------------------------------------------
// Host 代码
// ---------------------------------------------------------
int main()
{
    float *h_input, *h_output_gpu;
    float *d_input, *d_output;
    float expected_result = N * 1.0f; // 结果应该是 1048576.0

    h_input = (float *)malloc(N * sizeof(float));
    h_output_gpu = (float *)malloc(sizeof(float));
    for (int i = 0; i < N; i++)
        h_input[i] = 1.0f;

    cudaMalloc((void **)&d_input, N * sizeof(float));
    cudaMalloc((void **)&d_output, sizeof(float));

    // 拷贝数据
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // [重要]: atomicAdd 是在原值基础上累加。
    // 所以在启动 Kernel 前，必须把 output 初始化为 0！
    cudaMemset(d_output, 0, sizeof(float));

    // 计算 Grid 大小
    // 每个线程处理 2 个元素，每个 Block 有 THREADS_PER_BLOCK 个线程
    // 所以每个 Block 处理 2 * THREADS_PER_BLOCK 个元素
    int elementsPerBlock = 2 * THREADS_PER_BLOCK;
    int numBlocks = (N + elementsPerBlock - 1) / elementsPerBlock;

    printf("Launching Multi-Block Kernel...\n");
    printf("Num Elements: %d, Threads per Block: %d, Num Blocks: %d\n", N, THREADS_PER_BLOCK, numBlocks);

    multiBlockReduction<<<numBlocks, THREADS_PER_BLOCK>>>(d_input, d_output);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    cudaMemcpy(h_output_gpu, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Expected Sum: %.2f\n", expected_result);
    printf("GPU Result:   %.2f\n", *h_output_gpu);

    if (fabs(*h_output_gpu - expected_result) < 1.0f)
    { // 浮点大数累加可能有微小误差
        printf("✅ TEST PASSED! 成功处理百万级数据！\n");
    }
    else
    {
        printf("❌ TEST FAILED.\n");
    }

    free(h_input);
    free(h_output_gpu);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}