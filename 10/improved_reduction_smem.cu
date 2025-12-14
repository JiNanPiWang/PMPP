#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define N 2048                 // 数组元素总数
#define THREADS_PER_BLOCK 1024 // 线程数 = N / 2

// ---------------------------------------------------------
// 你的任务：完成 10.6 节的 "Shared Memory" 规约 Kernel
// ---------------------------------------------------------
__global__ void sharedMemReduction(float *input, float *output)
{
    // [TODO 1]: 声明共享内存
    // 这是一个静态大小的 Shared Memory 数组。
    // 因为我们有 1024 个线程，所以需要 1024 个 float 的空间。
    // 关键字：__shared__
    // 变量名建议：sdata

    // 你的代码:
    __shared__ float sdata[THREADS_PER_BLOCK];

    // 线程 ID
    unsigned int tid = threadIdx.x;

    // [TODO 2]: "Load & Add" (搬运并初次相加)
    // 这是 10.6 节 Fig 10.11 的核心优化。
    // 我们不只是单纯把 input[i] 搬进去，而是把 (input[i] + input[i + 1024]) 搬进去。
    // 这样 input 数组有 2048 个元素，被 1024 个线程一次性处理成了 1024 个结果存入 sdata。
    // 提示：读取 input[i] 和 input[i + blockDim.x]，相加后存入 sdata[tid]

    // 你的代码:
    sdata[tid] = input[tid] + input[tid + THREADS_PER_BLOCK];

    // [TODO 3]: 第一次同步
    // 在开始对 sdata 进行归约之前，必须保证所有线程都完成了 TODO 2 的搬运工作。
    // 否则 Thread 0 可能去读 Thread 100 还没写好的 sdata。

    // 你的代码:
    __syncthreads();

    // [TODO 4]: 在 Shared Memory 中进行规约
    // 逻辑和 10.4 节一模一样 (Sequential Addressing)，但这次操作的对象是 sdata。
    // 循环变量 s (stride) 从 blockDim.x / 2 开始 (因为我们已经做了一次加法了)。
    // 每次 s 减半。
    //
    // 注意：每次加法 sdata[tid] += sdata[tid + s] 后，都需要同步！

    for (int stride = THREADS_PER_BLOCK / 2; stride >= 1; stride /= 2)
    {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];

        __syncthreads();
    }

    // [TODO 5]: 写回结果
    // 最终结果现在保存在 sdata[0] 中。
    // 只有线程 0 负责把它写回全局内存 output。
    if (tid == 0)
    {
        *output = sdata[0];
    }
}

// ---------------------------------------------------------
// Host 代码 (无需修改)
// ---------------------------------------------------------
int main()
{
    float *h_input, *h_output_gpu;
    float *d_input, *d_output;
    float expected_result = N * 1.0f;

    h_input = (float *)malloc(N * sizeof(float));
    h_output_gpu = (float *)malloc(sizeof(float));
    for (int i = 0; i < N; i++)
        h_input[i] = 1.0f;

    cudaMalloc((void **)&d_input, N * sizeof(float));
    cudaMalloc((void **)&d_output, sizeof(float));

    // 拷贝数据
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // 启动 Kernel
    printf("Launching Shared Memory Reduction Kernel...\n");
    // 1 个 Block，1024 个线程，处理 2048 个数据
    sharedMemReduction<<<1, THREADS_PER_BLOCK>>>(d_input, d_output);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // 拷贝结果
    cudaMemcpy(h_output_gpu, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    // 验证
    printf("Expected Sum: %.2f\n", expected_result);
    printf("GPU Result:   %.2f\n", *h_output_gpu);

    if (fabs(*h_output_gpu - expected_result) < 1e-5)
    {
        printf("✅ TEST PASSED! Shared Memory 优化成功！\n");
    }
    else
    {
        printf("❌ TEST FAILED. 请检查 Load 逻辑或同步位置。\n");
    }

    free(h_input);
    free(h_output_gpu);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}