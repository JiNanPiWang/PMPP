#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define N 2048                 // 数组元素总数
#define THREADS_PER_BLOCK 1024 // 线程数 = N / 2

// ---------------------------------------------------------
// 你的任务：完成 10.4 节的 "最小化发散" Kernel
// ---------------------------------------------------------
__global__ void improvedSumReduction(float *input, float *output)
{
    // [TODO 1]: 定义数据索引
    // 在 10.3 中，我们用了 i = 2 * threadIdx.x (因为是交错的)
    // 在 10.4 的新策略中，每个线程最初只负责它自己 ID 对应的位置
    // Thread 0 -> input[0], Thread 1 -> input[1]...
    // 另外一半的数据 (input[1024]...input[2047]) 将在第一次迭代中被加进来
    unsigned int tid = threadIdx.x;
    unsigned int i = threadIdx.x;

    // [TODO 2]: 实现 "顺序寻址" 的 for 循环
    // 这里的 stride 代表“去远处搬运数据”的距离
    // 1. 初始化：stride 等于 blockDim.x (本例中为 1024)
    //    (注：原文 Fig 10.9 逻辑是先处理后半段，所以第一次加法是 input[i] + input[i+1024])
    // 2. 终止条件：stride > 0
    // 3. 更新：每次迭代 stride 右移一位 (除以 2) -> 1024, 512, 256...

    // 你的 for 循环写在这里:
    for (int stride = blockDim.x; stride >= 1; stride /= 2)
    {
        // [TODO 3]: 编写优化的 if 判断和加法逻辑
        // 关键点：我们需要活跃线程是连续的 (0 ~ stride-1)
        // 比如 stride=512 时，只有 tid < 512 的线程工作
        // 这样前 16 个 Warp 全力工作，后 16 个 Warp 全力休眠 -> 没有发散！

        if (tid < stride)
        {
            input[i] += input[i + stride];
        }

        // [TODO 4]: 它是树形结构，别忘了同步！
        __syncthreads();
    }

    // 3. 将最终结果写回 output
    if (tid == 0)
    {
        *output = input[0];
    }
}

// ---------------------------------------------------------
// Host 代码 (与上一节类似，用于验证)
// ---------------------------------------------------------
int main()
{
    float *h_input, *h_output_gpu;
    float *d_input, *d_output;
    float expected_result = N * 1.0f;

    // 内存分配与初始化
    h_input = (float *)malloc(N * sizeof(float));
    h_output_gpu = (float *)malloc(sizeof(float));
    for (int i = 0; i < N; i++)
        h_input[i] = 1.0f;

    cudaMalloc((void **)&d_input, N * sizeof(float));
    cudaMalloc((void **)&d_output, sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // 启动 Kernel
    printf("Launching Improved Kernel (Sequential Addressing)...\n");
    // 注意：这里我们启动 1024 个线程来处理 2048 个数据
    // 第一次迭代时，Thread 0 会把 input[0+1024] 加到 input[0] 上
    improvedSumReduction<<<1, THREADS_PER_BLOCK>>>(d_input, d_output);

    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // 获取结果
    cudaMemcpy(h_output_gpu, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    // 验证
    printf("Expected Sum: %.2f\n", expected_result);
    printf("GPU Result:   %.2f\n", *h_output_gpu);

    if (fabs(*h_output_gpu - expected_result) < 1e-5)
    {
        printf("✅ TEST PASSED! 控制发散已最小化！\n");
    }
    else
    {
        printf("❌ TEST FAILED. 检查 stride 的方向和 if 条件。\n");
    }

    // 清理
    free(h_input);
    free(h_output_gpu);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}