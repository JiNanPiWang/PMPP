#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define N 2048                 // 数组元素总数
#define THREADS_PER_BLOCK 1024 // 线程数 = N / 2

// ---------------------------------------------------------
// 你的任务在这里：完成核心的规约 Kernel
// ---------------------------------------------------------
__global__ void simpleSumReduction(float *input, float *output)
{
    // 1. 计算当前线程负责的数据索引
    // 根据 10.3 节：每个线程负责 input 数组中的偶数位置
    // Thread 0 -> index 0, Thread 1 -> index 2, Thread 2 -> index 4...
    // 提示：使用 threadIdx.x
    unsigned int i = 2 * threadIdx.x;

    // 2. 核心规约循环 (The Reduction Loop)
    // 这是一个原地(in-place)规约，直接修改 input 数组
    // TODO: 实现 for 循环
    // 循环变量 stride 从 1 开始，每次迭代翻倍 (*=2)，直到大于等于 blockDim.x
    // 在循环内部：
    //    a. 使用 if 语句筛选活跃线程 (模拟 10.3 的控制发散问题)
    //       条件是：threadIdx.x % stride == 0 (注意：这里是否需要仔细思考逻辑？)
    //       书中 10.3 的逻辑是：随着 stride 增加，只有 threadIdx.x 是 stride 倍数的线程工作
    //    b. 活跃线程执行加法： input[i] = input[i] + input[i + stride]
    //    c. 重要！！不要忘记在每轮迭代后同步线程

    // ----------- 你的代码开始 -----------

    // [TODO 1]: 编写 for 循环结构
    // 这里是小于等于，因为i = 2 * threadIdx.x
    for (int stride = 1; stride <= blockDim.x; stride *= 2)
    {
        // [TODO 2]: 编写 if 判断和加法逻辑 (注意：这里是发散的源头)
        if (threadIdx.x % stride == 0)
            input[i] += input[i + stride];
        // [TODO 3]: 线程同步栅栏
        __syncthreads();
    }

    // ----------- 你的代码结束 -----------

    // 3. 将最终结果写回 output
    // 只有线程 0 负责把最终结果 (存放在 input[0]) 写入 output 指针指向的内存
    if (threadIdx.x == 0)
    {
        *output = input[0];
    }
}

// ---------------------------------------------------------
// Host 端代码 (已写好，用于验证你的 Kernel)
// ---------------------------------------------------------
int main()
{
    float *h_input, *h_output_gpu;
    float *d_input, *d_output;
    float expected_result = N * 1.0f; // 因为我们将数组初始化为 1.0

    // 1. 分配 Host 内存
    h_input = (float *)malloc(N * sizeof(float));
    h_output_gpu = (float *)malloc(sizeof(float));

    // 2. 初始化数据
    for (int i = 0; i < N; i++)
    {
        h_input[i] = 1.0f;
    }

    // 3. 分配 Device 内存
    cudaMalloc((void **)&d_input, N * sizeof(float));
    cudaMalloc((void **)&d_output, sizeof(float));

    // 4. 拷贝数据 Host -> Device
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // 5. 启动 Kernel
    // 我们只需要 1 个 Block，线程数是元素数量的一半 (1024 个线程处理 2048 个数据)
    printf("Launching kernel with 1 block, %d threads...\n", THREADS_PER_BLOCK);
    simpleSumReduction<<<1, THREADS_PER_BLOCK>>>(d_input, d_output);

    // 检查 Kernel 是否出错
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // 6. 拷贝结果 Device -> Host
    cudaMemcpy(h_output_gpu, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    // 7. 验证结果
    printf("Expected Sum: %.2f\n", expected_result);
    printf("GPU Result:   %.2f\n", *h_output_gpu);

    if (fabs(*h_output_gpu - expected_result) < 1e-5)
    {
        printf("✅ TEST PASSED! 恭喜你完成了简单的规约树！\n");
    }
    else
    {
        printf("❌ TEST FAILED. 请检查你的循环逻辑和同步。\n");
    }

    // 清理内存
    free(h_input);
    free(h_output_gpu);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}