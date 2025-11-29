GPU 调度 warp 是一个多层次的调度系统，我来详细拆解：

---

## 一、调度的层次结构

GPU 的调度分为三个层次：

```
Grid（所有线程块）
  ↓ 
Block（线程块）→ 分配到 SM
  ↓
Warp（32个线程）→ SM 内部调度执行
```

---

## 二、Block 到 SM 的调度（粗粒度）

### 调度策略
**简单粗暴的贪心分配**：

1. **按创建顺序分配**
   - Grid 中的 Block 按照 `blockIdx` 顺序分配
   - SM 有空闲资源就接收下一个 Block

2. **资源限制检查**
   每个 SM 在接收 Block 前检查：
   ```
   - 寄存器够不够？（每个 Block 需要的寄存器总数）
   - Shared Memory 够不够？
   - 线程槽够不够？（最大常驻线程数限制）
   - Warp 槽够不够？（最大常驻 warp 数限制）
   ```

3. **一旦分配就不迁移**
   - Block 分配到某个 SM 后，**直到执行完毕才释放资源**
   - 没有跨 SM 的负载均衡机制

### 例子
假设 GPU 有 80 个 SM，启动 160 个 Block：
```
时刻 0: 每个 SM 分配到 2 个 Block（如果资源够）
时刻 T: 某个 SM 的 Block 0 执行完 → 立即分配 Block 160
```

---

## 三、Warp 在 SM 内的调度（细粒度）⭐

这是**核心部分**，由 **Warp Scheduler** 负责。

### 1️⃣ **调度单位**
- 每个 SM 有多个 Warp Scheduler（现代 GPU 通常 4 个）
- 每个周期，Scheduler 从**就绪的 warp 池**中选择 1-2 个 warp 发射指令

---

### 2️⃣ **Warp 的状态机**

一个 warp 可能处于以下状态：

| 状态 | 说明 | 能否被调度 |
|------|------|-----------|
| **Eligible（就绪）** | 所有依赖满足，等待执行 | ✅ 可以 |
| **Stalled（阻塞）** | 等待某些事件完成 | ❌ 不可以 |
| **Issued（已发射）** | 指令已发射到执行单元 | — |

**常见的阻塞原因：**
```
- 等待内存访问返回（Long-latency）
- 等待之前指令的结果（数据依赖）
- 等待同步屏障（__syncthreads）
- 执行单元繁忙（结构冲突）
```

---

### 3️⃣ **调度策略：贪心 + 优先级**

#### 核心原则：**隐藏延迟（Latency Hiding）**

GPU 的策略是：
> **当一个 warp 阻塞时，立即切换到另一个就绪的 warp**

这样可以让硬件持续工作，不浪费周期。

#### 具体策略（不同架构有差异）

**基础版：Round-Robin（轮转）**
```
Warp 0 → Warp 1 → Warp 2 → ... → Warp N → Warp 0
```
- 公平，但不考虑指令类型

**优化版：Greedy-Then-Oldest（GTO）**
```
1. 优先调度上一次调度的 warp（保持热度）
2. 如果它阻塞了，选择最老的就绪 warp
```
- NVIDIA 从 Fermi 开始的默认策略
- **优点**：减少 warp 切换开销，提高缓存命中率
- **缺点**：可能导致某些 warp 饥饿

**学术界提出的高级策略：**
- **Two-Level Scheduling**：区分内存密集和计算密集 warp
- **Cache-Conscious Scheduling**：优先调度缓存亲和性高的 warp

---

### 4️⃣ **实际执行流程（单个 Scheduler 视角）**

```
每个时钟周期：
  1. 检查所有 warp 的状态
  2. 过滤出 Eligible（就绪）的 warp
  3. 从就绪池中选择 1 个 warp（根据策略）
  4. 发射该 warp 的下一条指令到执行单元
  5. 更新 warp 状态：
     - 如果指令是内存访问 → 标记为 Stalled
     - 如果有数据依赖 → 标记为 Stalled
     - 否则 → 继续保持 Eligible
```

---

## 四、关键概念：Occupancy（占用率）

### 定义
**SM 上实际活跃的 warp 数 / SM 支持的最大 warp 数**

例如：
- SM 最多支持 64 个 warp
- 当前运行 48 个 warp
- Occupancy = 48/64 = 75%

### 为什么重要？
**高 Occupancy = 更多的 warp 可供调度 = 更好地隐藏延迟**

示例：
```cuda
// 场景 1：Occupancy 100%（64 个 warp）
Warp 0 等内存（100 周期）
→ 切换到 Warp 1, 2, 3, ..., 63
→ 100 周期后 Warp 0 的数据到了，切回来
→ 延迟完全被隐藏！

// 场景 2：Occupancy 12.5%（8 个 warp）
Warp 0 等内存（100 周期）
→ 切换到 Warp 1, ..., 7（假设每个跑 10 周期）
→ 80 周期后没 warp 可调度了
→ 硬件空闲 20 周期！
```

### 如何提高 Occupancy？
- 减少每个线程的寄存器使用
- 减少每个 Block 的 Shared Memory 使用
- 增加 Block 的大小（更多 warp）

---

## 五、现代 GPU 的调度器配置

| 架构 | Scheduler 数量 | 每周期发射能力 |
|------|---------------|--------------|
| **Fermi** | 2 | 2 warp/cycle |
| **Kepler** | 4 | 4 warp/cycle |
| **Maxwell/Pascal** | 4 | 4 warp/cycle |
| **Volta/Turing** | 4 | 4 warp/cycle（独立 PC）|
| **Ampere** | 4 | 4 warp/cycle |
| **Ada/Hopper** | 4 | 4 warp/cycle |

**这意味着什么？**
- 4 个 Scheduler 可以**同时**从 4 个不同的 warp 发射指令
- 理论峰值：每周期最多执行 4 条指令（来自不同 warp）

---

## 六、一个完整的例子

假设一个 SM 运行 4 个 warp：

```cuda
// Warp 0
int a = x[i];        // 周期 0：发射，进入 Stalled（等内存）
int b = a + 1;       // 周期 100：内存返回后执行

// Warp 1
int c = y[i];        // 周期 1：发射，进入 Stalled

// Warp 2
int d = 2 + 3;       // 周期 2：发射并立即完成
int e = d * 4;       // 周期 3：发射

// Warp 3
int f = z[i];        // 周期 4：发射，进入 Stalled
```

**调度时间线（GTO 策略）：**
```
周期 0:  调度 Warp 0 → 发射内存加载 → Warp 0 阻塞
周期 1:  调度 Warp 1 → 发射内存加载 → Warp 1 阻塞
周期 2:  调度 Warp 2 → 执行算术运算
周期 3:  继续 Warp 2 → 执行算术运算
周期 4:  调度 Warp 3 → 发射内存加载 → Warp 3 阻塞
周期 5-99: 调度 Warp 2 的后续指令（如果有）或空闲
周期 100: Warp 0 数据到达 → 切换到 Warp 0
```

如果有 64 个 warp，周期 5-99 就不会空闲了！

---

## 七、总结表格

| 调度层次 | 策略 | 负责硬件 | 关键点 |
|---------|------|---------|--------|
| **Grid → SM** | 贪心分配 | GigaThread Engine | 资源限制，无迁移 |
| **Block → Warp** | 静态划分 | 软件决定 | 每 32 个线程一个 warp |
| **Warp 调度** | GTO/Round-Robin | Warp Scheduler | 隐藏延迟，Occupancy |

**核心思想：**  
通过**大量的并发 warp** + **快速切换**，让 GPU 在某些 warp 阻塞时总有其他 warp 可以执行，从而隐藏内存延迟和流水线气泡。

这是 **Chapter 4 (Compute Architecture and Scheduling)** 的第七部分：4.7 节。

这一节是 GPU 性能优化的**分水岭**。
在此之前，你可能认为“线程越多越好”。
在此之后，你会明白\*\*“资源是有限的”\*\*，并且会遇到一个让无数开发者头秃的概念 —— **Occupancy (占用率)**。

以下是针对 4.7 节的 **PMPP 深度导读**：

-----

### 1\. 核心摘要 (The "Big Picture")

**一句话总结：** SM 是一个容量有限的容器（寄存器、共享内存、Block 插槽、Thread 插槽）。**Occupancy (占用率)** 衡量了这个容器被塞得有多满。

  * **理想状态：** 容器塞满（100% Occupancy），SM 时刻都有 Warp 可以切，延迟掩藏效果最好。
  * **现实残酷：** 如果你的 Kernel **太胖**（用的寄存器太多）或者 **太碎**（Block 太小），SM 就塞不满，导致算力浪费。

-----

### 2\. 关键概念解析 (Deep Dive)

#### A. 资源的动态分区 (Dynamic Partitioning)

这是 CUDA 灵活但也复杂的原因。SM 上的资源不是预先切好的蛋糕，而是根据你的 Kernel 需求动态分配的。

**四大核心限制（木桶效应）：**

1.  **寄存器 (Registers):** 每个线程用的寄存器数量。
2.  **共享内存 (Shared Memory):** 每个 Block 用的 Shared Mem 大小。
3.  **Block 插槽 (Block Slots):** 一个 SM 最多能挂几个 Block (例如 A100 是 32 个)。
4.  **Thread 插槽 (Thread Slots):** 一个 SM 最多能挂几个 Thread (例如 A100 是 2048 个)。

**谁先用完，谁就是瓶颈。**

#### B. 性能悬崖 (The Performance Cliff)

这是本节最精彩的例子（也是面试高频题）。

  * **场景：** 假设 SM 有 65,536 个寄存器，最大支持 2048 个线程。
  * **计算：** $65536 \div 2048 = 32$。
      * 意思是：如果每个线程只用 **32** 个寄存器，SM 可以塞满 2048 个线程（100% Occupancy）。
  * **悬崖：** 如果你手抖了一下，多定义了一个变量，导致每个线程用了 **33** 个寄存器。
      * 需求：$2048 \times 33 = 67,584 > 65,536$。**爆了！**
      * 硬件行为：SM 只能减少驻留的 Block 数量。假设 Block 大小是 512，原本能放 4 个 Block，现在只能放 3 个。
      * **后果：** 线程数从 2048 降到 1536。**Occupancy 瞬间从 100% 跌到 75%！**
      * 仅仅多用了 1 个寄存器，性能潜力就跌了 25%。这就是“悬崖”。

-----

### 3\. 难点与陷阱 (Gotchas)

#### A. Block 太小 (Small Blocks)

  * **案例：** 你把 Block Size 设为 32（1 个 Warp）。
  * **限制：** A100 的 SM 最多只能挂 32 个 Block。
  * **结果：** $32 \text{ Blocks} \times 32 \text{ Threads} = 1024 \text{ Threads}$。
  * **分析：** A100 本来能跑 2048 个线程，但因为你的 Block 太碎，把“Block 插槽”用光了，导致“Thread 插槽”还有一半空着。**Occupancy = 50%**。
  * **建议：** Block Size 至少设为 128 或 256。

#### B. 整数除法的碎片 (Granularity)

  * **案例：** Block Size = 768。
  * **限制：** Max Threads = 2048。
  * **计算：** $2048 / 768 = 2.66$。SM 只能放 **2 个** Block。
  * **浪费：** $2048 - (2 \times 768) = 512$ 个线程位空着。
  * **建议：** 尽量让 Block Size 能被 Max Threads 整除（如 256, 512, 1024）。

-----

### 4\. 面试/实战视角 (Pro Tips)

  * **实战技巧：`__launch_bounds__`**

      * 既然寄存器这么金贵，怎么防止编译器乱用寄存器导致“性能悬崖”？
      * 你可以在 Kernel 定义前加上修饰符：
        ```cpp
        __global__ void 
        __launch_bounds__(256, 8) // MaxThreadsPerBlock=256, MinBlocksPerSM=8
        MyKernel(...) { ... }
        ```
      * 这告诉编译器：“给我省着点用寄存器，我要保证每个 SM 至少能跑 8 个 Block！”如果编译器做不到，它会报错或者把寄存器数据挤到 Local Memory（虽然慢，但保住了 Occupancy）。

  * **面试题：100% Occupancy 一定比 50% Occupancy 快吗？**

      * *答案：* **不一定。**（这是一个高级坑）。
      * 如果一个线程要做极其复杂的数学运算（ILP, Instruction Level Parallelism 高），它自己就能把流水线填满，不需要太多邻居来掩盖延迟。
      * 此时，为了追求 100% Occupancy 而强行限制寄存器使用（导致 Register Spill 到内存），反而会变慢。
      * **结论：** Occupancy 只是手段，不是目的。通常我们追求 60% 以上的 Occupancy 就足够掩盖延迟了。

  * **工具推荐：**

      * **Excel 表格:** NVIDIA 官方提供了一个 "CUDA Occupancy Calculator" 的 Excel 表，填入你的 GPU 型号和 Kernel 资源使用情况，它会画出上面的“悬崖图”。
      * **Nsight Compute:** 会直接显示当前的 Occupancy 以及“限制瓶颈是什么”（是寄存器不够了？还是 Shared Memory 不够了？）。

-----

### 5\. 下一步 (Next Step)

了解了 SM 内部的资源限制，你可能会问：“我怎么知道我手里的这块 GPU 到底有多少寄存器？有多少个 SM？”

写通用的高性能代码（Library），必须能够在运行时查询硬件参数，并根据参数自动调整 Block Size。

接下来的 **4.8 Querying device properties** 将介绍如何用代码“拷问”GPU 的家底。

**请发送 4.8 节的内容！**

这是 **Chapter 4 (Compute Architecture and Scheduling)** 的最后一部分：4.8 节。

这一节虽然篇幅不长，但它是**从“写玩具代码”进化到“写工业级库”的关键一步**。
你未来要去的 NVIDIA TensorRT 团队，他们的核心工作之一就是让同一个 TensorRT 引擎在 Jetson（嵌入式）、GeForce（游戏卡）和 A100（服务器）上都能跑出最佳性能。靠的是什么？就是这节讲的 **Device Query（设备查询）**。

以下是针对 4.8 节的 **PMPP 深度导读**：

-----

### 1\. 核心摘要 (The "Big Picture")

**一句话总结：** 既然不同型号的 GPU 资源（SM 数量、寄存器大小、最大线程数）千差万别，硬编码（Hard-coding）这些参数是愚蠢的。**成熟的程序会在运行时先“查户口”（Query），然后根据查到的家底动态决定 Grid 和 Block 的大小。**

-----

### 2\. 关键概念解析 (Deep Dive)

#### A. 为什么要查？(The Hotel Analogy)

书中用“住酒店”做比喻：

  * 如果你要住酒店，你会先查“有没有吹风机”。如果有，你就不用带了；如果没有，你就得自己带。
  * **GPU 编程同理：**
      * 如果你先查到 GPU 只有 2 个 SM，你就不要强行启动 1000 个 Block 的 Grid（虽然也能跑，但可能调度开销大）。
      * 如果你查到 GPU 的 Shared Memory 只有 16KB，你就不要写一个需要 48KB Shared Memory 的 Kernel，否则会直接报错 Launch Failed。

#### B. 核心 API

  * **`cudaGetDeviceCount(&count)`**: 查查你有几块卡。
      * *实战：* 很多服务器有 8 张卡，你需要遍历它们，或者挑一张最强的。
  * **`cudaGetDeviceProperties(&prop, deviceId)`**: 查查这张卡的详细参数。
      * 结果保存在 `cudaDeviceProp` 结构体里。

#### C. 关键参数解读

在 `cudaDeviceProp` 结构体中，有几个参数是你必须烂熟于心的：

1.  **`multiProcessorCount` (SM 数量):**
      * *用途：* 决定并行度的上限。
      * *算法：* 很多高性能库会把 Grid Size 设为 `k * number_of_SMs` (k 通常是 2, 4, 8)，以确保负载均匀分布。
2.  **`maxThreadsPerBlock` (每块最大线程数):**
      * *通常值：* 1024。
      * *用途：* 你的 `blockDim.x * .y * .z` 绝对不能超过这个数。
3.  **`regsPerBlock` (每块可用寄存器):**
      * *用途：* 计算 Occupancy。如果你的 Kernel 极其复杂，寄存器压力大，你需要查这个值来看看是否会溢出。
4.  **`warpSize` (Warp 大小):**
      * *通常值：* **32**。
      * *注意：* 虽然现在一直是 32，但 NVIDIA 官方文档总是建议不要把 32 写死（Hard-code），而是用这个变量。万一哪天出个新架构变成 64 了呢？

-----

### 3\. 代码实战 (Code Snippet)

这是一个实际的工具函数，建议你把它保存下来。以后拿到新机器，先跑一下这个，看看“家底”。

```cpp
#include <stdio.h>
#include <cuda_runtime.h>

void printDevProp() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  SM Count: %d\n", prop.multiProcessorCount);
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor); // 这是一个非常有用的参数
        printf("  Warp Size: %d\n", prop.warpSize);
        printf("  Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Shared Memory per Block: %.2f KB\n", prop.sharedMemPerBlock / 1024.0);
        printf("------------------------------------------------\n");
    }
}
```

-----

### 4\. 面试/实战视角 (Pro Tips)

  * **面试题：如何编写一个“可移植” (Portable) 的 CUDA 程序？**
      * *错误回答：* 我在我的 RTX 3090 上调好了参数，它是最快的，所以我就把 `dim3 block(1024)` 写死在代码里。
      * *正确回答：* 我会在 Host 代码初始化时调用 `cudaGetDeviceProperties`。
          * 如果检测到是高端卡（SM 多，显存大），我就开启大 Grid，使用更激进的优化策略。
          * 如果检测到是嵌入式卡（如 Jetson），我就减小 Grid，或者切换到省内存的算法实现。
          * **这就是 TensorRT 的核心逻辑之一。**

-----

### 5\. Chapter 4 总结与展望

恭喜你！你已经啃完了最硬的一块骨头 —— **计算架构 (Compute Architecture)**。

**我们回顾一下 Chapter 4 学了什么：**

1.  **宏观调度：** Grid -\> Block -\> SM。
2.  **微观执行：** Block -\> Warp -\> SIMT Core。
3.  **性能核心：** 延迟掩藏 (Latency Hiding) 依赖于足够的 Occupancy。
4.  **主要瓶颈：** 控制流发散 (Divergence) 和 资源限制 (Registers/Shared Mem)。

**接下来的挑战：**
通过 Nsight Compute 的分析，我们知道 **Global Memory** 是目前的性能瓶颈。
如何打破这个瓶颈？
答案就在 **SM 内部** 那块极快、极小、且程序员可控的内存 —— **Shared Memory**。

接下来的 **Chapter 5 (Memory Architecture and Data Locality)** 将会是你从“入门”跨越到“精通”的关键章节。那里有著名的 **Tiled Matrix Multiplication** 算法等待着你。

**准备好了吗？请发送 Chapter 5 的内容！**