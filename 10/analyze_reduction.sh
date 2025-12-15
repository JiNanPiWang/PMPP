#!/bin/bash

# --- 核心函数：编译和分析单个文件 ---
analyze_file() {
    LOCAL_SRC_FILE=$1
    # 示例: improved_reduction_smem.cu -> improved_reduction_smem
    LOCAL_BASE_NAME=$(basename "$LOCAL_SRC_FILE" .cu)
    # 示例: improved_reduction_smem_analysis_results
    LOCAL_OUTPUT_DIR="${LOCAL_BASE_NAME}_analysis_results"

    echo "====================================================="
    echo "▶️ 开始分析文件: $LOCAL_SRC_FILE"
    echo "====================================================="

    # 1. 清理和创建文件夹
    if [ -d "$LOCAL_OUTPUT_DIR" ]; then
        echo "清理旧的分析结果文件夹: $LOCAL_OUTPUT_DIR"
        rm -rf "$LOCAL_OUTPUT_DIR"
    fi
    mkdir -p "$LOCAL_OUTPUT_DIR"
    echo "创建输出文件夹: $LOCAL_OUTPUT_DIR"

    # 2. 编译阶段
    echo "--- 编译程序 ---"
    # 使用 -lineinfo 以便 ncu 报告能准确对应源代码行
    # 注意：共享内存版本可能需要更高的编译优化等级，虽然默认即可，但为了兼容性保留 nvcc 默认设置。
    nvcc -o "$LOCAL_OUTPUT_DIR/$LOCAL_BASE_NAME" "$LOCAL_SRC_FILE" -lineinfo

    if [ $? -ne 0 ]; then
        echo "❌ 编译 $LOCAL_SRC_FILE 失败，跳过分析。"
        return 1
    fi

    # 3. Nsight Compute (ncu) 性能分析
    # -f --set full: 运行全量指标分析。用于 Kernel 级的详细指标。
    echo "--- 运行 Nsight Compute 分析 (ncu) ---"
    ncu -o "$LOCAL_OUTPUT_DIR/${LOCAL_BASE_NAME}_ncu_report" -f --set full "$LOCAL_OUTPUT_DIR/$LOCAL_BASE_NAME"

    # 4. Nsight Systems (nsys) 系统级分析
    # --stats=true: 打印简单统计信息。用于系统级的整体执行时间。
    echo "--- 运行 Nsight Systems 分析 (nsys) ---"
    nsys profile -o "$LOCAL_OUTPUT_DIR/${LOCAL_BASE_NAME}_nsys_report" --stats=true "$LOCAL_OUTPUT_DIR/$LOCAL_BASE_NAME"

    echo "✅ $LOCAL_SRC_FILE 分析完成！结果在 $LOCAL_OUTPUT_DIR。"
}

# --- 执行分析 ---

echo "--- 正在分析所有三个规约优化版本 ---"
# 1. 基线规约 (Divergence, Global Memory)
analyze_file "simple_reduction.cu"

# 2. 最小化发散规约 (No Divergence, Global Memory)
analyze_file "improved_reduction.cu"

# 3. 共享内存规约 (Shared Memory, Fastest version expected)
analyze_file "improved_reduction_smem.cu"


echo ""
echo "====================================================="
echo "所有分析任务已完成！"
echo "请对比以下三个文件夹中的报告："
echo "* simple_reduction_analysis_results"
echo "* improved_reduction_analysis_results"
echo "* improved_reduction_smem_analysis_results"
echo "====================================================="