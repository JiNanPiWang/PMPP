#!/bin/bash

# --- 核心函数：编译和分析单个文件 ---
analyze_file() {
    LOCAL_SRC_FILE=$1
    # 示例: conv_basic.cu -> conv_basic
    LOCAL_BASE_NAME=$(basename "$LOCAL_SRC_FILE" .cu)
    # 示例: conv_basic_results
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
    nvcc -o "$LOCAL_OUTPUT_DIR/$LOCAL_BASE_NAME" "$LOCAL_SRC_FILE" -lineinfo

    if [ $? -ne 0 ]; then
        echo "❌ 编译 $LOCAL_SRC_FILE 失败，跳过分析。"
        return 1
    fi

    # 3. Nsight Compute (ncu) 性能分析
    # -f --set full: 运行全量指标分析
    echo "--- 运行 Nsight Compute 分析 (ncu) ---"
    ncu -o "$LOCAL_OUTPUT_DIR/${LOCAL_BASE_NAME}_ncu_report" -f --set full "$LOCAL_OUTPUT_DIR/$LOCAL_BASE_NAME"

    # 4. Nsight Systems (nsys) 系统级分析
    # --stats=true: 打印简单统计信息
    echo "--- 运行 Nsight Systems 分析 (nsys) ---"
    nsys profile -o "$LOCAL_OUTPUT_DIR/${LOCAL_BASE_NAME}_nsys_report" --stats=true "$LOCAL_OUTPUT_DIR/$LOCAL_BASE_NAME"

    echo "✅ $LOCAL_SRC_FILE 分析完成！结果在 $LOCAL_OUTPUT_DIR。"
}

# --- 执行分析 ---

# 1. 分析 conv_basic.cu (基线)
analyze_file "conv_basic.cu"

# 2. 分析 conv_tiled.cu (Shared Memory 优化)
analyze_file "conv_tiled.cu"

# 3. 分析 conv_l2_cache.cu (L2 Cache 优化)
analyze_file "conv_l2_cache.cu"

echo ""
echo "====================================================="
echo "所有分析任务已完成！"
echo "请对比以下三个文件夹中的报告："
echo "* conv_basic_analysis_results"
echo "* conv_tiled_analysis_results"
echo "* conv_l2_cache_analysis_results"
echo "====================================================="