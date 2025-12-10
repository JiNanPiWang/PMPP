#!/bin/bash

# 1. 定义输出文件夹的名称
OUTPUT_DIR="conv_basic_analysis_results"

# 2. 清理旧的文件夹 (可选，但推荐)
if [ -d "$OUTPUT_DIR" ]; then
    echo "清理旧的分析结果文件夹: $OUTPUT_DIR"
    rm -rf "$OUTPUT_DIR"
fi

# 3. 创建新的输出文件夹
echo "创建输出文件夹: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# --- 编译阶段 ---

# 4. 使用 nvcc 编译 CUDA 程序
# -o: 指定输出文件名
# -lineinfo: 包含行号信息，方便 ncu 关联性能数据和源代码
echo "--- 编译程序 ---"
nvcc -o "$OUTPUT_DIR/conv_basic" conv_basic.cu -lineinfo

# 检查编译是否成功
if [ $? -ne 0 ]; then
    echo "编译失败，停止后续操作。"
    exit 1
fi

# --- Nsight Compute (ncu) 性能分析阶段 ---

# 5. 运行 Nsight Compute
# -o: 指定输出文件名 (放在新文件夹内)
# -f: 强制覆盖旧文件
# --set full: 运行全量指标分析 (适用于学习和深度优化)
echo "--- 运行 Nsight Compute 分析 (ncu) ---"
ncu -o "$OUTPUT_DIR/conv_basic_ncu_report" -f --set full "$OUTPUT_DIR/conv_basic"

# --- Nsight Systems (nsys) 系统级分析阶段 ---

# 6. 运行 Nsight Systems
# -o: 指定输出文件名 (放在新文件夹内)
# --stats=true: 在终端直接打印简单的统计信息
echo "--- 运行 Nsight Systems 分析 (nsys) ---"
nsys profile -o "$OUTPUT_DIR/conv_basic_nsys_report" --stats=true "$OUTPUT_DIR/conv_basic"

echo "✅ 分析完成！所有生成的文件已放入 $OUTPUT_DIR 文件夹内。"
echo "生成的文件包括:"
echo "* 可执行文件: $OUTPUT_DIR/conv_basic"
echo "* Nsight Compute 报告: $OUTPUT_DIR/conv_basic_ncu_report.ncu-rep"
echo "* Nsight Systems 报告: $OUTPUT_DIR/conv_basic_nsys_report.qdrep (和 .sqlite 文件，如果生成)"