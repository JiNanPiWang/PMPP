#!/bin/bash

nvcc -o matrix_mul matrix_mul_tiled.cu -lineinfo

# -o: 指定输出文件名
# -f: 强制覆盖旧文件
# --set full: 跑全量分析 (虽然慢一点，但为了学习一次到位吧！)
ncu -o matrix_mul_ncu_report -f --set full ./matrix_mul

# -o: 指定输出文件名
# --stats=true: 在终端直接打印简单的统计信息（可选）
nsys profile -o matrix_mul_report --stats=true ./matrix_mul