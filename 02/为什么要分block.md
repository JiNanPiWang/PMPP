1. 协作 (Cooperation): 同一个 Block 内的线程可以通过 Shared Memory（共享内存）和 Barrier Synchronization（栅栏同步）进行通信和协作。不同 Block 之间通常无法通信。

2. 扩展性 (Scalability): 这是文中 Fig 2.13 后面强调的。Block 是独立的执行单元。如果你有一个很弱的 GPU（只有 2 个核），它可以一次跑 2 个 Block，慢慢跑完。如果你有一个很强的 GPU（有 100 个核），它可以一次跑 100 个 Block。硬件越强，跑得越快，但代码不用改。