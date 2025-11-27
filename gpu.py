import torch
import time
import sys

def occupy_memory(target_gb=32):
    """
    在每张 AMD 显卡上占用指定大小的显存。
    """
    # 检查 CUDA (ROCm) 是否可用
    if not torch.cuda.is_available():
        print("错误: 未检测到 GPU 或 ROCm 环境。")
        sys.exit(1)

    num_gpus = torch.cuda.device_count()
    print(f"检测到 {num_gpus} 张显卡，准备每张占用 {target_gb}GB 显存...")

    # 计算 float32 类型所需的元素数量
    # 1 GB = 1024^3 bytes
    # float32 = 4 bytes
    elements_per_gpu = (target_gb * (1024 ** 3)) // 4
    
    tensors = []

    try:
        for i in range(num_gpus):
            device = torch.device(f"cuda:{i}")
            print(f"正在显卡 {i} 上分配内存...", end="")
            
            # 使用 torch.empty 分配内存（速度最快，几乎无计算消耗）
            # 如果发现 rocm-smi 中显存未显示被占用，可以将 .empty 改为 .zeros
            t = torch.empty(elements_per_gpu, dtype=torch.float32, device=device)
            
            # 将张量存入列表，防止被 Python 垃圾回收机制释放
            tensors.append(t)
            print("完成")
            
    except RuntimeError as e:
        print(f"\n错误: 显存分配失败。可能显存不足以分配完整的 {target_gb}GB。")
        print(f"详细错误: {e}")
        # 如果需要，可以在这里添加逻辑尝试减少分配量
        sys.exit(1)

    print("\n所有显卡显存已占用。脚本正在休眠以保持占用状态...")
    print("请按 Ctrl+C 停止脚本并释放显存。")

    # 进入无限循环休眠，保持进程存活但不占用 CPU/GPU 算力
    try:
        while True:
            time.sleep(3600) # 每小时醒一次，几乎不消耗资源
    except KeyboardInterrupt:
        print("\n脚本停止，正在释放显存...")

if __name__ == "__main__":
    # 如果显卡物理显存刚好是 32GB，建议设置为 31GB 或 31.5GB，
    # 因为系统上下文和显示输出也会占用少量显存，占满会导致 OOM。
    occupy_memory(target_gb=16)