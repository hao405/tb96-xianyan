if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/test" ]; then
    mkdir ./logs/test
fi

if [ ! -d "./logs/test/new" ]; then
    mkdir ./logs/test/new
fi

# =================================================================
# 新增：自动寻找空闲 GPU 的函数
# =================================================================
get_free_gpu() {
    # 1. 尝试使用 nvidia-smi (NVIDIA)
    if command -v nvidia-smi &> /dev/null; then
        # 获取显存剩余量最大的 GPU ID
        # query-gpu: index, memory.free
        # sort -k2 -n -r: 按第二列(显存)数字降序排列
        # head -n 1: 取第一个
        # awk: 提取 ID
        nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -n -r | head -n 1 | awk -F, '{print $1}'
        
    # 2. 尝试使用 rocm-smi (AMD) -> 鉴于你有 export MIOPEN... 可能是 AMD 环境
    elif command -v rocm-smi &> /dev/null; then
        # 获取 GPU 使用率最低的卡 (ROCm 的显存 info 格式较复杂，这里用 usage% 近似)
        # --showusage --csv 输出格式通常为: device, usage%
        # tail -n +2: 跳过标题
        # sort -k2 -n: 按使用率升序
        # head -n 1: 取最低的一个
        # sed: 去掉 'card' 前缀 (如 card0 -> 0)
        rocm-smi --showusage --csv | tail -n +2 | awk -F, '{print $1, $2}' | sort -k2 -n | head -n 1 | awk '{print $1}' | sed 's/card//g'
        
    # 3. 如果都没有，默认返回 0
    else
        echo "0"
    fi
}

model_name=TimeBridge
seq_len=96
# 移除硬编码的 GPU=0
# GPU=0
root=./data
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_DEBUG_DISABLE_FIND_DB=1

# 注意：这里我们每次循环都重新检测一次 GPU，以防多任务并行时冲突
# 如果你想整个脚本只用一张卡，可以把 GPU=$(get_free_gpu) 放在 for 循环外面

alpha=0.1
data_name=weather
for pred_len in 336
do
  # 自动获取当前最空闲的 GPU ID
  GPU=$(get_free_gpu)
  # 打印一下选中的 GPU，方便 log 查看
  echo "Auto-selected GPU: $GPU for pred_len: $pred_len"

  export HIP_VISIBLE_DEVICES=$GPU
  
  # 为了兼容 NVIDIA 环境，同时也设置 CUDA_VISIBLE_DEVICES
  export CUDA_VISIBLE_DEVICES=$GPU

  HIP_VISIBLE_DEVICES=$GPU \
  python -u tune3.py \
    --is_training 1 \
    --root_path $root/weather/ \
    --data_path weather.csv \
    --model_id $data_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --enc_in 21 \
    --ca_layers 1 \
    --pd_layers 1 \
    --ia_layers 1 \
    --des 'Exp' \
    --period 48 \
    --num_p 12 \
    --d_model 128 \
    --d_ff 128 \
    --alpha $alpha \
    --itr 1 | tee logs/test/new/$data_name'_'$alpha'_'$model_name'_'$pred_len.logs
done