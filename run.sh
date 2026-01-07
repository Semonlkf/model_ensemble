#!/bin/bash
# ============================================================
# 模型服务启动脚本
# 
# 部署规则:
#   - 生成模型: 使用 vLLM 部署
#   - 奖励模型: 使用 FastAPI 部署 (serves/math_shepherd_prm.py)
# ============================================================

set -e  # 遇到错误立即退出

# ============ 配置参数 ============
DATE=$(date +"%Y%m%d")
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="serve_logs/${DATE}"

# 端口配置
GENERATION_MODEL_PORT=23456
REWARD_MODEL_PORT=8001

# GPU 配置 (根据实际情况修改)
GENERATION_GPU=0
REWARD_GPU=1

# 模型路径 (根据实际情况修改)
GENERATION_MODEL_PATH="/mnt/shared-storage-user/marti/models/Qwen3-8B"
REWARD_MODEL_PATH="peiyi9979/math-shepherd-mistral-7b-prm"

# ============ 创建日志目录 ============
mkdir -p "$LOG_DIR"
echo "📁 Log directory: $LOG_DIR"

# ============ 启动奖励模型 (FastAPI) ============
echo "🚀 Starting Reward Model (Math-Shepherd) on GPU $REWARD_GPU, port $REWARD_MODEL_PORT..."
nohup python serves/math_shepherd_prm.py \
    --model_path "$REWARD_MODEL_PATH" \
    --port $REWARD_MODEL_PORT \
    --gpu $REWARD_GPU \
    > "$LOG_DIR/math_shepherd_prm_${TIMESTAMP}.log" 2>&1 &
REWARD_PID=$!
echo "   PID: $REWARD_PID"

# ============ 启动生成模型 (vLLM) ============
echo "🚀 Starting Generation Model (Qwen3-8B) on GPU $GENERATION_GPU, port $GENERATION_MODEL_PORT..."
nohup env CUDA_VISIBLE_DEVICES=$GENERATION_GPU vllm serve "$GENERATION_MODEL_PATH" \
    --port $GENERATION_MODEL_PORT \
    --max-num-seqs 512 \
    --gpu-memory-utilization 0.8 \
    --max-logprobs 10 \
    > "$LOG_DIR/vllm_qwen3_8b_${TIMESTAMP}.log" 2>&1 &
VLLM_PID=$!
echo "   PID: $VLLM_PID"

# ============ 保存 PID 信息 ============
echo "REWARD_PID=$REWARD_PID" > "$LOG_DIR/pids_${TIMESTAMP}.txt"
echo "VLLM_PID=$VLLM_PID" >> "$LOG_DIR/pids_${TIMESTAMP}.txt"
echo "📝 PIDs saved to $LOG_DIR/pids_${TIMESTAMP}.txt"

# ============ 等待服务启动 ============
echo ""
echo "⏳ Waiting for services to start..."
echo "   - Reward Model log: $LOG_DIR/math_shepherd_prm_${TIMESTAMP}.log"
echo "   - Generation Model log: $LOG_DIR/vllm_qwen3_8b_${TIMESTAMP}.log"
echo ""

# 等待 vLLM 启动 (通常需要 30-60 秒)
sleep 10

# ============ 健康检查 ============
echo "🔍 Checking service health..."

# 检查奖励模型
if curl -s "http://localhost:$REWARD_MODEL_PORT/health" > /dev/null 2>&1; then
    echo "   ✅ Reward Model is ready"
else
    echo "   ⏳ Reward Model is still loading (check log for details)"
fi

# 检查生成模型
if curl -s "http://localhost:$GENERATION_MODEL_PORT/health" > /dev/null 2>&1; then
    echo "   ✅ Generation Model is ready"
else
    echo "   ⏳ Generation Model is still loading (check log for details)"
fi

echo ""
echo "============================================================"
echo "📋 Service Summary:"
echo "   Reward Model:     http://localhost:$REWARD_MODEL_PORT (GPU $REWARD_GPU)"
echo "   Generation Model: http://localhost:$GENERATION_MODEL_PORT (GPU $GENERATION_GPU)"
echo ""
echo "📂 Logs: $LOG_DIR/"
echo ""
echo "🛑 To stop services:"
echo "   kill $REWARD_PID $VLLM_PID"
echo "   or: kill \$(cat $LOG_DIR/pids_${TIMESTAMP}.txt | cut -d'=' -f2 | tr '\\n' ' ')"
echo "============================================================"

# ============ 可选: 运行主程序 ============
# 取消下面的注释以自动运行主程序
# echo ""
# echo "🏃 Starting main program..."
python -u run.py \
    --backend Qwen3-8B \
    --task gsm8k \
    --task_start_index 0 \
    --task_end_index 100 \
    --prompt_sample cot \
    --method_generate sample \
    --method_evaluate random \
    --n_generate_sample 4 \
    --n_evaluate_sample 3 \
    --baseline mcts \
    --model_pool_config configs/ensemble_example.yaml

python -u run.py \
    --task gsm8k \
    --task_start_index 0 \
    --task_end_index 100 \
    --prompt_sample cot \
    --method_generate sample \
    --method_evaluate llm_as_process_reward \
    --baseline lemcts \
    --backend_prm math_shepherd \
    --model_pool_config configs/ensemble_example.yaml