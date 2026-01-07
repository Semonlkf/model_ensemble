"""
FastAPI 服务: math-shepherd-mistral-7b-prm 奖励模型

模型来源: peiyi9979/math-shepherd-mistral-7b-prm
模型特点: 
    - 使用特殊 token 'ки' 标记每个推理步骤的结束
    - 输出 '+' (正确) 和 '-' (错误) 的概率作为步骤分数

启动方式:
    # 方式1: 使用 --gpu 参数指定 GPU
    python serves/math_shepherd_prm.py --port 8001 --gpu 0
    
    # 方式2: 使用 CUDA_VISIBLE_DEVICES 环境变量
    CUDA_VISIBLE_DEVICES=1 python serves/math_shepherd_prm.py --port 8001
    
    # 方式3: 指定本地模型路径
    python serves/math_shepherd_prm.py --model_path /path/to/local/model --port 8001

API 接口规范 (所有奖励模型服务必须遵循):
    POST /v1/scores
    请求体: {"model": "model_name", "input": "text with step markers"}
    响应体: {"data": [{"score": float, "step_scores": [{"step_index": int, "score": float}, ...]}]}
"""

import os
import argparse
from typing import List, Optional
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============ 全局变量 ============
model = None
tokenizer = None
candidate_tokens = None
step_tag_id = None
device = None

# ============ Math-Shepherd 特殊 token 定义 ============
GOOD_TOKEN = '+'
BAD_TOKEN = '-'
STEP_TAG = 'ки'  # Math-Shepherd 使用此 token 标记步骤结束


# ============ 通用请求/响应模型 (所有奖励模型服务共用) ============
class ScoreRequest(BaseModel):
    """
    奖励模型评分请求 (通用接口)
    
    Attributes:
        model: 模型名称标识符
        input: 待评分的文本，包含问题和推理步骤
    """
    model: str
    input: str


class ScoreResponse(BaseModel):
    """
    奖励模型评分响应 (通用接口)
    
    Attributes:
        data: 评分结果列表，每个元素包含:
            - score: 整体评分 (float)
            - step_scores: 各步骤评分列表 [{"step_index": int, "score": float}, ...]
    """
    data: List[dict]


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    model_loaded: bool
    model_name: Optional[str] = None


# ============ Math-Shepherd 模型加载 ============
def load_model(model_path: str, gpu_id: Optional[int] = None):
    """
    加载 Math-Shepherd 奖励模型
    
    Args:
        model_path: 模型路径 (HuggingFace 或本地路径)
        gpu_id: 指定 GPU ID (0, 1, 2, ...)，如果为 None 则使用 CUDA_VISIBLE_DEVICES 或默认 GPU
    """
    global model, tokenizer, candidate_tokens, step_tag_id, device
    
    # 确定使用的设备
    if not torch.cuda.is_available():
        device = torch.device("cpu")
        print("⚠️ CUDA not available, using CPU")
    elif gpu_id is not None:
        device = torch.device(f"cuda:{gpu_id}")
        print(f"🎯 Using GPU {gpu_id}")
    else:
        device = torch.device("cuda")
        print(f"🎯 Using default GPU (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')})")
    
    print(f"🔄 Loading Math-Shepherd tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 获取 Math-Shepherd 特殊 token 的 ID
    candidate_tokens = tokenizer.encode(f"{GOOD_TOKEN} {BAD_TOKEN}")[1:]  # [648, 387]
    step_tag_id = tokenizer.encode(f"{STEP_TAG}")[-1]  # 12902
    
    print(f"📝 Candidate tokens ('+', '-'): {candidate_tokens}")
    print(f"📝 Step tag ID ('ки'): {step_tag_id}")
    
    print(f"🔄 Loading Math-Shepherd model from {model_path}...")
    
    # 根据设备类型选择加载方式
    if device.type == "cuda":
        if gpu_id is not None:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map={"": device}
            ).eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32
        ).eval().to(device)
    
    print(f"✅ Math-Shepherd model loaded on {device}")


# ============ Math-Shepherd 推理逻辑 ============
def compute_step_scores(text: str) -> dict:
    """
    计算 Math-Shepherd 每个推理步骤的得分
    
    Math-Shepherd 模型要求:
        - 每个步骤以 'ки' token 结尾
        - 模型在 'ки' 位置输出 '+' 和 '-' 的概率
        - '+' 的概率越高表示该步骤越正确
    
    Args:
        text: 包含问题和推理步骤的文本，格式如:
              "Question... Step 1: xxx ки Step 2: xxx ки ..."
        
    Returns:
        dict: {
            "score": float,  # 整体评分 (取所有步骤的最小分数)
            "step_scores": [{"step_index": int, "score": float}, ...]
        }
    """
    global model, tokenizer, candidate_tokens, step_tag_id, device
    
    input_ids = torch.tensor([tokenizer.encode(text)]).to(device)
    
    with torch.no_grad():
        logits = model(input_ids).logits[:, :, candidate_tokens]
        # softmax 后取第一个 token (good token '+') 的概率作为分数
        scores = logits.softmax(dim=-1)[:, :, 0]
        
        # 找到所有 step_tag ('ки') 位置的分数
        mask = input_ids[0] == step_tag_id
        step_scores = scores[0][mask].cpu().tolist()
    
    # 计算整体分数
    if step_scores:
        # 使用最小分数作为整体评分 (最弱步骤决定整体质量)
        overall_score = min(step_scores)
    else:
        overall_score = 0.0
    
    return {
        "score": overall_score,
        "step_scores": [
            {"step_index": i, "score": s} 
            for i, s in enumerate(step_scores)
        ]
    }


# ============ FastAPI 应用 ============
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    yield
    print("🛑 Shutting down Math-Shepherd server...")


app = FastAPI(
    title="Math-Shepherd PRM API",
    description="FastAPI 服务: math-shepherd-mistral-7b-prm 过程奖励模型",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口"""
    return HealthResponse(
        status="ok" if model is not None else "not_ready",
        model_loaded=model is not None,
        model_name="math-shepherd-mistral-7b-prm" if model else None
    )


@app.post("/v1/scores", response_model=ScoreResponse)
async def get_scores(request: ScoreRequest):
    """
    获取奖励分数
    
    请求体示例:
        {
            "model": "math-shepherd-mistral-7b-prm",
            "input": "Janet's ducks lay 16 eggs per day... Step 1: xxx ки Step 2: xxx ки"
        }
    
    响应体示例:
        {
            "data": [{
                "score": 0.95,
                "step_scores": [
                    {"step_index": 0, "score": 0.99},
                    {"step_index": 1, "score": 0.95}
                ]
            }]
        }
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = compute_step_scores(request.input)
        return ScoreResponse(data=[result])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 兼容不带 /v1 前缀的路径
@app.post("/scores", response_model=ScoreResponse)
async def get_scores_compat(request: ScoreRequest):
    """兼容旧版 API 路径"""
    return await get_scores(request)


def main():
    import uvicorn
    
    parser = argparse.ArgumentParser(description="Math-Shepherd PRM FastAPI Server")
    parser.add_argument("--model_path", type=str, default="/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--peiyi9979--math-shepherd-mistral-7b-prm",
                        help="HuggingFace model path or local path")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind")
    parser.add_argument("--gpu", type=int, default=None, 
                        help="GPU ID to use (e.g., 0, 1, 2). If not specified, uses CUDA_VISIBLE_DEVICES or default GPU")
    
    args = parser.parse_args()
    
    load_model(args.model_path, gpu_id=args.gpu)
    
    print(f"🚀 Starting Math-Shepherd PRM Server at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

