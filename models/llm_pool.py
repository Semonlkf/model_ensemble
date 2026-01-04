import time
from typing import List, Dict, Optional, TypedDict, Union
import openai
import math
import requests
class ModelConfig(TypedDict):
    api_base: str
    api_key: str
    model_name: str
    default_temp: float
    is_reward_model: bool

class ModelInstance:
    def __init__(self, name: str, config: ModelConfig):
        self.name = name
        self.config = config
        self.is_reward_model = config['is_reward_model']
        self.client = openai.OpenAI(
            base_url=config['api_base'],
            api_key=config['api_key']
        )

class LLMModelPool:
    """
    模型池管理器：只负责维护连接和执行 API 调用
    """
    def __init__(self):
        self.instances: Dict[str, ModelInstance] = {} # 改用字典方便查找
        self.call_count = 0

    def register_model(self, name: str, config: ModelConfig):
        instance = ModelInstance(name, config)
        self.instances[name] = instance
        print(f"✅ [ModelPool] Registered: {name} ({config['model_name']}) | Default Temp: {config['default_temp']}")

    def get_all_model_names(self) -> List[str]:
        return list(self.instances.keys())

    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        if model_name in self.instances:
            return self.instances[model_name].config
        return None

    def generate(self, prompt: str, model_name: str, n: int = 1, top_p: Optional[float] = None, temperature: Optional[float] = None , stop: Optional[str] = None, max_tokens: Optional[int] = None) -> Optional[List[str]]:
        """
        纯粹的生成接口。
        :param model_name: 必须指定要使用的模型名称
        :param temperature:如果不指定，使用模型默认配置
        """
        if model_name not in self.instances:
            print(f"❌ Model '{model_name}' not found!") # 建议 print 而不是 raise，防止崩溃
            return None
        self.call_count += 1
        selected_inst = self.instances[model_name]
        
        actual_temp = temperature if temperature is not None else selected_inst.config['default_temp']        
        try:
            response_stream = selected_inst.client.chat.completions.create(
                model=selected_inst.config['model_name'],
                messages=[{"role": "user", "content": prompt}],
                temperature=actual_temp,
                max_tokens=max_tokens if max_tokens is not None else 2048,
                n=n,
                stream=True,
                top_p=top_p if top_p is not None else 0.9,
                stop=stop
            )
            #content = response.choices[0].message.content
            samples_data = {i: [] for i in range(n)}

            for chunk in response_stream:
                if not chunk.choices:
                    continue
                
                delta = chunk.choices[0].delta
                index = chunk.choices[0].index  # 获取当前碎片属于哪一个采样
                
                if hasattr(delta, 'content') and delta.content is not None:
                    samples_data[index].append(delta.content)

            # 将各个 index 的碎片拼接成完整的字符串列表
            full_replies = [
                "".join(samples_data[i]) for i in range(n)
            ]

            return full_replies

        except Exception as e:
            print(f"❌ [API Error] {model_name}: {e}")
            return None
    
    def calculate_ppl(self, text: str, model_name: str) -> Optional[float]:
            """
            计算给定文本的困惑度 (Perplexity, PPL)。
            
            注意：
            1. 这需要后端支持 `/completions` (Legacy/Base) 接口。
            2. 必须支持 `echo=True` 参数以返回 Prompt 的 logprobs。
            """
            if model_name not in self.instances:
                print(f"❌ Model '{model_name}' not found!")
                return None

            selected_inst = self.instances[model_name]
            
            try:
                # 使用 completions 接口而非 chat
                # max_tokens=0 + echo=True 是获取 prompt logprobs 的标准技巧
                response = selected_inst.client.completions.create(
                    model=selected_inst.config['model_name'],
                    prompt=text,
                    max_tokens=0,    # 我们不需要生成新内容
                    echo=True,       # 关键：让 API 回显输入的 logprobs
                    logprobs=1       # 请求 logprobs
                )

                # 获取 logprobs 数据
                if not response.choices or not response.choices[0].logprobs:
                    print(f"❌ [PPL Error] No logprobs returned for {model_name}")
                    return None

                token_logprobs = response.choices[0].logprobs.token_logprobs
                tokens = response.choices[0].logprobs.tokens

                # 数据清洗：通常第一个 token 的 logprob 是 None (因为没有前文)，需要过滤掉
                valid_logprobs = [lp for lp in token_logprobs if lp is not None]
                
                if not valid_logprobs:
                    print("❌ [PPL Error] No valid logprobs found (text too short?)")
                    return None

                # 计算 PPL 公式: exp( - (1/N) * sum(log_probs) )
                # N 是 token 的数量
                sum_logprobs = sum(valid_logprobs)
                n_tokens = len(valid_logprobs)
                
                mean_logprob = sum_logprobs / n_tokens
                ppl = math.exp(-mean_logprob)

                # 可选：打印调试信息看看 token 切分情况
                # print(f"DEBUG: Tokens: {tokens[:5]}... Count: {n_tokens}")
                
                return ppl

            except Exception as e:
                print(f"❌ [PPL Calculation Error] {model_name}: {e}")
                # 常见错误提示：如果模型不支持 completions (如 gpt-3.5-turbo 聊天版)，这里会报错
                if "This is a chat model" in str(e):
                    print("⚠️ Hint: This model seems to be a Chat model. PPL calculation usually requires a Base/Completion model endpoint.")
                return None

    def get_reward_score(self, text: str, model_name: str, return_step_scores: bool = False) -> Optional[Union[float, Dict]]:
            """
            通用奖励模型评分接口。
            
            此函数是通用的，可以连接任何遵循以下 API 规范的奖励模型服务。
            
            ==================== 奖励模型 API 规范 ====================
            
            所有奖励模型的 FastAPI 服务必须实现以下接口:
            
            端点: POST /v1/scores (或 /scores)
            
            请求体 (Request Body):
                {
                    "model": str,   # 模型名称标识符
                    "input": str    # 待评分的文本
                }
            
            响应体 (Response Body):
                {
                    "data": [
                        {
                            "score": float,        # 整体评分 (0.0 ~ 1.0)
                            "step_scores": [       # 各步骤评分 (可选)
                                {"step_index": int, "score": float},
                                ...
                            ]
                        }
                    ]
                }
            
            已实现的奖励模型服务:
                - serves/math_shepherd_prm.py  (math-shepherd-mistral-7b-prm)
                - (可扩展添加更多奖励模型...)
            
            ===========================================================
            
            Args:
                text: 待评分的文本 (包含问题和推理步骤)
                model_name: 在模型池中注册的模型名称
                return_step_scores: 是否返回详细的步骤分数
                
            Returns:
                如果 return_step_scores=False: 返回整体分数 (float)
                如果 return_step_scores=True: 返回完整结果 (dict)，包含 score 和 step_scores
            """
            if model_name not in self.instances:
                print(f"❌ Model '{model_name}' not found!")
                return None

            selected_inst = self.instances[model_name]
            
            # 确保该模型在配置中被标记为奖励模型
            if not selected_inst.is_reward_model:
                print(f"⚠️ Warning: Model '{model_name}' is not marked as a reward model in config.")

            try:
                # 奖励模型使用 FastAPI 部署，通过 requests 直接调用 HTTP API
                # api_base 格式应为: http://host:port (不带 /v1 后缀)
                api_base = selected_inst.config['api_base'].rstrip('/')
                # 移除可能的 /v1 后缀以获取基础 URL
                if api_base.endswith('/v1'):
                    api_base = api_base[:-3]
                
                url = f"{api_base}/v1/scores"
                
                response = requests.post(
                    url,
                    json={
                        "model": selected_inst.config['model_name'],
                        "input": text,
                    },
                    timeout=60  # 60秒超时
                )
                response.raise_for_status()
                
                res_json = response.json()
                
                if 'data' in res_json and len(res_json['data']) > 0:
                    result = res_json['data'][0]
                    
                    if return_step_scores:
                        # 返回完整结果，包含整体分数和各步骤分数
                        return {
                            "score": float(result.get('score', 0.0)),
                            "step_scores": result.get('step_scores', [])
                        }
                    else:
                        # 只返回整体分数
                        return float(result.get('score', 0.0))
                else:
                    print(f"❌ [RM Error] Unexpected response format from {model_name}: {res_json}")
                    return None

            except requests.exceptions.Timeout:
                print(f"❌ [Reward Score Error] {model_name}: Request timeout")
                return None
            except requests.exceptions.RequestException as e:
                print(f"❌ [Reward Score Error] {model_name}: {e}")
                return None
            except Exception as e:
                print(f"❌ [Reward Score Error] {model_name}: {e}")
                return None