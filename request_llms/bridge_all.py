
"""
    该文件中主要包含2个函数，是所有LLM的通用接口，它们会继续向下调用更底层的LLM模型，处理多模型并行等细节

    不具备多线程能力的函数：正常对话时使用，具备完备的交互功能，不可多线程
    1. predict(...)

    具备多线程调用能力的函数：在函数插件中被调用，灵活而简洁
    2. predict_no_ui_long_connection(...)
"""
import tiktoken, copy
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from toolbox import get_conf, trimmed_format_exc, apply_gpt_academic_string_mask

from .bridge_chatgpt import predict_no_ui_long_connection as chatgpt_noui
from .bridge_chatgpt import predict as chatgpt_ui

from .bridge_chatgpt_vision import predict_no_ui_long_connection as chatgpt_vision_noui
from .bridge_chatgpt_vision import predict as chatgpt_vision_ui

#from .bridge_minicpm import predict_no_ui_long_connection as chatglm_noui
#from .bridge_minicpm import predict as chatglm_ui

from .bridge_chatglm import predict_no_ui_long_connection as chatglm_noui
from .bridge_chatglm import predict as chatglm_ui

from .bridge_chatglm3 import predict_no_ui_long_connection as chatglm3_noui
from .bridge_chatglm3 import predict as chatglm3_ui

from .bridge_qianfan import predict_no_ui_long_connection as qianfan_noui
from .bridge_qianfan import predict as qianfan_ui

from .bridge_google_gemini import predict as genai_ui
from .bridge_google_gemini import predict_no_ui_long_connection  as genai_noui

from .bridge_ollama_gemma import predict as ollama_ui
from .bridge_ollama_gemma import predict_no_ui_long_connection  as ollama_noui

from .bridge_ollama_internlm2 import predict as ollama_ui
from .bridge_ollama_internlm2 import predict_no_ui_long_connection  as ollama_noui

from .bridge_llamafile_gemma import predict as llamafile_ui
from .bridge_llamafile_gemma import predict_no_ui_long_connection  as llamafile_noui

from .bridge_llamafile_internlm2 import predict as llamafile_ui
from .bridge_llamafile_internlm2 import predict_no_ui_long_connection  as llamafile_noui

from .bridge_llamafile_qwen import predict as llamafile_ui
from .bridge_llamafile_qwen import predict_no_ui_long_connection  as llamafile_noui

#from .bridge_ollama_internlm2 import predict as llamafile_ui
#from .bridge_ollama_internlm2 import predict_no_ui_long_connection  as llamafile_noui

colors = ['#FF00FF', '#00FFFF', '#FF0000', '#990099', '#009999', '#990044']

class LazyloadTiktoken(object):
    def __init__(self, model):
        self.model = model

    @staticmethod
    @lru_cache(maxsize=128)
    def get_encoder(model):
        print('正在加载tokenizer，如果是第一次运行，可能需要一点时间下载参数')
        tmp = tiktoken.encoding_for_model(model)
        print('加载tokenizer完毕')
        return tmp
    
    def encode(self, *args, **kwargs):
        encoder = self.get_encoder(self.model) 
        return encoder.encode(*args, **kwargs)
    
    def decode(self, *args, **kwargs):
        encoder = self.get_encoder(self.model) 
        return encoder.decode(*args, **kwargs)

# Endpoint 重定向
API_URL_REDIRECT, AZURE_ENDPOINT, AZURE_ENGINE = get_conf("API_URL_REDIRECT", "AZURE_ENDPOINT", "AZURE_ENGINE")
openai_endpoint = "https://api.openai.com/v1/chat/completions"
api2d_endpoint = "https://openai.api2d.net/v1/chat/completions"
newbing_endpoint = "wss://sydney.bing.com/sydney/ChatHub"
if not AZURE_ENDPOINT.endswith('/'): AZURE_ENDPOINT += '/'
azure_endpoint = AZURE_ENDPOINT + f'openai/deployments/{AZURE_ENGINE}/chat/completions?api-version=2023-05-15'
# 兼容旧版的配置
try:
    API_URL = get_conf("API_URL")
    if API_URL != "https://api.openai.com/v1/chat/completions": 
        openai_endpoint = API_URL
        print("警告！API_URL配置选项将被弃用，请更换为API_URL_REDIRECT配置")
except:
    pass
# 新版配置
if openai_endpoint in API_URL_REDIRECT: openai_endpoint = API_URL_REDIRECT[openai_endpoint]
if api2d_endpoint in API_URL_REDIRECT: api2d_endpoint = API_URL_REDIRECT[api2d_endpoint]
if newbing_endpoint in API_URL_REDIRECT: newbing_endpoint = API_URL_REDIRECT[newbing_endpoint]


# 获取tokenizer
tokenizer_gpt35 = LazyloadTiktoken("gpt-3.5-turbo")
tokenizer_gpt4 = LazyloadTiktoken("gpt-4")
get_token_num_gpt35 = lambda txt: len(tokenizer_gpt35.encode(txt, disallowed_special=()))
get_token_num_gpt4 = lambda txt: len(tokenizer_gpt4.encode(txt, disallowed_special=()))


# 开始初始化模型
AVAIL_LLM_MODELS, LLM_MODEL = get_conf("AVAIL_LLM_MODELS", "LLM_MODEL")
AVAIL_LLM_MODELS = AVAIL_LLM_MODELS + [LLM_MODEL]
# -=-=-=-=-=-=- 以下这部分是最早加入的最稳定的模型 -=-=-=-=-=-=-
model_info = {
# 新加入部分
    "ollama_gemma": {
        "fn_with_ui": ollama_ui,
        "fn_without_ui": ollama_noui,
        "endpoint": 'http://localhost:11434/api/chat',
        "max_token": 1024 * 32,
        "tokenizer": tokenizer_gpt35,
        "token_cnt": get_token_num_gpt35,
    },
    "ollama_internlm2": {
        "fn_with_ui": ollama_ui,
        "fn_without_ui": ollama_noui,
        "endpoint": 'http://localhost:11434/api/chat',
        "max_token": 1024 * 32,
        "tokenizer": tokenizer_gpt35,
        "token_cnt": get_token_num_gpt35,
    },
    "llamafile_gemma": {
        "fn_with_ui": llamafile_ui,
        "fn_without_ui": llamafile_noui,
        "endpoint": 'http://localhost:8080/api/chat',
        "max_token": 1024 * 32,
        "tokenizer": tokenizer_gpt35,
        "token_cnt": get_token_num_gpt35,
    },
    "llamafile_internlm2": {
        "fn_with_ui": llamafile_ui,
        "fn_without_ui": llamafile_noui,
        "endpoint": 'http://localhost:8080',
        "max_token": 1024 * 32,
        "tokenizer": tokenizer_gpt35,
        "token_cnt": get_token_num_gpt35,
    },
    "llamafile_qwen": {
        "fn_with_ui": llamafile_ui,
        "fn_without_ui": llamafile_noui,
        "endpoint": 'http://localhost:8080/api/chat',
        "max_token": 1024 * 32,
        "tokenizer": tokenizer_gpt35,
        "token_cnt": get_token_num_gpt35,
    },
    

    # openai
    "gpt-3.5-turbo": {
        "fn_with_ui": chatgpt_ui,
        "fn_without_ui": chatgpt_noui,
        "endpoint": openai_endpoint,
        "max_token": 4096,
        "tokenizer": tokenizer_gpt35,
        "token_cnt": get_token_num_gpt35,
    },
    
    "gpt-3.5-turbo-16k": {
        "fn_with_ui": chatgpt_ui,
        "fn_without_ui": chatgpt_noui,
        "endpoint": openai_endpoint,
        "max_token": 16385,
        "tokenizer": tokenizer_gpt35,
        "token_cnt": get_token_num_gpt35,
    },

    "gpt-3.5-turbo-0613": {
        "fn_with_ui": chatgpt_ui,
        "fn_without_ui": chatgpt_noui,
        "endpoint": openai_endpoint,
        "max_token": 4096,
        "tokenizer": tokenizer_gpt35,
        "token_cnt": get_token_num_gpt35,
    },

    "gpt-3.5-turbo-16k-0613": {
        "fn_with_ui": chatgpt_ui,
        "fn_without_ui": chatgpt_noui,
        "endpoint": openai_endpoint,
        "max_token": 16385,
        "tokenizer": tokenizer_gpt35,
        "token_cnt": get_token_num_gpt35,
    },

    "gpt-3.5-turbo-1106": {#16k
        "fn_with_ui": chatgpt_ui,
        "fn_without_ui": chatgpt_noui,
        "endpoint": openai_endpoint,
        "max_token": 16385,
        "tokenizer": tokenizer_gpt35,
        "token_cnt": get_token_num_gpt35,
    },

    "gpt-4": {
        "fn_with_ui": chatgpt_ui,
        "fn_without_ui": chatgpt_noui,
        "endpoint": openai_endpoint,
        "max_token": 8192,
        "tokenizer": tokenizer_gpt4,
        "token_cnt": get_token_num_gpt4,
    },

    "gpt-4-32k": {
        "fn_with_ui": chatgpt_ui,
        "fn_without_ui": chatgpt_noui,
        "endpoint": openai_endpoint,
        "max_token": 32768,
        "tokenizer": tokenizer_gpt4,
        "token_cnt": get_token_num_gpt4,
    },

    "gpt-4-turbo-preview": {
        "fn_with_ui": chatgpt_ui,
        "fn_without_ui": chatgpt_noui,
        "endpoint": openai_endpoint,
        "max_token": 128000,
        "tokenizer": tokenizer_gpt4,
        "token_cnt": get_token_num_gpt4,
    },

    "gpt-4-1106-preview": {
        "fn_with_ui": chatgpt_ui,
        "fn_without_ui": chatgpt_noui,
        "endpoint": openai_endpoint,
        "max_token": 128000,
        "tokenizer": tokenizer_gpt4,
        "token_cnt": get_token_num_gpt4,
    },

    "gpt-4-0125-preview": {
        "fn_with_ui": chatgpt_ui,
        "fn_without_ui": chatgpt_noui,
        "endpoint": openai_endpoint,
        "max_token": 128000,
        "tokenizer": tokenizer_gpt4,
        "token_cnt": get_token_num_gpt4,
    },

    "gpt-3.5-random": {
        "fn_with_ui": chatgpt_ui,
        "fn_without_ui": chatgpt_noui,
        "endpoint": openai_endpoint,
        "max_token": 4096,
        "tokenizer": tokenizer_gpt4,
        "token_cnt": get_token_num_gpt4,
    },
    
    "gpt-4-vision-preview": {
        "fn_with_ui": chatgpt_vision_ui,
        "fn_without_ui": chatgpt_vision_noui,
        "endpoint": openai_endpoint,
        "max_token": 4096,
        "tokenizer": tokenizer_gpt4,
        "token_cnt": get_token_num_gpt4,
    },


    # azure openai
    "azure-gpt-3.5":{
        "fn_with_ui": chatgpt_ui,
        "fn_without_ui": chatgpt_noui,
        "endpoint": azure_endpoint,
        "max_token": 4096,
        "tokenizer": tokenizer_gpt35,
        "token_cnt": get_token_num_gpt35,
    },

    "azure-gpt-4":{
        "fn_with_ui": chatgpt_ui,
        "fn_without_ui": chatgpt_noui,
        "endpoint": azure_endpoint,
        "max_token": 8192,
        "tokenizer": tokenizer_gpt4,
        "token_cnt": get_token_num_gpt4,
    },

    # api_2d (此后不需要在此处添加api2d的接口了，因为下面的代码会自动添加)
    "api2d-gpt-3.5-turbo": {
        "fn_with_ui": chatgpt_ui,
        "fn_without_ui": chatgpt_noui,
        "endpoint": api2d_endpoint,
        "max_token": 4096,
        "tokenizer": tokenizer_gpt35,
        "token_cnt": get_token_num_gpt35,
    },

    "api2d-gpt-4": {
        "fn_with_ui": chatgpt_ui,
        "fn_without_ui": chatgpt_noui,
        "endpoint": api2d_endpoint,
        "max_token": 8192,
        "tokenizer": tokenizer_gpt4,
        "token_cnt": get_token_num_gpt4,
    },


    # 将 chatglm 直接对齐到 chatglm2
    "chatglm": {
        "fn_with_ui": chatglm_ui,
        "fn_without_ui": chatglm_noui,
        "endpoint": None,
        "max_token": 1024,
        "tokenizer": tokenizer_gpt35,
        "token_cnt": get_token_num_gpt35,
    },
    "chatglm2": {
        "fn_with_ui": chatglm_ui,
        "fn_without_ui": chatglm_noui,
        "endpoint": None,
        "max_token": 1024,
        "tokenizer": tokenizer_gpt35,
        "token_cnt": get_token_num_gpt35,
    },
    "chatglm3": {
        "fn_with_ui": chatglm3_ui,
        "fn_without_ui": chatglm3_noui,
        "endpoint": None,
        "max_token": 8192,
        "tokenizer": tokenizer_gpt35,
        "token_cnt": get_token_num_gpt35,
    },
    "qianfan": {
        "fn_with_ui": qianfan_ui,
        "fn_without_ui": qianfan_noui,
        "endpoint": None,
        "max_token": 2000,
        "tokenizer": tokenizer_gpt35,
        "token_cnt": get_token_num_gpt35,
    },
    "gemini-pro": {
        "fn_with_ui": genai_ui,
        "fn_without_ui": genai_noui,
        "endpoint": None,
        "max_token": 1024 * 32,
        "tokenizer": tokenizer_gpt35,
        "token_cnt": get_token_num_gpt35,
    },
    "gemini-pro-vision": {
        "fn_with_ui": genai_ui,
        "fn_without_ui": genai_noui,
        "endpoint": None,
        "max_token": 1024 * 32,
        "tokenizer": tokenizer_gpt35,
        "token_cnt": get_token_num_gpt35,
    },
    
}

# -=-=-=-=-=-=- api2d 对齐支持 -=-=-=-=-=-=-
for model in AVAIL_LLM_MODELS:
    if model.startswith('api2d-') and (model.replace('api2d-','') in model_info.keys()):
        mi = copy.deepcopy(model_info[model.replace('api2d-','')])
        mi.update({"endpoint": api2d_endpoint})
        model_info.update({model: mi})

# -=-=-=-=-=-=- azure 对齐支持 -=-=-=-=-=-=-
for model in AVAIL_LLM_MODELS:
    if model.startswith('azure-') and (model.replace('azure-','') in model_info.keys()):
        mi = copy.deepcopy(model_info[model.replace('azure-','')])
        mi.update({"endpoint": azure_endpoint})
        model_info.update({model: mi})

# -=-=-=-=-=-=- 以下部分是新加入的模型，可能附带额外依赖 -=-=-=-=-=-=-
if "claude-1-100k" in AVAIL_LLM_MODELS or "claude-2" in AVAIL_LLM_MODELS:
    from .bridge_claude import predict_no_ui_long_connection as claude_noui
    from .bridge_claude import predict as claude_ui
    model_info.update({
        "claude-1-100k": {
            "fn_with_ui": claude_ui,
            "fn_without_ui": claude_noui,
            "endpoint": None,
            "max_token": 8196,
            "tokenizer": tokenizer_gpt35,
            "token_cnt": get_token_num_gpt35,
        },
    })
    model_info.update({
        "claude-2": {
            "fn_with_ui": claude_ui,
            "fn_without_ui": claude_noui,
            "endpoint": None,
            "max_token": 8196,
            "tokenizer": tokenizer_gpt35,
            "token_cnt": get_token_num_gpt35,
        },
    })
if "jittorllms_rwkv" in AVAIL_LLM_MODELS:
    from .bridge_jittorllms_rwkv import predict_no_ui_long_connection as rwkv_noui
    from .bridge_jittorllms_rwkv import predict as rwkv_ui
    model_info.update({
        "jittorllms_rwkv": {
            "fn_with_ui": rwkv_ui,
            "fn_without_ui": rwkv_noui,
            "endpoint": None,
            "max_token": 1024,
            "tokenizer": tokenizer_gpt35,
            "token_cnt": get_token_num_gpt35,
        },
    })
if "jittorllms_llama" in AVAIL_LLM_MODELS:
    from .bridge_jittorllms_llama import predict_no_ui_long_connection as llama_noui
    from .bridge_jittorllms_llama import predict as llama_ui
    model_info.update({
        "jittorllms_llama": {
            "fn_with_ui": llama_ui,
            "fn_without_ui": llama_noui,
            "endpoint": None,
            "max_token": 1024,
            "tokenizer": tokenizer_gpt35,
            "token_cnt": get_token_num_gpt35,
        },
    })
if "jittorllms_pangualpha" in AVAIL_LLM_MODELS:
    from .bridge_jittorllms_pangualpha import predict_no_ui_long_connection as pangualpha_noui
    from .bridge_jittorllms_pangualpha import predict as pangualpha_ui
    model_info.update({
        "jittorllms_pangualpha": {
            "fn_with_ui": pangualpha_ui,
            "fn_without_ui": pangualpha_noui,
            "endpoint": None,
            "max_token": 1024,
            "tokenizer": tokenizer_gpt35,
            "token_cnt": get_token_num_gpt35,
        },
    })
if "moss" in AVAIL_LLM_MODELS:
    from .bridge_moss import predict_no_ui_long_connection as moss_noui
    from .bridge_moss import predict as moss_ui
    model_info.update({
        "moss": {
            "fn_with_ui": moss_ui,
            "fn_without_ui": moss_noui,
            "endpoint": None,
            "max_token": 1024,
            "tokenizer": tokenizer_gpt35,
            "token_cnt": get_token_num_gpt35,
        },
    })
if "stack-claude" in AVAIL_LLM_MODELS:
    from .bridge_stackclaude import predict_no_ui_long_connection as claude_noui
    from .bridge_stackclaude import predict as claude_ui
    model_info.update({
        "stack-claude": {
            "fn_with_ui": claude_ui,
            "fn_without_ui": claude_noui,
            "endpoint": None,
            "max_token": 8192,
            "tokenizer": tokenizer_gpt35,
            "token_cnt": get_token_num_gpt35,
        }
    })
if "newbing-free" in AVAIL_LLM_MODELS:
    try:
        from .bridge_newbingfree import predict_no_ui_long_connection as newbingfree_noui
        from .bridge_newbingfree import predict as newbingfree_ui
        model_info.update({
            "newbing-free": {
                "fn_with_ui": newbingfree_ui,
                "fn_without_ui": newbingfree_noui,
                "endpoint": newbing_endpoint,
                "max_token": 4096,
                "tokenizer": tokenizer_gpt35,
                "token_cnt": get_token_num_gpt35,
            }
        })
    except:
        print(trimmed_format_exc())
if "newbing" in AVAIL_LLM_MODELS:   # same with newbing-free
    try:
        from .bridge_newbingfree import predict_no_ui_long_connection as newbingfree_noui
        from .bridge_newbingfree import predict as newbingfree_ui
        model_info.update({
            "newbing": {
                "fn_with_ui": newbingfree_ui,
                "fn_without_ui": newbingfree_noui,
                "endpoint": newbing_endpoint,
                "max_token": 4096,
                "tokenizer": tokenizer_gpt35,
                "token_cnt": get_token_num_gpt35,
            }
        })
    except:
        print(trimmed_format_exc())
if "chatglmft" in AVAIL_LLM_MODELS:   # same with newbing-free
    try:
        from .bridge_chatglmft import predict_no_ui_long_connection as chatglmft_noui
        from .bridge_chatglmft import predict as chatglmft_ui
        model_info.update({
            "chatglmft": {
                "fn_with_ui": chatglmft_ui,
                "fn_without_ui": chatglmft_noui,
                "endpoint": None,
                "max_token": 4096,
                "tokenizer": tokenizer_gpt35,
                "token_cnt": get_token_num_gpt35,
            }
        })
    except:
        print(trimmed_format_exc())
if "internlm" in AVAIL_LLM_MODELS:
    try:
        from .bridge_internlm import predict_no_ui_long_connection as internlm_noui
        from .bridge_internlm import predict as internlm_ui
        model_info.update({
            "internlm": {
                "fn_with_ui": internlm_ui,
                "fn_without_ui": internlm_noui,
                "endpoint": None,
                "max_token": 4096,
                "tokenizer": tokenizer_gpt35,
                "token_cnt": get_token_num_gpt35,
            }
        })
    except:
        print(trimmed_format_exc())
if "chatglm_onnx" in AVAIL_LLM_MODELS:
    try:
        from .bridge_chatglmonnx import predict_no_ui_long_connection as chatglm_onnx_noui
        from .bridge_chatglmonnx import predict as chatglm_onnx_ui
        model_info.update({
            "chatglm_onnx": {
                "fn_with_ui": chatglm_onnx_ui,
                "fn_without_ui": chatglm_onnx_noui,
                "endpoint": None,
                "max_token": 4096,
                "tokenizer": tokenizer_gpt35,
                "token_cnt": get_token_num_gpt35,
            }
        })
    except:
        print(trimmed_format_exc())
if "qwen-local" in AVAIL_LLM_MODELS:
    try:
        from .bridge_qwen_local import predict_no_ui_long_connection as qwen_local_noui
        from .bridge_qwen_local import predict as qwen_local_ui
        model_info.update({
            "qwen-local": {
                "fn_with_ui": qwen_local_ui,
                "fn_without_ui": qwen_local_noui,
                "endpoint": None,
                "max_token": 4096,
                "tokenizer": tokenizer_gpt35,
                "token_cnt": get_token_num_gpt35,
            }
        })
    except:
        print(trimmed_format_exc())
if "qwen-turbo" in AVAIL_LLM_MODELS or "qwen-plus" in AVAIL_LLM_MODELS or "qwen-max" in AVAIL_LLM_MODELS:   # zhipuai
    try:
        from .bridge_qwen import predict_no_ui_long_connection as qwen_noui
        from .bridge_qwen import predict as qwen_ui
        model_info.update({
            "qwen-turbo": {
                "fn_with_ui": qwen_ui,
                "fn_without_ui": qwen_noui,
                "endpoint": None,
                "max_token": 6144,
                "tokenizer": tokenizer_gpt35,
                "token_cnt": get_token_num_gpt35,
            },
            "qwen-plus": {
                "fn_with_ui": qwen_ui,
                "fn_without_ui": qwen_noui,
                "endpoint": None,
                "max_token": 30720,
                "tokenizer": tokenizer_gpt35,
                "token_cnt": get_token_num_gpt35,
            },
            "qwen-max": {
                "fn_with_ui": qwen_ui,
                "fn_without_ui": qwen_noui,
                "endpoint": None,
                "max_token": 28672,
                "tokenizer": tokenizer_gpt35,
                "token_cnt": get_token_num_gpt35,
            }
        })
    except:
        print(trimmed_format_exc())
if "spark" in AVAIL_LLM_MODELS:   # 讯飞星火认知大模型
    try:
        from .bridge_spark import predict_no_ui_long_connection as spark_noui
        from .bridge_spark import predict as spark_ui
        model_info.update({
            "spark": {
                "fn_with_ui": spark_ui,
                "fn_without_ui": spark_noui,
                "endpoint": None,
                "max_token": 4096,
                "tokenizer": tokenizer_gpt35,
                "token_cnt": get_token_num_gpt35,
            }
        })
    except:
        print(trimmed_format_exc())
if "sparkv2" in AVAIL_LLM_MODELS:   # 讯飞星火认知大模型
    try:
        from .bridge_spark import predict_no_ui_long_connection as spark_noui
        from .bridge_spark import predict as spark_ui
        model_info.update({
            "sparkv2": {
                "fn_with_ui": spark_ui,
                "fn_without_ui": spark_noui,
                "endpoint": None,
                "max_token": 4096,
                "tokenizer": tokenizer_gpt35,
                "token_cnt": get_token_num_gpt35,
            }
        })
    except:
        print(trimmed_format_exc())
if "sparkv3" in AVAIL_LLM_MODELS:   # 讯飞星火认知大模型
    try:
        from .bridge_spark import predict_no_ui_long_connection as spark_noui
        from .bridge_spark import predict as spark_ui
        model_info.update({
            "sparkv3": {
                "fn_with_ui": spark_ui,
                "fn_without_ui": spark_noui,
                "endpoint": None,
                "max_token": 4096,
                "tokenizer": tokenizer_gpt35,
                "token_cnt": get_token_num_gpt35,
            }
        })
    except:
        print(trimmed_format_exc())
if "llama2" in AVAIL_LLM_MODELS:   # llama2
    try:
        from .bridge_llama2 import predict_no_ui_long_connection as llama2_noui
        from .bridge_llama2 import predict as llama2_ui
        model_info.update({
            "llama2": {
                "fn_with_ui": llama2_ui,
                "fn_without_ui": llama2_noui,
                "endpoint": None,
                "max_token": 4096,
                "tokenizer": tokenizer_gpt35,
                "token_cnt": get_token_num_gpt35,
            }
        })
    except:
        print(trimmed_format_exc())
if "zhipuai" in AVAIL_LLM_MODELS:   # zhipuai
    try:
        from .bridge_zhipu import predict_no_ui_long_connection as zhipu_noui
        from .bridge_zhipu import predict as zhipu_ui
        model_info.update({
            "zhipuai": {
                "fn_with_ui": zhipu_ui,
                "fn_without_ui": zhipu_noui,
                "endpoint": None,
                "max_token": 4096,
                "tokenizer": tokenizer_gpt35,
                "token_cnt": get_token_num_gpt35,
            }
        })
    except:
        print(trimmed_format_exc())
if "deepseekcoder" in AVAIL_LLM_MODELS:   # deepseekcoder
    try:
        from .bridge_deepseekcoder import predict_no_ui_long_connection as deepseekcoder_noui
        from .bridge_deepseekcoder import predict as deepseekcoder_ui
        model_info.update({
            "deepseekcoder": {
                "fn_with_ui": deepseekcoder_ui,
                "fn_without_ui": deepseekcoder_noui,
                "endpoint": None,
                "max_token": 2048,
                "tokenizer": tokenizer_gpt35,
                "token_cnt": get_token_num_gpt35,
            }
        })
    except:
        print(trimmed_format_exc())
# if "skylark" in AVAIL_LLM_MODELS:
#     try:
#         from .bridge_skylark2 import predict_no_ui_long_connection as skylark_noui
#         from .bridge_skylark2 import predict as skylark_ui
#         model_info.update({
#             "skylark": {
#                 "fn_with_ui": skylark_ui,
#                 "fn_without_ui": skylark_noui,
#                 "endpoint": None,
#                 "max_token": 4096,
#                 "tokenizer": tokenizer_gpt35,
#                 "token_cnt": get_token_num_gpt35,
#             }
#         })
#     except:
#         print(trimmed_format_exc())

# <-- 用于定义和切换多个azure模型 -->
AZURE_CFG_ARRAY = get_conf("AZURE_CFG_ARRAY")
if len(AZURE_CFG_ARRAY) > 0:
    for azure_model_name, azure_cfg_dict in AZURE_CFG_ARRAY.items():
        # 可能会覆盖之前的配置，但这是意料之中的
        if not azure_model_name.startswith('azure'): 
            raise ValueError("AZURE_CFG_ARRAY中配置的模型必须以azure开头")
        endpoint_ = azure_cfg_dict["AZURE_ENDPOINT"] + \
            f'openai/deployments/{azure_cfg_dict["AZURE_ENGINE"]}/chat/completions?api-version=2023-05-15'
        model_info.update({
            azure_model_name: {
                "fn_with_ui": chatgpt_ui,
                "fn_without_ui": chatgpt_noui,
                "endpoint": endpoint_,
                "azure_api_key": azure_cfg_dict["AZURE_API_KEY"],
                "max_token": azure_cfg_dict["AZURE_MODEL_MAX_TOKEN"],
                "tokenizer": tokenizer_gpt35,   # tokenizer只用于粗估token数量
                "token_cnt": get_token_num_gpt35,
            }
        })
        if azure_model_name not in AVAIL_LLM_MODELS:
            AVAIL_LLM_MODELS += [azure_model_name]




def LLM_CATCH_EXCEPTION(f):
    """
    装饰器函数，将错误显示出来
    """
    def decorated(inputs, llm_kwargs, history, sys_prompt, observe_window, console_slience):
        try:
            return f(inputs, llm_kwargs, history, sys_prompt, observe_window, console_slience)
        except Exception as e:
            tb_str = '\n```\n' + trimmed_format_exc() + '\n```\n'
            observe_window[0] = tb_str
            return tb_str
    return decorated


def predict_no_ui_long_connection(inputs, llm_kwargs, history, sys_prompt, observe_window=[], console_slience=False):
    """
    发送至LLM，等待回复，一次性完成，不显示中间过程。但内部用stream的方法避免中途网线被掐。
    inputs：
        是本次问询的输入
    sys_prompt:
        系统静默prompt
    llm_kwargs：
        LLM的内部调优参数
    history：
        是之前的对话列表
    observe_window = None：
        用于负责跨越线程传递已经输出的部分，大部分时候仅仅为了fancy的视觉效果，留空即可。observe_window[0]：观测窗。observe_window[1]：看门狗
    """
    import threading, time, copy

    inputs = apply_gpt_academic_string_mask(inputs, mode="show_llm")
    model = llm_kwargs['llm_model']
    n_model = 1
    if '&' not in model:
        assert not model.startswith("tgui"), "TGUI不支持函数插件的实现"

        # 如果只询问1个大语言模型：
        method = model_info[model]["fn_without_ui"]
        return method(inputs, llm_kwargs, history, sys_prompt, observe_window, console_slience)
    else:

        # 如果同时询问多个大语言模型，这个稍微啰嗦一点，但思路相同，您不必读这个else分支
        executor = ThreadPoolExecutor(max_workers=4)
        models = model.split('&')
        n_model = len(models)
        
        window_len = len(observe_window)
        assert window_len==3
        window_mutex = [["", time.time(), ""] for _ in range(n_model)] + [True]

        futures = []
        for i in range(n_model):
            model = models[i]
            method = model_info[model]["fn_without_ui"]
            llm_kwargs_feedin = copy.deepcopy(llm_kwargs)
            llm_kwargs_feedin['llm_model'] = model
            future = executor.submit(LLM_CATCH_EXCEPTION(method), inputs, llm_kwargs_feedin, history, sys_prompt, window_mutex[i], console_slience)
            futures.append(future)

        def mutex_manager(window_mutex, observe_window):
            while True:
                time.sleep(0.25)
                if not window_mutex[-1]: break
                # 看门狗（watchdog）
                for i in range(n_model): 
                    window_mutex[i][1] = observe_window[1]
                # 观察窗（window）
                chat_string = []
                for i in range(n_model):
                    chat_string.append( f"【{str(models[i])} 说】: <font color=\"{colors[i]}\"> {window_mutex[i][0]} </font>" )
                res = '<br/><br/>\n\n---\n\n'.join(chat_string)
                # # # # # # # # # # #
                observe_window[0] = res

        t_model = threading.Thread(target=mutex_manager, args=(window_mutex, observe_window), daemon=True)
        t_model.start()

        return_string_collect = []
        while True:
            worker_done = [h.done() for h in futures]
            if all(worker_done):
                executor.shutdown()
                break
            time.sleep(1)

        for i, future in enumerate(futures):  # wait and get
            return_string_collect.append( f"【{str(models[i])} 说】: <font color=\"{colors[i]}\"> {future.result()} </font>" )

        window_mutex[-1] = False # stop mutex thread
        res = '<br/><br/>\n\n---\n\n'.join(return_string_collect)
        return res


def predict(inputs, llm_kwargs, *args, **kwargs):
    """
    发送至LLM，流式获取输出。
    用于基础的对话功能。
    inputs 是本次问询的输入
    top_p, temperature是LLM的内部调优参数
    history 是之前的对话列表（注意无论是inputs还是history，内容太长了都会触发token数量溢出的错误）
    chatbot 为WebUI中显示的对话列表，修改它，然后yeild出去，可以直接修改对话界面内容
    additional_fn代表点击的哪个按钮，按钮见functional.py
    """

    inputs = apply_gpt_academic_string_mask(inputs, mode="show_llm")
    method = model_info[llm_kwargs['llm_model']]["fn_with_ui"]  # 如果这里报错，检查config中的AVAIL_LLM_MODELS选项
    yield from method(inputs, llm_kwargs, *args, **kwargs)

