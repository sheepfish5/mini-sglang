from pathlib import Path

from transformers import AutoTokenizer

model_path = Path("./models").resolve()
tok = AutoTokenizer.from_pretrained(str(model_path))

messages = [
    {"role": "user", "content": "Who are you?"}
]

prompt = tok.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
# input_ids = tok.encode(prompt, add_special_tokens=False)
input_ids = tok.encode(prompt, add_special_tokens=False)

print(prompt)
print(input_ids)

"""
<|begin_of_text|><|header_start|>user<|header_end|>

Who are you?<|eot|><|header_start|>assistant<|header_end|>


[200000, 200005, 1556, 200006, 368, 33267, 583, 650, 43, 200008, 200005, 140680, 200006, 368]

"""


"""

mini-sglang 启动命令：

python3 -m minisgl --model-path "./models" --shell --attention-backend fi --tp 4

sglang 启动命令：
python -m sglang.launch_server  \
  --model-path "/root/autodl-tmp/mini-sglang/models"   \
  --port 30000   \
  --mem-fraction-static 0.9   \
  --context-length 8192   \
  --max-running-requests 4   \
  --disable-cuda-graph \
  --tp-size 4


sglang 测试命令：

curl http://localhost:30000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "default",
    "messages": [
      {"role": "user", "content": "Who are you?"}
    ],
    "max_tokens": 1    
  }'

"""