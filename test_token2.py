from pathlib import Path

from transformers import AutoTokenizer

model_path = "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(model_path)

messages = [
    # {"role": "user", "content": "Who are you?"}
    {"role": "user", "content": "Hello!"}
]

prompt = tok.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
input_ids = tok.encode(prompt, add_special_tokens=False)
# input_ids = tok.encode(prompt)

print(prompt)
print(input_ids)

"""
<|begin_of_text|><|header_start|>user<|header_end|>

Who are you?<|eot|><|header_start|>assistant<|header_end|>


[200000, 200005, 1556, 200006, 368, 33267, 583, 650, 43, 200008, 200005, 140680, 200006, 368]

"""


"""
<|im_start|>user
Hello!<|im_end|>
<|im_start|>assistant

[151644, 872, 198, 9707, 0, 151645, 198, 151644, 77091, 198]
"""


"""
sglang 启动命令：

python -m sglang.launch_server \
  --model-path "Qwen/Qwen3-0.6B" \
  --port 30000 \
  --mem-fraction-static 0.7 \
  --context-length 8192 \
  --max-running-requests 4 \
  --attention-backend flashinfer \
  --disable-cuda-graph

  
sglang 测试命令：

curl http://localhost:30000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "prompt": [200000, 200005, 1556, 200006, 368, 33267, 583, 650, 43, 200008, 200005, 140680, 200006, 368],
    "max_tokens": 1,
    "logprobs": 5
  }'
"""