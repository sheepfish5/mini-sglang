import torch
import torch.nn.functional as F

a = torch.load("tmp/mini_logits.pt")
b = torch.load("tmp/sglang_logits.pt")

print("shape:", a.shape, b.shape)
print("max_abs_diff:", (a - b).abs().max().item())
print("mean_abs_diff:", (a - b).abs().mean().item())
print("cosine:", F.cosine_similarity(a, b, dim=0).item())

ak = torch.topk(a, 20)
bk = torch.topk(b, 20)

print("mini top20 ids :", ak.indices.tolist())
print("sgl  top20 ids :", bk.indices.tolist())
print("mini top20 vals:", ak.values.tolist()[:10])
print("sgl  top20 vals:", bk.values.tolist()[:10])
