```python
                if r == 0:
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3
                    memory_reserved = torch.cuda.memory_reserved() / 1024**3
                    print(f"[load_weight]: name=={name}, shape=={raw.shape}\n  Before sharding: allocated={memory_allocated} GB, reserved={memory_reserved} GB")

                gc.collect()
                torch.cuda.empty_cache()
```

