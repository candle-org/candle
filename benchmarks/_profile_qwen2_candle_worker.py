"""Profile Candle tiny Qwen2 train-step hot rows (post-rsqrt-node build).

Run inside torchnpu311 (has transformers) with PYTHONPATH=<worktree>/src.
"""
import cProfile
import io
import os
import pstats
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from compat.transformers.conftest import apply_all_patches

apply_all_patches()
import candle as torch

sys.modules["torch"] = torch

from transformers import Qwen2Config, Qwen2ForCausalLM

CONFIG = dict(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=64,
    attention_dropout=0.0,
    use_cache=False,
)


def main():
    device = torch.Device("npu:0")
    dtype = torch.float16
    config = Qwen2Config(**CONFIG)
    torch.manual_seed(20260608)
    model = Qwen2ForCausalLM(config).to(device).to(dtype)
    model.train()

    input_ids = torch.arange(0, 8, device=device, dtype=torch.int64).reshape(1, 8) % 128
    labels = (input_ids + 1) % 128
    attention_mask = torch.ones((1, 8), device=device, dtype=torch.int64)

    def step():
        for param in model.parameters():
            if getattr(param, "grad", None) is not None:
                param.grad = None
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
        out.loss.backward()

    for _ in range(3):
        step()
    torch.npu.synchronize()

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(3):
        step()
    torch.npu.synchronize()
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative").print_stats(60)
    print(stream.getvalue())

    stream2 = io.StringIO()
    stats2 = pstats.Stats(profiler, stream=stream2)
    stats2.sort_stats("tottime").print_stats(40)
    print(stream2.getvalue())


if __name__ == "__main__":
    main()
