import cs336_basics
import argparse
from cs336_basics.transformer import TransformerLM
from cs336_basics.transformer import scaled_dot_product_attention
import torch
import timeit   
import logging
import pandas as pd
import datetime
from contextlib import nullcontext
import torch
import os
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class Attention(nn.Module):
    def forward(self, Q, K, V):
        return scaled_dot_product_attention(Q, K, V)

def benchmark_attention(seq_len, d_model, warmup_steps, compile: bool = False):
    batch_size = 8
    device = torch.device('cuda')

    # generate random input tensors Q, K, V
    rand_Q = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    rand_K = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    rand_V = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)

    forward_time = []
    backward_time = []

    forward_memory = []
    backward_memory = []

    attention_module = Attention()
    if compile:
        attention_module = torch.compile(attention_module)

    for i in range(warmup_steps + 100):
        torch.cuda.synchronize()
        start_time = timeit.default_timer()

        out = attention_module(rand_Q, rand_K, rand_V)

        torch.cuda.synchronize()
        mid_time = timeit.default_timer()
        mid_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        torch.sum(out, [0,1,2]).backward()

        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        end_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        if i > warmup_steps:
            backward_time.append(end_time - mid_time)
            forward_time.append(mid_time - start_time)
            forward_memory.append(mid_memory)
            backward_memory.append(end_memory)

        torch.cuda.reset_peak_memory_stats(device)

    return np.mean(forward_time) * 1000, np.mean(backward_time) * 1000, np.mean(forward_memory), np.mean(backward_memory)

def run_attention_benchmark():
    seq_lens = [256, 1024, 4096, 8192, 16384]
    d_models = [16, 32, 64, 128]
    warmup_steps = 3

    rows = []

    for compile in [False]:
        for d_model in d_models:
            for seq_len in seq_lens:
                try:
                    logger.info(f"Running benchmark for seq len {seq_len} and d model {d_model}")
                    forward_time, backward_time, forward_memory, backward_memory = benchmark_attention(seq_len, d_model, warmup_steps)
                    rows.append({
                        'seq_len': seq_len,
                        'd_model': d_model,
                        'forward_time': forward_time,
                        'backward_time': backward_time,
                        'forward_memory': forward_memory,
                        'backward_memory': backward_memory,
                        'compile': compile
                    })
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.warning(f"OOM for seq_len={seq_len}, d_model={d_model}")
                        torch.cuda.empty_cache()
                        rows.append({
                            'seq_len': seq_len,
                            'd_model': d_model,
                            'forward_time': None,
                            'backward_time': None,
                            'forward_memory': None,
                            'backward_memory': None,
                            'compile': compile
                        })
                    else:
                        raise
    
    df = pd.DataFrame(rows)
    print(df.to_latex(index=False, float_format="%.2f"))

if __name__ == "__main__":
    run_attention_benchmark()





