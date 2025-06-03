import cs336_basics
import argparse
from cs336_basics.transformer import TransformerLM
from cs336_basics.optimizers import AdamW
from cs336_basics.training import cross_entropy_loss
from cs336_systems import transformer_annotated
import torch
import timeit   
import logging
import pandas as pd
import datetime
from contextlib import nullcontext
cs336_basics.transformer = transformer_annotated
import torch
import os

logger = logging.Logger("benchmark")

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def run_simple_benchmark(d_model: int, num_heads: int, d_ff: int, context_length: int, num_layers: int, 
                num_steps, warmup_steps,
                batch_size: int = 4,
                vocab_size: int = 10000, 
                theta: float = None,
                dtype: torch.dtype | None = None,
                model_size: str = None, mixed_precision: bool = False, compile: bool = False):
    
    # initialize and do not compile model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = TransformerLM(d_model = d_model, d_ff = d_ff, vocab_size = vocab_size, 
                          context_length = context_length, num_heads = num_heads, num_layers = num_layers, 
                          theta = theta, device = device, dtype = dtype)
    
    if compile:
        model = torch.compile(model)
    model.to(device)
   
    optimizer = AdamW(
        model.parameters(),
        lr=1e-4, 
        weight_decay=0.01,
        eps=1e-8,
        betas=(0.9, 0.95)
    )
    loss_fn = cross_entropy_loss  

    # generate random batch of data
    batch = torch.randint(low=0, high=vocab_size, size=(batch_size, context_length))
    batch = batch.to(device)

 
    inputs = batch
    targets = batch

    # time forward and backward passes

    forward_pass = []
    backward_pass = []
    full_step_times = []  
    optimizer_times = []

    casting = torch.autocast(device_type='cuda',dtype=dtype) if mixed_precision else nullcontext()

    with casting:
        logger.info(f"Starting benchmarking")
        # do not time for warmup
        torch.cuda.synchronize()
        for i in range(warmup_steps):
            optimizer.zero_grad()
            logits = model(inputs)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()


        torch.cuda.memory._record_memory_history(max_entries=1000000)

        # time here, run on same batch of data each time without change
        for i in range(num_steps):
            logger.info(f"Starting timing steps")
            torch.cuda.synchronize()
            f_start = timeit.default_timer()

            output = model(batch)
            torch.cuda.synchronize()
            f_end = timeit.default_timer()

            output.mean().backward()
            torch.cuda.synchronize()

            b_end = timeit.default_timer()
            backward_pass.append(b_end - f_end)
            forward_pass.append(f_end - f_start)
            torch.cuda.synchronize()

            # Full training step timing with optimizer
            torch.cuda.synchronize()
            full_start = timeit.default_timer()
            optimizer.zero_grad()
            logits = model(inputs)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            full_end = timeit.default_timer()
            optimizer_times.append(full_end - full_start - backward_pass[-1])
            full_step_times.append(full_end - full_start)

    torch.cuda.memory._dump_snapshot(f"memory_ctx{context_length}.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)
    # calculate statistics
    forward_mean = sum(forward_pass) / len(forward_pass)
    forward_std = (sum((x - forward_mean) ** 2 for x in forward_pass) / len(forward_pass)) ** 0.5
    backward_mean = sum(backward_pass) / len(backward_pass)
    backward_std = (sum((x - backward_mean) ** 2 for x in backward_pass) / len(backward_pass)) ** 0.5
    full_step_mean = sum(full_step_times) / len(full_step_times)
    full_step_std = (sum((x - full_step_mean) ** 2 for x in full_step_times) / len(full_step_times)) ** 0.5

    # Print model settings
    print("\nModel Settings:")
    # # print(f"Device: {device}")
    # print(f"Model Architecture:")
    print(f"  model size: {model_size}")
    # print(f"  d_model: {d_model}")
    # print(f"  num_heads: {num_heads}")
    # print(f"  d_ff: {d_ff}")
    print(f"  context_length: {context_length}")
    # print(f"  num_layers: {num_layers}")
    # print(f"  vocab_size: {vocab_size}")
    # print(f"  batch_size: {batch_size}")
    print(f"  dtype: {dtype}")
    # print(f"  theta: {theta}")
    
    # print("\nBenchmark Results:")
    # print(f"Forward Pass:")
    # print(f"  Mean: {forward_mean*1000:.2f} ms")
    # print(f"  Std:  {forward_std*1000:.2f} ms")
    # print(f"Backward Pass:")
    # print(f"  Mean: {backward_mean*1000:.2f} ms")
    # print(f"  Std:  {backward_std*1000:.2f} ms")
    # print(f"Total Time per Step:")
    # print(f"  Mean: {(forward_mean + backward_mean)*1000:.2f} ms")
    # print(f"  Std:  {(forward_std + backward_std)*1000:.2f} ms")
    # print(f"Full Training Step:")
    # print(f"  Mean: {full_step_mean*1000:.2f} ms")
    # print(f"  Std:  {full_step_std*1000:.2f} ms")

    # # Prepare data for LaTeX table
    results_data = {
        "model size": [model_size],
        "context length": [context_length],
        "forward mean (ms)": [forward_mean * 1000],
        "forward std (ms)": [forward_std * 1000],
        "full step mean (ms)": [full_step_mean * 1000],
        "full step std (ms)": [full_step_std * 1000],
        "compiled?": [compile],
    }
    df = pd.DataFrame(results_data)
    print("\nLaTeX Table:")
    print(df.to_latex(index=False, float_format="%.2f"))

    logger.info("Benchmark complete.")
    return results_data

def run_memory_foward_fullstep(d_model: int, num_heads: int, d_ff: int, context_length: int, num_layers: int, 
                batch_size: int = 4,
                vocab_size: int = 10000, 
                theta: float = None,
                dtype: torch.dtype | None = None,
                model_size: str = None, mixed_precision: bool = False, forward_only: bool = False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = TransformerLM(d_model = d_model, d_ff = d_ff, vocab_size = vocab_size, 
                          context_length = context_length, num_heads = num_heads, num_layers = num_layers, 
                          theta = theta, device = device, dtype = dtype)
    #model = torch.compile(model)
    model.to(device)

    # Add optimizer and loss function (custom, matching your usual setup)
    optimizer = AdamW(
        model.parameters(),
        lr=1e-4,  # You can adjust this or make it an argument
        weight_decay=0.01,
        eps=1e-8,
        betas=(0.9, 0.95)
    )
    loss_fn = cross_entropy_loss  # Use as a function, not instantiated

    # generate random batch of data
    batch = torch.randint(low=0, high=vocab_size, size=(batch_size, context_length))
    batch = batch.to(device)

    # Prepare inputs and targets for language modeling
    inputs = batch
    targets = batch

    casting = torch.autocast(device_type='cuda',dtype=dtype) if mixed_precision else nullcontext()


    with casting:
        logger.info(f"Starting benchmarking memory for {context_length} context length")
        torch.cuda.memory._record_memory_history(max_entries=1000000)


        if forward_only:
            logger.info(f"Starting timing forward pass only")
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logits = model(inputs)
            torch.cuda.synchronize()
        else:
        # Full training step timing with optimizer
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            optimizer.zero_grad()
            logits = model(inputs)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
    
    peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # Convert bytes to MB

    memory_dir = "memory_files"
    filename = f"memory_ctx{context_length}_{'forward' if forward_only else 'fullstep'}_{'mixed' if mixed_precision else ''}.pickle"
    full_path = os.path.join(memory_dir, filename)

    torch.cuda.memory._dump_snapshot(full_path)
    logger.info(f"File saved to {full_path}")
    torch.cuda.memory._record_memory_history(enabled=None)

    return peak_memory

def get_model_config(model_size):
    if model_size == 'small':
        return dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12)
    elif model_size == 'medium':
        return dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16)
    elif model_size == 'large':
        return dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20)
    elif model_size == 'xl':
        return dict(d_model=1600, d_ff=6400, num_layers=48, num_heads=25)
    elif model_size == '2.7b':
        return dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32)
    else:
        raise ValueError(f"Unknown model size: {model_size}")


def run_all_benchmarks():
    batch_size = 4
    vocab_size = 10000
    context_length = 256
    theta = 10000
    warmup_steps = 2
    num_steps = 10
    model_sizes = ['small', 'medium', 'large', 'xl', '2.7b']
    mixed_precision = False
    results = []
    
    if mixed_precision:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
        torch.set_float32_matmul_precision('high')

    for model_size in model_sizes:
        config = get_model_config(model_size)
        for compile in [True, False]:
            print(f"\n[INFO] Running benchmark for {model_size} compile={compile})")
            results_data = run_simple_benchmark(
                d_model=config['d_model'],
                num_heads=config['num_heads'],
                d_ff=config['d_ff'],
                context_length=context_length,
                num_layers=config['num_layers'],
                num_steps=num_steps,
                warmup_steps=warmup_steps,
                batch_size=batch_size,
                vocab_size=vocab_size,
                theta=theta,
                dtype=dtype,
                model_size=model_size,
                mixed_precision=mixed_precision
            )
            # Flatten results_data dict to a single row
            row = {k: v[0] for k, v in results_data.items()}
            results.append(row)
    # Aggregate all results into a DataFrame and print a single LaTeX table
    df = pd.DataFrame(results)
    print("\nAggregated LaTeX Table:")
    print(df.to_latex(index=False, float_format="%.2f"))

def run_benchmark_suite():
    # Generate a unique nsys output filename using timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    nsys_output = f"result_{timestamp}"
    print(f"[INFO] Recommended nsys command:")
    print(f"nsys profile -o {nsys_output} python3 -m cs336_systems.benchmark")
    run_all_benchmarks()

def profile_memory_forward_fullstep():
    batch_size = 4
    vocab_size = 10000
    theta = 10000
    device = torch.device('cuda')

    context_lengths = [128, 256, 512]
    model_sizes = ['2.7b']

    mixed_precision = True
    if mixed_precision:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
        torch.set_float32_matmul_precision('high')

    rows = []
    for forward_only in [True, False]:
        for model_size in model_sizes:
            for context_length in context_lengths:
                print(f"\n[INFO] Starting memory profiling for context length {context_length} with mixed precision {mixed_precision}...")
                config = get_model_config(model_size)
                peak_memory = run_memory_foward_fullstep(
                    d_model=config['d_model'],
                    num_heads=config['num_heads'],
                    d_ff=config['d_ff'],
                    context_length=context_length,
                    num_layers=config['num_layers'],
                    batch_size=batch_size,
                    vocab_size=vocab_size,
                    theta=theta,
                    dtype=dtype,
                    model_size=model_size,
                    mixed_precision=mixed_precision,
                    forward_only=forward_only
                )
                rows.append({'context length': context_length, 'forward only': forward_only, 'peak memory (MB)': peak_memory})

    df = pd.DataFrame(rows)
    print(df.to_latex(index=False, float_format="%.2f"))

def profile_2_7b_memory_by_context_length():
    batch_size = 4
    vocab_size = 10000
    theta = 10000
    model_size = '2.7b'
    device = torch.device('cuda')

    warmup_steps = 0
    num_steps = 1

    mixed_precision = False
    if mixed_precision:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
        torch.set_float32_matmul_precision('high')

    results = []
    for context_length in [128, 256, 512, 1024]:
        print(f"\n[INFO] Starting memory profiling for context length {context_length}...")

        config = get_model_config(model_size)
        run_simple_benchmark(
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            d_ff=config['d_ff'],
            context_length=context_length,
            num_layers=config['num_layers'],
            num_steps=num_steps,
            warmup_steps=warmup_steps,
            batch_size=batch_size,
            vocab_size=vocab_size,
            theta=theta,
            dtype=dtype,
            model_size=model_size,
            mixed_precision=mixed_precision
        )
  
    #     optimizer = AdamW(
    #         model.parameters(),
    #         lr=1e-4, weight_decay=0.01, eps=1e-8, betas=(0.9, 0.95)
    #     )
    #     loss_fn = cross_entropy_loss
    #     batch = torch.randint(low=0, high=vocab_size, size=(batch_size, context_length), device=device)
    #     inputs = batch
    #     targets = batch
    #     # Start memory history recording for all phases
    #     torch.cuda.memory._record_memory_history(max_entries=1000000)
    #     # Forward pass memory
    #     print(f"[INFO]   Forward pass...")
    #     torch.cuda.empty_cache()
    #     torch.cuda.reset_peak_memory_stats(device)
    #     with torch.autocast(device_type='cuda', dtype=dtype):
    #         logits = model(inputs)
    #     torch.cuda.synchronize()
    #     forward_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    #     # Backward pass memory
    #     print(f"[INFO]   Backward pass...")
    #     torch.cuda.empty_cache()
    #     torch.cuda.reset_peak_memory_stats(device)
    #     with torch.autocast(device_type='cuda', dtype=dtype):
    #         logits = model(inputs)
    #         loss = loss_fn(logits, targets)
    #         loss.backward()
    #     torch.cuda.synchronize()
    #     backward_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    #     # Optimizer step memory
    #     print(f"[INFO]   Optimizer step...")
    #     torch.cuda.empty_cache()
    #     torch.cuda.reset_peak_memory_stats(device)
    #     with torch.autocast(device_type='cuda', dtype=dtype):
    #         optimizer.zero_grad()
    #         logits = model(inputs)
    #         loss = loss_fn(logits, targets)
    #         loss.backward()
    #         optimizer.step()
    #     torch.cuda.synchronize()
    #     optimizer_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    #     # Dump memory history for this context length
    #     torch.cuda.memory._dump_snapshot(f"memory_ctx{context_length}.pickle")
    #     torch.cuda.memory._record_memory_history(enabled=None)
    #     print(f"[INFO] Finished memory profiling for context length {context_length}.")
    #     results.append({
    #         'Context Length': context_length,
    #         'Forward Peak Memory (MB)': forward_mem,
    #         'Backward Peak Memory (MB)': backward_mem,
    #         'Optimizer Step Peak Memory (MB)': optimizer_mem
    #     })
    # df = pd.DataFrame(results)
    # print("\n2.7B Model Memory Profiling by Context Length:")
    # print(df.to_string(index=False, float_format="%.2f"))
    # print("\nLaTeX Table:")
    # print(df.to_latex(index=False, float_format="%.2f"))

if __name__ == '__main__':
    # run_benchmark_suite()
    # profile_2_7b_memory_by_context_length()
    # profile_2_7b_memory_forward_and_fullstep()
    # profile_memory_forward_fullstep()
    run_all_benchmarks()
