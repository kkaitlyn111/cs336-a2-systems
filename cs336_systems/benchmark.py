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
cs336_basics.transformer = transformer_annotated

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
                model_size: str = None):
    
    # initialize and compile model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = TransformerLM(d_model = d_model, d_ff = d_ff, vocab_size = vocab_size, 
                          context_length = context_length, num_heads = num_heads, num_layers = num_layers, 
                          theta = theta, device = device, dtype = dtype)
    model = torch.compile(model)
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

    # time forward and backward passes

    forward_pass = []
    backward_pass = []
    full_step_times = []  
    optimizer_times = []

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

    # start profiling memory after warmup
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
        

        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)


    # Calculate statistics
    forward_mean = sum(forward_pass) / len(forward_pass)
    forward_std = (sum((x - forward_mean) ** 2 for x in forward_pass) / len(forward_pass)) ** 0.5
    backward_mean = sum(backward_pass) / len(backward_pass)
    backward_std = (sum((x - backward_mean) ** 2 for x in backward_pass) / len(backward_pass)) ** 0.5
    full_step_mean = sum(full_step_times) / len(full_step_times)
    full_step_std = (sum((x - full_step_mean) ** 2 for x in full_step_times) / len(full_step_times)) ** 0.5

    # Print model settings
    print("\nModel Settings:")
    print(f"Device: {device}")
    print(f"Model Architecture:")
    print(f"  model size: {model_size}")
    print(f"  d_model: {d_model}")
    print(f"  num_heads: {num_heads}")
    print(f"  d_ff: {d_ff}")
    print(f"  context_length: {context_length}")
    print(f"  num_layers: {num_layers}")
    print(f"  vocab_size: {vocab_size}")
    print(f"  batch_size: {batch_size}")
    print(f"  dtype: {dtype}")
    print(f"  theta: {theta}")
    
    print("\nBenchmark Results:")
    print(f"Forward Pass:")
    print(f"  Mean: {forward_mean*1000:.2f} ms")
    print(f"  Std:  {forward_std*1000:.2f} ms")
    print(f"Backward Pass:")
    print(f"  Mean: {backward_mean*1000:.2f} ms")
    print(f"  Std:  {backward_std*1000:.2f} ms")
    print(f"Total Time per Step:")
    print(f"  Mean: {(forward_mean + backward_mean)*1000:.2f} ms")
    print(f"  Std:  {(forward_std + backward_std)*1000:.2f} ms")
    print(f"Full Training Step:")
    print(f"  Mean: {full_step_mean*1000:.2f} ms")
    print(f"  Std:  {full_step_std*1000:.2f} ms")

    # Prepare data for LaTeX table
    results_data = {
        "Model Size": [model_size],
        "context_length": [context_length],
        "Forward Mean (ms)": [forward_mean * 1000],
        "Forward Std (ms)": [forward_std * 1000],
        "Backward Mean (ms)": [backward_mean * 1000],
        "Backward Std (ms)": [backward_std * 1000],
        "Total Mean (ms)": [(forward_mean + backward_mean) * 1000],
        "Total Std (ms)": [(forward_std + backward_std) * 1000],
        "Full Step Mean (ms)": [full_step_mean * 1000],
        "Full Step Std (ms)": [full_step_std * 1000],
    }
    df = pd.DataFrame(results_data)
    print("\nLaTeX Table:")
    print(df.to_latex(index=False, float_format="%.2f"))

    logger.info("Benchmark complete.")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    batch_size = 4
    vocab_size = 10000
    context_length = 256
    d_model = 512
    d_ff = 1344
    n_layers = 4
    n_heads = 16
    theta = 10000

    model_size = 'medium'

    if model_size == 'small':
        d_model = 768
        d_ff = 3072
        num_layers = 12
        num_heads = 12
    elif model_size == 'medium':
        d_model = 1024
        d_ff = 4096
        num_layers = 24
        num_heads = 16
    elif model_size == 'large':
        d_model = 1280
        d_ff = 5120
        num_layers = 36
        num_heads = 20
    elif model_size == 'xl':
        d_model = 1600
        d_ff = 6400
        num_layers = 48
        num_heads = 25
    elif model_size == '2.7b':
        d_model = 2560
        d_ff = 10240
        num_layers = 32
        num_heads = 32

    warmup_steps = 1
    num_steps = 10
    dtype = torch.float32

    torch.set_float32_matmul_precision('high')

    # Generate a unique nsys output filename using timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    nsys_output = f"result_{timestamp}"
    print(f"[INFO] Recommended nsys command:")
    print(f"nsys profile -o {nsys_output} python3 -m cs336_systems.benchmark")

    run_simple_benchmark(
        d_model=d_model,
        num_heads=n_heads,
        d_ff=d_ff,
        context_length=context_length,
        num_layers=n_layers,
        num_steps=num_steps,
        warmup_steps=warmup_steps,
        batch_size=batch_size,
        vocab_size=vocab_size,
        theta=theta,
        dtype=dtype,
        model_size=model_size
    )
