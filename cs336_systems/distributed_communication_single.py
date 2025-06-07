import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Queue
import pandas as pd
import numpy as np

def setup(rank, world_size, device):
    if device.type == 'cuda':
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29502'
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size
        )
    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29502'
        dist.init_process_group(
            backend="gloo",
            rank=rank,
            world_size=world_size
        )

def benchmark_allreduce(rank, world_size, warmup, benchmark, device, data_size, result_queue):
    setup(rank, world_size, torch.device(device, rank) if device == 'cuda' else torch.device(device))
    device_obj = torch.device(device, rank) if device == 'cuda' else torch.device(device)
    numel = (data_size * 1024 * 1024) // 4
    data = torch.rand(numel, dtype=torch.float32, device=device_obj)

    for _ in range(warmup):
        dist.all_reduce(data, async_op=False)
        if device_obj.type == 'cuda':
            torch.cuda.synchronize(device_obj)

    times = []
    for _ in range(benchmark):
        data.uniform_()  # randomize data
        if device_obj.type == 'cuda':
            torch.cuda.synchronize(device_obj)
        start = time.time()
        dist.all_reduce(data, async_op=False)
        if device_obj.type == 'cuda':
            torch.cuda.synchronize(device_obj)
        end = time.time()
        times.append((end - start) * 1000)  # Convert to milliseconds

    # Calculate statistics for this process
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    # Create record for this process
    record = {
        'backend': 'nccl' if device_obj.type == 'cuda' else 'gloo',
        'device_type': device_obj.type,
        'world_size': world_size,
        'data_mb': data_size,
        'time_ms': mean_time,
        'time_std_ms': std_time
    }
    
    # Convert record to tensor for gathering
    if rank == 0:
        # Create a list to store all records
        all_records = [None] * world_size
        all_records[0] = record
    else:
        all_records = None
    
    # Convert record values to tensor
    record_tensor = torch.tensor([
        float(mean_time),
        float(std_time)
    ], dtype=torch.float64, device=device_obj)
    
    # Gather all records to rank 0
    if device_obj.type == 'cuda':
        gathered_tensors = [torch.zeros_like(record_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_tensors, record_tensor)
        if rank == 0:
            for i, tensor in enumerate(gathered_tensors[1:], 1):  # Skip rank 0 as we already have it
                all_records[i] = {
                    'backend': 'nccl' if device_obj.type == 'cuda' else 'gloo',
                    'device_type': device_obj.type,
                    'world_size': world_size,
                    'data_mb': data_size,
                    'time_ms': tensor[0].item(),
                    'time_std_ms': tensor[1].item()
                }
    else:
        gathered_tensors = [torch.zeros_like(record_tensor) for _ in range(world_size)]
        dist.gather(record_tensor, gather_list=gathered_tensors if rank == 0 else None, dst=0)
        if rank == 0:
            for i, tensor in enumerate(gathered_tensors[1:], 1):  # Skip rank 0 as we already have it
                all_records[i] = {
                    'backend': 'nccl' if device_obj.type == 'cuda' else 'gloo',
                    'device_type': device_obj.type,
                    'world_size': world_size,
                    'data_mb': data_size,
                    'time_ms': tensor[0].item(),
                    'time_std_ms': tensor[1].item()
                }

    dist.destroy_process_group()
    if rank == 0:
        result_queue.put(all_records)

def main():
    world_sizes = [2]
    warmup = 5
    benchmark = 3
    data_sizes = [1, 10, 100]
    device = 'cpu'
    all_results = []

    for world_size in world_sizes:
        for data_size in data_sizes:
            result_queue = Queue()
            mp.spawn(
                benchmark_allreduce,
                args=(world_size, warmup, benchmark, device, data_size, result_queue),
                nprocs=world_size,
                join=True
            )
            # Get results from the queue
            results = result_queue.get()
            if results is not None:
                all_results.extend(results)

    # Create DataFrame from results and print summary
    if all_results:
        df = pd.DataFrame(all_results)
        # Group by relevant columns and calculate statistics
        summary = df.groupby(['backend', 'device_type', 'world_size', 'data_mb']).agg({
            'time_ms': ['mean', 'std']
        }).reset_index()
        
        # Flatten column names
        summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
        
        # Format the output for better readability
        print("\nBenchmark Results Summary (times in milliseconds):")
        print(summary.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
        print("\nLaTeX Table:")
        print(summary.to_latex(index=False, float_format=lambda x: f'{x:.2f}'))

if __name__ == "__main__":
    main()
        

