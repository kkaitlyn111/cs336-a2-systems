import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pandas as pd
import numpy as np

def setup(rank, world_size, device):
    if device.type == 'cuda':
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size
        )
    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group(
            backend="gloo",
            rank=rank,
            world_size=world_size
        )

def benchmark_allreduce(rank, world_size, warmup, benchmark, device, data_size):
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
        times.append(end - start)

    # Gather all times to rank 0
    times_tensor = torch.tensor(times, dtype=torch.float64, device=device_obj)
    if device_obj.type == 'cuda':
        all_times = [torch.zeros_like(times_tensor) for _ in range(world_size)]
        dist.all_gather(all_times, times_tensor)
        all_times = [t.cpu().numpy() for t in all_times]
    else:
        all_times = [torch.zeros_like(times_tensor) for _ in range(world_size)]
        dist.gather(times_tensor, gather_list=all_times if rank == 0 else None, dst=0)
        if rank == 0:
            all_times = [t.cpu().numpy() for t in all_times]
        else:
            all_times = None

    if rank == 0:
        # all_times: list of arrays, one per rank
        all_times_concat = np.concatenate(all_times)
        mean_time = all_times_concat.mean()
        std_time = all_times_concat.std()
        record = {
            'backend': 'nccl' if device_obj.type == 'cuda' else 'gloo',
            'world_size': world_size,
            'data_mb': data_size,
            'mean_time_s': mean_time,
            'std_time_s': std_time
        }
        df = pd.DataFrame([record])
        df.to_csv('allreduce_results.csv', mode='a', header=not os.path.exists('allreduce_results.csv'), index=False)

    dist.destroy_process_group()
    
def main():
    world_sizes = [2, 4, 6]
    warmup = 5
    benchmark = 20
    data_sizes = [1, 10, 100, 1000]
    device = 'cpu'

    for world_size in world_sizes:
        for data_size in data_sizes:
            mp.spawn(benchmark_allreduce, args=(world_size, warmup, benchmark, device, data_size), nprocs=world_size, join=True)

    # Print LaTeX table at the end (only once, after all runs)
    if os.path.exists('allreduce_results.csv'):
        df = pd.read_csv('allreduce_results.csv')
        summary = df.groupby(['backend', 'device', 'world_size', 'data_mb'])['time_s'].agg(['mean', 'std']).reset_index()
        print(summary.to_latex(index=False))

if __name__ == "__main__":
    main()
        

