import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Queue
import pandas as pd
import numpy as np 

from cs336_basics.transformer import TransformerLM
from cs336_basics.optimizers import AdamW
from cs336_basics.training import cross_entropy_loss, gradient_clipping



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

def generate_random_data(batch_size, context_length, vocab_size):
    # Generate data in [batch_size, context_length] shape
    return torch.randint(0, vocab_size, (context_length, batch_size))
    

def train(rank, world_size, 
          input_data, target_data,
          d_model, num_heads, d_ff, vocab_size, context_length, num_layers, max_seq_len, rope_theta, device,
          lr, weight_decay, epsilon, beta1, beta2,
          batch_size,
          warmup, steps):
    
    try:
        setup(rank, world_size, device)

        model = TransformerLM(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            vocab_size=vocab_size,
            context_length=context_length,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            theta=rope_theta,
            device=device
        )
        if device.type == 'cuda':
            model.cuda(rank)
        else:
            model.to(device)

        if rank == 0:
            for param in model.parameters():
                param.data = torch.randn_like(param.data)
                dist.broadcast(param.data, src=0)
            print("broadcasted parameters from rank 0!")

        my_batch_size = batch_size // world_size
        my_start = rank * my_batch_size
        my_end = my_start + my_batch_size

        # move data onto correct worker
        # shape of input data is [context_length batch_size]
        if device.type == 'cuda':
            my_data_input = input_data[:,my_start:my_end].cuda(rank)
            my_data_target = target_data[:,my_start:my_end].cuda(rank)
        else:
            my_data_input = input_data[:,my_start:my_end].to(device)
            my_data_target = target_data[:,my_start:my_end].to(device)

        optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            eps=epsilon,
            betas=(beta1, beta2)
        )

        for step in range(warmup + steps):
            
            logits = model(my_data_input)
            loss = cross_entropy_loss(logits, my_data_target)

            optimizer.zero_grad()
            loss.backward()
            gradient_clip_M = 1.0
            gradient_clipping(model.parameters(), gradient_clip_M)
            
            # Average gradients across all processes
            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVERAGE)
            dist.barrier()
        
            optimizer.step()
            
            my_params = model.state_dict()
            
            if step % 100 == 0 and rank == 0: 
                print(f"rank {rank}: step {step}, loss {loss.item()}, params {my_params['layers.1.ln1.weight']}")
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

def main():
    world_size = 2

    batch_size = 20
    context_length = 128

    vocab_size = 10000
    d_model = 16
    num_layers = 2
    d_ff = 32
    num_heads = 4
    rope_theta = 10000
    max_seq_len = 128
    lr = 0.0001
    weight_decay = 0.01
    epsilon = 1e-8
    beta1 = 0.9
    beta2 = 0.95

    warmup = 0
    steps = 1

    input_data = generate_random_data(batch_size, context_length, vocab_size)
    target_data = generate_random_data(batch_size, context_length, vocab_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mp.spawn(train, args=(world_size,
          input_data, target_data,
          d_model, num_heads, d_ff, vocab_size, context_length, num_layers, max_seq_len, rope_theta, device,
          lr, weight_decay, epsilon, beta1, beta2, batch_size, warmup, steps), nprocs=world_size, join=True)






    



    


