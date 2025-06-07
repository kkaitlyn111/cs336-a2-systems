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

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def generate_random_data(batch_size, context_length, vocab_size, seed=None):
    if seed is not None:
        set_seed(seed)
    data = torch.randint(0, vocab_size, (context_length, batch_size))
    # Save the data for reproducibility
    torch.save(data, 'random_data.pt')
    return data

def save_model_params(model, path='initial_params.pt'):
    """Save initial model parameters"""
    torch.save(model.state_dict(), path)

def load_model_params(model, path='initial_params.pt'):
    """Load initial model parameters"""
    model.load_state_dict(torch.load(path))

def setup(rank, world_size, device):
    if device.type == 'cuda':
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29503'
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

def train_regular(input_data, target_data, d_model, num_heads, d_ff, vocab_size, context_length, num_layers, max_seq_len, rope_theta, device, lr, weight_decay, epsilon, beta1, beta2, batch_size, warmup, steps, seed=None):
    """Run regular (non-DDP) training with the same data and parameters"""
    if seed is not None:
        set_seed(seed)
    
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
    model.to(device)

    # Load saved parameters if they exist, otherwise initialize and save
    #if os.path.exists('initial_params.pt'):
    if 0:
        load_model_params(model)
    else:
        for param in model.parameters():
            param.data = torch.randn_like(param.data)
        save_model_params(model)

    # Transpose data to match expected shape [batch_size, context_length]
    input_data = input_data.t()  # Now shape is [batch_size, context_length]
    target_data = target_data.t()  # Now shape is [batch_size, context_length]

    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        eps=epsilon,
        betas=(beta1, beta2)
    )

    print("Starting regular training")
    losses = []
    times = []
    start_time = time.time()
    
    for step in range(warmup + steps):
        step_start_time = time.time()
        
        # Forward pass
        forward_start = time.time()
        logits = model(input_data)
        forward_time = time.time() - forward_start
        
        loss = cross_entropy_loss(logits, target_data)
        losses.append(loss.item())

        # Backward pass
        backward_start = time.time()
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), 1.0)
        backward_time = time.time() - backward_start
        
        # Optimizer step
        optimizer.step()
        
        # Record timing for this step
        full_step_time = time.time() - step_start_time
        times.append({
            'forward_time': forward_time,
            'backward_time': backward_time,
            'send_grad_time': 0.0,  # No gradient communication in regular training
            'full_step_time': full_step_time
        })
        
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # Calculate mean times across all steps
    mean_times = {
        'forward_time': np.mean([t['forward_time'] for t in times]) * 1000,  # convert to ms
        'backward_time': np.mean([t['backward_time'] for t in times]) * 1000,
        'send_grad_time': 0.0,  # No gradient communication in regular training
        'full_step_time': np.mean([t['full_step_time'] for t in times]) * 1000
    }
    
    # Create DataFrame with timing results
    timing_df = pd.DataFrame({
        'Metric': ['Forward', 'Backward', 'Gradient Communication', 'Total Step'],
        'Time (ms)': [
            mean_times['forward_time'],
            mean_times['backward_time'],
            mean_times['send_grad_time'],
            mean_times['full_step_time']
        ],
        'Processes': [1] * 4,  # Regular training uses 1 process
        'Batch Size': [batch_size] * 4
    })
    
    # Save timing results to CSV
    timing_df.to_csv('regular_timing_results.csv', index=False)
    
    # Print timing table
    print("\nRegular Training Timing Results:")
    print(timing_df.to_string(index=False))
    
    # Save final parameters for comparison
    torch.save(model.state_dict(), 'regular_final_params.pt')
    
    return {
        'model': model,
        'losses': losses,
        'final_loss': losses[-1],
        'training_time': training_time,
        'params': {name: param.data.clone() for name, param in model.named_parameters()},
        'timing_stats': mean_times
    }

def train(rank, world_size, 
          input_data, target_data,
          d_model, num_heads, d_ff, vocab_size, context_length, num_layers, max_seq_len, rope_theta, device,
          lr, weight_decay, epsilon, beta1, beta2,
          batch_size,
          warmup, steps):
    
    try:
        setup(rank, world_size, device)
        print(f"rank {rank}: setup complete")

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
        print(f"rank {rank}: model created")
        if device.type == 'cuda':
            model.cuda(rank)
        else:
            model.to(device)

        if rank == 0:
            print(f"hi im rank {rank}")
            # Initialize parameters with random values on rank 0
            for param in model.parameters():
                param.data = torch.randn_like(param.data)
        
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
        
        if rank == 0:
            print("broadcasted parameters from rank 0!")
        dist.barrier()

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

        # Transpose data to match expected shape [batch_size, context_length]
        my_data_input = my_data_input.t()  # Now shape is [my_batch_size, context_length]
        my_data_target = my_data_target.t()  # Now shape is [my_batch_size, context_length]

        optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            eps=epsilon,
            betas=(beta1, beta2)
        )

        print(f"starting training")

        for step in range(warmup + steps):
            print(f"rank {rank}: step {step}")
            logits = model(my_data_input)
            loss = cross_entropy_loss(logits, my_data_target)

            optimizer.zero_grad()
            loss.backward()
            gradient_clip_M = 1.0
            gradient_clipping(model.parameters(), gradient_clip_M)
            
            # Average gradients across all processes
            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                    param.grad.div_(world_size)  # Divide by world_size to get average
            dist.barrier()
        
            optimizer.step()
            
            my_params = model.state_dict()
            
            if step % 100 == 0 and rank == 0: 
                print(f"rank {rank}: step {step}, loss {loss.item()}, params {my_params['layers.1.ln1.weight']}")
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

def train_ddp(rank, world_size, 
          input_data, target_data,
          d_model, num_heads, d_ff, vocab_size, context_length, num_layers, max_seq_len, rope_theta, device,
          lr, weight_decay, epsilon, beta1, beta2,
          batch_size, warmup, steps, seed=None):
    
    try:
        if seed is not None:
            set_seed(seed + rank)  # Different seed for each rank but deterministic
        setup(rank, world_size, device)
        print(f"rank {rank}: setup complete")

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

        # Load saved parameters on rank 0 and broadcast
        if rank == 0:
            if os.path.exists('initial_params.pt'):
                load_model_params(model)
            else:
                # Initialize and save parameters if they don't exist
                for param in model.parameters():
                    param.data = torch.randn_like(param.data)
                save_model_params(model)
        
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
        
        if rank == 0:
            print("broadcasted parameters from rank 0!")

        dist.barrier()

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

        # Transpose data to match expected shape [batch_size, context_length]
        my_data_input = my_data_input.t()  # Now shape is [my_batch_size, context_length]
        my_data_target = my_data_target.t()  # Now shape is [my_batch_size, context_length]

        optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            eps=epsilon,
            betas=(beta1, beta2)
        )

        print(f"Starting DDP training")
        losses = []
        times = []
        
        for step in range(warmup + steps):
            start_time = time.time()
            logits = model(my_data_input)
            forward_time = time.time() - start_time
            loss = cross_entropy_loss(logits, my_data_target)
            
            if rank == 0:
                losses.append(loss.item())

            optimizer.zero_grad()
            start_backward_time = time.time()
            loss.backward()
            gradient_clipping(model.parameters(), 1.0)
            backward_time = time.time() - start_backward_time
            
            start_send_grad_time = time.time()
            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                    param.grad.div_(world_size)
            send_grad_time = time.time() - start_send_grad_time

            optimizer.step()
            full_step_time = time.time() - start_time
            
            if step % 100 == 0 and rank == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}")
            
            times.append({
                'forward_time': forward_time,
                'backward_time': backward_time,
                'send_grad_time': send_grad_time,
                'full_step_time': full_step_time
            })
        
        end_time = time.time()
        training_time = end_time - start_time

        # Calculate mean times across all steps (one value per metric)
        mean_times = {
            'forward_time': np.mean([t['forward_time'] for t in times]) * 1000,  # convert to ms
            'backward_time': np.mean([t['backward_time'] for t in times]) * 1000,
            'send_grad_time': np.mean([t['send_grad_time'] for t in times]) * 1000,
            'full_step_time': np.mean([t['full_step_time'] for t in times]) * 1000
        }
        
        # Convert to tensor for gathering
        times_tensor = torch.tensor([
            mean_times['forward_time'],
            mean_times['backward_time'],
            mean_times['send_grad_time'],
            mean_times['full_step_time']
        ], device=device)
        
        # Create a tensor to gather into on rank 0
        if rank == 0:
            gathered_times = [torch.zeros_like(times_tensor) for _ in range(world_size)]
        else:
            gathered_times = None
            
        # Gather all times onto rank 0
        dist.gather(times_tensor, gathered_times, dst=0)
        
        if rank == 0:
            # Average the times across all processes
            avg_times = torch.stack(gathered_times).mean(dim=0)
            
            # Create DataFrame with timing results
            timing_df = pd.DataFrame({
                'Metric': ['Forward', 'Backward', 'Gradient Communication', 'Total Step'],
                'Time (ms)': avg_times.tolist(),
                'Processes': [world_size] * 4,
                'Batch Size': [batch_size] * 4
            })
            
            # Save timing results to CSV
            timing_df.to_csv('timing_results.csv', index=False)
            
            # Print timing table
            print("\nTiming Results:")
            print(timing_df.to_string(index=False))
            
            # Save final parameters and results
            torch.save(model.state_dict(), 'ddp_final_params.pt')
            torch.save({
                'losses': losses,
                'final_loss': losses[-1],
                'training_time': training_time,
                'timing_stats': {
                    'forward_time_ms': avg_times[0].item(),
                    'backward_time_ms': avg_times[1].item(),
                    'grad_comm_time_ms': avg_times[2].item(),
                    'total_step_time_ms': avg_times[3].item()
                }
            }, 'ddp_results.pt')

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

def train_ddp_flat(rank, world_size, 
          input_data, target_data,
          d_model, num_heads, d_ff, vocab_size, context_length, num_layers, max_seq_len, rope_theta, device,
          lr, weight_decay, epsilon, beta1, beta2,
          batch_size, warmup, steps, seed=None):
    
    try:
        if seed is not None:
            set_seed(seed + rank)  # Different seed for each rank but deterministic
        setup(rank, world_size, device)
        print(f"rank {rank}: setup complete")

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

        # Load saved parameters on rank 0 and broadcast
        if rank == 0:
            if os.path.exists('initial_params.pt'):
                load_model_params(model)
            else:
                # Initialize and save parameters if they don't exist
                for param in model.parameters():
                    param.data = torch.randn_like(param.data)
                save_model_params(model)
        
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
        
        if rank == 0:
            print("broadcasted parameters from rank 0!")

        dist.barrier()

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

        # Transpose data to match expected shape [batch_size, context_length]
        my_data_input = my_data_input.t()  # Now shape is [my_batch_size, context_length]
        my_data_target = my_data_target.t()  # Now shape is [my_batch_size, context_length]

        optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            eps=epsilon,
            betas=(beta1, beta2)
        )

        print(f"Starting DDP training")
        losses = []
        times = []
        
        for step in range(warmup + steps):
            start_time = time.time()
            logits = model(my_data_input)
            forward_time = time.time() - start_time
            loss = cross_entropy_loss(logits, my_data_target)
            
            if rank == 0:
                losses.append(loss.item())

            optimizer.zero_grad()
            start_backward_time = time.time()
            loss.backward()
            gradient_clipping(model.parameters(), 1.0)
            backward_time = time.time() - start_backward_time
            
            # Gradient communication with detailed timing
            start_grad_comm_time = time.time()
            
            # for param in model.parameters():
            #     if param.grad is not None:
            #         dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            #         param.grad.div_(world_size)

            # =======
            start_send_grad_time = time.time()
            param_list = model.state_dict()
        
            flat_grads = torch._utils._flatten_dense_tensors([param.grad for param in model.parameters()])
            dist.all_reduce(tensor=flat_grads, op=dist.ReduceOp.SUM, async_op=False)
            flat_grads.div_(world_size)

            unflat_grads = torch._utils._unflatten_dense_tensors(flat_grads, [tensor for tensor in param_list.values()])

            for param, tensor in zip(model.parameters(), unflat_grads):
                param.grad = tensor

            send_grad_time = time.time() - start_send_grad_time


            optimizer.step()
            full_step_time = time.time() - start_time
            
            if step % 100 == 0 and rank == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}")
            
            times.append({
                'forward_time': forward_time,
                'backward_time': backward_time,
                'send_grad_time': send_grad_time,
                'full_step_time': full_step_time
            })
        
        end_time = time.time()
        training_time = end_time - start_time

        # Calculate mean times across all steps (one value per metric)
        mean_times = {
            'forward_time': np.mean([t['forward_time'] for t in times]) * 1000,  # convert to ms
            'backward_time': np.mean([t['backward_time'] for t in times]) * 1000,
            'send_grad_time': np.mean([t['send_grad_time'] for t in times]) * 1000,
            'full_step_time': np.mean([t['full_step_time'] for t in times]) * 1000
        }
        
        # Convert to tensor for gathering
        times_tensor = torch.tensor([
            mean_times['forward_time'],
            mean_times['backward_time'],
            mean_times['send_grad_time'],
            mean_times['full_step_time']
        ], device=device)
        
        # Create a tensor to gather into on rank 0
        if rank == 0:
            gathered_times = [torch.zeros_like(times_tensor) for _ in range(world_size)]
        else:
            gathered_times = None
            
        # Gather all times onto rank 0
        dist.gather(times_tensor, gathered_times, dst=0)
        
        if rank == 0:
            # Average the times across all processes
            avg_times = torch.stack(gathered_times).mean(dim=0)
            
            # Create DataFrame with timing results
            timing_df = pd.DataFrame({
                'Metric': ['Forward', 'Backward', 'Gradient Communication', 'Total Step'],
                'Time (ms)': avg_times.tolist(),
                'Processes': [world_size] * 4,
                'Batch Size': [batch_size] * 4
            })
            
            # Save timing results to CSV
            timing_df.to_csv('timing_results.csv', index=False)
            
            # Print timing table
            print("\nTiming Results:")
            print(timing_df.to_string(index=False))
            
            # Save final parameters and results
            torch.save(model.state_dict(), 'ddp_final_params.pt')
            torch.save({
                'losses': losses,
                'final_loss': losses[-1],
                'training_time': training_time,
                'timing_stats': {
                    'forward_time_ms': avg_times[0].item(),
                    'backward_time_ms': avg_times[1].item(),
                    'grad_comm_time_ms': avg_times[2].item(),
                    'total_step_time_ms': avg_times[3].item()
                }
            }, 'ddp_results.pt')

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

def compare_results(regular_results, ddp_results):
    """Print comparison of regular and DDP training results"""
    print("\n" + "="*50)
    print("Training Results Comparison")
    print("="*50)
    
    print("\nLoss Comparison:")
    print(f"Regular Training Final Loss: {regular_results['final_loss']:.4f}")
    print(f"DDP Training Final Loss: {ddp_results['final_loss']:.4f}")
    print(f"Loss Difference: {abs(regular_results['final_loss'] - ddp_results['final_loss']):.4f}")
    
    print("\nTraining Time:")
    print(f"Regular Training Time: {regular_results['training_time']:.2f} seconds")
    print(f"DDP Training Time: {ddp_results['training_time']:.2f} seconds")
    print(f"Speedup: {regular_results['training_time'] / ddp_results['training_time']:.2f}x")
    
    print("\nParameter Comparison:")
    max_diff = 0
    for name in regular_results['params'].keys():
        diff = torch.max(torch.abs(regular_results['params'][name] - ddp_results['params'][name]))
        max_diff = max(max_diff, diff.item())
    print(f"Maximum Parameter Difference: {max_diff:.6f}")
    print("="*50 + "\n")

def main():
    # Set a fixed seed for reproducibility
    seed = 42
    set_seed(seed)
    
    world_size = 2

    batch_size = 128
    context_length = 32

    vocab_size = 10000
    d_model = 768
    num_layers = 12
    d_ff = 3072
    num_heads = 12
    rope_theta = 10000
    max_seq_len = 64
    lr = 0.0001
    weight_decay = 0.01
    epsilon = 1e-8
    beta1 = 0.9
    beta2 = 0.95

    warmup = 0
    steps = 10

    # Generate or load the same random data
    if os.path.exists('random_data.pt'):
        input_data = torch.load('random_data.pt')
        target_data = torch.load('random_data.pt')
    else:
        input_data = generate_random_data(batch_size, context_length, vocab_size, seed)
        target_data = generate_random_data(batch_size, context_length, vocab_size, seed + 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # Run regular training first
    # print("\nRunning regular training...")
    # regular_results = train_regular(
    #     input_data, target_data,
    #     d_model, num_heads, d_ff, vocab_size, context_length, num_layers, max_seq_len, rope_theta, device,
    #     lr, weight_decay, epsilon, beta1, beta2, batch_size, warmup, steps, seed
    # )
    
    # Run DDP training
    print("\nRunning DDP training...")
    mp.spawn(train_ddp, args=(world_size,
          input_data, target_data,
          d_model, num_heads, d_ff, vocab_size, context_length, num_layers, max_seq_len, rope_theta, device,
          lr, weight_decay, epsilon, beta1, beta2, batch_size, warmup, steps, seed), 
          nprocs=world_size, join=True)
    
    # Load DDP results
    ddp_results = torch.load('ddp_results.pt')
    ddp_model = TransformerLM(
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
    ddp_model.load_state_dict(torch.load('ddp_final_params.pt'))
    ddp_results['params'] = {name: param.data.clone() for name, param in ddp_model.named_parameters()}
    
    # Compare results
    compare_results(regular_results, ddp_results)

if __name__ == "__main__":
    main()






    



    


