from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.cuda.nvtx as nvtx
import os
import timeit
import pickle
from cs336_basics.transformer import TransformerLM
from cs336_basics.training import cross_entropy_loss as cross_entropy
from cs336_basics.optimizers import AdamW
import argparse
import collections
from typing import Any, Type, Optional, Callable


def train_single_process(hparams):
    d_model, d_ff, num_layers, num_heads, batch_size, theta, context_length, vocab_size, lr, betas, eps, weight_decay, num_steps = hparams["d_model"], hparams["d_ff"], hparams["num_layers"], hparams["num_heads"], hparams["batch_size"], hparams["theta"], hparams["context_length"], hparams["vocab_size"], hparams["lr"], hparams["betas"], hparams["eps"], hparams["weight_decay"], hparams["num_steps"]
    
    device = torch.device("cuda")
    model = TransformerLM(vocab_size=vocab_size, context_length=context_length, d_model=d_model, num_layers=num_layers, num_heads=num_heads, d_ff=d_ff, theta=theta, device=device, dtype=torch.float32).to(device)
    
    optimizer = AdamW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    with open("data_large.pkl", "rb") as f:
        data = pickle.load(f)
    data = (data[0].to(device), data[1].to(device))

    for i in range(num_steps):
        batch = data[0][i * batch_size: (i+1) * batch_size]
        labels = data[1][i * batch_size: (i+1) * batch_size]
        logits = model(batch)
        loss = cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            print(f"Step {i} loss: {loss.item()}")
    
    print("Final loss: ", loss.item())


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29504"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
        
def train_process(rank, world_size, hparams):
    try:
        setup(rank, world_size)
        device = torch.device(f"cuda:{rank}")
        d_model, d_ff, num_layers, num_heads, batch_size, theta, context_length, vocab_size, lr, betas, eps, weight_decay, num_steps = hparams["d_model"], hparams["d_ff"], hparams["num_layers"], hparams["num_heads"], hparams["batch_size"], hparams["theta"], hparams["context_length"], hparams["vocab_size"], hparams["lr"], hparams["betas"], hparams["eps"], hparams["weight_decay"], hparams["num_steps"]

        model = TransformerLM(vocab_size=vocab_size, context_length=context_length, d_model=d_model, num_layers=num_layers, num_heads=num_heads, d_ff=d_ff, theta=theta, device=device, dtype=torch.float32).to(device)
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
        optimizer = AdamW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        with open("data_large.pkl", "rb") as f:
            data = pickle.load(f)
        data = (data[0].to(device), data[1].to(device))

        batch_size_P = batch_size // world_size
        durations_training_steps = []
        durations_communication = []
        for i in range(num_steps):
            step_start = timeit.default_timer()
            batch_full = data[0][i * batch_size: (i+1) * batch_size]
            labels_full = data[1][i * batch_size: (i+1) * batch_size]
            batch = batch_full[rank * batch_size_P: (rank+1) * batch_size_P]
            labels = labels_full[rank * batch_size_P: (rank+1) * batch_size_P]

            logits = model(batch)
            loss = cross_entropy(logits, labels)
            optimizer.zero_grad()
            with nvtx.range("Backward"):
                loss.backward()
            communication_start = timeit.default_timer()
            # grads = [param.grad.data for param in model.parameters()]
            # grad_flat = torch._utils._flatten_dense_tensors(grads)
            # dist.all_reduce(grad_flat, op=dist.ReduceOp.AVG)
            # grad_unflat = torch._utils._unflatten_dense_tensors(grad_flat, grads)
            # for param, grad in zip(model.parameters(), grad_unflat):
            #     param.grad.data = grad
            for param in model.parameters():
                with nvtx.range(f"Communication {id(param)}"):
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.AVG)
            torch.cuda.synchronize()
            communication_end = timeit.default_timer()
            durations_communication.append(communication_end - communication_start)

            optimizer.step()

            torch.cuda.synchronize()
            step_end = timeit.default_timer()
            durations_training_steps.append(step_end - step_start)

            if i % 10 == 0:
                dist.all_reduce(loss.data, op=dist.ReduceOp.AVG)
                if rank == 0:
                    print(f"Step {i} Average loss: {loss.item()}")
        
        print("Final loss: ", loss.item())

        
        durations_training_steps = torch.tensor(durations_training_steps, device=device)
        durations_communication = torch.tensor(durations_communication, device=device)
        dist.all_reduce(durations_training_steps, op=dist.ReduceOp.AVG)
        dist.all_reduce(durations_communication, op=dist.ReduceOp.AVG)
        d_training_steps = torch.sum(durations_training_steps).item()
        d_communication = torch.sum(durations_communication).item()
        if rank == 0:
            print(f"Total Time Per Training Step: {d_training_steps:.2f}s")
            print(f"Total Time Per Gradient Communication: {d_communication:.2f}s")
            print(f"Percentage of time spent in gradient communication: {(d_communication / d_training_steps * 100):.2f}%")
        
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def train_distributed(hparams):
    world_size = 2
    mp.spawn(train_process, args=(world_size, hparams), nprocs=world_size, join=True)
    



def train_process_ddp(rank, world_size, hparams, wrap, sharded=False):
    try:
        setup(rank, world_size)
        device = torch.device(f"cuda:{rank}")
        d_model, d_ff, num_layers, num_heads, batch_size, theta, context_length, vocab_size, lr, betas, eps, weight_decay, num_steps = hparams["d_model"], hparams["d_ff"], hparams["num_layers"], hparams["num_heads"], hparams["batch_size"], hparams["theta"], hparams["context_length"], hparams["vocab_size"], hparams["lr"], hparams["betas"], hparams["eps"], hparams["weight_decay"], hparams["num_steps"]

        model = TransformerLM(vocab_size=vocab_size, context_length=context_length, d_model=d_model, num_layers=num_layers, num_heads=num_heads, d_ff=d_ff, theta=theta, device=device, dtype=torch.float32).to(device)
        if wrap == "DDP":
            model_ddp = DDP(model)
        elif wrap == "DDP_Bucketed":
            bucket_size_mb = 1000
            model_ddp = DDP_Bucketed(model, bucket_size_mb=bucket_size_mb)
        if sharded:
            optimizer = ShardedStateOptimizer(model.parameters(), AdamW, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
            print(f"Sharded optimizer initialized")
        else:
            optimizer = AdamW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        
        torch.cuda.synchronize()
        memory = torch.tensor(torch.cuda.max_memory_allocated() / 1024**2, device=device)
        dist.all_reduce(memory, op=dist.ReduceOp.AVG)
        if rank == 0:
            print(f"Memory right after model initialization: {memory.item():.2f}MB")

        with open("data_large.pkl", "rb") as f:
            data = pickle.load(f)
        data = (data[0].to(device), data[1].to(device))

        batch_size_P = batch_size // world_size
        durations_training_steps = []
        for i in range(num_steps):
            step_start = timeit.default_timer()
            batch_full = data[0][i * batch_size: (i+1) * batch_size]
            labels_full = data[1][i * batch_size: (i+1) * batch_size]
            batch = batch_full[rank * batch_size_P: (rank+1) * batch_size_P]
            labels = labels_full[rank * batch_size_P: (rank+1) * batch_size_P]

            logits = model_ddp(batch)
            loss = cross_entropy(logits, labels)
            optimizer.zero_grad()
            with nvtx.range("Backward"):
                loss.backward()
            model_ddp.finish_gradient_synchronization()

            torch.cuda.synchronize()
            memory = torch.tensor(torch.cuda.max_memory_allocated() / 1024**2, device=device)
            dist.all_reduce(memory, op=dist.ReduceOp.AVG)
            if rank == 0:
                print(f"Memory right before optimizer step: {memory.item():.2f}MB")
            optimizer.step()
            torch.cuda.synchronize()
            memory = torch.tensor(torch.cuda.max_memory_allocated() / 1024**2, device=device)
            dist.all_reduce(memory, op=dist.ReduceOp.AVG)
            if rank == 0:
                print(f"Memory right after optimizer step: {memory.item():.2f}MB")

            torch.cuda.reset_peak_memory_stats()


            torch.cuda.synchronize()
            step_end = timeit.default_timer()
            durations_training_steps.append(step_end - step_start)

            if i % 10 == 0:
                dist.all_reduce(loss.data, op=dist.ReduceOp.AVG)
                if rank == 0:
                    print(f"Step {i} Average loss: {loss.item()}")
        
        print("Final loss: ", loss.item())

        
        durations_training_steps = torch.tensor(durations_training_steps, device=device)
        dist.all_reduce(durations_training_steps, op=dist.ReduceOp.AVG)
        d_training_steps = torch.sum(durations_training_steps).item()
        if rank == 0:
            print(f"Total Time Per Training Step {f'(bucket size {bucket_size_mb}MB)' if wrap == 'DDP_Bucketed' else ''}: {d_training_steps:.2f}s")
        
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

def train_distributed_ddp(hparams, model_wrapper, sharded=False):
    world_size = 2
    mp.spawn(train_process_ddp, args=(world_size, hparams, model_wrapper, sharded), nprocs=world_size, join=True)


class DDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
        
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self.all_reduce_gradients)
        
        self.handles = []
    
    def all_reduce_gradients(self, param):
        with nvtx.range(f"Communication {id(param)}"):
            handle = dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, async_op=True)
        self.handles.append(handle)
    
    def forward(self, *inputs, **kwargs):
        return self.module.forward(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        for handle in reversed(self.handles):
            handle.wait()
        for param in self.module.parameters():
            if param.requires_grad:
                param.grad.data /= self.world_size
        self.handles.clear()


class DDP_Bucketed(nn.Module):
    def __init__(self, module: nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.bucket_size_mb = bucket_size_mb

        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
        
        self.param_buckets = {} # param_id -> bucket_id
        self.bucket_params = {} # bucket_id -> list of param
        curr_bucket, curr_bucket_id, curr_size = [], 0, 0
        temp = []
        for param in self.module.parameters():
            temp.append(param)
        for param in reversed(temp):
            if curr_size + param.numel() * param.element_size() > self.bucket_size_mb * 1024**2:
                self.bucket_params[curr_bucket_id] = curr_bucket
                curr_bucket = []
                curr_bucket_id += 1
                curr_size = 0
            self.param_buckets[id(param)] = curr_bucket_id
            curr_bucket.append(param)      
            curr_size += param.numel() * param.element_size()
        if curr_bucket:
            self.bucket_params[curr_bucket_id] = curr_bucket
        self.bucket_counter = collections.defaultdict(int) #bucket id -> number of gradients computed in bucket
        self.bucket_grads_flat = {} #bucket id -> grads_flat
        
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self.all_reduce_gradients)
        
        self.handles = []
    
    def all_reduce_gradients(self, param):
        #we need to check if all other in bucket have already had gradients, otherwise do nothing here
        bucket_id = self.param_buckets[id(param)]
        self.bucket_counter[bucket_id] += 1
        if self.bucket_counter[bucket_id] < len([p for p in self.bucket_params[bucket_id] if p.requires_grad]):
            return
        with nvtx.range(f"Communication: bucket {bucket_id}"):
            grads = [bucket_param.grad.data for bucket_param in self.bucket_params[bucket_id] if bucket_param.requires_grad]
            grads_flat = torch._utils._flatten_dense_tensors(grads)
            self.bucket_grads_flat[bucket_id] = grads_flat
            handle = dist.all_reduce(grads_flat, op=dist.ReduceOp.SUM, async_op=True)
        self.handles.append(handle)
    
    def forward(self, *inputs, **kwargs):
        return self.module.forward(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        for handle in reversed(self.handles):
            handle.wait()
        for bucket_id in self.bucket_grads_flat.keys():
            grads_flat = self.bucket_grads_flat[bucket_id]
            grads = [param.grad.data for param in self.bucket_params[bucket_id] if param.requires_grad]
            grads_unflat = torch._utils._unflatten_dense_tensors(grads_flat, grads)
            for param, grad in zip([p for p in self.bucket_params[bucket_id] if p.requires_grad], grads_unflat):
                if param.requires_grad:
                    param.grad.data = grad / self.world_size
        self.bucket_grads_flat.clear()
        for bucket_id in self.bucket_counter.keys():
            self.bucket_counter[bucket_id] = 0
        self.handles.clear()



class ShardedStateOptimizer(optim.Optimizer):
    def __init__(self, params, optimizer_cls: Type[optim.Optimizer], **kwargs: Any):
        self.optimizer = None #
        self.kwargs = kwargs
        self.optimizer_cls = optimizer_cls

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.param_to_rank = {} #id(param) -> rank
        self.rank_sizes =[0] * self.world_size #rank -> MB of parameters on this rank currently
        super().__init__(params, defaults={})

    def step(self, closure: Optional[Callable] = None, **kwargs):
        self.optimizer.step(closure, **kwargs)
        for group in self.param_groups:
            for param in group["params"]:
                param_rank = self.param_to_rank[id(param)]
                dist.broadcast(param.data, src=param_rank)
    
    def add_param_group(self, param_group: dict[str, Any]):
        super().add_param_group(param_group)
        for param in param_group["params"]:
            param_rank = np.argmin(self.rank_sizes)
            self.param_to_rank[id(param)] = param_rank
            self.rank_sizes[param_rank] += param.numel() * param.element_size()
        
        local_params = []
        for param in param_group["params"]:
            if self.param_to_rank[id(param)] == self.rank:
                local_params.append(param)
                assert not isinstance(param, str)
            
        
        if local_params:
            cfg = {k: v for k, v in param_group.items() if k != "params"}
            local_param_group = {"params": local_params, **cfg}
            if self.optimizer is None:
                self.optimizer = self.optimizer_cls([local_param_group], **self.kwargs)
            else:
                self.optimizer.add_param_group(local_param_group)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--distributed", action="store_true", help="Whether to run distributed training")
    parser.add_argument("--ddp", action="store_true", help="Whether to run DDP training")
    parser.add_argument("--ddp_bucketed", action="store_true", help="Whether to run DDP Bucketed training")
    parser.add_argument("--sharded", action="store_true", help="Whether to run sharded training")
    args = parser.parse_args()
    distributed = args.distributed
    ddp = args.ddp
    ddp_bucketed = args.ddp_bucketed
    sharded = args.sharded
    if distributed:
        print(f"Running distributed training")
    elif ddp and not sharded:
        print(f"Running DDP training")
    elif ddp_bucketed and not sharded:
        print(f"Running DDP Bucketed training")
    elif ddp and sharded:
        print(f"Running DDP + sharded training")
    elif ddp_bucketed and sharded:
        print(f"Running DDP Bucketed + sharded training")
    else:
        print(f"Running single process training")

    hparams = {
        'd_model': 768,
        'd_ff': 3072,
        'num_layers': 12,
        'num_heads': 12,
        'batch_size': 128,
        'theta': 10000,
        'context_length': 128,
        'vocab_size': 10000,
        'lr': 1e-3,
        'betas': (0.9, 0.95),
        'eps': 1e-8,
        'weight_decay': 0.1,
        'num_steps': 50,
    }

    # #generate random data and write it to a file
    # data = torch.randint(0, hparams['vocab_size'], (hparams['batch_size'] * hparams['num_steps'], hparams['context_length']))
    # labels = torch.randint(0, hparams['vocab_size'], (hparams['batch_size'] * hparams['num_steps'], hparams['context_length']))
    # with open("data_large.pkl", "wb") as f:
    #     pickle.dump((data, labels), f)

    if distributed:
        train_distributed(hparams)
    elif ddp or ddp_bucketed:
        model_wrapper = "DDP_Bucketed" if ddp_bucketed else "DDP"
        sharded = True if sharded else False
        train_distributed_ddp(hparams, model_wrapper, sharded)
    else:
        train_single_process(hparams)