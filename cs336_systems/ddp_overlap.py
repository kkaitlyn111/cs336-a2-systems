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

class DDPOverlap(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()

        rank = 0
        self.handles = []
        self.module = module

        for param in self.module.parameters():
            dist.broadcast(tensor=param.data, src=rank)

        # use hook to enable communication in between grad accumulations!
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self.add_hook())

        def forward(self, *inputs, **kwargs):
            return self.module(*inputs, **kwargs)
        
        def finish_gradient_synchronization(self):
            for handle in self.handles:
                handle.wait()

            for param in self.module.parameters():
                if param.requires_grad:
                    param.grad.div_(2)

            self.handles = []

        def add_hook(self):
            def hook(grad):
                self.handles.append(dist.all_reduce(tensor=grad, op=dist.ReduceOp.SUM, async_op=True))
                # we want to enable async now, so true!
                # avg not allwoed here, divide by n workers later
            return hook

