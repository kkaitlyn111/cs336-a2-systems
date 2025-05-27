import torch 
import torch.nn as nn


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        print('x data type after fc1', x.dtype)
        print('fc1 weight type:', self.fc1.weight.dtype)
        self.ln = nn.LayerNorm(10)
        print('x data type after layer norm', x.dtype)
        print('ln weight type:', self.ln.weight.dtype)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.ln(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':

    bfloat16 = False
    dtype = torch.bfloat16 if bfloat16 else torch.float16

    model = ToyModel(10, 10)
    model.to('cuda')
    x = torch.randn(size=(4, 10), device='cuda')

    with torch.autocast(device_type='cuda',dtype=dtype):
        print('model parameters data type:')
        for name, param in model.named_parameters():
            print(name, param.dtype)

        print('\n')

        y = model(x)
        loss = nn.CrossEntropyLoss()
        output = loss(y, torch.randn(size=(4, 10), device='cuda'))
        output.backward()

        print('logits type:', y.dtype, 'loss type:', output.dtype)

        print('\n')

        print('gradients type:')
        for name, param in model.named_parameters():
            print(name, param.grad.dtype)
