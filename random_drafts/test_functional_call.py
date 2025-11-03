import torch
import time
import torch.nn as nn
import torch.func as F

# Define a simple module
class SimpleLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

# Create instance
model = SimpleLinear(3, 2)

# Extract parameters
params = {k: v for k, v in model.named_parameters()}

print(params)

# Functional call (fast, no copying)
x = torch.randn(5, 3)  # Input batch
output = F.functional_call(model, params, (x,))  # Apply module functionally
print(output)

def funct(model, params, x):    
    return F.functional_call(model, params, (x,))


model = torch.compile(model, fullgraph=True)
funct = torch.compile(funct, fullgraph=True)

num_iterations = 1000

# Warmup iterations for JIT and cache stabilization
for _ in range(100):
    x = torch.randn(128, 3)  # Input batch
    funct(model, params, x)
    model(x)
    
# Measure functional_call execution time
start_time = time.perf_counter()
for _ in range(num_iterations):
    # x = torch.randn(5, 3)  # Input batch
    funct(model, params, x)
functional_time = time.perf_counter() - start_time

# Measure module.forward execution time
start_time = time.perf_counter()
for _ in range(num_iterations):
    # x = torch.randn(5, 3)  # Input batch
    model(x)
forward_time = time.perf_counter() - start_time

print(f"Functional call time: {functional_time:.6f} seconds for {num_iterations} iterations")
print(f"Module forward time: {forward_time:.6f} seconds for {num_iterations} iterations")