import torch
from time import perf_counter



class TestModel(torch.nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 6),
        )
        
    def forward(self, x, u):
        xu = torch.cat([x, u], dim=-1)
        return self.fc(xu)
    
    
def rollout_list(model, x0, u):
    x = x0.clone()
    x_list = []
    for i in range(100):
        x = model(x, u[:, i, :])
        x_list.append(x)
    return torch.stack(x_list, dim=1)


@torch.compile(mode="default", fullgraph=True)
def rollout_list_compile(model, x0, u):
    x = x0.clone()
    x_list = []
    for i in range(100):
        x = model(x, u[:, i, :])
        x_list.append(x)
    return torch.stack(x_list, dim=1)


def test_compile_loop():
    model = TestModel()
    x0 = torch.randn(1, 6)
    u = torch.randn(1, 100, 4)
    
    xro = rollout_list(model, x0, u)
    print(xro.shape)
    rollout_list(model, x0, u)
    
    time_now = perf_counter()
    rollout_list(model, x0, u)
    print(f"Time taken: {perf_counter() - time_now}")    
    
    rollout_list_compile(model, x0, u)
    rollout_list_compile(model, x0, u)
    
    time_now = perf_counter()
    rollout_list_compile(model, x0, u)
    print(f"Time taken: {perf_counter() - time_now}")
    
    
    
if __name__ == "__main__":
    test_compile_loop()
    