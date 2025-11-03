import torch
import math
from typing import List
from robot_model.mlp.mlp_external_p import MlpExternalParams


if __name__ == "__main__":
    import time

    # Define layer sizes and activation
    layer_sizes = [5, 32, 32, 4]
    activation = torch.nn.Tanh()

    # Create MlpExternalParams model
    model = MlpExternalParams(layer_sizes, activation, compile_model=True)
    
    list_of_names = model.param_wrapper.generate_list_of_params_names()
    print(list_of_names)
    print(len(list_of_names))

    # Calculate and print model parameter count
    paramcount = model.parameter_count()
    print(f"Model parameter count: {paramcount}")
    
    assert len(list_of_names) == paramcount, "Parameter count does not match the number of parameter names!"

    # Initialize default parameters
    p_default = model.get_default_params()

    # Create torch.nn.Sequential model with the same architecture
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        if i < len(layer_sizes) - 2:
            layers.append(activation)

    seq_model = torch.nn.Sequential(*layers)

    # Set seq_model's parameters to match p_default
    param_idx = 0
    for layer in seq_model:
        if isinstance(layer, torch.nn.Linear):
            in_features = layer.in_features
            out_features = layer.out_features

            weight_size = out_features * in_features
            bias_size = out_features

            # Extract weights and biases from p_default
            weight = p_default[param_idx:param_idx + weight_size]
            weight = weight.view(out_features, in_features)
            param_idx += weight_size

            bias = p_default[param_idx:param_idx + bias_size]
            param_idx += bias_size

            # Assign weights and biases to seq_model
            layer.weight.data = weight.clone()
            layer.bias.data = bias.clone()

    # Generate random inputs
    batch_size = 32
    t = torch.randn(batch_size)
    x = torch.randn(batch_size, 4)
    u = torch.randn(batch_size, 1)

    # Prepare parameter vector for the model
    p_expanded = p_default.unsqueeze(0).expand(batch_size, -1)

    # Run MlpExternalParams model
    model_output = model(t, x, u, p_expanded)

    # Run seq_model
    xu = torch.cat([x, u], dim=-1)
    seq_output = seq_model(xu)

    # Compare outputs
    difference = torch.abs(model_output - seq_output)
    max_difference = difference.max()
    print(f"Maximum difference between model outputs: {max_difference.item()}")

    # Assert that the outputs are close
    assert torch.allclose(model_output, seq_output, atol=1e-6), "Outputs are not the same!"

    # Benchmarking the model
    time_list = []
    x_list = []
    
    for i in range(100):
        now = time.time()
        p = model.get_default_params().unsqueeze(0).expand(32, -1) * 0.1
        p.requires_grad = True
        t = torch.randn(32)
        x = torch.randn(32, 4) * 4.0
        u = torch.randn(32, 1)
        x = model(t, x, u, p)
        x_list.append(x)
        elapsed = time.time() - now
        if i > 10:
            time_list.append(elapsed)


    x = torch.cat(x_list)
    print(f'Output shape: {(x**2).mean(dim=0).sqrt()}')
    
    avg_time = sum(time_list) / len(time_list)
    print(f'Average time: {avg_time:.10f}')
