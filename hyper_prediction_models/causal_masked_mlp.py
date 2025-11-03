import torch


class CausalMaskedLinear(torch.nn.Module):
    def __init__(self, num_time_steps, input_feature_size, output_feature_size, triangular_block_size=-1, bias=True, ):
        """
        num_time_steps: Number of sequential time steps (e.g., t, t+1, t+2...)
        input_feature_size: Size of the input feature dimension for each time step
        output_feature_size: Size of the output feature dimension for each time step
        """
        super(CausalMaskedLinear, self).__init__()
        self.num_time_steps = num_time_steps

        input_size = num_time_steps * input_feature_size
        output_size = num_time_steps * output_feature_size

        self.linear = torch.nn.Linear(input_size, output_size, bias=bias)
        
        # Create block lower triangular mask
        mask = torch.zeros(output_size, input_size, dtype=torch.bool)
        for i in range(num_time_steps):
            mask[
                i * output_feature_size : (i + 1) * output_feature_size,
                : (i + 1) * input_feature_size
            ] = True  # Allow each output to depend on the current and previous inputs
               
        # Apply triangular block size to reduce overfitting
        if triangular_block_size != -1:
            for i in range(num_time_steps):
                mask[
                    i * output_feature_size : (i + 1) * output_feature_size,
                    : max(0, (i - triangular_block_size + 1)) * input_feature_size
                ] = False
       
        self.register_buffer('mask', mask)
        
        
    def inactive_parameters_count(self):
        return (self.mask == False).sum().item()

    def forward(self, x):
        masked_weight = self.linear.weight * self.mask
        return torch.nn.functional.linear(x, masked_weight, self.linear.bias)
    
    
    
if __name__ == "__main__":

    points = 5
    input_feature_size = 2
    output_feature_size = 3
    
    causal_masked_linear = CausalMaskedLinear(points, input_feature_size, output_feature_size,
                                              bias=True, triangular_block_size=2)
    print(causal_masked_linear.mask)

    # replace weight with ones for testing
    causal_masked_linear.linear.weight.data.fill_(1.0)
    # replace bias with zeros for testing
    causal_masked_linear.linear.bias.data.fill_(0.0)

    x = torch.ones(1, points * input_feature_size)
    y = causal_masked_linear(x)
    print(y.shape)
    
    print(y)
    
    print(causal_masked_linear.inactive_parameters_count())
    
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Hyper param_count {count_parameters(causal_masked_linear)}")
    