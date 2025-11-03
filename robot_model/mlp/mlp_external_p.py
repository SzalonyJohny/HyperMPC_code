import torch
import math
from robot_model.acrobot.acrobot_params import ActobotParams
from typing import List

class MlpExternalParams(torch.nn.Module):

    def __init__(self,
                 preprocessor: torch.nn.Module,
                 layer_sizes: List, # [input_dim, hidden1_dim, ..., output_dim]
                 activation: torch.nn.Module,
                 compile_model: bool = False,
                 *args, **kwargs) -> None:
        super(MlpExternalParams, self).__init__(*args, **kwargs)

        # print("MlpExternalParams init")
        # print(f"layer_sizes: {layer_sizes}, layer_sizes type {type(layer_sizes)}")
        
        self.layer_sizes = list(layer_sizes)
        
        self.layer_count = len(layer_sizes)
        self.activation = activation
        self.init_weights_scale = 0.1
        self.preprocessor = preprocessor
        
        class MlpParamWrapper():
            def __init__(self, layer_sizes):
                self.layer_sizes = layer_sizes
                self.list_of_names = self.generate_list_of_params_names()
            
            def generate_list_of_params_names(self):
                list_of_names = []
                for i in range(len(self.layer_sizes) - 1):
                    for j in range(self.layer_sizes[i] * self.layer_sizes[i+1]):
                        list_of_names.append(f'W_{i}_{j}')
                    for j in range(self.layer_sizes[i+1]):
                        list_of_names.append(f'b_{i}_{j}')                    
                return list_of_names
        
            def get_params_names(self):
                return self.list_of_names
            
            @staticmethod
            def positive_params():
                # TODO implement as index tensor
                return False
        
        self.param_wrapper = MlpParamWrapper(layer_sizes)
        
        if compile_model:
            self.forward = torch.compile(self.forward, fullgraph=True, dynamic=False)

    def get_params_names(self):
        return self.param_wrapper.get_params_names()

    def parameter_count(self):
        count = 0
        for i in range(len(self.layer_sizes) - 1):
            count += self.layer_sizes[i] * self.layer_sizes[i+1] + self.layer_sizes[i+1]
        return count

    def extract_weights(self, p, i):
        """
            p = [batch, param_count]
            i = layer index
        """
        param_size = self.layer_sizes[i] * self.layer_sizes[i+1]
        W = p[..., :param_size]
        W = W.view(p.shape[0], self.layer_sizes[i+1], self.layer_sizes[i])
        return W, p[..., param_size:]

    def extract_bias(self, p, i):
        """
            p = [batch, param_count]
            i : layer index
        """
        b = p[..., :self.layer_sizes[i+1]]
        return b.unsqueeze(-1), p[..., self.layer_sizes[i+1]:]

    def get_default_params(self):
        """
            Returns a parameter vector p of shape [param_count],
            initialized with default PyTorch initializations.
        """
        params = []
        for i in range(len(self.layer_sizes) - 1):
            in_features = self.layer_sizes[i]
            out_features = self.layer_sizes[i + 1]

            # Initialize weights
            weight = torch.empty(out_features, in_features)
            torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            weight = weight.view(-1)
            params.append(weight)

            # Initialize biases
            bias = torch.empty(out_features)
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
                weight.view(out_features, in_features))
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(bias, -bound, bound)
            params.append(bias)

        p = torch.cat(params) * self.init_weights_scale
        p = p.unsqueeze(0)
        return p

    def forward(self, t, x, u, p):
        """
            t = [batch]
            x = [batch, state dim -> 4]
            u = [batch, torque -> 1]
            p = [batch, param_count]
        """
        xu = torch.cat([x, u], dim=-1)

        # unsqueeze to get vector that can be multiplied with weights
        xu = self.preprocessor(xu).unsqueeze(-1) 

        for i in range(self.layer_count - 1):
            W, p = self.extract_weights(p, i)
            b, p = self.extract_bias(p, i)

            xu = torch.bmm(W, xu) + b

            # Apply activation after all layers except the last
            if i < self.layer_count - 2:
                xu = self.activation(xu)

        assert p.shape[-1] == 0
        
        # ddq = xu.squeeze(-1)
        # dq = x[:, 2:4]
        # return torch.cat([dq, ddq], dim=-1)
        return xu.squeeze(-1)
