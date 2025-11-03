import torch
from utils.bspline import BSpline


class FuturePredictionEncoder(torch.nn.Module):
    def __init__(self, 
                 channels_count: int,
                 input_len, 
                 output_ctrl_point,
                 return_spline_and_error,
                 device: str, 
                 channel_divider = None) -> None:
        
        super(FuturePredictionEncoder, self).__init__()
        
        self.bspline = BSpline(output_ctrl_point, 4,
                               input_len, "prediction_sampling")
        self.Ns, self.dNs, self.ddNs, self.dddNs = self.bspline.calculate_N()
        self.Ns = torch.from_numpy(self.Ns).float().contiguous()
        
        self.inv_Ns = torch.linalg.pinv(self.Ns).to(device)
        
        self.return_spline_and_error = return_spline_and_error    
        
        self.channels_count = channels_count
        
        if channel_divider is None:
            self.channels_deviders = torch.ones(channels_count, device=device)
        else:        
            self.channels_deviders = torch.tensor(channel_divider, device=device)
            assert self.channels_deviders.shape[-1] == self.channels_count

        print(f'Channel deviders: {self.channels_deviders} and shape: {self.channels_deviders.shape}')
    
    def get_channel_count(self):
        return self.channels_count

    def forward(self, P: torch.Tensor, sigma: float) -> torch.Tensor:
        """ From the input tensor P, [batch, input_len, channels],
            return the control points of the spline function with noise added to the control points.
        
        Args:
            P (torch.Tensor): prediction horizon [batch, input_len, channels]
            sigma (float): standard deviation of the noise to add to the control points

        Returns:
            torch.Tensor: [batch, output_ctrl_point, channels]
        """
        
        assert P.shape[-1] == self.channels_count,\
            f"Expected {self.channels_count} channels, got {P.shape[-1]}"
        
        P = P / self.channels_deviders
        
        ctrl_points = torch.matmul(self.inv_Ns, P)
        ctrl_points_org = ctrl_points.clone()

        if self.training:
            ctrl_points[:, 1:, :] *= (torch.randn_like(ctrl_points[:, 1:, :]) * sigma + 1.0)

        if self.return_spline_and_error:
            # Calculate spline from control points
            spline = torch.matmul(self.Ns, ctrl_points)
            
            # Calculate reproduction error of original signal
            spline_org = torch.matmul(self.Ns, ctrl_points_org)
            rms_error = torch.sqrt(torch.mean((spline_org - P) ** 2, dim=[1]))
            nrms_error = rms_error / torch.sqrt(torch.mean(P ** 2))
            
            return spline, ctrl_points, ctrl_points_org, nrms_error
        
        return ctrl_points


    
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    points_count = 85
    
    sampled_points_count = 12

    sampler = FuturePredictionEncoder(10, points_count,
                                     sampled_points_count, return_spline_and_error=True, device='cpu', channel_divider = [2.0] * 10)
    
    # sampler = torch.compile(sampler, fullgraph=True)
    
    t_ctrl = torch.linspace(0, 20, sampled_points_count).unsqueeze(0).unsqueeze(-1)

    t = torch.linspace(0, 20, points_count).unsqueeze(0).unsqueeze(-1)
    y = torch.sin(t * 0.85) + 2.0 + torch.cos(t * 0.1)**2 * 3
    y -= y.mean()
    
    # y = torch.sign(y) 
    y = y.repeat(1, 1, 10)
    
    print(y.shape)

    sample_y, ctrl_points, ctrl_points_org, nrms_error = sampler(y, 0.2)

    for i in range(sample_y.shape[-1]):
        plt.plot(t[0, :, 0].numpy(), sample_y[0, :, i].numpy(), label='spline', linestyle='--')   
        plt.scatter(t_ctrl[0, :, 0].numpy(), ctrl_points[0, :, i].numpy(), label='control points with noise', marker='x')
    
    plt.plot(t[0, :, 0].numpy(), y[0, :, 0].numpy(), label='original')
    plt.scatter(t_ctrl[0, :, 0].numpy(), ctrl_points_org[0, :, 0].numpy(), label='control points')
    
    print(nrms_error)

    plt.title('Samples from the original signal')
    plt.grid()
    plt.show()
        
        
    