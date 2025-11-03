import torch
from robot_model.car.pacejka_tire_model import PacejkaTireModel
from robot_model.car.pacejka_params import PacejkaParameters


def plot_tire_model():
        
    import matplotlib.pyplot as plt

    # Initialize Pacejka parameters
    pacejka_params = PacejkaParameters()
    wp_tensor = pacejka_params.default_params_tensor()
    wp = pacejka_params.forward(wp_tensor)[0]

    # Define slip angles and slip ratios for the plot
    slip_angles = torch.linspace(0, 20, 100)  # Slip angles from 0 to 20 degrees
    slip_ratios = [0, 0.03, 0.06, 0.09, 0.15, 0.30]  # Different slip ratios

    # Create a plot for the tire model "Fy vs slip angle under different slip ratios"
    plt.figure(figsize=(10, 6))

    for Sx in slip_ratios:
        Fy_values = []
        for Alpha in slip_angles:
            Fy = PacejkaTireModel.tire_forces_model(Alpha * torch.pi / 180.0, Sx, wp)[1]
            Fy_values.append(- Fy.item())
        
        plt.plot(slip_angles.numpy(), Fy_values, label=f'Slip ratio = {Sx * 100:.0f}%')

    # Plot settings
    plt.title('Lateral Force (Fy) vs Slip Angle under Different Slip Ratios')
    plt.xlabel('Slip Angle (degrees)')
    plt.ylabel('Lateral Force (N)')
    plt.legend()
    plt.grid(True)

    # Define slip ratios and slip angles for the plot
    slip_ratios = torch.linspace(0, 0.6, 100)  # Slip ratios from 0 to 0.6 (60%)
    slip_angles = [0, 2, 4, 6, 8, 12, 20]  # Different slip angles in degrees

    # Create a plot for the tire model "Fx vs slip ratio under different slip angles"
    plt.figure(figsize=(10, 6))

    for Alpha in slip_angles:
        Fx_values = []
        for Sx in slip_ratios:
            Fx = PacejkaTireModel.tire_forces_model(Alpha * torch.pi / 180.0, Sx.item(), wp)[0]
            Fx_values.append(Fx.item())
        
        plt.plot(slip_ratios.numpy(), Fx_values, label=f'Slip angle = {Alpha}°')

    # Plot settings
    plt.title('Longitudinal Force (Fx) vs Slip Ratio under Different Slip Angles')
    plt.xlabel('Slip Ratio')
    plt.ylabel('Longitudinal Force (N)')
    plt.legend()
    plt.grid(True)

    # Generate a grid of slip angle and slip ratio pairs for testing
    slip_ratios_grid = torch.linspace(0, 0.95, 200)  # Slip ratios from 0 to 0.95
    slip_angles_grid = torch.linspace(0, 45, 200)  # Slip angles from 0 to 45 degrees

    # Create meshgrid for the slip ratios and slip angles
    Slip_ratios, Slip_angles = torch.meshgrid(slip_ratios_grid, slip_angles_grid, indexing='ij')

    # Compute forces for each pair and store them
    Fx_values = []
    Fy_values = []

    for i in range(Slip_ratios.shape[0]):
        for j in range(Slip_ratios.shape[1]):
            Sx = Slip_ratios[i, j]
            Alpha = Slip_angles[i, j]
            Fx, Fy = PacejkaTireModel.tire_forces_model(Alpha * torch.pi / 180.0, Sx.item(), wp)
            Fx_values.append(Fx.item())
            Fy_values.append(- Fy.item())

    # Create the scatter plot (tire ellipse)
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(Fx_values, Fy_values, c=Slip_angles, s=10, alpha=0.5, cmap='viridis', label='Tire Force Ellipse')
    plt.colorbar(scatter, label='Slip Angle (degrees)')
    plt.title('Tire Force Ellipse: Fx vs Fy')
    plt.xlabel('Longitudinal Force (Fx) [N]')
    plt.ylabel('Lateral Force (Fy) [N]')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True)
    plt.axis('equal')
    plt.legend()

    # Create the scatter plot (tire ellipse)
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(Fx_values, Fy_values, c=Slip_ratios, s=10, alpha=0.5, cmap='viridis', label='Tire Force Ellipse')
    plt.colorbar(scatter, label='Slip Ratio (degrees)')
    plt.title('Tire Force Ellipse: Fx vs Fy')
    plt.xlabel('Longitudinal Force (Fx) [N]')
    plt.ylabel('Lateral Force (Fy) [N]')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.axis('equal')
    plt.grid(True)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    plot_tire_model()