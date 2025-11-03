import numpy as np
import matplotlib.pyplot as plt

# Configuration
class cfg:
    track_width = 0.10
    soft_constraint_lambda = 200.0

# Track constraints as soft constraints
def soft_constraint(h, lambda_=cfg.soft_constraint_lambda):
    return np.log(1 + np.exp( - lambda_ * h))

# Generate data for plotting
n_values = np.linspace(-0.5, 0.5, 1000)

h_left_values =  cfg.track_width - n_values 
h_right_values =  cfg.track_width + n_values

cost = soft_constraint(h_left_values) + soft_constraint(h_right_values)

# Plotting
plt.plot(n_values, soft_constraint(h_left_values))
plt.plot(n_values, soft_constraint(h_right_values))
plt.plot(n_values, cost, marker='o')
plt.xlabel('n')
plt.ylabel('Soft Constraint Cost')
plt.title('Track Bound Soft Constraint Function')
plt.grid(True)
plt.show()
