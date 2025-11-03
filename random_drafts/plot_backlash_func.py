import torch
import matplotlib.pyplot as plt


def backlash_func(x):
    return torch.sigmoid(25 * (x - 0.75))

p = torch.linspace(-10, 10, 1000)

y = backlash_func(p)

p_test = torch.tensor([1.0])
p_max = p_test * 1.5
p_min = p_test * 0.5

p_test_tensor = torch.tensor([p_min, p_test, p_max])
y_test = backlash_func(p_test_tensor)

plt.plot(p_test_tensor, y_test, 'o')

plt.plot(p, y)


plt.grid()
plt.show()