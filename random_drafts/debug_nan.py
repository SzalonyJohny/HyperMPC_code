import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# list all file in data dir
import os


# for i in range(30):
#     x_sim = np.load(f"data/debug_nan/train_X_sim_{i}.npz")
#     is_nan = np.isnan(x_sim['arr_0'])
#     correct = np.all(is_nan == False)
#     print(f"train_X_sim_{i} : {correct}")
    
# x_sim = np.load(f"data/debug_nan/train_X_sim_{15}.npz")
# x_sim = x_sim['arr_0']

# is_nan = np.isnan(x_sim)

# print(is_nan[0, 1, :])

# print(x_sim[0, 0, :])
# print(x_sim[0, 1, :])


    
x_sim = np.load(f"data/debug_nan/train_X_sim_{14}.npz")
x_sim = x_sim['arr_0']

is_nan = np.isinf(x_sim)

is_nan = np.any(is_nan, axis=-1)

# print(is_nan.shape)

plt.imshow(is_nan, aspect='auto', cmap='hot', interpolation='nearest')
plt.colorbar(label='NaN values')
plt.title('NaN values in x_sim')
plt.xlabel('batch element index')
plt.ylabel('Time step')
plt.show()

b = 436
# plt.plot(x_sim[b, :, 0], label='x')
# plt.plot(x_sim[b, :, 1], label='y')
plt.plot(x_sim[b, :, 2], label='z')

# plt.plot(x_sim[b, :, 7], label='vx')
# plt.plot(x_sim[b, :, 8], label='vy')
plt.plot(x_sim[b, :, 9], label='vz')


# axis limi
plt.xlim(0, 100)
plt.ylim(-100, 100)
plt.legend()


plt.show()