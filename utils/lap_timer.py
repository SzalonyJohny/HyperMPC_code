import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("test_data/gcz_mpc_2_s_only.csv")

s = df['/mpc_debug/debug_mpc/x_mpc_in/s']
t = df['__time']

plt.scatter(t, s)

s_diff = np.diff(s)
s_diff = np.append(s_diff, 0)
new_lap = np.where(s_diff < -0.5, 1, 0)

plt.scatter(t, new_lap, color='red')
plt.show()

# count time between new laps
lap_times = []
lap_start = 0
for i in range(len(new_lap)):
    if new_lap[i] == 1:
        lap_times.append(t[i] - lap_start)
        lap_start = t[i]
    
lap_times = lap_times[1:]

sorted_lap_times = np.sort(lap_times)
print(sorted_lap_times)


l = len(sorted_lap_times) * 0.8
print(np.mean(sorted_lap_times[:int(l)]))