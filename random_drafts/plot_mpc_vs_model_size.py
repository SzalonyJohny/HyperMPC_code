import matplotlib.pyplot as plt

# Define the data as a dictionary with layer size as key and episode cost as value
data = {
    # 64: 131232,
    # 128: 123506,
    48: 11273.3,
    8: 10885.7,
    32: 10823,
    24: 11707.7,
    # 4: 1.57684e+07,
    16: 11300.7,
    256: 11610
}

# Sort data by layer size for a cleaner plot
sorted_items = sorted(data.items())
layer_sizes, episode_costs = zip(*sorted_items)

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(layer_sizes, episode_costs, marker='o', linestyle='-')
plt.xlabel('Layer Size')
plt.ylabel('Episode Cost')
plt.title('MPC Result vs Model Size')
plt.grid(True)
# log scale
# plt.yscale('log')
plt.xscale('log')
plt.tight_layout()
plt.show()
