import numpy as np

import matplotlib
import matplotlib.pyplot as plt
plt.ion

all_data = np.loadtxt('all_results.csv', delimiter=',')
ordered_results = {}

for i in range(2000):
    ordered_results[i] = []

for row in all_data:
    ordered_results[row[1]].append(row[0])

means = []

for i in range(1000):
    means.append(np.mean(ordered_results[i]))
    
overview = plt.figure()
results_over_time_graph = overview.add_subplot(1, 1, 1)

results_over_time_graph.cla()
results_over_time_graph.plot(means, label="Mean reward")

overview.canvas.draw()
plt.pause(100)