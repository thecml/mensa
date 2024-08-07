import matplotlib.pyplot as plt
import numpy as np

# Bins array
data = np.array([359.80058085, 300.12872946, 312.23091776, 337.41995048, 373.84406443,
                 357.1915057, 404.53702511, 450.14494603, 472.39366052, 481.30861966])

# Calculate the histogram data
hist, bin_edges = np.histogram(data)

# Calculate the width of each bin
bin_widths = bin_edges[1:] - bin_edges[:-1]

# Plotting the bar chart
plt.bar(bin_edges[:-1], hist, width=bin_widths, edgecolor='black')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Bar Chart of Data Array with Custom Bins')
plt.show()