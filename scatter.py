"""Normalized script generated from scatter.py."""

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['font.sans-serif'] = ['SimSun']  # Configure Chinese font rendering for Matplotlib labels.
plt.rcParams['axes.unicode_minus'] = False  # Render minus signs correctly in Matplotlib.

# Assume the data has been saved to a CSV file.
# CSV file path.
csv_file = "tide_optimization_history.csv"

# Read CSV data.
data = pd.read_csv(csv_file)

# Extract required columns.
x = data['input_chunk_length']
y = data['output_chunk_length']
z = data['hidden_size']
mape = data['MAPE']

# Create a 3D figure.
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Draw a 3D scatter plot.
scatter = ax.scatter(x, y, z, c=mape, cmap='viridis', s=50, alpha=0.7)

# Set axis labels.
ax.set_xlabel('输入块长度')
ax.set_ylabel('输出块长度')
ax.set_zlabel('隐藏层大小')

# Add the color bar.
cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
cbar.set_label('MAPE')

# Display the figure.
plt.title('WOA利用MAPE优化神经网络全局参数散点图')
plt.tight_layout()
plt.show()
