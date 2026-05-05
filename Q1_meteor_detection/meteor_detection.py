import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Load data
data = pd.read_csv("data/meteor_data.csv")

# Extract signal column
signal = data['signal']

# Plot signal
plt.plot(signal)
plt.title("Signal Data")
plt.show()

# Detect peaks (meteor events)
peaks, _ = find_peaks(signal, height=threshold)

# Print result
print("Number of meteors detected:", len(peaks))
