# ----------------------------------------
# Q1: Meteor Detection
# ----------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Load dataset
data = pd.read_csv("data/meteor_data.csv")

# Check columns
print("Columns:", data.columns)

# Change column name if needed
signal = data['signal']

# Plot signal
plt.plot(signal)
plt.title("Signal Data")
plt.xlabel("Time")
plt.ylabel("Signal Strength")
plt.show()

# Set threshold
threshold = signal.mean() + 2 * signal.std()

# Detect peaks
peaks, _ = find_peaks(signal, height=threshold)

# Output
print("Number of meteors detected:", len(peaks))

# Plot peaks
plt.plot(signal)
plt.plot(peaks, signal[peaks], "rx")
plt.title("Detected Meteors")
plt.show()
