# ----------------------------------------
# Q4: Exoplanet Detection using Light Curve
# ----------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Load dataset
data = pd.read_csv("data/light_curve.csv")

# Check columns
print(data.columns)

time = data['time']
brightness = data['brightness']

# Plot light curve
plt.plot(time, brightness)
plt.title("Light Curve")
plt.xlabel("Time")
plt.ylabel("Brightness")
plt.show()

# Detect dips (negative peaks)
dips, _ = find_peaks(-brightness)

# Output
print("Number of possible exoplanets:", len(dips))

# Plot dips
plt.plot(time, brightness)
plt.plot(time[dips], brightness[dips], "rx")
plt.title("Detected Transit Signals")
plt.show()
