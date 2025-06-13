"""
test_fit.py - Simple test for pendulum fitting
"""

import numpy as np
import matplotlib.pyplot as plt
from fitting_utils import fit_pendulum, predict_positions

# Load data
data = np.load('../data/pendulum_data.npz')
t, x, y = data['t'], data['x'], data['y']

# Fit and predict
fit_result = fit_pendulum(t, x, y)
x_pred, y_pred = predict_positions(t, fit_result)

# Parameters
print(f"L={fit_result['length']:.3f}m, d={fit_result['damping']:.3f}, tilt={np.degrees(fit_result['tilt_angle']):.1f}°")

# Plot
plt.figure(figsize=(10, 4))
plt.plot(t, x, 'b-', label='X data')
plt.plot(t, x_pred, 'r--', label='X fit')
plt.plot(t, y, 'g-', label='Y data') 
plt.plot(t, y_pred, 'm--', label='Y fit')
plt.xlabel('Time [s]')
plt.ylabel('Position [pixels]')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()