"""
fitting_utils.py
----------------
Simple pendulum motion fitting to damped harmonic oscillator.
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp

def pendulum_ode(t, y, length, damping, g=9.81):
    """Damped pendulum: θ'' + damping*θ' + (g/L)*sin(θ) = 0"""
    theta, theta_dot = y
    theta_ddot = -(g/length) * np.sin(theta) - damping * theta_dot
    return [theta_dot, theta_ddot]

def solve_pendulum(times, length, damping, theta0, omega0):
    """Solve pendulum ODE and return angles"""
    sol = solve_ivp(
        lambda t, y: pendulum_ode(t, y, length, damping),
        [times[0], times[-1]], 
        [theta0, omega0], 
        t_eval=times,
        method='RK45'
    )
    return sol.y[0] if sol.success else np.zeros_like(times)

def residual_function(params, times, angles):
    """Residual for least squares fitting"""
    length, damping, theta0, omega0 = params
    try:
        predicted = solve_pendulum(times, length, damping, theta0, omega0)
        return predicted - angles
    except:
        return np.full_like(angles, 1e6)

def fit_pendulum(times, x_positions, y_positions, pivot_x, pivot_y):
    """
    Fit pendulum motion to tracking data.
    
    Args:
        times: Time points [s]
        x_positions, y_positions: Pendulum bob coordinates [pixels]
        pivot_x, pivot_y: Pivot point coordinates [pixels]
    
    Returns:
        dict: {'length': L, 'damping': γ, 'theta0': θ₀, 'omega0': ω₀, 'success': bool, 'length_pixels': L_pix}
    """
    # Convert to angles
    dx = np.array(x_positions) - pivot_x
    dy = np.array(y_positions) - pivot_y
    angles = np.arctan2(dx, dy)
    
    # Estimate pendulum length in pixels from amplitude
    amplitude_pixels = np.sqrt(np.var(dx) + np.var(dy))
    length_pixels_guess = amplitude_pixels / np.std(angles)  # approximate length in pixels
    
    # Initial guess - use typical 400 pixel amplitude as reference
    x0 = [length_pixels_guess, 0.1, angles[0], 0.0]  # [length_pixels, damping, theta0, omega0]
    
    # Bounds: length [50, 1000] pixels, damping [0, 2], angles [-π, π], omega [-10, 10]
    bounds = ([50.0, 0.0, -np.pi, -10.0], [1000.0, 2.0, np.pi, 10.0])
    
    result = least_squares(residual_function, x0, args=(times, angles), bounds=bounds)
    
    return {
        'length_pixels': result.x[0],
        'damping': result.x[1], 
        'theta0': result.x[2],
        'omega0': result.x[3],
        'success': result.success,
        'residual': np.linalg.norm(result.fun)
    }

def predict_positions(times, length_pixels, damping, theta0, omega0, pivot_x, pivot_y):
    """
    Predict x,y positions from pendulum parameters.
    
    Args:
        times: Time points [s]
        length_pixels: Pendulum length [pixels]
        damping: Damping coefficient [1/s]
        theta0: Initial angle [rad]
        omega0: Initial angular velocity [rad/s]
        pivot_x, pivot_y: Pivot coordinates [pixels]
    
    Returns:
        tuple: (x_positions, y_positions) arrays
    """
    angles = solve_pendulum(times, length_pixels, damping, theta0, omega0)
    
    x_positions = pivot_x + length_pixels * np.sin(angles)
    y_positions = pivot_y + length_pixels * np.cos(angles)
    
    return x_positions, y_positions
