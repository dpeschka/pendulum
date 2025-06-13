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

def residual_function(params, times, x_positions, y_positions):
    """Residual for least squares fitting"""
    length_pixels, damping, theta0, omega0, pivot_x, pivot_y = params
    try:
        # Convert to angles
        dx = np.array(x_positions) - pivot_x
        dy = np.array(y_positions) - pivot_y
        angles = np.arctan2(dx, dy)
        
        predicted_angles = solve_pendulum(times, length_pixels, damping, theta0, omega0)
        return predicted_angles - angles
    except:
        return np.full_like(x_positions, 1e6)

def fit_pendulum(times, x_positions, y_positions, pivot_x_guess=None, pivot_y_guess=None):
    """
    Fit pendulum motion to tracking data.
    
    Args:
        times: Time points [s]
        x_positions, y_positions: Pendulum bob coordinates [pixels]
        pivot_x_guess, pivot_y_guess: Initial guess for pivot (optional)
    
    Returns:
        dict: {'length_pixels': L, 'damping': γ, 'theta0': θ₀, 'omega0': ω₀, 
               'pivot_x': px, 'pivot_y': py, 'success': bool, 'residual': r}
    """
    x_arr = np.array(x_positions)
    y_arr = np.array(y_positions)
    
    # Initial guess for pivot if not provided
    if pivot_x_guess is None:
        pivot_x_guess = np.mean(x_arr)
    if pivot_y_guess is None:
        pivot_y_guess = np.min(y_arr) - 200  # assume pivot above the motion
    
    # Estimate pendulum length from range
    amplitude_pixels = np.sqrt(np.var(x_arr) + np.var(y_arr))
    length_pixels_guess = max(400, amplitude_pixels * 2)  # typical 400 pixel length
    
    # Initial guess: [length_pixels, damping, theta0, omega0, pivot_x, pivot_y]
    x0 = [length_pixels_guess, 0.1, 0.0, 0.0, pivot_x_guess, pivot_y_guess]
    
    # Bounds: length [50, 1000], damping [0, 2], angles [-π, π], omega [-10, 10], pivot ranges
    x_range = np.ptp(x_arr) + 200
    y_range = np.ptp(y_arr) + 200
    bounds = ([0.05, 0.0, -np.pi, -10.0, np.min(x_arr) - x_range, -5000.0], 
              [2.00, 5.0, np.pi, 10.0, np.max(x_arr) + x_range, 5000.0])
    
    result = least_squares(residual_function, x0, args=(times, x_positions, y_positions))#, bounds=bounds)
    
    return {
        'length_pixels': result.x[0],
        'damping': result.x[1], 
        'theta0': result.x[2],
        'omega0': result.x[3],
        'pivot_x': result.x[4],
        'pivot_y': result.x[5],
        'success': result.success,
        'residual': np.linalg.norm(result.fun)
    }

def predict_positions(times, fit_result):
    """
    Predict x,y positions from fit result.
    
    Args:
        times: Time points [s]
        fit_result: Dictionary from fit_pendulum
    
    Returns:
        tuple: (x_positions, y_positions) arrays
    """
    angles = solve_pendulum(times, fit_result['length_pixels'], fit_result['damping'], 
                           fit_result['theta0'], fit_result['omega0'])
    
    x_positions = fit_result['pivot_x'] + fit_result['length_pixels'] * np.sin(angles)
    y_positions = fit_result['pivot_y'] + fit_result['length_pixels'] * np.cos(angles)
    
    return x_positions, y_positions
