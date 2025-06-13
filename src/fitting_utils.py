"""
fitting_utils.py
----------------
Simple pendulum motion fitting with camera tilt compensation.
Fits pendulum trajectory data to damped harmonic oscillator model.
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp

def solve_pendulum(times, length, damping, theta0, omega0):
    """Solve damped pendulum ODE and return angles"""
    def ode(t, y):
        theta, theta_dot = y
        return [theta_dot, -(9.81/length) * np.sin(theta) - damping * theta_dot]
    
    sol = solve_ivp(ode, [times[0], times[-1]], [theta0, omega0], t_eval=times)
    return sol.y[0] if sol.success else np.zeros_like(times)

def fit_pendulum(times, x_positions, y_positions):
    """
    Fit pendulum motion to tracking data using two-step optimization.
    
    Returns:
        dict: Fitted parameters including length, damping, pivot position, etc.
    """
    x_arr, y_arr = np.array(x_positions), np.array(y_positions)
    
    # Step 1: Find optimal pivot point
    def pivot_cost(pivot):
        px, py = pivot
        radii = np.sqrt((x_arr - px)**2 + (y_arr - py)**2)
        return radii - np.mean(radii)  # minimize radius variance
    
    pivot_guess = [np.mean(x_arr), np.mean(y_arr) - np.std(y_arr) * 2]
    pivot_result = least_squares(pivot_cost, pivot_guess, max_nfev=1000)
    px, py = pivot_result.x
    
    # Step 2: Fit pendulum parameters with fixed pivot
    dx, dy = x_arr - px, y_arr - py
    radius = np.mean(np.sqrt(dx**2 + dy**2))
    
    # Initial parameter estimates from data
    angles_data = np.arctan2(dx, dy)
    theta0_est = angles_data[0]
    omega0_est = (angles_data[1] - angles_data[0]) / (times[1] - times[0]) if len(times) > 1 else 0.0
    
    def pendulum_cost(params):
        length, damping, theta0, omega0, tilt = params
        
        # Apply rotation to compensate for camera tilt
        cos_t, sin_t = np.cos(-tilt), np.sin(-tilt)
        dx_rot = dx * cos_t - dy * sin_t
        dy_rot = dx * sin_t + dy * cos_t
        
        # Calculate angles and compare with prediction
        angles_measured = np.arctan2(dx_rot, dy_rot)
        angles_predicted = solve_pendulum(times, length, damping, theta0, omega0)
        
        return angles_predicted - angles_measured
    
    # Optimize pendulum parameters
    x0 = [1.0, 0.1, theta0_est, omega0_est, 0.0]
    bounds = ([0.01, 0.0, -np.pi, -20.0, -np.pi/6], 
              [20.0, 5.0, np.pi, 20.0, np.pi/6])
    
    result = least_squares(pendulum_cost, x0, bounds=bounds, max_nfev=3000)
    
    return {
        'length': result.x[0],
        'damping': result.x[1], 
        'theta0': result.x[2],
        'omega0': result.x[3],
        'tilt_angle': result.x[4],
        'pivot_x': px,
        'pivot_y': py,
        'radius_pixels': radius,
        'success': pivot_result.success and result.success,
        'residual': np.linalg.norm(result.fun)
    }

def predict_positions(times, fit_result):
    """Predict pendulum positions from fitted parameters"""
    # Solve pendulum motion
    angles = solve_pendulum(times, fit_result['length'], fit_result['damping'], 
                           fit_result['theta0'], fit_result['omega0'])
    
    # Convert to cartesian coordinates
    x_pend = fit_result['radius_pixels'] * np.sin(angles)
    y_pend = fit_result['radius_pixels'] * np.cos(angles)
    
    # Apply camera tilt and translate to image coordinates
    tilt = fit_result['tilt_angle']
    cos_t, sin_t = np.cos(tilt), np.sin(tilt)
    x_rot = x_pend * cos_t - y_pend * sin_t
    y_rot = x_pend * sin_t + y_pend * cos_t
    
    return (fit_result['pivot_x'] + x_rot, fit_result['pivot_y'] + y_rot)
