"""
fitting_utils.py
----------------
Simple pendulum motion fitting with camera tilt compensation.
Fits pendulum trajectory data to damped harmonic oscillator model.
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp

def solve_pendulum(ode_func,times, length, damping, theta0, omega0, tilt=0.0):
    """Solve damped pendulum ODE and return angles"""
    def ode(t, y):
        theta, theta_dot = y
        dydt = ode_func(t, y, length, damping, tilt)
        return dydt

    #def ode(t, y):
    #    theta, theta_dot = y
    #    return [theta_dot, -(9.81/length) * np.sin(theta - tilt) - damping * theta_dot]
    
    sol = solve_ivp(ode, [times[0], times[-1]], [theta0, omega0], t_eval=times)
    return sol.y[0] if sol.success else np.zeros_like(times)

def fit_pendulum(ode_func,times, x_positions, y_positions, initial_values=None):
    """
    Fit pendulum motion to tracking data using two-step optimization.
    
    Args:
        times: Time array
        x_positions: X coordinates of pendulum bob
        y_positions: Y coordinates of pendulum bob
        initial_values: Initial guess for parameters in format 
                        {'length': float, 'damping': float, 'theta0': float, 'omega0': float, 'tilt': float}
    
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
    
    if initial_values is None:
        initial_values = {
            'length': 1.0,
            'damping': 0.1,
            'theta0': theta0_est,
            'omega0': omega0_est,
            'tilt': 0.0
        }
    
    # Convert initial values to array format for least_squares
    x0 = [initial_values['length'], initial_values['damping'], initial_values['theta0'], initial_values['omega0'], initial_values['tilt']]
    
    def pendulum_cost(params):
        length, damping, theta0, omega0, tilt = params
        
        # Calculate angles directly without coordinate rotation
        angles_measured = np.arctan2(dx, dy)
        angles_predicted = solve_pendulum(ode_func,times, length, damping, theta0, omega0, tilt)
        
        return angles_predicted - angles_measured
    
    # Optimize pendulum parameters
    bounds = ([0.01, 0.0, -np.pi, -20.0, -np.pi/6], 
              [20.0, 5.0, np.pi, 20.0, np.pi/6])
    
    result = least_squares(pendulum_cost, x0, bounds=bounds, max_nfev=3000)
    
    return {
        'length': result.x[0],
        'damping': result.x[1], 
        'theta0': result.x[2],
        'omega0': result.x[3],
        'tilt': result.x[4],
        'pivot_x': px,
        'pivot_y': py,
        'radius_pixels': radius,
        'success': pivot_result.success and result.success,
        'residual': np.linalg.norm(result.fun)
    }

def predict_positions(ode_func,times, fit_result):
    """Predict pendulum positions from fitted parameters"""
    # Solve pendulum motion with tilt
    angles = solve_pendulum(ode_func,times, fit_result['length'], fit_result['damping'], 
                           fit_result['theta0'], fit_result['omega0'], fit_result['tilt'])
    
    # Convert to cartesian coordinates
    x_pend = fit_result['radius_pixels'] * np.sin(angles)
    y_pend = fit_result['radius_pixels'] * np.cos(angles)
    
    # Translate to image coordinates (no rotation needed)
    return (fit_result['pivot_x'] + x_pend, fit_result['pivot_y'] + y_pend)
