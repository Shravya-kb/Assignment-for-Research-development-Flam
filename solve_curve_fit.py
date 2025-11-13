
import pandas as pd
import numpy as np
import io
from scipy.optimize import minimize

# --- Helper Functions ---

# == STEP 1: DEFINE THE PARAMETRIC MODEL ==
def model_equations(params, t):
    """
    Calculates the x and y values from the parametric equations.
    params: A list or array [theta_deg, M, X]
    t: A NumPy array of t-values
    """
    theta_deg, M, X = params
    
    # Convert degrees to radians for numpy functions
    theta_rad = np.radians(theta_deg)
    
    t_abs = np.abs(t) # |t|
    sin_0_3t = np.sin(0.3 * t)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    
    # This term appears in both equations: e^(M*|t|) * sin(0.3t)
    exp_term = np.exp(M * t_abs) * sin_0_3t
    
    # Calculate x(t) and y(t)
    # x = (t * cos(theta) - e^(M*|t|) * sin(0.3t) * sin(theta)) + X
    x_pred = (t * cos_theta - exp_term * sin_theta) + X
    
    # y = (42 + t * sin(theta) + e^(M*|t|) * sin(0.3t) * cos(theta))
    y_pred = (42 + t * sin_theta + exp_term * cos_theta)
    
    return x_pred, y_pred

# == STEP 2: DEFINE THE LOSS FUNCTION (L1 DISTANCE / MAE) ==
def loss_function(params, t_data, x_data, y_data):
    """
    Calculates the total L1 distance (Mean Absolute Error) between
    predicted data and actual data.
    """
    try:
        x_pred, y_pred = model_equations(params, t_data)
        
        # Check for NaNs or Infs, which can happen with bad `M` values
        if not np.all(np.isfinite(x_pred)) or not np.all(np.isfinite(y_pred)):
            return 1e12 # Return a very large error

        mae_x = np.mean(np.abs(x_pred - x_data))
        mae_y = np.mean(np.abs(y_pred - y_data))
        
        total_loss = mae_x + mae_y
        
        if not np.isfinite(total_loss):
            return 1e12
            
        return total_loss
    except Exception as e:
        # Catch any other math errors (e.g., overflow)
        return 1e12 # Penalize errors heavily

# --- Main Execution ---

print("--- Starting Assignment Solver ---")

try:
    # == STEP 3: LOAD AND PREPARE DATA ==
    # This file name must match the file you uploaded to Colab
    file_name = "xy_data.csv"
    df = pd.read_csv(file_name)
    
    print(f"Successfully loaded {file_name}.")
    print("Data Head:")
    print(df.head())
    
    # Extract data into NumPy arrays for speed
    x_data = df['x'].values
    y_data = df['y'].values
    N = len(x_data)

    if N == 0:
        raise ValueError("No data found in the CSV.")

    # Create the corresponding 't' vector
    t_vector = np.linspace(6, 60, N)
    print(f"\nLoaded {N} data points. Created 't' vector from 6 to 60.")

    # == STEP 4: RUN THE OPTIMIZATION ==
    print("\nStarting optimization... This may take a few seconds.")
    
    # Initial Guess: [theta_deg, M, X]
    initial_guess = [25.0, 0.0, 50.0]
    
    # Bounds: The strict ranges given in the problem
    bounds = [
        (0.01, 49.99),  # 0 < theta < 50 deg
        (-0.049, 0.049), # -0.05 < M < 0.05
        (0.01, 99.99)   # 0 < X < 100
    ]

    # Run the minimizer
    result = minimize(
        loss_function,
        initial_guess,
        args=(t_vector, x_data, y_data),
        method='L-BFGS-B',
        bounds=bounds,
        options={'disp': True, 'maxiter': 500, 'ftol': 1e-10}
    )

    print("\n--- Optimization Complete! ---")

    if result.success:
        theta_final, M_final, X_final = result.x
        final_loss = result.fun

        print("\n Found optimal parameters:")
        print(f"  theta (θ) = {theta_final:.8f} degrees")
        print(f"      M     = {M_final:.8f}")
        print(f"      X     = {X_final:.8f}")
        print(f"\n  Final L1 Loss (Error): {final_loss:.8f}")

        # == STEP 5: GENERATE THE FINAL SUBMISSION STRING ==
        # Convert final theta from degrees to radians for the string
        theta_rad_final = np.radians(theta_final)
        
        print("\n--- Generating Submission String ---")
        
        # Build the (x, y) tuple for Desmos
        latex_x = f"\\left(t\\cos({theta_rad_final:.8f}) - e^{{{M_final:.8f}\\left|t\\right|}} \\cdot \\sin(0.3t)\\sin({theta_rad_final:.8f})\\right) + {X_final:.8f}"
        latex_y = f"42 + t\\sin({theta_rad_final:.8f}) + e^{{{M_final:.8f}\\left|t\\right|}} \\cdot \\sin(0.3t)\\cos({theta_rad_final:.8f})"
        
        final_submission_string = f"({latex_x}, {latex_y})"
        
        print("\nHere is the final string for README:")
        print("=" * 80)
        print(final_submission_string)
        print("=" * 80)

    else:
        print("\n Optimization FAILED.")
        print("Message:", result.message)

except FileNotFoundError:
    print(f"\n*** ERROR: FileNotFoundError ***")
    print(f"Could not find the file: '{file_name}'")
    print("Please make sure you have uploaded the file to your Colab session.")
    print("Use the 'Files' icon in the left sidebar to upload.")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")
