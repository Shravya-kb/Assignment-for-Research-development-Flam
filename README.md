# Assignment-for-Research-development-Flam
flam interview assignment 
# Parametric Curve Fitting for R&D/AI Assignment

> **Creator:** Shravya K B

This repository contains the solution for the R&D/AI assignment. The goal was to find the unknown parameters ($\theta$, $M$, $X$) of a given parametric equation to best fit a provided `xy_data.csv` dataset.

---

## 1. Final Unknown Variables

After running the optimization script, the following optimal parameters were found:

* **$\theta$ (theta):** `28.11842217` degrees
* **$M$:** `0.02138895`
* **$X$:** `54.90030319`

---

## 2. Desmos Submission String

This is the required result for the submission, formatted for the Desmos calculator.

```
(\left(t\cos(0.49075905) - e^{0.02138895\left|t\right|} \cdot \sin(0.3t)\sin(0.49075905)\right) + 54.90030319, 42 + t\sin(0.49075905) + e^{0.02138895\left|t\right|} \cdot \sin(0.3t)\cos(0.49075905))
```

---

## 3. Explanation of Process and Steps

The solution was found using a numerical optimization process in Python. Here is a summary of the steps involved:

### Step 1: Data Loading & Preparation
* The `xy_data.csv` file, containing 1,500 `(x, y)` data points, was loaded into a `pandas` DataFrame.
* The parametric equations are dependent on the parameter $t$, which ranges from 6 to 60. To map the data points to $t$, a `numpy.linspace(6, 60, 1500)` vector was created. This assumes the `(x, y)` points were sampled uniformly across the range of $t$.

### Step 2: Defining the Model
* A Python function (`model_equations`) was created to implement the two parametric equations.
* This function takes a guess for the parameters ($\theta, M, X$) and the $t$-vector, and returns the *predicted* $x$ and $y$ values.
* The function also handles the conversion of $\theta$ (in degrees) to radians, as required by `numpy`'s trigonometric functions (`np.cos`, `np.sin`).

### Step 3: The Loss Function (L1 Distance)
* As per the assessment criteria, a **L1 distance** (Mean Absolute Error) loss function was implemented.
* This function calculates the model's error for a given set of parameters. It works as follows:
    1.  It calls the `model_equations` function to get the predicted `x_pred` and `y_pred` values.
    2.  It calculates the absolute difference between the predictions and the actual data: `abs(x_pred - x_data)` and `abs(y_pred - y_data)`.
    3.  It returns the sum of the mean of these errors. This single number represents the total "error" of the fit.

### Step 4: Optimization
* The core of the solution uses the `scipy.optimize.minimize` function.
* This function was given our `loss_function`, an initial guess, and the strict **bounds** specified in the problem (e.g., $0 < \theta < 50$).
* The `L-BFGS-B` method was used because it is efficient and respects these bounds.
* The optimizer's job is to systematically test combinations of ($\theta, M, X$) to find the single set of values that results in the **lowest possible L1 loss**.

### Step 5: Result Generation
* The `minimize` function returned the optimal values for $\theta$, $M$, and $X$.
* These values were then formatted into the final LaTeX string required for the Desmos submission, with the final $\theta$ value being converted back to radians.

---

## 4. Final L1 Distance (Error Score)

The final score for the L1 distance, as reported by the loss function using the optimal parameters, is:

**Final L1 Loss:** `25.24339589`

---

## 5. Submitted Code

The complete Python script used to find these results is included in this repository.

* **File:** `solve_curve_fit.py`
  
* **Dependencies:** `pandas`, `numpy`, `scipy`, `matplotlib`
