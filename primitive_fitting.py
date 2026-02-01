import torch
import numpy as np
from scipy.optimize import minimize, least_squares
from typing import Tuple, Optional, Dict

def sqrt_guard(x, tol = 1e-6):
    return (max(x, tol)) ** 0.5

def fit_plane(points: torch.Tensor):
    c = points.mean(dim = 0)
    X = points - c
    
    U, S, Vh = torch.linalg.svd(X, full_matrices = False)
    # Right eigenspace is 3x3
    # Normal vector of plane spanned by first two eigenvectors is exactly the third, reamining eigenvector!
    a = Vh[-1]
    d = a @ c

    return a, d

def fit_plane_numpy(points : np.ndarray):
    c = points.mean(axis = 0)
    X = points - c

    U, S, Vh = np.linalg.svd(X, full_matrices = False)
    a = Vh[-1]
    d = a @ c

    return a, d

def fit_sphere_numpy(points: np.ndarray, rcond : float = 1e-5, sqrt_tol : float = 1e-6):
    A = np.concatenate((2 * points, np.ones((points.shape[0], 1))), axis = 1)
    y = (points ** 2).sum(axis = 1)
    w, _, rank, _ = np.linalg.lstsq(A, y, rcond = rcond)

    center = w[:3]
    radius = w[3] + (center ** 2).sum()
    radius = sqrt_guard(radius, tol = sqrt_tol)

    return center, radius

def fit_cylinder(data, guess_angles=None):
    """Fit a list of data points to a cylinder surface. The algorithm implemented
    here is from David Eberly's paper "Fitting 3D Data with a Cylinder" from
    https://www.geometrictools.com/Documentation/CylinderFitting.pdf
    Arguments:
        data - A list of 3D data points to be fitted.
        guess_angles[0] - Guess of the theta angle of the axis direction
        guess_angles[1] - Guess of the phi angle of the axis direction

    Return:
        Direction of the cylinder axis
        A point on the cylinder axis
        Radius of the cylinder
        Fitting error (G function)
    """

    def direction(theta, phi):
        """Return the direction vector of a cylinder defined
        by the spherical coordinates theta and phi.
        """
        return np.array(
            [np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)]
        )

    def projection_matrix(w):
        """Return the projection matrix  of a direction w."""
        return np.identity(3) - np.dot(np.reshape(w, (3, 1)), np.reshape(w, (1, 3)))

    def skew_matrix(w):
        """Return the skew matrix of a direction w."""
        return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])

    def calc_A(Ys):
        """Return the matrix A from a list of Y vectors."""
        return sum(np.dot(np.reshape(Y, (3, 1)), np.reshape(Y, (1, 3))) for Y in Ys)

    def calc_A_hat(A, S):
        """Return the A_hat matrix of A given the skew matrix S"""
        return np.dot(S, np.dot(A, np.transpose(S)))

    def preprocess_data(Xs_raw):
        """Translate the center of mass (COM) of the data to the origin.
        Return the prossed data and the shift of the COM"""
        n = len(Xs_raw)
        Xs_raw_mean = sum(X for X in Xs_raw) / n

        return [X - Xs_raw_mean for X in Xs_raw], Xs_raw_mean

    def G(w, Xs):
        """Calculate the G function given a cylinder direction w and a
        list of data points Xs to be fitted."""
        n = len(Xs)
        P = projection_matrix(w)
        Ys = [np.dot(P, X) for X in Xs]
        A = calc_A(Ys)
        A_hat = calc_A_hat(A, skew_matrix(w))

        u = sum(np.dot(Y, Y) for Y in Ys) / n
        v = np.dot(A_hat, sum(np.dot(Y, Y) * Y for Y in Ys)) / np.trace(
            np.dot(A_hat, A)
        )

        return sum((np.dot(Y, Y) - u - 2 * np.dot(Y, v)) ** 2 for Y in Ys)

    def C(w, Xs):
        """Calculate the cylinder center given the cylinder direction and
        a list of data points.
        """
        n = len(Xs)
        P = projection_matrix(w)
        Ys = [np.dot(P, X) for X in Xs]
        A = calc_A(Ys)
        A_hat = calc_A_hat(A, skew_matrix(w))

        return np.dot(A_hat, sum(np.dot(Y, Y) * Y for Y in Ys)) / np.trace(
            np.dot(A_hat, A)
        )

    def r(w, Xs):
        """Calculate the radius given the cylinder direction and a list
        of data points.
        """
        n = len(Xs)
        P = projection_matrix(w)
        c = C(w, Xs)

        return np.sqrt(sum(np.dot(c - X, np.dot(P, c - X)) for X in Xs) / n)

    Xs, t = preprocess_data(data)

    # Set the start points

    start_points = [(0, 0), (np.pi / 2, 0), (np.pi / 2, np.pi / 2)]
    if guess_angles:
        start_points = guess_angles

    # Fit the cylinder from different start points

    best_fit = None
    best_score = float("inf")

    for sp in start_points:
        fitted = minimize(
            lambda x: G(direction(x[0], x[1]), Xs), sp, method = "Powell", tol = 1e-6
        )

        if fitted.fun < best_score:
            best_score = fitted.fun
            best_fit = fitted

    w = direction(best_fit.x[0], best_fit.x[1])

    return w, C(w, Xs) + t, r(w, Xs), best_fit.fun

def fit_cylinder_optimized(data, guess_angles = None, sqrt_tol : float = 1e-6):
    """Optimized vectorized cylinder fitting. Based on Eberly's least squares cylinder fitting algorithm: https://www.geometrictools.com/Documentation/LeastSquaresFitting.pdf
       As oposed to original point2cad implementation, we use vectorized expression for dramatic increase in execution time.
    """
    
    def direction(theta, phi):
        return np.array([
            np.cos(phi) * np.sin(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(theta)
        ])
    
    def preprocess_data(Xs_raw):
        """Convert to matrix form and center"""
        X = np.array(Xs_raw)  # n × 3 matrix
        X_mean = X.mean(axis=0)
        X_centered = X - X_mean
        return X_centered, X_mean
    
    def G_vectorized(w, X):
        """
        Vectorized G function computation
        X: n × 3 matrix of centered points
        w: 3D unit vector (axis direction)
        """
        n = X.shape[0]
        
        P = np.eye(3) - np.outer(w, w) # [3, 3]
        
        Y = X @ P.T # [N, 3]
        
        Y_norm_sq = np.sum(Y * Y, axis = 1)  # [N]
        
        A = Y.T @ Y # [3, 3]
        
        S = np.array([
            [0, -w[2], w[1]],
            [w[2], 0, -w[0]],
            [-w[1], w[0], 0]
        ])
        
        A_hat = S @ A @ S.T # [3, 3]
        
        trace_AA_hat = np.trace(A_hat @ A)
        u = Y_norm_sq.mean()
        
        weighted_Y = X.T @ Y_norm_sq  # [3,]
        v = A_hat @ weighted_Y / trace_AA_hat # [3,]
        
        residuals = Y_norm_sq - u - 2 * (X @ v)
        
        # Original implementation does not scale residuals by 1 / n.
        # Morhpology of the scoring function for cylinder direction vector remains the same.
        return np.sum(residuals ** 2) / n
    
    def C_vectorized(w, X):
        """Compute center (vectorized). Formula does not include scaling by 1 / n. Scaling A and weighted_y by 1 / n is equivalent to this expression."""
        n = X.shape[0]
        P = np.eye(3) - np.outer(w, w)
        Y = X @ P.T
        Y_norm_sq = np.sum(Y * Y, axis = 1)
        
        A = Y.T @ Y
        S = np.array([
            [0, -w[2], w[1]],
            [w[2], 0, -w[0]],
            [-w[1], w[0], 0]
        ])
        A_hat = S @ A @ S.T
        
        weighted_Y = X.T @ Y_norm_sq
        return A_hat @ weighted_Y / np.trace(A_hat @ A)
    
    def r_vectorized(w, X):
        """Compute radius (vectorized)"""
        P = np.eye(3) - np.outer(w, w)
        c = C_vectorized(w, X)
        d = X - c
        perp_dist_sq = np.sum(d @ P * d, axis = 1).mean()
        return sqrt_guard(perp_dist_sq, tol = sqrt_tol)

    X, t = preprocess_data(data)
    
    # Multiple initial points for better stability. Refine further perhaps...
    start_points = [(0, 0), (np.pi / 2, 0), (np.pi / 2, np.pi / 2)]
    if guess_angles:
        start_points = guess_angles

    best_fit = None
    best_score = float("inf")
    for sp in start_points:
        fitted = minimize(
            lambda angles: G_vectorized(direction(angles[0], angles[1]), X),
            sp,
            method = "Powell",
            tol = 1e-6
        )
        if fitted.fun < best_score:
            best_score = fitted.fun
            best_fit = fitted
    
    w = direction(best_fit.x[0], best_fit.x[1])
    return w, C_vectorized(w, X) + t, r_vectorized(w, X), best_fit.fun, best_fit.nit

def plane_error(points, a, d):
    assert np.allclose(np.linalg.norm(a), 1, rtol = 1e-6)

    shifted = points - d
    projections = shifted - np.dot(shifted, a) * a
    projections += a * d
    
    return np.linalg.norm(projections - points).mean()

def sphere_error(points, center, radius):
    return np.abs(np.linalg.norm(points - center) - radius).mean()

def cylinder_error(points, center, axis, radius):
    assert np.allclose(np.linalg.norm(axis), 1, rtol = 1e-6)

    shifted = points - center
    orth_projection = shifted - np.dot(shifted, axis) * axis
    orth_distance = np.linalg.norm(orth_projection)

    return np.abs(orth_distance - radius).mean()

def cone_error(points : np.ndarray, vertex: np.ndarray, axis: np.ndarray, theta: float):  
    assert np.allclose(np.linalg.norm(axis), 1, rtol = 1e-6)
     
    d = points - vertex
    t = d @ axis
    
    d_perp = d - t[:, np.newaxis] * axis
    r_perp = np.linalg.norm(d_perp, axis = 1)
    
    distances = np.abs(r_perp * np.cos(theta) - np.abs(t) * np.sin(theta))
    
    return np.mean(distances)

def cone_residuals(params: np.ndarray, points: np.ndarray) -> np.ndarray:
    # n = points.shape[0]
    theta = params[0]
    axis = params[1:4]
    vertex = params[4:7]
    
    axis_unit = axis / np.linalg.norm(axis)
    
    d = points - vertex
    M = np.cos(theta)**2 * np.eye(3) - np.outer(axis_unit, axis_unit)
    
    residuals = np.sum((d @ M) * d, axis = 1)
    
    return residuals

def cone_jacobian(params: np.ndarray, points: np.ndarray) -> np.ndarray:
    theta = params[0]
    axis = params[1:4]
    vertex = params[4:7]
    
    axis_unit = axis / np.linalg.norm(axis)
    
    n = len(points)
    J = np.zeros((n, 7))
    
    d = points - vertex
    M = np.cos(theta)**2 * np.eye(3) - np.outer(axis_unit, axis_unit)
    
    d_norm_sq = np.sum(d * d, axis = 1)
    J[:, 0] = -np.sin(2 * theta) * d_norm_sq
    
    axis_dot_d = d @ axis_unit
    J[:, 1:4] = -2 * axis_dot_d[:, np.newaxis] * d
    
    M_d = d @ M.T
    J[:, 4:7] = -2 * M_d
    
    return J

def fit_cone(points: np.ndarray, initial_guess: Optional[np.ndarray] = None) -> Dict:
    residual_fn = lambda p: cone_residuals(p, points)
    jacobian_fn = lambda p: cone_jacobian(p, points)
    
    if initial_guess is not None:
        initial_guesses = [initial_guess]
    else:
        centroid = points.mean(axis = 0)
        
        initial_guesses = [
            np.array([0.1, 1.0, 0.0, 0.0, *centroid]),
            np.array([0.1, 0.0, 1.0, 0.0, *centroid]),
            np.array([0.1, 0.0, 0.0, 1.0, *centroid])
        ]
    
    best_result = None
    best_cost = np.inf
    
    for x0 in initial_guesses:
        try:
            result = least_squares(
                residual_fn,
                x0,
                jac = jacobian_fn,
                method = "lm",
                # ftol = 1e-10,
                # xtol = 1e-10,
                # gtol = 1e-10,
                # max_nfev = 1000
            )
            
            if result.success and result.cost < best_cost:
                best_cost = result.cost
                best_result = result
        except:
            continue
    
    if best_result is None:
        return {
            "success": False,
            "vertex": None,
            "axis": None,
            "theta": None,
            "error": None
        }
    
    theta_opt = best_result.x[0]
    axis_opt = best_result.x[1:4]
    vertex_opt = best_result.x[4:7]
    
    axis_opt = axis_opt / np.linalg.norm(axis_opt)
    theta_opt = np.clip(theta_opt, 0.0, np.pi / 2)
    
    return {
        "success": True,
        "vertex": vertex_opt,
        "axis": axis_opt,
        "theta": theta_opt,
        "error": best_result.cost,
        "iterations": best_result.nfev,
        "optimality": best_result.optimality,
        "error": cone_error(points, vertex_opt, axis_opt, theta_opt)
    }