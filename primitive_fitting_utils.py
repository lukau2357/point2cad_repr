import numpy as np

def plane_error(points, a, d):
    assert np.allclose(np.linalg.norm(a), 1, rtol = 1e-6)

    plane_projections = points - np.dot(points, a)[:, np.newaxis] * a
    plane_projections += a * d
    return np.linalg.norm(plane_projections - points, axis = 1).mean()

def sphere_error(points, center, radius):
    return np.abs(np.linalg.norm(points - center, axis = 1) - radius).mean()

def cylinder_error(points, center, axis, radius):
    assert np.allclose(np.linalg.norm(axis), 1, rtol = 1e-6)

    shifted = points - center
    plane_projection = shifted - np.dot(shifted, axis)[:, np.newaxis] * axis
    orth_distance = np.linalg.norm(plane_projection, axis = 1)
    return np.abs(orth_distance - radius).mean()

def cone_error(points : np.ndarray, vertex: np.ndarray, axis: np.ndarray, theta: float):  
    assert np.allclose(np.linalg.norm(axis), 1, rtol = 1e-6)
     
    z = points - vertex
    scalar_projections = z @ axis
    h = np.abs(scalar_projections)
    r = z - scalar_projections[:, np.newaxis] * axis
    r = np.linalg.norm(r, axis = 1)
    
    errors = np.abs(h * np.sin(theta) - r * np.cos(theta))
    return np.mean(errors)