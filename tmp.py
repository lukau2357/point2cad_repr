import numpy as np

def rotation_z(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]])

if __name__ == "__main__":
    M = rotation_z(np.pi / 3)
    prod = M @ M.T
    print(np.allclose(prod[~np.eye(M.shape[0], dtype = np.bool)], 0))