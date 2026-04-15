# INR Boundary Smoothness Regularization

## Problem

Extending the UV grid beyond the encoder's bounding box causes the decoder to extrapolate into unseen UV regions, producing noisy/hallucinated geometry. With `uv_margin=0` the mesh is clean; even 10% margin introduces artifacts.

## Proposed loss

For each training step, sample UV points $u_{\text{ext}}$ in the margin region (outside the current UV bounding box). For each, find the nearest point $u_{\text{bnd}}$ on the bounding box boundary. Penalize deviation from a first-order Taylor expansion:

$$\mathcal{L}_{\text{smooth}} = \left\| f(u_{\text{ext}}) - f(u_{\text{bnd}}) - J_f(u_{\text{bnd}}) \cdot (u_{\text{ext}} - u_{\text{bnd}}) \right\|$$

where $f: \mathbb{R}^2 \to \mathbb{R}^3$ is the decoder and $J_f(u_{\text{bnd}}) \in \mathbb{R}^{3 \times 2}$ is its Jacobian at the boundary point.

### Interpretation

The loss says: *the surface in the margin should be approximately a linear continuation of the surface at the boundary*. This is the mildest possible assumption — the surface just keeps going in the same direction it was heading at the edge.

### Computing the Jacobian

The Jacobian is cheap via one `torch.autograd.functional.jacobian` call, or equivalently two backward passes (one per UV coordinate):

$$J_f(u) = \begin{bmatrix} \frac{\partial f_x}{\partial u} & \frac{\partial f_x}{\partial v} \\ \frac{\partial f_y}{\partial u} & \frac{\partial f_y}{\partial v} \\ \frac{\partial f_z}{\partial u} & \frac{\partial f_z}{\partial v} \end{bmatrix}$$

In practice, for a batch of $B$ boundary points, use `torch.autograd.grad` with `create_graph=True` to keep it differentiable.

## Alternative: MLS upsampling

Instead of regularizing the loss, extend the point cloud itself before training using Moving Least Squares projection:

1. Generate candidate points beyond the cluster boundary
2. Project onto the MLS surface (smooth extrapolation)
3. Train INR on original + extended points with `uv_margin=0`

This avoids decoder extrapolation entirely. The two approaches can also be combined.
