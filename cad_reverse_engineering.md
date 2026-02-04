Triangle mesh:

$V = \{ v_i \in \mathbb{R}^{3}\}_{i=1}^{N}$, $E = \{ (i, j) \mid v_i, v_j \in V\}$, $F = \{(i, j, k) \mid v_i, v_j, v_k \in \mathbb{R}^3\}$. Mesh is defined as a set of vertices, edges and faces. A triangular face defines a planar sufrace patch:
$$
F(u, v) = (1 - u - v) v_i + u v_j + v v_k, u, v \geq 0, u + v \leq 1
$$

* What does open and closed surface fitting mean?
* How exactly does INR represent a surface over a given point cluster, when the latent space is 2D? "For extrapolation and latent space traversal, we encode all cluster points into the latent uv space and store the bounding box parameters along with the autoencoder. We reset the corresponding axis range for surfaces with closed dimensions to [−1, 1]. To sample the extended surface with the margin, we extend the bounding box by 10% in both dimensions and compute 3D surface points using the decoder."

## Point2CAD curve fitting
In Point2CAD, the first stage after being provided a clustering of a set of 3D points is primitive curve fitting. Firstly, they fit the following parametric models:

* Place parametrized by $(n, d)$, $n$ is the normal vector of unit norm, and $d$ is the distance to the origin. Every point $u$ from the plane must satisfy $un^{T} = d$.

* Sphere, $(c, r)$ $c$ is the center of the sphere and $r$ is it's radius.

* Cylinder, $(a, c, r)$ center $c$, unit length vector $a$ that describes the axis of the cylinder, and $r$ is the cylinder radius.

* Cone, $(v, a, \theta)$, $v$ denotes the apex point, $a$ is a unit-length vector that describes the axis of the cone, and $\theta \in (0, \frac{\pi}{2}) $ is half the angle of a cone - maximum angular deviation from vector $a$ to points on the surface of the cone. Any point $p$ from the cone surface must satisfy $\frac{\langle a, p - v \rangle}{||p - v||_2} = cos(\theta)$

Additionally, they fit an **INR** implicit neural representation surface to a given set of points. INR is an autoencoder neural network, the encoder maps 3D points into a 2D latent space, and the decoder takes these latent representations and maps them back to the original 3D space. Before we continue, let us define what an open/closed parametrization of a surface means. Consider cylindric coordinates: $(Rcos\theta, Rsin\theta, h)$. This is a 2D parametrization, $\theta \in [0, 2\pi]$ and $h \in [0, H]$. This parametrization is **closed** in $\theta$ - periodic, and **open** in $h$ - non-periodic. Our INR generates a latent parametrization of the surface, in the form of $(u, v)$ coordinates. However, we do not know in advance if the optimal surface should be open/closed in $uv$, so the authors fit all 4 possible combinations of openness/closedness in $uv$. The INR neural network accomodates for this: (https://github.com/prs-eth/point2cad/blob/main/point2cad/fitting_one_surface.py#L758, https://github.com/prs-eth/point2cad/blob/main/point2cad/fitting_one_surface.py#L772).

![no figure](./figures/inr_1.png)

Regardless of openness/closedness, the latent space ends up being a unit square $[-1, 1]^{2}$. In the code, they also talke about **lifted latent space**, those are coordiantes $(u_1, u_2, v_1, v_2)$, this is important for loss components not included in the paper, it's arguable if these components are even important - if they were important for SOTA results they would have surely added them to the paper. Regardless, the main component of the loss function is reconstruction error. There are 2 additional loss components not included in the original paper:

* **UV tightness** TODO
* **Metric learning** - We give pseudocode for computing this component:
```
cdist_3d = torch.cdist(x, x)  # Pairwise distances in 3D
cdist_uv = torch.cdist(uv_lifted, uv_lifted)  # Pairwise distances in UV

# Find K pairs that are FAR in 3D
far_pairs = topk(cdist_3d, K, largest=True)
# Find K pairs that are CLOSE in 3D  
close_pairs = topk(cdist_3d, K, largest=False)

# Ranking loss: close pairs should have smaller UV distance than far pairs
loss_metric = mean(relu(cdist_uv[close_pairs] - cdist_uv[far_pairs] + margin))
```

Basically this is a contrastive component, real points close to one another should have close UV representations, similar for dissimilar real points. Default value for the margin parameter seems to be $0.2$. The best INR is the one that achieves lowest reconstruction, and recall that there are 4 INRs being fitted to every point cluster.

So, after this procedure we obtain 5 different surfaces for a given cluster of points. For INR, the error is simply reconstruction error, for other 4 primitive surfaces, we project all real points to the interpolated surface, and we measure the distance between the given point and the projection. We take **the simplest surface with lowest error**. Now what does this mean precisely? If INR achieves the lowest error (and it usually will), we will check if the minimum error of remaining 4 surfaces is within a given error margin, if it is we take the simpler surface - otherwise we settle with INR (**slightly more logic behind the choosing procedure, but this is the main idea, elaborate futher**). (https://github.com/prs-eth/point2cad/blob/main/point2cad/fitting_one_surface.py#L96). 

Instead of using analytical representations for all surfaces, the authors instead use triangle mesh representations. Using analytical representations could potentially lead to very numeircally unstable calculations. A triangle mesh is created by first sampling points from the given surface and then creating a triangle mesh from it. In particular, for INR, we take the $uv$ bounding box for the current cluster - determined by the minimum and maximum values in both coordinates, recall that the latent space is $[-1, 1]^{2}$. Next, we extend the bounding box by 10%, and from this extended bounding box we sample sufficient number of points, for which we create a triangle mesh.

During training of the INR, they optionally add noise to encoder/decoder inputs, in order to make the network more robust to small pertubations:
```
langevin_noise_schedule = (num_fit_steps - step - 1) / (num_fit_steps - 1)
x_input = x
if langevin_noise_magnitude_3d > 0:
    x_input = x + (
        langevin_noise_magnitude_3d * langevin_noise_schedule
    ) * torch.randn_like(x)
uv = model.encoder(x_input)
uv_input = uv
if langevin_noise_magnitude_uv > 0:
    uv_input = uv + (
        langevin_noise_magnitude_uv * langevin_noise_schedule
    ) * torch.randn_like(uv)
x_hat = model.decoder(uv_input)
```

SIREN layer:
$$
\Phi(x) = sin(\omega Wx + b), x \in \mathbb{R}^{n}, X \in \mathbb{R}^{m \times n}, b \in \mathbb{R}^{m}
$$

Starting from the second hidden layer of SIREN, authors argue that the initialization of weights should be $U(-\sqrt{\frac{6}{\text{n}}}, \sqrt{\frac{6}{n}})$, in order for the matrix product to be uniformly distributed across components. However, Point2CAD INR uses only a single hidden layer, even though this extended initialization is present in the original codebase, it is not relevant. They use $U(-\frac{1}{n}, \frac{1}{n})$, without clear justification. So, according to performed ablations, this is how hidden layer of the INR should look like:
```
# Each "hidden layer" in block_type="combined"
┌─────────────────────────────────────┐
│ SIREN path (50% channels):          │
│   Linear with custom init           │
│   → multiply by ω=10                │
│   → sinc(x)                         │
├─────────────────────────────────────┤
│ "ResBlock" path (50% channels):     │
│   Linear (standard init)            │
│   → BatchNorm                       │
│   → (NO skip connection by default) │
│   → SiLU(x)                         │
└─────────────────────────────────────┘
         ↓
    Concatenate
```

SIREN official implementation, first layer initialization:

https://github.com/vsitzmann/siren/blob/master/modules.py#L630

If a skip connection is turned on, then an additional learnable weight parameter is added - controls the strength of the skip connection, the weight affects the downstream path of the network. Dimensionality of the hidden layer is 64, with evenly distributed neurons among SIREN and SiLU blocks.

## Analytical formulas for sufraces
* Scalar projection of vector $a$ to vector $b$: 

$$||a|| \cos \theta = |||a|| \frac{a^{T}b}{||a|| ||b||} = \frac{a^{T}b}{||b||}.
$$

* Vector projection of vector $a$ to vector $b$: scalar projection multiplied by $b$ itself (normed to unit length): $$\frac{a^{T}b}{||b||} \frac{b}{||b||} = \frac{a^{T}b}{||b||^{2}}b = proj_b(a)$$

* Orthogonal projection of vector $a$ to vector $b$:
$$
a - \frac{a^{T}b}{||b||^{2}}b = orth_b(a) = a - proj_b(a)
$$

It's easy to verify that $orth_b(a)^{T}b = (a - \frac{a^T{b}}{||b||^{2}}b)^{T}b = a^{T}b - \frac{b^{T}b^{T}a}{||b||^{2}}b = a^{T}b - \frac{||b||^{2}a^{T}b}{||b||^2} = a^{T}b - a^{T}b = 0$

* **Hyperplane**: describ ed by a set of points satsifying $a^{T}x = d$, where $a \in \mathbb{R}^{n}$ is the direction vector typically of unit length, and $d \in \mathbb{R}$. Only in the case of $d = 0$ can we construct a corresponding linear opeartor.

* Projecting a vector to a plane: if we do $orth_a(x)$ we will obtain a vector perpendicular to $a$, but we need to satisfy the hyperplane condition, so we add $ad$ to the orthogonal projection to satisfy the requirement $a^{T}x = d$. The final formula is:
$$
proj\_plane_{a, d}(x) = orth_a(x) + ad
$$

* Plane reconstruction error: $||proj\_plane_{a, d}(x) - x||_2^{2}$. For a set of ground trugh points take the mean over this expression.

* Hypersphere: $\sum_{i=1}^{N}(x_i - a_i)^{2} = r$, where $r > 0$ is the radius parameter, and $a \in \mathbb{R}^{n}$ is the center point.

* Hypersphere error: $|||x - a|| - r|$. Essentially, we take the difference vector between the center and the given point, and measure it's deviation from the radius, ignoring direction.

* Cylinder. Characterized by $(a, c, r)$, center point $c$, unit length vector $a$ that defines the axis of the cylinder, and radius parameter $r > 0$. Point $x$ lies on the cylinder if and only if:
$$
(x - c)^{T}(I - aa^{T})(x - c) = r^2
$$

Some elaborations regarding the previous equation. For a plane centered at 0 ($d = 0$), $I - aa^{T}$ is the plane projection operator, easily verifiable. First subtract the center point from the input $x$, and then project it to the plane with normal vector $a$ centered at $c$, this is $(I - aa^{T})(x - c)$ part of the expression. Next we measure the norm of this orthogonal component, it must be equal to $r$. Using the symmetry and idempotency of plane projection operator, the squared norm of $(I - aa^{T})(x - c)$ turns out to be $(x - c)^{T}(I - aa^{T})(x - c)$. This must be equal to $r^2$ - recall the squaring part.

* Cylinder error: It's simply the absolute deviation of $(I - aa^{T})(x - c)$ from $r$. Note that we do not look at the squared norm, that is why the square in r is removed. Mean of this is taken over a set of points, if multiple points are present. For least-squares cylinder fitting we observe the square of the deviation, for simply measuring the fit error we take the absolute value.

* Cone can be parametrized by $(v, a, r)$, where $v$ is the apex point, $a$ is the unit vector that represents the axis of the cone, and $\theta \in (0, \frac{\pi}{2})$ represents the half-angle of a cone. Membership condition is the following:
$$
a^{T}(\frac{x - v}{||x - v||}) = \cos \theta
$$
or equivalenty, as a quadratic form:
$$
(x - v)^{T}(\cos^2 \theta I - aa^{T})(x - v) = 0
$$

Geometrical interpretation of this would be that point $x$ lies on the given cone if and only if the angle between the axis vector and the vector pointing from the apex to the given point is $\theta$. 

* Cone error. We want to measure how well our cone fits the data, disregarding the least-squares expression for cone fitting with LM or other algorithm. Let $x_i$ be the point for which we want to measure the error. We observe the vector $z_i = x_i - v$, it's scalar projection to the axis is $h_i = a^{T}z_i$, while the orthogonal projection is $r_i = z_i - h_ia$. We observe the right triangle with catheti $||r_i||$ and $|h_i|$, the hypothenuse being $||z_i||$. If $x_i$ is actually on the cone, then we would have:
$$
\tan \theta = \frac{||r_i||}{|h_i|} \Rightarrow |h_i| \sin \theta = ||r_i|| \cos \theta
$$

If a point does not belong to the cone, we simply measure the deviation between these two quantities:
$$
||h_i| \sin \theta  - ||r_i|| \cos \theta|
$$

and average over a set of points, if a set of points is given.

* In general, all surfaces in $R^{3}$ can be written as:
$$
x^{T}Qx + a^{T}x + b
$$

## Flaws/bugs in Point2CAD implementation. Differences from the original implementation
* For SIREN layers, the use $sinc(x) = \frac{sin(\pi x)}{\pi x}$ activation, instead of ordinary sine. SIREN paper precisely derived modified initialization for linear layers, this initialization is not theoretically justified for $sinc$, yet they still use it in Point2CAD implementation. (https://github.com/prs-eth/point2cad/blob/main/point2cad/layers.py#L77)

* Further experiments with cone fitting, is current mathematical setup the best? The official implementation uses actual distances with double absolute values, and they use LM optimization algorithm that does not support giving bounds for the half-angle that should lie in $(0, \pi / 2)$. The TRF algorithm seemingly gave better results, and it does support bounds. Investigate mathematical details behind both algorithms, and further think about cone fitting intricacies!

* Default learning rate for INR is 1e-1, we will try with slightly lower learning rate 1e-2. Use as mutch GPU memory as possible, in our implementation resolved by `automatic_batch_size` function in inr.py. 

* Default noise magnitude for both 3D and UV seems to be 0.005: https://github.com/prs-eth/point2cad/blob/81e15bfa952aee62cf06cdf4b0897c552fe4fb3a/point2cad/fitting_one_surface.py#L313

* Our INR implementation vs actual INR implementation:
![no figure](./figures/ours_vs_theirs_inr.png)

* Error threshold for simple sufraces, kept the same for current implementation. Instead of additive error threshold, look at ratio between INR error and simple error. Instead of absolute error threshold for fixing degenerate cones, look at ratio between plane error and cone error. Mention that unlike in the original implementation, we sidestep INR fitting unless it is absolutely necessary.

* Point2CAD parallelism: each cluster is fitted in parallel, but 4 INR are fitted sequentially on a single process. Perhaps paralellize this operation as well? Relevant code lines: https://github.com/prs-eth/point2cad/blob/81e15bfa952aee62cf06cdf4b0897c552fe4fb3a/point2cad/main.py#L14, https://github.com/prs-eth/point2cad/blob/main/point2cad/fitting_one_surface.py#L17

* Paper sent by the professor does not containt Point2CAD as a reference, which is a bit concerning. I found one paper way back that goes from meshes to NURBS - https://ieeexplore.ieee.org/document/10824954/

## Misc
* Outer product quadratic form partial derivative wrt. outer product vector (derived by hand):
$$
\frac{\partial}{\partial w} y^{T}(ww^{T})y = 2(y^{T}w)y
$$

* Derivative of unit length normalization (derived by hand):
$$
\frac{\partial}{\partial w} (\frac{w}{||w||}) = \frac{1}{||w||}(I - \hat{w} \hat{w}^{T}), \hat{w} = \frac{w}{||w||}
$$

* General chain rule for vector differentiation (write it out correctly!):
$$
f : \mathbb{R}^{n} \rightarrow \mathbb{R}^{m}, g : \mathbb{R}^{m} \rightarrow \mathbb{R}^{k}, h(x) = g(f(x)), \text{ then}:

\frac{\partial h}{\partial x} = 
$$

* Parameters of the INR network used by Point2CAD: hidden_dim = 64, fraction_siren = 0.5, use_shortcut = False. 4 networks are trained in parallel using ADAM for 1000 steps, with 50 warmup steps.

* Native cylinder fitting and our cylinder fitting can differ only in the direction of the axis vector, only as sever as $a = -a$, but this is not important. That is why in some logs for error measurements, we see that $||a - a'|| = ||2a|| = 2 ||a|| = 2$, the L2 error between fitted axis vectrors is 2.

* A useful blog that explains how sphere fitting can be linearized. This implementation uses the same algorithm, write it down mathematically in the master thesis report.
https://jekel.me/2015/Least-Squares-Sphere-Fit/