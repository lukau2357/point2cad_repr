# Uniform Sampling on a Triangle via Barycentric Coordinates

## Setup

Let $v_0, v_1, v_2 \in \mathbb{R}^3$ be the vertices of a triangle $T$. Any point
$P \in T$ can be written as

$$P = \lambda_0 v_0 + \lambda_1 v_1 + \lambda_2 v_2, \qquad \lambda_0 + \lambda_1 + \lambda_2 = 1, \quad \lambda_i \geq 0.$$

Eliminating $\lambda_0 = 1 - \lambda_1 - \lambda_2$ gives the parameterization

$$P(\lambda_1, \lambda_2) = v_0 + \lambda_1 (v_1 - v_0) + \lambda_2 (v_2 - v_0),$$

where $(\lambda_1, \lambda_2)$ ranges over the standard 2-simplex

$$\Delta = \{(\lambda_1, \lambda_2) \in \mathbb{R}^2 : \lambda_1 \geq 0,\ \lambda_2 \geq 0,\ \lambda_1 + \lambda_2 \leq 1\}.$$

The area element on $T$ induced by this parameterization is

$$dA = \|(v_1 - v_0) \times (v_2 - v_0)\|\, d\lambda_1\, d\lambda_2 = 2\,\text{Area}(T)\, d\lambda_1\, d\lambda_2.$$

Since the prefactor $2\,\text{Area}(T)$ is constant, sampling $P$ uniformly over $T$
(with respect to surface area) is equivalent to sampling $(\lambda_1, \lambda_2)$ uniformly
over $\Delta$.

## Probability Density and the Change of Variables Formula

**Definition.** Let $X$ be a continuous random vector in $\mathbb{R}^n$. Its
*probability density function* (pdf) $f_X$ is the function satisfying

$$\Pr[X \in A] = \int_A f_X(x)\, dx$$

for every measurable set $A \subseteq \mathbb{R}^n$, where $dx$ denotes the Lebesgue
measure on $\mathbb{R}^n$. Equivalently, $f_X(x)\, dx$ is the Radon-Nikodym derivative
of the distribution of $X$ with respect to the Lebesgue measure.

**Uniform density on a region $S$.** If $X$ is uniformly distributed over a region
$S \subset \mathbb{R}^n$ of finite Lebesgue measure $|S|$, then

$$f_X(x) = \frac{1}{|S|} \mathbf{1}_S(x).$$

For the standard 2-simplex $\Delta$, we have $|\Delta| = 1/2$, so the uniform density is
$f(\lambda_1, \lambda_2) = 2$ on $\Delta$.

**Change of variables.** Let $g : U \to V$ be a $C^1$-diffeomorphism between open subsets
of $\mathbb{R}^n$. If $R = (R_1, \dots, R_n)$ has density $f_R$ on $U$, and we define
$\Lambda = g(R)$, then $\Lambda$ has density

$$f_\Lambda(\lambda) = f_R\!\left(g^{-1}(\lambda)\right) \cdot \left|\det Dg^{-1}(\lambda)\right|$$

where $Dg^{-1}$ is the Jacobian matrix of the inverse map. Equivalently, if $J_g$ denotes
the Jacobian matrix of $g$,

$$f_\Lambda(\lambda) = \frac{f_R(r)}{|\det J_g(r)|} \bigg|_{r = g^{-1}(\lambda)}.$$

This is the formula used below. The key point: **uniform density on $\Delta$ requires
$|\det J_g|$ to be constant**, because $f_R \equiv 1$ on $[0,1]^2$.

## Case 1: Without Square Root (Non-Uniform)

Define $g : [0,1]^2 \to \Delta$ by

$$\lambda_1 = r_1(1 - r_2), \qquad \lambda_2 = r_1 r_2.$$

The image satisfies $\lambda_1 + \lambda_2 = r_1 \in [0,1]$, $\lambda_1 \geq 0$,
$\lambda_2 \geq 0$, so $g$ maps into $\Delta$. The Jacobian matrix is

$$J_g = \begin{pmatrix} \partial \lambda_1 / \partial r_1 & \partial \lambda_1 / \partial r_2 \\ \partial \lambda_2 / \partial r_1 & \partial \lambda_2 / \partial r_2 \end{pmatrix} = \begin{pmatrix} 1 - r_2 & -r_1 \\ r_2 & r_1 \end{pmatrix}.$$

Its determinant is

$$\det J_g = r_1(1 - r_2) + r_1 r_2 = r_1.$$

Since $(r_1, r_2) \sim \text{Uniform}([0,1]^2)$, we have $f_R \equiv 1$. Applying the
change-of-variables formula:

$$f_\Lambda(\lambda_1, \lambda_2) = \frac{1}{|r_1|} = \frac{1}{\lambda_1 + \lambda_2}.$$

This density is **not constant** on $\Delta$. It diverges as
$\lambda_1 + \lambda_2 \to 0$, meaning the sampling concentrates near vertex $v_0$
(where $\lambda_0 \to 1$). Therefore this transformation does not produce uniform samples
on the triangle.

## Case 2: With Square Root (Uniform)

Define $g : [0,1]^2 \to \Delta$ by

$$\lambda_1 = \sqrt{r_1}\,(1 - r_2), \qquad \lambda_2 = \sqrt{r_1}\, r_2.$$

Again $\lambda_1 + \lambda_2 = \sqrt{r_1} \in [0,1]$, so the image lies in $\Delta$. The
Jacobian matrix is

$$J_g = \begin{pmatrix} \dfrac{1 - r_2}{2\sqrt{r_1}} & -\sqrt{r_1} \\[6pt] \dfrac{r_2}{2\sqrt{r_1}} & \sqrt{r_1} \end{pmatrix}.$$

Its determinant is

$$\det J_g = \frac{(1 - r_2)\sqrt{r_1}}{2\sqrt{r_1}} + \frac{r_2 \sqrt{r_1}}{2\sqrt{r_1}} = \frac{1 - r_2}{2} + \frac{r_2}{2} = \frac{1}{2}.$$

Applying the change-of-variables formula:

$$f_\Lambda(\lambda_1, \lambda_2) = \frac{1}{1/2} = 2 = \frac{1}{|\Delta|}.$$

This is the uniform density on $\Delta$. Therefore sampling with $\sqrt{r_1}$ produces
points that are uniformly distributed over the triangle.

## Why the Square Root is Necessary

The determinant $\det J_g$ measures how the transformation locally scales area. In Case 1,
$\det J_g = r_1 = \lambda_1 + \lambda_2$, which means the map compresses area near
$r_1 = 0$ (the vertex $v_0$) and stretches it near $r_1 = 1$ (the opposite edge). This
non-uniform scaling distorts the originally uniform distribution on $[0,1]^2$.

The square root in Case 2 exactly compensates: replacing $r_1$ with $\sqrt{r_1}$ transforms
the linearly growing determinant into a constant. Concretely, $\sqrt{r_1}$ is the inverse
CDF of the distribution with pdf $f(t) = 2t$ on $[0,1]$. This is precisely the marginal
distribution that $\lambda_1 + \lambda_2$ must follow for the joint density on $\Delta$ to
be uniform.

To see this directly: if $(\lambda_1, \lambda_2)$ is uniform on $\Delta$, then the marginal
density of $s = \lambda_1 + \lambda_2$ is obtained by integrating out the ratio
$\lambda_2 / s$ over $[0, s]$:

$$f_S(s) = \int_0^s 2\, d\lambda_2 = 2s, \qquad s \in [0,1].$$

The CDF is $F_S(s) = s^2$, so to generate $s$ from a uniform $r_1$, we set
$s = F_S^{-1}(r_1) = \sqrt{r_1}$.
