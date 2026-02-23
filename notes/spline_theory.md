# B-Spline and NURBS Theory

## 1. The Problem with Polynomial Interpolation

Given n distinct points on a plane (or in 3D), there exists a unique polynomial of degree
n - 1 passing through all of them. This is the classical **Lagrange interpolation** result.

For n points $(t_1, y_1), ..., (t_n, y_n)$ with distinct $t_i$, the interpolating polynomial is:

$$
P(t) = \sum_{i=1}^{n} y_i l_i(t), l_i(t) = \prod_{j=1, j \neq i}^{n} \frac{t - t_{j}}{t_{i} - t_{j}}
$$

This polynomial is unique and passes through every data point exactly. So why not use it?

**Runge's phenomenon**: As the degree grows, the polynomial develops increasingly violent
oscillations between the interpolation points, especially near the boundaries of the domain.
For example, interpolating $f(x) = 1 / (1 + 25x^2)$ on $[-1, 1]$ with $n$ equispaced points produces a degree $n-1$ polynomial that diverges catastrophically as $n$ increases. The
interpolant passes through all $n$ points but swings wildly between them.

**No local control**: Moving a single data point changes the *entire* polynomial. Every
coefficient depends on every data point. This makes interactive editing impossible and
amplifies numerical instability.

**Numerical conditioning**: The Vandermonde matrix for high-degree polynomial
interpolation becomes extremely ill-conditioned. Small perturbations in the data produce
large changes in the polynomial coefficients.

These problems are fundamental to *global* polynomial representations and motivated the
development of *piecewise* polynomial methods.

## 2. Piecewise Polynomials: The Naive Approach

The natural fix for Runge's phenomenon: instead of one high-degree polynomial, use many
low-degree polynomials joined at designated points called **knots**.

Consider approximating a function on $[a, b]$ using $K$ intervals defined by knots
$a = t_0 \leq t_1 \leq \cdots \leq t_K = b$. On each interval $[t_{k}, t_{k+1})$, define a separate
polynomial of degree $p$:

$$
s(u) = q_k(u) \quad \text{for } u \in [t_k, t_{k+1}), \quad k = 0, \ldots, K-1
$$

Each $q_k$ has $p + 1$ coefficients, giving $K(p + 1)$ unknowns total. To ensure smoothness,
we impose continuity conditions at each interior knot $t_k$ ($k = 1, \ldots, K-1$):

$$
q_{k-1}^{(j)}(t_k) = q_k^{(j)}(t_k), \quad j = 0, 1, \ldots, r
$$

where $r$ is the desired order of continuity. For $C^{p-1}$ continuity (the maximum possible
for degree $p$), this gives $p$ constraints per interior knot.

**Counting degrees of freedom** for $C^{p-1}$ continuity:

$$
\dim = K(p + 1) - (K - 1) \cdot p = K + p
$$

For cubic splines ($p = 3$) with $K = 10$ intervals: $\dim = 13$ free parameters, regardless
of the number of data points. The degree stays at 3 even for millions of data points ---
only the number of pieces grows.

**The problem with this direct representation:**

While piecewise polynomials solve Runge's phenomenon, specifying them directly by
their per-piece coefficients $\{a_{k,0}, a_{k,1}, \ldots, a_{k,p}\}_{k=0}^{K-1}$ has serious drawbacks:

1. **No locality**: The continuity constraints couple adjacent pieces. Changing the shape
   in one region requires re-solving the constraints globally, propagating changes to
   distant pieces.

2. **Redundant parameterization**: We have $K(p+1)$ raw coefficients but only $K + p$
   degrees of freedom. The continuity constraints must be enforced explicitly, either
   by elimination (complicated bookkeeping) or by constrained optimization (expensive).

3. **No geometric intuition**: The coefficients $a_{k,j}$ of each polynomial piece have no
   direct geometric meaning. You cannot look at the coefficients and understand the
   shape, nor can you edit the shape by moving intuitive control handles.

This motivates the search for a **basis** of the spline space: a set of $K + p$ functions
$\{B_i(u)\}$ such that every spline can be written as $s(u) = \sum_i c_i B_i(u)$, where the
coefficients $c_i$ have geometric meaning and local influence.

## 3. The Spline Space, Smoothness Vector, and Dimension

**Definition.** Let $p \geq 0$ be the polynomial degree. Let
$\mathbf{t} = (t_0, t_1, \ldots, t_K)$ be a sequence of $K + 1$ **distinct** knot positions with
$t_0 < t_1 < \cdots < t_K$. Let $\mathbf{r} = (r_1, r_2, \ldots, r_{K-1})$ be a **smoothness
vector**, where $r_k \in \{0, 1, \ldots, p-1\}$ specifies the required continuity class
$C^{r_k}$ at interior knot $t_k$.

The **spline space** $\mathbb{S}_{p, \mathbf{t}, \mathbf{r}}$ is the set of all functions
$s: [t_0, t_K] \to \mathbb{R}$ such that:

1. On each interval $[t_k, t_{k+1})$, $s$ is a polynomial of degree $\leq p$.
2. At each interior knot $t_k$ ($k = 1, \ldots, K-1$), $s$ is $C^{r_k}$ continuous,
   i.e., the left and right polynomial pieces agree in derivatives of order $0, 1, \ldots, r_k$.

The maximal smoothness is $r_k = p - 1$, which imposes $C^{p-1}$
continuity.

#### Knot multiplicity as an encoding of smoothness

In the B-spline representation, the smoothness at each knot is not specified by a
separate vector $\mathbf{r}$ but is instead encoded into the knot vector through **repeated
entries**. The **multiplicity** $m_k$ of a knot is the number of times its value appears in
the B-spline knot vector, and it relates to the smoothness by:

$$
m_k = p - r_k
$$

or equivalently $r_k = p - m_k$. Higher multiplicity = lower smoothness:

| Smoothness $r_k$ | Multiplicity $m_k = p - r_k$ | Continuity class | Meaning |
|---|---|---|---|
| $p - 1$ | $1$ (simple knot) | $C^{p-1}$ | Maximal smoothness |
| $p - 2$ | $2$ (double knot) | $C^{p-2}$ | One derivative less |
| $\vdots$ | $\vdots$ | $\vdots$ | $\vdots$ |
| $1$ | $p - 1$ | $C^{1}$ | Tangent continuous |
| $0$ | $p$ | $C^{0}$ | Positional continuity only (corner) |

For example, the B-spline knot vector $U = [0, 0, 0, 0, 1, 2, 2, 3, 4, 4, 4, 4]$
with $p = 3$ encodes:
- $\xi_1 = 1$: $m_1 = 1$, so $r_1 = 3 - 1 = 2$ ($C^2$, smooth)
- $\xi_2 = 2$: $m_2 = 2$, so $r_2 = 3 - 2 = 1$ ($C^1$, tangent continuous but curvature can jump)
- $\xi_3 = 3$: $m_3 = 1$, so $r_3 = 3 - 1 = 2$ ($C^2$, smooth)
- Boundary knots $0$ and $4$: $m = 4 = p + 1$, forcing endpoint interpolation

#### Effect on the curve (cubic, $p = 3$)

- $r_k = 2$ ($m = 1$, simple knot): value, first and second derivatives continuous.
  The curve passes smoothly through the knot with no visible join.

- $r_k = 1$ ($m = 2$, double knot): value and first derivative continuous, but second
  derivative can jump. Tangent-continuous, but curvature changes abruptly.

- $r_k = 0$ ($m = 3$, triple knot): only the value is continuous. The curve has a sharp
  **corner** (tangent direction changes abruptly). The curve passes directly through the
  corresponding control point.

- $r_k = -1$ ($m = 4$, quadruple knot): the curve is **discontinuous** — it breaks into
  two separate pieces.

> **Important.** The four cases above apply exclusively to **interior** knots,
> where two polynomial pieces meet and continuity between them is well-defined.
> At the **boundary** knots $t_0$ and $t_{n+p+1}$ there is only one adjacent
> piece, so the concept of $C^r$ continuity between two pieces does not apply.
> Multiplicity $p+1$ at a boundary knot does **not** produce a discontinuity;
> instead it forces **endpoint interpolation** — see Section 5 (Clamped knot
> vectors) for the proof.

#### Dimension of the spline space

**Theorem.** Let $p \geq 0$. Let $\mathbf{t} = (t_0, t_1, \ldots, t_K)$ with
$t_0 < t_1 < \cdots < t_K$ and smoothness vector
$\mathbf{r} = (r_1, \ldots, r_{K-1})$ with $r_k \in \{0, \ldots, p-1\}$. Then:

$$
\dim \mathbb{S}_{p, \mathbf{t}, \mathbf{r}} = (p + 1) + \sum_{k=1}^{K-1} (p - r_k)
$$

*Proof.* The $K$ distinct knots define $K$ intervals
$[t_0, t_1), [t_1, t_2), \ldots, [t_{K-1}, t_K]$. On each interval, $s$ is a polynomial of
degree $\leq p$ with $p + 1$ coefficients. The unconstrained dimension is $K(p + 1)$.

At interior knot $t_k$, the spline transitions from polynomial piece $q_{k-1}$
(defined on $[t_{k-1}, t_k)$) to piece $q_k$ (defined on $[t_k, t_{k+1})$). The
$C^{r_k}$ condition requires:

$$
q_{k-1}^{(j)}(t_k) = q_k^{(j)}(t_k), \quad j = 0, 1, \ldots, r_k
$$

That is, the $j$-th derivative of the left polynomial piece $q_{k-1}$ and the $j$-th
derivative of the right polynomial piece $q_k$, both evaluated at the knot $t_k$,
must be equal. This imposes $r_k + 1$ independent linear constraints (each involves
only the coefficients of the two adjacent pieces $q_{k-1}$ and $q_k$). Summing over all interior knots:

$$
\dim \mathbb{S}_{p, \mathbf{t}, \mathbf{r}} = K(p + 1) - \sum_{k=1}^{K-1} (r_k + 1)
$$

Separating the contribution of the first interval from the remaining $K - 1$:

$$
= (p+1) + (K-1)(p+1) - \sum_{k=1}^{K-1}(r_k + 1)
$$

$$
= (p+1) + \sum_{k=1}^{K-1}\bigl[(p+1) - (r_k + 1)\bigr]
$$

$$
= (p+1) + \sum_{k=1}^{K-1}(p - r_k) \qquad \blacksquare
$$

**Corollary (B-spline knot vector form).** Defining $m_k = p - r_k$, the dimension
becomes $(p + 1) + \sum_{k=1}^{K-1} m_k$. A clamped B-spline knot vector encodes this
configuration by repeating each interior knot $t_k$ exactly $m_k$ times, and repeating
each boundary knot $p + 1$ times. The total number of knot entries is:

$$
N = (p + 1) + \sum_{k=1}^{K-1} m_k + (p + 1)
$$

Therefore:

$$
\dim \mathbb{S}_{p, \mathbf{t}, \mathbf{r}} = N - p - 1
$$

A B-spline knot vector with $N$ entries defines $N - p - 1$ basis functions of degree $p$. If we give the degree $p$ and the number of basis functions $n + 1$ in advance, then the required number of knots becomes:
$$
N - p - 1 = n + 1 \Rightarrow N = n + p + 2
$$ 

This parametrization is more common in practice - we give in advance the desired degree and number of basis functions, from which the solvers typically infer the required number of konts.

**Example.** Cubic ($p = 3$), $\mathbf{t} = (0, 1, 2, 3, 4)$,
$\mathbf{r} = (2, 1, 2)$:
- At $t_1 = 1$: $r_1 = 2$ ($C^2$, smooth), multiplicity $m_1 = 3 - 2 = 1$.
- At $t_2 = 2$: $r_2 = 1$ ($C^1$, tangent continuous), multiplicity $m_2 = 3 - 1 = 2$.
- At $t_3 = 3$: $r_3 = 2$ ($C^2$, smooth), multiplicity $m_3 = 3 - 2 = 1$.
- $\dim = 4 + (1 + 2 + 1) = 8$ control points.
- Clamped knot vector: $U = [0, 0, 0, 0, 1, 2, 2, 3, 4, 4, 4, 4]$,
  $N = 12$, and $12 - 3 - 1 = 8$. Consistent.

Replacing $r_2 = 1$ with $r_2 = 2$ (making all interior knots maximally smooth) gives
$\dim = 4 + (1 + 1 + 1) = 7$. The reduction in smoothness at $t_2$ added one degree
of freedom.

## 4. Why Basis Functions? Truncated Powers vs. B-Splines

Any $(K + p)$-dimensional vector space can be spanned by many different bases. The choice
of basis determines the numerical and practical properties of the representation.

**Truncated power basis.** A mathematically natural basis for $\mathbb{S}_{p, \mathbf{t}}$ with simple
interior knots $t_1 < \cdots < t_{K-1}$ is:

$$
\{1, u, u^2, \ldots, u^p, (u - t_1)_+^p, (u - t_2)_+^p, \ldots, (u - t_{K-1})_+^p\}
$$

where $(u - t_k)_+^p = \max(0, u - t_k)^p$ is the **truncated power function**. This basis has
exactly $K + p$ elements and spans $\mathbb{S}_{p, \mathbf{t}}$.

The problem: truncated power functions have **global support**. Each $(u - t_k)_+^p$ is
nonzero on the entire interval $[t_k, t_K]$. This means:
- The coefficient matrix for fitting is dense and ill-conditioned (condition number
  grows exponentially with the number of knots).
- Changing one coefficient affects the spline over a large region.
- Numerical cancellation renders the basis unusable for practical computation with
  more than a handful of knots.

**B-spline basis.** The B-spline basis functions $\{N_{i,p}\}$ span the same space
$\mathbb{S}_{p, \mathbf{t}}$ but with fundamentally better properties:

| Property | Truncated powers | B-splines |
|---|---|---|
| Support | Global: $[t_0, t_K]$ | Local: $[u_i, u_{i+p+1})$ |
| Sign | Alternating | Non-negative |
| Conditioning | Exponentially ill-conditioned | Well-conditioned (banded) |
| Geometric meaning | None | Control points define a control polygon |
| Editing locality | Global | Local: moving $P_i$ affects only $[u_i, u_{i+p+1})$ |

The B-spline basis is *the* standard basis for splines in all practical applications. It was
introduced by Isaac Schoenberg (1946) and made computationally practical by Carl de Boor
and Maurice Cox (1972) through the recursive evaluation formula.

## 5. B-Spline Basis Functions

**Definition** (Cox-de Boor recursion):

Given a **knot vector** $U = [u_0, u_1, \ldots, u_{n+p+1}]$ (a non-decreasing sequence of
$n + p + 2$ real numbers), the $n + 1$ B-spline basis functions of degree $p$ are defined
recursively:

**Degree 0** (piecewise constant):

$$
N_{i, 0}(u) = \begin{cases} 1, & u_i \leq u < u_{i + 1} \\
0, & \text{otherwise}
\end{cases}
$$

**Degree $p$** (recursive):

$$
N_{i, p}(u) = \frac{u - u_i}{u_{i + p} - u_i} N_{i, p - 1}(u) + \frac{u_{i + p + 1} - u}{u_{i + p + 1} - u_{i + 1}} N_{i + 1, p - 1}(u)
$$

with the convention $0 / 0 = 0$.

**Relationship between knot vector size and basis function count**: A knot vector with
$n + p + 2$ entries defines exactly $n + 1$ basis functions $N_{0,p}, N_{1,p}, \ldots, N_{n,p}$.
Equivalently: $\text{(number of basis functions)} = \text{(number of knots)} - p - 1$.

**Key properties:**

1. **Non-negativity**: $N_{i,p}(u) \geq 0$ for all $u$. (easy to verify by induction)

2. **Compact support**: $N_{i,p}(u) \neq 0$ only on $[u_i, u_{i+p+1})$. Each basis function is "local" --- it influences only $p + 1$ knot spans. (easy to verify with induction, using the previous property also)

3. **Partition of unity**: $\sum_{i=0}^{n} N_{i,p}(u) = 1$ for all $u$ in the domain $[a, b)$.

   *Proof by induction on $p$, for a clamped knot vector.*

   **Base case** ($p = 0$): For $u \in [u_j, u_{j+1})$ with $u_j < u_{j+1}$, exactly one
   basis function satisfies $N_{j,0}(u) = 1$; all others are $0$. The sum is $1$.

   **Inductive step**: Assume $\sum_{i=0}^{n+1} N_{i,p-1}(u) = 1$ (the degree-$(p-1)$
   basis functions on the same knot vector; there are $n + 2$ of them since
   $(N - (p-1) - 1) = n + 2$).

   Define $\alpha_i = \frac{u - u_i}{u_{i+p} - u_i}$ (with $0/0 = 0$). The second coefficient
   in the Cox-de Boor recursion satisfies:

   $$\frac{u_{i+p+1} - u}{u_{i+p+1} - u_{i+1}} = 1 - \frac{u - u_{i+1}}{u_{i+p+1} - u_{i+1}} = 1 - \alpha_{i+1}$$

   (When a denominator is zero, both sides evaluate to $0$ by the $0/0 = 0$
   convention, and the corresponding basis function has zero-length support
   and is identically zero, so the product vanishes regardless.)

   The recursion becomes $N_{i,p}(u) = \alpha_i \, N_{i,p-1}(u) + (1 - \alpha_{i+1}) \, N_{i+1,p-1}(u)$.
   Summing over $i = 0, \ldots, n$:

   $$\sum_{i=0}^{n} N_{i,p}(u) = \sum_{i=0}^{n} \alpha_i \, N_{i,p-1}(u) + \sum_{i=0}^{n} (1 - \alpha_{i+1}) \, N_{i+1,p-1}(u)$$

   Re-index the second sum with $j = i + 1$:

   $$= \sum_{i=0}^{n} \alpha_i \, N_{i,p-1}(u) + \sum_{j=1}^{n+1} (1 - \alpha_j) \, N_{j,p-1}(u)$$

   The degree-$(p-1)$ functions are $N_{0,p-1}, \ldots, N_{n+1,p-1}$. Collecting
   coefficients of each $N_{i,p-1}(u)$:

   - $i = 0$: coefficient $\alpha_0$ (first sum only)
   - $1 \leq i \leq n$: coefficient $\alpha_i + (1 - \alpha_i) = 1$ (both sums)
   - $i = n + 1$: coefficient $1 - \alpha_{n+1}$ (second sum only)

   Therefore:

   $$\sum_{i=0}^{n} N_{i,p}(u) = \alpha_0 \, N_{0,p-1}(u) + \sum_{i=1}^{n} N_{i,p-1}(u) + (1 - \alpha_{n+1}) \, N_{n+1,p-1}(u)$$

   By the inductive hypothesis $\sum_{i=0}^{n+1} N_{i,p-1}(u) = 1$, so
   $\sum_{i=1}^{n} N_{i,p-1}(u) = 1 - N_{0,p-1}(u) - N_{n+1,p-1}(u)$.
   Substituting:

   $$= 1 + (\alpha_0 - 1) \, N_{0,p-1}(u) - \alpha_{n+1} \, N_{n+1,p-1}(u)$$

   It remains to show the two extra terms vanish. For a clamped knot vector,
   $u_0 = \cdots = u_p = a$ and $u_{n+1} = \cdots = u_{n+p+1} = b$:

   - $N_{0,p-1}(u)$ has support $[u_0, u_p) = [a, a)$, which is empty for $p \geq 1$.
     Therefore $N_{0,p-1}(u) = 0$.
   - $N_{n+1,p-1}(u)$ has support $[u_{n+1}, u_{n+p+1}) = [b, b)$, which is empty.
     Therefore $N_{n+1,p-1}(u) = 0$.

   Hence $\sum_{i=0}^{n} N_{i,p}(u) = 1$. $\blacksquare$

4. **Smoothness**: At an **interior** knot of multiplicity $m$, the basis function is
   $C^{p-m}$.  A simple knot gives $C^{p-1}$ continuity; multiplicity $p$ gives $C^0$
   (positional continuity only, a sharp corner); multiplicity $p+1$ at an interior knot
   gives $C^{-1}$ — a genuine discontinuity between two adjacent pieces.  This rule
   applies **only to interior knots**; at boundary knots multiplicity $p+1$ has a
   different meaning (see below).

5. **Linear independence**: The $n + 1$ basis functions are linearly independent and form
   a basis for $\mathbb{S}_{p, \mathbf{t}}$.

**Clamped knot vectors.** In practice, the first and last knots are repeated $p + 1$ times:

$$
U = [\underbrace{a, \ldots, a}_{p+1}, u_{p+1}, \ldots, u_{n}, \underbrace{b, \ldots, b}_{p+1}]
$$

At the boundary $t = a$ there is only one adjacent polynomial piece $q_0$, so no
continuity condition between two pieces exists there.  The formula $r = p - m = -1$
is vacuously satisfied and does not imply a discontinuity.  What multiplicity $p+1$
*does* enforce is **endpoint interpolation**: with $u_0 = \cdots = u_p = a$, the
degree-0 basis satisfies $N_{p,0}(a) = 1$ and $N_{i,0}(a) = 0$ for $i \neq p$
(the only non-empty interval $[u_i, u_{i+1})$ containing $a$ is $[u_p, u_{p+1})$).
Propagating through $p$ levels of the Cox–de Boor recursion gives $N_{0,p}(a) = 1$
and $N_{i,p}(a) = 0$ for $i > 0$, so:

$$C(a) = \sum_{i=0}^{n} N_{i,p}(a)\,P_i = P_0$$

The same argument at $t = b$ gives $C(b) = P_n$.  Endpoint interpolation is almost
always desired in practice.

## 6. B-Spline Curves

A **B-spline curve** of degree $p$ with $n + 1$ control points $P_0, \ldots, P_n \in \mathbb{R}^d$
and knot vector $U = [u_0, \ldots, u_{n+p+1}]$ is:

$$
C(u) = \sum_{i=0}^{n} N_{i,p}(u) \, P_i
$$

The control points do not in general lie on the curve. They form a **control polygon** that
approximates the shape of the curve. Due to the partition of unity and non-negativity of
the basis functions, the curve is a convex combination of control points at each parameter
value, so $C(u)$ lies within the convex hull of the active control points.

**Local modification**: Because $N_{i,p}$ is nonzero only on $[u_i, u_{i+p+1})$, moving control
point $P_i$ affects the curve only in that interval. At most $p + 1$ control points influence
any given point on the curve.

**Example**: A cubic B-spline ($p = 3$) with 7 control points ($n = 6$) and clamped knot vector
$U = [0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4]$ (11 knots = $n + p + 2 = 11$):
- The curve starts at $P_0$ and ends at $P_6$ (clamped endpoints).
- Moving $P_3$ only affects the curve on approximately $u \in [1, 3]$.
- The curve has $C^2$ continuity at each interior knot (simple knots, $p - 1 = 2$).

## 7. NURBS: Adding Weights

**Non-Uniform Rational B-Splines (NURBS)** extend B-splines by assigning a **weight**
$w_i > 0$ to each control point. The curve becomes:

$$
C(u) = \sum_{i=0}^{n} R_{i,p}(u) \, P_i
$$

where the **rational basis functions** are:

$$
R_{i,p}(u) = \frac{N_{i,p}(u) \, w_i}{\sum_{j=0}^{n} N_{j,p}(u) \, w_j}
$$

The denominator normalizes so that $\sum_i R_{i,p}(u) = 1$.

**Why weights?** Plain B-splines (polynomial, all $w_i = 1$) cannot exactly represent conic
sections (circles, ellipses, parabolas, hyperbolas). NURBS can. A circle can be represented
exactly as a degree-2 NURBS with specific control points and weights:

$$
\text{Quarter circle: } P_0 = (1, 0), \; P_1 = (1, 1), \; P_2 = (0, 1), \quad w_0 = 1, \; w_1 = \frac{\sqrt{2}}{2}, \; w_2 = 1
$$

This is critical for CAD, where circles and arcs are fundamental primitives.

**Geometric interpretation**: A NURBS curve in $\mathbb{R}^d$ is the projection of a non-rational
B-spline curve in $\mathbb{R}^{d+1}$ (homogeneous coordinates). Define the **weighted control
points** $P_i^w = (w_i P_i, w_i) \in \mathbb{R}^{d+1}$. The B-spline curve through these points in
$\mathbb{R}^{d+1}$ is $C^w(u) = \sum_i N_{i,p}(u) P_i^w$. Projecting back to $\mathbb{R}^d$ by
dividing by the last coordinate gives the NURBS curve $C(u)$.

The weight $w_i$ controls how strongly $P_i$ "pulls" the curve. Higher weight = curve passes
closer to that control point. Setting all weights equal recovers the plain B-spline.

## 8. NURBS Surfaces (Tensor Product)

A **NURBS surface** is the tensor product of two NURBS curves. Given an
$(n+1) \times (m+1)$ grid of control points $P_{i,j} \in \mathbb{R}^3$ with weights $w_{i,j} > 0$:

$$
S(u, v) = \sum_{i=0}^{n} \sum_{j=0}^{m} R_{i,j}(u, v) \, P_{i,j}
$$

where:

$$
R_{i,j}(u, v) = \frac{N_{i,p}(u) \, N_{j,q}(v) \, w_{i,j}}{\sum_{k=0}^{n} \sum_{l=0}^{m} N_{k,p}(u) \, N_{l,q}(v) \, w_{k,l}}
$$

The surface has:
- Degree $p$ in the $u$-direction with knot vector $U = [u_0, \ldots, u_{n+p+1}]$.
- Degree $q$ in the $v$-direction with knot vector $V = [v_0, \ldots, v_{m+q+1}]$.
- An $(n+1) \times (m+1)$ net of control points forming a **control mesh**.
- Total parameters: $(n+1)(m+1)$ control points $\times$ (3 coordinates + 1 weight) each.

## 9. Fitting B-Splines to Data

This is the central practical question: given a set of data points, how does one compute
the B-spline (or NURBS) that best represents them? The procedure has distinct stages,
and crucially, **the knot vector is fixed before solving for control points**, making the
fitting problem linear.

#### Step 1: Choose the degree $p$

Almost always $p = 3$ (cubic). Cubic B-splines have $C^2$ continuity, which is sufficient
for visual and mechanical smoothness. Degree 2 is used when exact conic representation
is needed (via NURBS weights). Degrees above 4 are rare.

#### Step 2: Parameterize the data

Given data points $Q_0, Q_1, \ldots, Q_M \in \mathbb{R}^d$, assign a parameter value $\bar{u}_k$ to
each point. This defines *where* on the parameter axis each data point should correspond to.

**Uniform parameterization** (simplest):

$$
\bar{u}_k = \frac{k}{M}, \quad k = 0, \ldots, M
$$

**Chord-length parameterization** (accounts for point spacing):

$$
\bar{u}_0 = 0, \quad \bar{u}_k = \bar{u}_{k-1} + \frac{\|Q_k - Q_{k-1}\|}{L}, \quad L = \sum_{j=1}^{M} \|Q_j - Q_{j-1}\|
$$

**Centripetal parameterization** (often best for curves with sharp turns):

$$
\bar{u}_0 = 0, \quad \bar{u}_k = \bar{u}_{k-1} + \frac{\sqrt{\|Q_k - Q_{k-1}\|}}{L'}, \quad L' = \sum_{j=1}^{M} \sqrt{\|Q_j - Q_{j-1}\|}
$$

For surface fitting with grid-sampled data (as in our INR case), uniform parameterization
is appropriate since the UV grid is already regular.

#### Step 3: Choose the knot vector $U$

The knot vector must have $n + p + 2$ entries for $n + 1$ control points of degree $p$.
With a clamped knot vector, the first and last $p + 1$ values are fixed to the domain
boundaries. The $n - p$ interior knots must be chosen.

**Uniform interior knots** (simplest, good for uniformly distributed data):

$$
u_{p+j} = \frac{j}{n - p + 1}, \quad j = 1, \ldots, n - p
$$

**Averaging method** (de Boor, adapts to data distribution):

$$
u_{p+j} = \frac{1}{p} \sum_{i=j}^{j+p-1} \bar{u}_i, \quad j = 1, \ldots, n - p
$$

This places interior knots at the average of $p$ consecutive parameter values, ensuring
each knot span contains data points. It is the standard method recommended by Piegl
and Tiller (*The NURBS Book*).

**The knot vector is not optimized jointly with control points.** If it were, the basis
functions $N_{i,p}(u)$ would depend on the unknowns, making the fitting problem nonlinear.
Some advanced methods do optimize knot positions (see references below), but the standard
approach fixes the knot vector first and solves a linear system for the control points.

#### Step 4a: Interpolation ($n = M$, exact fit through data points)

If we want the curve to pass through every data point exactly, we need $n + 1 = M + 1$
control points (as many as data points). The interpolation conditions are:

$$
C(\bar{u}_k) = Q_k \quad \Longleftrightarrow \quad \sum_{i=0}^{n} N_{i,p}(\bar{u}_k) \, P_i = Q_k, \quad k = 0, \ldots, M
$$

In matrix form:

$$
\mathbf{N} \, \mathbf{P} = \mathbf{Q}
$$

where $\mathbf{N}$ is the $(M+1) \times (M+1)$ matrix with entries
$\mathbf{N}_{k,i} = N_{i,p}(\bar{u}_k)$, $\mathbf{P}$ is the $(M+1) \times d$ matrix of unknown control
points, and $\mathbf{Q}$ is the $(M+1) \times d$ matrix of data points.

**Key property**: Because each $N_{i,p}$ has compact support (nonzero on at most $p + 1$
knot spans), the matrix $\mathbf{N}$ is **banded** with bandwidth $p + 1$. This means:
- The system can be solved in $O(M \cdot p^2)$ time (not $O(M^3)$).
- The system is well-conditioned (unlike Vandermonde matrices for global polynomials).

With a clamped knot vector and properly chosen parameterization, $\mathbf{N}$ is guaranteed
to be non-singular, so a unique solution exists.

**Comparison with Lagrange interpolation**: Both pass through all data points. But
Lagrange uses a single degree-$M$ polynomial (ill-conditioned, oscillatory). B-spline
interpolation uses $M + 1$ basis functions of fixed degree $p$ (well-conditioned, local).
The B-spline interpolant avoids Runge's phenomenon because each basis function
influences only a local region.

#### Step 4b: Approximation ($n < M$, least-squares fit)

When fitting noisy data or when a compact representation is desired, we use fewer control
points than data points: $n + 1 < M + 1$. The **least-squares fitting problem** is:

$$
\min_{\mathbf{P}} \sum_{k=0}^{M} \left\| \sum_{i=0}^{n} N_{i,p}(\bar{u}_k) \, P_i - Q_k \right\|^2
= \min_{\mathbf{P}} \| \mathbf{N} \mathbf{P} - \mathbf{Q} \|_F^2
$$

where $\mathbf{N}$ is now a rectangular $(M+1) \times (n+1)$ matrix (more rows than columns).
Setting the gradient to zero gives the **normal equations**:

$$
\mathbf{N}^\top \mathbf{N} \, \mathbf{P} = \mathbf{N}^\top \mathbf{Q}
$$

The matrix $\mathbf{N}^\top \mathbf{N}$ is $(n+1) \times (n+1)$, symmetric positive definite, and
**banded** with bandwidth $2p + 1$ (because $\mathbf{N}$ is banded with bandwidth $p + 1$).
This is solvable in $O(n \cdot p^2)$ time via Cholesky decomposition.

**The number of control points $n + 1$ acts as a regularization parameter:**
- Fewer control points $\to$ smoother curve, higher fitting error (underfitting).
- More control points $\to$ lower fitting error, risk of wiggles (overfitting).
- The knot vector distributes the control points' influence across the domain.

**Optionally, with clamped endpoints** (common): fix $P_0 = Q_0$ and $P_n = Q_M$,
and solve the reduced $(n-1) \times (n-1)$ system for the interior control points.

#### Surface fitting (tensor product)

For surface data on a grid $Q_{k,l}$ with parameters $(\bar{u}_k, \bar{v}_l)$, evaluation
at a grid point is:

$$S(\bar{u}_k, \bar{v}_l) = \bigl[\mathbf{N}_u\,\mathbf{P}\,\mathbf{N}_v^\top\bigr]_{kl}$$

where $(\mathbf{N}_u)_{ki} = N_{i,p}(\bar{u}_k)$ and $(\mathbf{N}_v)_{lj} = N_{j,q}(\bar{v}_l)$,
and $\mathbf{P}$ is the $(n+1)\times(m+1)$ matrix of control points (one per spatial
coordinate).  The least-squares objective is:

$$\min_{\mathbf{P}}\;\|\mathbf{N}_u\,\mathbf{P}\,\mathbf{N}_v^\top - \mathbf{Q}\|_F^2$$

Setting the gradient to zero gives the **2D normal equations**:

$$(\mathbf{N}_u^\top \mathbf{N}_u)\,\mathbf{P}\,(\mathbf{N}_v^\top \mathbf{N}_v)
  = \mathbf{N}_u^\top\,\mathbf{Q}\,\mathbf{N}_v$$

Because the two Gram matrices act on independent indices, the solution factors as:

$$\mathbf{P} = (\mathbf{N}_u^\top \mathbf{N}_u)^{-1}\,\mathbf{N}_u^\top\,\mathbf{Q}
              \,\mathbf{N}_v\,(\mathbf{N}_v^\top \mathbf{N}_v)^{-1}$$

This is computed in **two sequential passes**, each consisting of independent 1D
least-squares problems:

**Pass 1 — fit in $u$, for each column $l = 0,\ldots,M_v$:**

$$\min_{\mathbf{r}_l}\;\|\mathbf{N}_u\,\mathbf{r}_l - \mathbf{q}_l\|^2
\;\Longrightarrow\;
\mathbf{R} = (\mathbf{N}_u^\top \mathbf{N}_u)^{-1}\,\mathbf{N}_u^\top\,\mathbf{Q}
\in \mathbb{R}^{(n+1)\times(M_v+1)}$$

The $M_v+1$ systems share the same coefficient matrix $\mathbf{N}_u^\top\mathbf{N}_u$
and differ only in their right-hand sides, so a single Cholesky factorisation suffices.

**Pass 2 — fit in $v$, for each row $i = 0,\ldots,n$:**

$$\min_{\mathbf{p}_i}\;\|\mathbf{N}_v\,\mathbf{p}_i - \mathbf{r}_i^\top\|^2
\;\Longrightarrow\;
\mathbf{P} = \mathbf{R}\,\mathbf{N}_v\,(\mathbf{N}_v^\top \mathbf{N}_v)^{-1}$$

The $n+1$ systems again share one factorisation of $\mathbf{N}_v^\top\mathbf{N}_v$.

The two-pass solution is **not an approximation** — it is algebraically identical to
solving the full 2D system jointly.  The equivalence relies on one essential assumption:
**the data must lie on a rectangular grid**, so that the same $\bar{u}$ values are used
for every $l$ and the same $\bar{v}$ values for every $k$.  For scattered (non-grid) data
the normal equations do not factor and the 2D system must be solved directly.

The intermediate matrix $\mathbf{R}$ has a natural interpretation: its $i$-th row
$R_{i,\cdot}$ gives the optimal $u$-direction control point $i$ as a function of the
$v$ sample index.  Pass 2 then fits a B-spline through those $M_v+1$ values to obtain
the final bivariate control net $P_{i,j}$.

## 10. NURBS Hyperparameters

Fitting a NURBS to data involves choosing several hyperparameters. This is fundamentally
different from Lagrange interpolation where the degree is forced by the number of points.

| Hyperparameter | What it controls | Typical values |
|---|---|---|
| Degree $p$ (and $q$ for surfaces) | Smoothness of each polynomial piece | 3 (cubic) is standard. 2 for conics. |
| Control point count $n+1$ (or $(n+1) \times (m+1)$) | Resolution and expressiveness. More = finer detail but risk of overfitting. | 10-50 per direction, depends on complexity. |
| Knot vector $U$ | Where polynomial pieces join. Controls parameterization and detail distribution. | Uniform or averaging method. |
| Weights $w_i$ | How strongly each control point attracts the surface. $w_i = 1$ reduces to plain B-spline. | 1.0 for most applications. |
| Parameterization method | How parameter values are assigned to data points. | Chord-length or centripetal for curves; uniform for grid data. |

**On knot vector optimization**: The standard approach fixes the knot vector heuristically
(uniform, averaging, or chord-length) and solves a linear system. Advanced methods optimize
knot positions jointly with control points, but this turns the problem into a nonlinear
optimization (the basis functions depend on the knot positions). This is computationally
expensive and typically unnecessary for smooth surfaces. See
[Plos One: Optimal Knots for B-Spline Curves](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0173857)
for an example of knot optimization.

## 11. NURBS vs. Polynomial Interpolation: Summary

| Property | Polynomial (Lagrange) | NURBS |
|---|---|---|
| Degree | Forced: $n-1$ for $n$ points | Free: typically 3, independent of data size |
| Locality | Global: moving 1 point changes entire curve | Local: moving 1 control point affects only nearby region |
| Oscillation | Runge's phenomenon for high degree | Controlled by keeping degree low (piecewise) |
| Exact conics | Cannot represent circles exactly | Exact via rational weights |
| Numerical stability | Ill-conditioned ($O(\kappa^n)$ Vandermonde) | Well-conditioned (banded, $O(n \cdot p^2)$ solve) |
| CAD compatibility | None | Universal standard (STEP, IGES) |
| Fitting mode | Always interpolates (degree = $n-1$) | Interpolation or least-squares approximation |
| Fitting linearity | Linear (Vandermonde system) | Linear (banded system, knot vector fixed) |
| Hyperparameters | None (everything forced by data) | Degree, control point count, knot vector, weights |

## 12. Why NURBS is the Right Choice Here

For the specific goal of producing CAD-compatible output:

1. **STEP format uses NURBS**: The STEP standard represents all surfaces (including
   analytic ones like planes and cylinders) as NURBS internally. Any surface we want to
   export must ultimately be expressible as NURBS.

2. **OCCT speaks NURBS**: OpenCASCADE's intersection, trimming, and topology
   algorithms all operate on NURBS representations. Converting the INR to NURBS
   makes all of OCCT's machinery available.

3. **The INR's UV parameterization is a natural fit**: The INR encoder already maps
   3D points to a 2D UV domain. A NURBS surface is a map from UV to 3D. The
   trained INR gives us dense UV-to-3D samples that we can directly fit a NURBS to.
   Furthermore, the INR samples on a regular UV grid, so uniform parameterization is
   appropriate, and the tensor product fitting procedure (Section 4.9) applies directly.

4. **Compact representation**: A NURBS surface with, say, $20 \times 20$ control points
   has 400 control points (1200 coordinates). The INR MLP has thousands of parameters.
   The NURBS is a more compact, interpretable, and CAD-native representation.

Alternative representations (subdivision surfaces, T-splines) either lack CAD software
support or add complexity without clear benefit for this use case.

## 13. Concrete Example: Fitting a NURBS to INR Output

**Step 1**: Train the INR as usual. The encoder maps points to UV, the decoder maps UV to 3D.

**Step 2**: Sample the trained decoder on a regular grid in UV:

$$
Q_{k,l} = \text{INR\_decoder}\left(\frac{k}{M_u}, \frac{l}{M_v}\right), \quad k = 0, \ldots, M_u, \quad l = 0, \ldots, M_v
$$

For example, $M_u = M_v = 49$ gives a $50 \times 50$ grid of 2500 sample points.

**Step 3**: Choose fitting parameters:
- Degree: $p = q = 3$ (cubic in both directions).
- Control points: e.g. $15 \times 15$ ($n = m = 14$).
- Knot vectors: clamped, uniform interior knots.

**Step 4**: Solve the tensor product least-squares problem (Section 4.9):
- First pass: for each $v$-row, solve the banded $(50 \times 15)$ least-squares system in $u$.
- Second pass: for each $u$-column of intermediate results, solve in $v$.
- Or use OCCT directly: `GeomAPI_PointsToBSplineSurface(grid_points, 3, 3, ...)`.

**Step 5**: Validate the approximation error:

$$
\epsilon_{\max} = \max_{k,l} \| S(\bar{u}_k, \bar{v}_l) - Q_{k,l} \|
$$

Sample at off-grid UV points to check generalization. Target: $\epsilon_{\max}$ within CAD
tolerance, e.g. $10^{-4}$ of model size.

Sources:
- [Least-Squares Fitting of Data with B-Spline Curves (Geometric Tools)](https://www.geometrictools.com/Documentation/BSplineCurveLeastSquaresFit.pdf)
- [A Direct Method to Solve Optimal Knots of B-Spline Curves (PLOS One)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0173857)
- [Introduction to Spline Theory (Floater, UiO lecture notes)](https://www.uio.no/studier/emner/matnat/math/MAT4170/v23/undervisningsmateriale/spline_notes.pdf)


## More useful links
* BSpline basis nice illustrations: https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/B-spline/bspline-basis.html

* Open Cascade BREP documentation: https://dev.opencascade.org/doc/overview/html/specification__brep_format.html