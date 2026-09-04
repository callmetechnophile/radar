# V9 Kalman Covariance Units & Dimensional Consistency Audit

## 1. Problem Formulation
In the V8.1 canonical estimator, the state vector is defined as:
$$x_t = [p_t, v_t]^T \in \mathbb{R}^{132}$$
where $p_t \in \mathbb{R}^{66}$ represents 3D positions in meters ($\text{m}$) and $v_t \in \mathbb{R}^{66}$ represents 3D velocities in meters per second ($\text{m/s}$).

The Kalman covariance matrix $P_t \in \mathbb{R}^{132 \times 132}$ has the block structure:
$$P_t = \begin{bmatrix} P_{pp} & P_{pv} \\ P_{vp} & P_{vv} \end{bmatrix}$$
with physical units:
- $P_{pp} \in \mathbb{R}^{66 \times 66}$: Units of **$\text{m}^2$** (positional variance)
- $P_{vv} \in \mathbb{R}^{66 \times 66}$: Units of **$\text{m}^2/\text{s}^2$** (velocity variance)
- $P_{pv}, P_{vp} \in \mathbb{R}^{66 \times 66}$: Units of **$\text{m}^2/\text{s}$** (cross-covariance)

## 2. Inconsistency of Full State Trace $\text{Tr}(P_t)$
Computing the full trace:
$$\text{Tr}(P_t) = \text{Tr}(P_{pp}) + \text{Tr}(P_{vv}) \quad [\text{m}^2 + \text{m}^2/\text{s}^2]$$
yields a dimensionally inhomogeneous scalar summing squared meters and squared velocity.
Therefore:
$$\sqrt{\text{Tr}(P_t)} \neq \text{meters}$$
Using $\sqrt{\text{Tr}(P_t)}$ directly in a Cartesian spatial bounding formula ($r_{\max} = k\sqrt{\text{Tr}(P)} + b$) is **dimensionally flawed**.

## 3. Derivation of Positional Covariance Trace $\text{Tr}(P_{\text{pos}})$
To restore strict dimensional purity:
1. Extract strictly the position diagonal elements:
   $$\text{Tr}(P_{\text{pos}}) = \sum_{j=0}^{21} \sum_{k=0}^2 P_{6j+k, 6j+k} \quad [\text{m}^2]$$
2. The root-mean-square spatial uncertainty is:
   $$\sigma_{\text{pos}} = \sqrt{\text{Tr}(P_{\text{pos}})} \quad [\text{m}]$$
3. Per-joint average spatial standard deviation:
   $$\bar{\sigma}_{\text{joint}} = \sqrt{\frac{1}{22} \text{Tr}(P_{\text{pos}})} \quad [\text{m}]$$

## 4. Audit Verdict
- Full state trace $\text{Tr}(P_t)$ for position bounding: **FAIL (Dimensional Inconsistency)**
- Explicit Positional Submatrix Trace $\text{Tr}(P_{\text{pos}})$: **PASS (Pure $\text{m}^2$, $\sqrt{\text{Tr}(P_{\text{pos}})} \in \text{m}$)**
- Action: V9 freeze specification replaces $\text{Tr}(P_t)$ with $\text{Tr}(P_{\text{pos}})$ in the safety bound and feature interface.
