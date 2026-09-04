# Scientific Hypothesis & Justification for Mamba/SSM Residual Modeling

## 1. Context & Motivation
In Phase V8.2, empirical failure-surface analysis revealed that the mathematical estimator's primary breakdown originates from **ballistic acceleration divergence during unobserved horizons >24 frames** and **high-jerk direction reversals**.
The empirical error residual $r_t = y_{\text{gt}} - y_{\text{math}}$ exhibits:
- Strong temporal autocorrelation ($r = 0.78$ at lag 1, $r = 0.29$ at lag 4)
- Highly dynamic motion dependence (concentrated during agile direction reversals)
- Non-stationary, non-Gaussian structure

## 2. Formal Hypotheses
- **Null Hypothesis ($H_0$):**
  A Selective State-Space Model (Mamba) provides no statistically significant reduction in 16-frame and 32-frame gap MPJPE compared to a parameter-matched Temporal Convolutional Network (TCN) or Gated Recurrent Unit (GRU) when trained on identical 136-D inputs.
- **Alternative Hypothesis ($H_1$):**
  Mamba's input-dependent selectivity ($B(x), C(x), \Delta(x)$) allows the model to dynamically compress or forget linear Kalman extrapolation during continuous observations and selectively activate long-horizon trajectory momentum during unobserved gaps, achieving $>15\%$ lower MPJPE than TCN/GRU under identical parameter budgets ($\le 25\text{k}$).

## 3. Discretization Advantage for Variable Temporal Gaps
Unlike fixed-kernel TCNs or discrete GRUs, State Space Models originate from continuous-time linear differential equations:
$$h'(t) = A h(t) + B x(t), \quad y(t) = C h(t) + D x(t)$$
When discretized with step size $\Delta_t = \text{softplus}(\text{Linear}(x_t))$, the effective transition matrix becomes:
$$\bar{A} = \exp(\Delta_t A)$$
During unobserved frames ($m_t = 0$), $\Delta_t$ dynamically expands, naturally matching the physics of extended unobserved integration without retuning architecture weights.

## 4. Controlled Experimental Protocol to Test $H_0$ vs $H_1$
Prior to claiming superiority:
1. Candidate B (MLP), Candidate C (TCN), Candidate D (GRU), and Candidate E (Mamba) will be trained under exact parameter equivalence ($25\text{k} \pm 2\text{k}$).
2. Identical loss $\mathcal{L}_{\text{total}}$, identical optimizer (AdamW, lr=$10^{-3}$), identical seed suite (42, 123, 456).
3. Evaluated on 16-frame gap, 32-frame gap, and agile direction reversals.
