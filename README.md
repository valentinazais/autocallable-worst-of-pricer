# Monte Carlo Worst-Of Autocallable Pricer

Interactive dashboard for pricing and analyzing multi-asset Worst-Of Autocallable structured products using vectorized Monte Carlo simulations with Cholesky decomposition.

---

## Model Formulas

### Stochastic Diffusion Model (Risk-Neutral Measure $\mathbb{Q}$)

The underlying assets follow a correlated Geometric Brownian Motion:

$$
dS_k(t) = S_k(t) \left( (r - q_k)dt + \sigma_k dW_k(t) \right) \quad \text{for } k \in \{1, 2\}
$$

With instantaneous correlation:

$$
\langle dW_1(t), dW_2(t) \rangle = \rho dt
$$

| Symbol | Definition |
|--------|-----------|
| $S_k$ | Current underlying price of asset $k$ |
| $T$ | Time to maturity (years) |
| $r$ | Risk-free rate |
| $q_k$ | Continuous dividend yield of asset $k$ |
| $\sigma_k$ | Volatility of asset $k$ |
| $\rho$ | Correlation between the two assets |

---

### Cholesky Decomposition

To generate correlated shocks efficiently, we apply the Cholesky decomposition $\Sigma = L L^T$ to independent Gaussian variables $Z \sim \mathcal{N}(0, I)$.

$$
L = \begin{pmatrix} 1 & 0 \\ \rho & \sqrt{1-\rho^2} \end{pmatrix}
$$

---

### Autocallable Mechanism

At each observation date $t_i$, the "Worst-Of" performance is evaluated:

$$
W_{perf}(t_i) = \min \left( \frac{S_1(t_i)}{S_1(0)}, \frac{S_2(t_i)}{S_2(0)} \right)
$$

**Early Redemption (Autocall):**
If $W_{perf}(t_i) \ge \text{Autocall Barrier}$, the product matures early. The payoff is:
$$
\text{Payoff} = \text{Nominal} \times (1 + \text{Coupon} \times i)
$$

**Maturity - Capital Protection:**
If no autocall occurs and $W_{perf}(T) \ge \text{Knock-In Barrier}$ at maturity, the capital is protected:
$$
\text{Payoff} = \text{Nominal} \times (1 + \text{Coupon} \times T)
$$

**Maturity - Capital Loss:**
If no autocall occurs and $W_{perf}(T) < \text{Knock-In Barrier}$ at maturity, capital protection is lost:
$$
\text{Payoff} = \text{Nominal} \times W_{perf}(T)
$$

---

## Features

### Market & Asset Parameters
- Risk-Free Rate ($r$)
- Correlation ($\rho$)
- Independent Volatilities ($\sigma_1, \sigma_2$)
- Independent Dividend Yields ($q_1, q_2$)

### Product Parameters
- Maturity (Years)
- Annual Coupon (%)
- Autocall Barrier (%)
- European Knock-In Barrier (%)

### Simulation Engine
- Configurable number of Monte Carlo paths (up to 100,000)
- Reproducible random seed

### Output Analytics
- Fair Value (Present Value)
- Capital Loss Risk (Probability)
- Expected Payoff (Undiscounted)
- Expected Annual Return

---

## Visualizations

### Exit Scenarios Distribution
Bar chart showing the probability of each outcome (Autocall by year, Capital Protection, or Capital Loss).

### Sample Paths Visualization
Line chart overlaying 50 random Monte Carlo paths of the Worst-Of performance against the Autocall and Knock-In barriers.

---

## Architecture
streamlit (Python)
│
autocallable_worst_of.py
│
├── Monte Carlo engine (numpy vectorization)
├── Cholesky decomposition (scipy.linalg)
├── Sidebar parameter controls
├── Matplotlib figure rendering
└── Streamlit metrics & charts grid

System properties:
- Python backend, Streamlit frontend
- Highly optimized vectorized simulations (`numpy`)
- Memory-efficient path generation

---

## Numerical Implementation

- `numpy.random.normal` for standard normal shocks
- `scipy.linalg.cholesky` for correlation modeling
- `numpy.cumsum` and `numpy.exp` for geometric path generation
- `matplotlib` for financial charting

---

## Technology

- Python 3
- Streamlit
- Matplotlib
- SciPy
- NumPy

---

## Result

A browser-accessible quantitative pricing terminal for exploring:
- Multi-asset path-dependent structured products
- Correlation risk (dispersion)
- The probability of capital loss vs. early redemption
- Visualizing complex Monte Carlo distributions

All directly in the browser without local installation.
