import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import cholesky

# Configuration de la page
st.set_page_config(page_title="Pricer Autocallable Worst-Of", layout="wide")

# CSS Personnalisé pour un look épuré (similaire à votre script)
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 0; }
    [data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.3);
        border-radius: 8px;
        padding: 12px 16px;
    }
    section[data-testid="stSidebar"] { width: 340px; }
</style>
""", unsafe_allow_html=True)

plt.style.use('ggplot')

st.title("Monte Carlo Worst-Of Autocallable Pricer")

# ─── SIDEBAR : Paramètres ───
st.sidebar.header("Market Parameters")
r = st.sidebar.number_input("Risk-Free Rate (r)", value=0.03, step=0.01)
rho = st.sidebar.slider("Correlation (ρ)", min_value=-0.99, max_value=0.99, value=0.60, step=0.05)

st.sidebar.header("Asset Parameters")
col1, col2 = st.sidebar.columns(2)
with col1:
    vol1 = st.number_input("Volatility 1 (σ1)", value=0.20, step=0.01)
    q1 = st.number_input("Dividend Yield 1 (q1)", value=0.02, step=0.01)
with col2:
    vol2 = st.number_input("Volatility 2 (σ2)", value=0.25, step=0.01)
    q2 = st.number_input("Dividend Yield 2 (q2)", value=0.03, step=0.01)

st.sidebar.header("Product Parameters")
nominal = 100.0
T = st.sidebar.number_input("Maturity (Years)", value=3, min_value=1, max_value=10, step=1)
coupon_pct = st.sidebar.number_input("Annual Coupon (%)", value=8.0, step=0.5) / 100.0
autocall_barrier_pct = st.sidebar.number_input("Autocall Barrier (%)", value=100.0, step=5.0) / 100.0
ki_barrier_pct = st.sidebar.number_input("Knock-In Barrier (%)", value=60.0, step=5.0) / 100.0

st.sidebar.header("Monte Carlo Parameters")
M = st.sidebar.selectbox("Number of Simulations", [1000, 5000, 10000, 50000, 100000], index=2)
seed_val = st.sidebar.number_input("Random Seed", value=42, step=1)

# ─── MOTEUR MONTE CARLO (Vectorisé & Caché) ───
@st.cache_data
def run_monte_carlo(r, rho, vol1, vol2, q1, q2, T, coupon_pct, autocall_barrier_pct, ki_barrier_pct, M, seed_val):
    # Setup temporel
    obs_freq = 1 # Observations annuelles (standard)
    N_obs = int(T)
    N_steps_per_year = 252 # Jours de trading
    N_steps = N_obs * N_steps_per_year
    dt = 1.0 / N_steps_per_year
    
    np.random.seed(seed_val)
    
    # Matrice de corrélation et Cholesky
    corr_matrix = np.array([[1.0, rho], [rho, 1.0]])
    L = cholesky(corr_matrix, lower=True)
    
    # Mouvement Brownien Correlé (Z gaussien standard -> W corrélé)
    Z = np.random.normal(0, 1, (2, M, N_steps))
    W = (L @ Z.reshape(2, -1)).reshape(2, M, N_steps)
    
    # Dérives (Drifts) sous probabilité Risque-Neutre Q
    nu1 = r - q1 - 0.5 * vol1**2
    nu2 = r - q2 - 0.5 * vol2**2
    
    # Génération des trajectoires (rendements logarithmiques cumulés)
    log_path1 = np.cumsum(nu1 * dt + vol1 * np.sqrt(dt) * W[0], axis=1)
    log_path2 = np.cumsum(nu2 * dt + vol2 * np.sqrt(dt) * W[1], axis=1)
    
    # Performance par rapport au Spot Initial (Base 1.0)
    perf1 = np.hstack([np.ones((M, 1)), np.exp(log_path1)])
    perf2 = np.hstack([np.ones((M, 1)), np.exp(log_path2)])
    
    # Dynamique "Worst-Of"
    worst_of_perf = np.minimum(perf1, perf2)
    
    # Extraction des performances aux dates d'observation
    obs_indices = [int(i * N_steps_per_year) for i in range(1, N_obs + 1)]
    obs_wo_perf = worst_of_perf[:, obs_indices]
    
    # Initialisation des vecteurs de résultats
    payoffs = np.zeros(M)
    discounted_payoffs = np.zeros(M)
    status = np.full(M, "Unknown", dtype=object)
    
    # --- 1. Logique Autocall ---
    autocall_mask = obs_wo_perf >= autocall_barrier_pct
    hit_any_autocall = np.any(autocall_mask, axis=1)
    first_autocall_idx = np.argmax(autocall_mask, axis=1) # Première année où l'autocall est déclenché
    
    for i in range(N_obs):
        year = i + 1
        # Masque pour les chemins qui font leur PREMIER autocall à l'année 'year'
        mask = hit_any_autocall & (first_autocall_idx == i)
        payoffs[mask] = nominal * (1.0 + coupon_pct * year)
        discounted_payoffs[mask] = payoffs[mask] * np.exp(-r * year)
        status[mask] = f"Autocall Year {year}"
        
    # --- 2. Logique à Maturité (chemins non-autocalled) ---
    not_autocalled_mask = ~hit_any_autocall
    final_wo_perf = worst_of_perf[not_autocalled_mask, -1]
    
    ki_breach = final_wo_perf < ki_barrier_pct
    
    # Scénario A : Capital Protégé (avec coupon mémoire final)
    prot_mask = not_autocalled_mask.copy()
    prot_mask[not_autocalled_mask] = ~ki_breach
    payoffs[prot_mask] = nominal * (1.0 + coupon_pct * T)
    discounted_payoffs[prot_mask] = payoffs[prot_mask] * np.exp(-r * T)
    status[prot_mask] = "Maturity - Protected"
    
    # Scénario B : Perte en Capital (Franchissement barrière Knock-In)
    loss_mask = not_autocalled_mask.copy()
    loss_mask[not_autocalled_mask] = ki_breach
    payoffs[loss_mask] = nominal * final_wo_perf[ki_breach]
    discounted_payoffs[loss_mask] = payoffs[loss_mask] * np.exp(-r * T)
    status[loss_mask] = "Maturity - Capital Loss"
    
    # Juste Valeur (Prix du produit)
    fair_value = np.mean(discounted_payoffs)
    
    # Probabilités des scénarios
    unique_status, counts = np.unique(status, return_counts=True)
    status_probs = {k: v / M for k, v in zip(unique_status, counts)}
    
    return fair_value, payoffs, status_probs, worst_of_perf

# Exécution de la simulation
with st.spinner("Running Monte Carlo simulation..."):
    fv, payoffs, status_probs, wo_perf = run_monte_carlo(
        r, rho, vol1, vol2, q1, q2, T, coupon_pct, autocall_barrier_pct, ki_barrier_pct, M, seed_val
    )

# ─── MAIN CONTENT : RESULTATS ───
st.header("Pricing Analysis")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Fair Value", f"{fv:.2f} %")
c2.metric("Capital Loss Risk (Prob)", f"{status_probs.get('Maturity - Capital Loss', 0)*100:.1f} %")
c3.metric("Expected Payoff (Undiscounted)", f"{np.mean(payoffs):.2f} %")
c4.metric("Expected Annual Return", f"{(np.mean(payoffs)/nominal)**(1/T) - 1:.2%}")

st.divider()

# Graphiques d'analyse
col_left, col_mid, col_right = st.columns([1, 2, 1])

with col_mid:
    st.subheader("Exit Scenarios Distribution")
    # Tri logique des statuts pour le graphique
    sorted_keys = [f"Autocall Year {i}" for i in range(1, int(T)+1)] + ["Maturity - Protected", "Maturity - Capital Loss"]
    probs = [status_probs.get(k, 0) * 100 for k in sorted_keys]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ['#2ca02c'] * int(T) + ['#1f77b4', '#d62728'] # Vert (Autocall), Bleu (Protégé), Rouge (Perte)
    ax.bar(sorted_keys, probs, color=colors)
    ax.set_ylabel("Probability (%)")
    plt.xticks(rotation=45, ha='right')
    ax.set_ylim(0, max(probs) + 15 if probs else 100)
    for i, p in enumerate(probs):
        if p > 0:
            ax.text(i, p + 2, f"{p:.1f}%", ha='center', fontsize=9, fontweight='bold')
    st.pyplot(fig)

st.divider()

# Tracé des trajectoires
st.subheader("Sample Paths Visualization (Worst-Of Performance)")
st.caption("Worst-Of performance evolution for a sample of 50 random paths.")

fig3, ax3 = plt.subplots(figsize=(10, 4.5))
time_axis = np.linspace(0, T, wo_perf.shape[1])

# On sélectionne jusqu'à 50 chemins au hasard pour ne pas surcharger le graphique
sample_indices = np.random.choice(M, min(50, M), replace=False)

for idx in sample_indices:
    ax3.plot(time_axis, wo_perf[idx, :], color='gray', alpha=0.25, linewidth=1)

# Tracé des Barrières
ax3.axhline(autocall_barrier_pct, color='green', linestyle='--', linewidth=2, label=f"Autocall Barrier ({autocall_barrier_pct*100:.0f}%)")
ax3.axhline(ki_barrier_pct, color='red', linestyle='--', linewidth=2, label=f"Knock-In Barrier ({ki_barrier_pct*100:.0f}%)")

# Lignes verticales pour les dates d'observation
for i in range(1, int(T)+1):
    ax3.axvline(i, color='black', linestyle=':', alpha=0.4)

ax3.set_xlabel("Time (Years)")
ax3.set_ylabel("Worst-Of Performance")
ax3.set_xlim(0, T)
ax3.legend(loc="upper right")
st.pyplot(fig3)

# ─── EXPLICATION QUANTITATIVE (Pour le recruteur) ───
with st.expander("Mathematical details (Risk-Neutral Measure)"):
    st.markdown(r'''
    ### Stochastic Diffusion Model
    The underlying assets follow a correlated Geometric Brownian Motion under the Risk-Neutral measure $\mathbb{Q}$:
    $$ dS_k(t) = S_k(t) \left( (r - q_k)dt + \sigma_k dW_k(t) \right) \quad \text{for } k \in \{1, 2\} $$
    With instantaneous correlation $ \langle dW_1(t), dW_2(t) \rangle = \rho dt $.
    
    **Vectorized Generation and Cholesky Decomposition:**
    To generate correlated shocks efficiently using matrix operations, we apply the Cholesky decomposition $\Sigma = L L^T$ to independent Gaussian variables $Z \sim \mathcal{N}(0, I)$.
    
    **Autocallable Mechanism (Discrete Path-Dependency):**
    - At each annual observation date $t_i$, we evaluate the worst performance: $ W_{perf}(t_i) = \min \left( \frac{S_1(t_i)}{S_1(0)}, \frac{S_2(t_i)}{S_2(0)} \right) $.
    - If $ W_{perf}(t_i) \ge \text{Autocall Barrier} $, the product is called early. The client receives the nominal plus a cumulative memory coupon.
    - Otherwise, at maturity $T$, if $ W_{perf}(T) < \text{KI Barrier} $ (European Knock-In), capital protection is lost, exposing the investor to the full downside risk of the worst-performing leg.
    ''')
