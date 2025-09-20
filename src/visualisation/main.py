"""
Main Streamlit application for Binomial Options Pricing Model

This module orchestrates the tab-based interface for theoretical pricing
and P&L analysis visualizations.
"""

import streamlit as st
from .theoretical import theoretical_prices_tab
from .pnl import pnl_analysis_tab


def main():
    st.set_page_config(
        page_title="Binomial Options Pricing Model",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("Binomial Options Pricing Model")
    
    # Sidebar
    with st.sidebar:
        st.title("📊 Binomial Options Pricing Model")
        
        # Show model parameters checkbox
        show_intermediate = st.checkbox("Show Model Parameters", help="Show intermediate calculation values (dt, u, d, p)")
        
        # Model parameters
        st.markdown("### Model Parameters")
        S0 = st.number_input("Initial Stock Price (S₀)", min_value=0.01, value=100.00, step=1.00, format="%.2f")
        K = st.number_input("Strike Price (K)", min_value=0.01, value=105.00, step=1.00, format="%.2f")
        T = st.number_input("Time to Maturity (T)", min_value=0.01, max_value=10.00, value=0.25, step=0.01, format="%.2f")
        r = st.number_input("Risk-free Rate (r)", min_value=0.000, max_value=1.000, value=0.050, step=0.001, format="%.3f")
        sigma = st.number_input("Volatility (σ)", min_value=0.010, max_value=2.000, value=0.200, step=0.010, format="%.3f")
        n_steps = st.slider("Number of Steps", 1, 10, 4, 1)
        
        # Option type (for Theoretical Prices tab)
        option_type = st.selectbox("Option Type", ["call", "put"], index=1)
    
    # Model parameters for both tabs
    model_params = {
        'S0': S0,
        'K': K,
        'T': T,
        'r': r,
        'sigma': sigma,
        'n_steps': n_steps,
        'option_type': option_type
    }
    
    # Create tabs
    tab1, tab2 = st.tabs(["📈 European vs American", "💰 P&L Analysis"])
    
    with tab1:
        theoretical_prices_tab(model_params, show_intermediate)
    
    with tab2:
        pnl_analysis_tab(model_params, show_intermediate)
