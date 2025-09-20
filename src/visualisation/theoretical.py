"""
Theoretical Prices tab for Binomial Options Pricing Model

This module handles the theoretical pricing visualization tab,
showing European vs American option pricing with early exercise analysis.
"""

import streamlit as st
from .core import calculate_node_coordinates, plot_american_options_tree, compare_european_american


def theoretical_prices_tab(model_params, show_intermediate):
    """Theoretical Prices tab content"""
    try:
        # Always compare European vs American
        european_model, american_model, european_price, american_price = compare_european_american(model_params)
        
        # Display comparison results
        st.markdown("### Option Pricing")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("European Price", f"${european_price:.4f}")
        
        with col2:
            st.metric("American Price", f"${american_price:.4f}")
        
        with col3:
            early_exercise_premium = american_price - european_price
            st.metric("Early Exercise Premium", f"${early_exercise_premium:.4f}")
        
        with col4:
            if european_price > 0:
                premium_pct = (early_exercise_premium / european_price) * 100
                st.metric("Premium %", f"{premium_pct:.2f}%")
        
        # Advanced metrics section
        adv_placeholder = st.container()
        if show_intermediate:
            with adv_placeholder:
                st.markdown("### Model Parameters")
                colp1, colp2, colp3, colp4 = st.columns(4)
                with colp1:
                    st.metric("Timestep Duration", f"{american_model.dt:.6f}")
                with colp2:
                    st.metric("Up Factor", f"{american_model.u:.6f}")
                with colp3:
                    st.metric("Down Factor", f"{american_model.d:.6f}")
                with colp4:
                    st.metric("Risk Neutral Probability", f"{american_model.p:.6f}")
        else:
            with adv_placeholder:
                # Small spacer to reduce layout shift without leaving large gaps
                st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)
        
        # Show both trees
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🇪🇺 European Options Tree")
            coordinates_eu = calculate_node_coordinates(european_model)
            fig_eu, _ = plot_american_options_tree(
                european_model, coordinates_eu, 
                show_pnl=False
            )
            st.pyplot(fig_eu, use_container_width=True)
        
        with col2:
            st.subheader("🇺🇸 American Options Tree")
            coordinates_us = calculate_node_coordinates(american_model)
            fig_us, early_exercise_nodes = plot_american_options_tree(
                american_model, coordinates_us,
                show_pnl=False
            )
            st.pyplot(fig_us, use_container_width=True)
        
        # Early exercise analysis
        if early_exercise_nodes:
            with st.expander("⚡ Early Exercise Analysis", expanded=False):
                st.caption(f"Nodes where early exercise is optimal: {len(early_exercise_nodes)}")
                
                # Compact table-style display
                rows = []
                for t, i in early_exercise_nodes:
                    node = american_model.tree.nodes[(t, i)]
                    exercise_value = node.get_exercise_value(american_model.K, american_model.option_type)
                    rows.append({
                        "Node": f"({t},{i})",
                        "S": f"${node.stock_price:.2f}",
                        "Exercise": f"${exercise_value:.2f}",
                        "Option": f"${node.option_price:.2f}"
                    })
                st.dataframe(rows, width='stretch', hide_index=True)
                
                # Optional deeper detail
                with st.expander("More detail", expanded=False):
                    for t, i in early_exercise_nodes:
                        node = american_model.tree.nodes[(t, i)]
                        exercise_value = node.get_exercise_value(american_model.K, american_model.option_type)
                        st.markdown(
                            f"- Node ({t},{i}) • S=${node.stock_price:.2f} • "
                            f"Exercise=${exercise_value:.2f} • Option=${node.option_price:.2f}"
                        )
        else:
            with st.expander("⚡ Early Exercise Analysis", expanded=False):
                st.write("No early exercise is optimal – American option behaves like European option")
        
        # Model details
        with st.expander("🔍 Model Details"):
            st.json(american_model.get_model_info())
        
    except ValueError as e:
        st.error(f"❌ Parameter Error: {e}")
    except Exception as e:
        st.error(f"❌ Unexpected Error: {e}")
        st.exception(e)
