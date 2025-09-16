"""
American Options Visualisation with Early Exercise Analysis

This creates a visualisation showing American options pricing with
early exercise decisions highlighted.
"""

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from binomial_model import BinomialModel

# Set matplotlib to use a non-interactive backend
plt.switch_backend('Agg')

def calculate_node_coordinates(model):
    """
    Calculate x, y coordinates for each node in the tree for visualisation.
    """
    coordinates = {}
    max_y_spread = model.n_steps / 2
    
    for t in range(model.n_steps + 1):
        nodes_at_time = model.tree.get_nodes_at_time(t)
        num_nodes_at_time = len(nodes_at_time)
        
        # Calculate y-offset to center the nodes vertically
        y_offset = (num_nodes_at_time - 1) / 2.0
        
        for i, node in enumerate(nodes_at_time):
            x = t  # Time step is the x-coordinate
            y = (i - y_offset) * (max_y_spread / (model.n_steps / 2))  # Flip so up moves go up
            coordinates[(t, node.node_index)] = (x, y)
    
    return coordinates

def plot_american_options_tree(model, coordinates):
    """
    Plot the binomial tree with early exercise decisions highlighted.
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    # Dark mode theme
    fig.patch.set_facecolor('#0f1115')
    ax.set_facecolor('#0f1115')
    for spine in ax.spines.values():
        spine.set_color('#444a55')
    ax.tick_params(colors='#d0d3d8')
    
    # Get all nodes for color mapping
    all_nodes = list(model.tree.nodes.values())
    all_option_prices = [node.option_price for node in all_nodes if node.option_price is not None]
    
    if not all_option_prices:
        min_price, max_price = 0, 1
    else:
        min_price, max_price = min(all_option_prices), max(all_option_prices)
    
    # Build a darker red→yellow→green colormap for dark backgrounds
    dark_ryg_colors = [
        '#8B1E1E',  # deep red
        '#9A3412',  # deep orange
        '#B45309',  # dark amber
        '#166534',  # dark green
        '#14532D',  # deeper green
    ]
    dark_ryg_cmap = mcolors.LinearSegmentedColormap.from_list('dark_ryg', dark_ryg_colors, N=256)

    # Draw connections first - iterate through all nodes and draw connections to their children
    for (t, i), (x, y) in coordinates.items():
        node = model.tree.nodes[(t, i)]
        
        # Draw connection to up child
        if node.up_child:
            up_child_coords = coordinates[(node.up_child.time_step, node.up_child.node_index)]
            ax.plot([x, up_child_coords[0]], [y, up_child_coords[1]], 
                   color='#8a8f98', linewidth=2, alpha=0.8, zorder=1)
            
            # Add up movement label
            mid_x = (x + up_child_coords[0]) / 2
            mid_y = (y + up_child_coords[1]) / 2
            # Use a softer green for up moves
            ax.text(mid_x, mid_y + 0.1, '↑', ha='center', va='center', 
                   fontsize=12, color='#66BB6A', weight='bold', zorder=3)
        
        # Draw connection to down child
        if node.down_child:
            down_child_coords = coordinates[(node.down_child.time_step, node.down_child.node_index)]
            ax.plot([x, down_child_coords[0]], [y, down_child_coords[1]], 
                   color='#8a8f98', linewidth=2, alpha=0.8, zorder=1)
            
            # Add down movement label
            mid_x = (x + down_child_coords[0]) / 2
            mid_y = (y + down_child_coords[1]) / 2
            # Use a softer red for down moves
            ax.text(mid_x, mid_y - 0.1, '↓', ha='center', va='center', 
                   fontsize=12, color='#EF5350', weight='bold', zorder=3)
    
    # Draw nodes
    early_exercise_nodes = []
    for (t, i), (x, y) in coordinates.items():
        node = model.tree.nodes[(t, i)]
        
        # Check if this node would be exercised early (for American options)
        is_early_exercise = False
        if model.option_style == "american" and not node.is_terminal():
            if node.option_price is not None:
                exercise_value = node.get_exercise_value(model.K, model.option_type)
                
                # Calculate what the holding value would be (without early exercise)
                if node.up_child and node.down_child:
                    expected_value = (
                        model.p * node.up_child.option_price + 
                        model.q * node.down_child.option_price
                    )
                    import math
                    discount_factor = math.exp(-model.r * model.dt)
                    holding_value = expected_value * discount_factor
                    
                    # Early exercise is optimal if exercise value > holding value
                    # Add a small tolerance to avoid numerical precision issues
                    # Also require a minimum difference to avoid marking trivial cases
                    tolerance = 1e-6
                    # Early exercise triggers whenever exercise value exceeds holding value
                    is_early_exercise = (exercise_value > holding_value + tolerance)
                    
                    if is_early_exercise:
                        early_exercise_nodes.append((t, i))
        
        # Color mapping based on option price
        if node.option_price is not None:
            color_norm = (node.option_price - min_price) / (max_price - min_price + 1e-9)
            color = dark_ryg_cmap(color_norm)
        else:
            color = 'lightgray'
        
        # Node size and style based on state
        if is_early_exercise:
            # Early exercise node (highlighted with blue border, keep same size as regular)
            circle = plt.Circle((x, y), 0.2, color=color, ec='#42A5F5', linewidth=3, zorder=5)
            ax.add_patch(circle)
        elif node.is_terminal():
            # Terminal node
            circle = plt.Circle((x, y), 0.25, color=color, ec='#9aa0a6', linewidth=2, zorder=4)
            ax.add_patch(circle)
        else:
            # Regular node
            circle = plt.Circle((x, y), 0.2, color=color, ec='#9aa0a6', linewidth=1, zorder=3)
            ax.add_patch(circle)
        
        # Node labels
        if node.option_price is not None:
            label = f"S: ${node.stock_price:.1f}\nO: ${node.option_price:.2f}"
            if is_early_exercise:
                label += "\nEXERCISE"
        else:
            label = f"S: ${node.stock_price:.1f}\nO: ?"
        
        ax.text(x, y, label, ha='center', va='center', fontsize=8, 
               color='#ffffff', weight='bold', zorder=6)
    
    ax.set_title(f"{model.option_style.title()} {model.option_type.title()} Options Pricing Model\n"
                f"(Blue borders = Early Exercise)", 
                fontsize=16, weight='bold', pad=20, color='#e8eaed')
    ax.set_xlabel("Time Steps", fontsize=12, color='#d0d3d8')
    ax.set_ylabel("Node Position", fontsize=12, color='#d0d3d8')
    ax.set_xticks(range(model.n_steps + 1))
    ax.grid(True, linestyle='--', alpha=0.5, color='#2a2e35')
    ax.set_aspect('equal')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#BBDEFB', 
                  markersize=10, label='Early Exercise', markeredgecolor='#42A5F5', markeredgewidth=2)
    ]
    leg = ax.legend(handles=legend_elements, loc='upper right', facecolor='#151922', edgecolor='#444a55')
    for text in leg.get_texts():
        text.set_color('#e8eaed')
    
    plt.tight_layout()
    return fig, early_exercise_nodes

def compare_european_american(model_params):
    """
    Compare European vs American option prices.
    """
    # Create European model
    european_model = BinomialModel(**model_params, option_style='european')
    european_model.build_stock_price_tree()
    european_model.build_option_price_tree()
    
    # Create American model
    american_model = BinomialModel(**model_params, option_style='american')
    american_model.build_stock_price_tree()
    american_model.build_option_price_tree()
    
    european_price = european_model.get_option_price()
    american_price = american_model.get_option_price()
    
    return european_model, american_model, european_price, american_price

def main():
    st.set_page_config(
        page_title="American Options Pricing Model",
        page_icon="🇺🇸",
        layout="wide"
    )
    
    st.title("Binomial Options Pricing Model")
    st.markdown("**Visualisation of European & American options with early exercise analysis**")
    
    # Sidebar for parameters
    st.sidebar.header("Model Parameters")
    
    # Single-column inputs (to match tree_viz), remain in sidebar
    S0 = st.sidebar.number_input("Initial Stock Price (S₀)", min_value=0.01, value=100.00, step=1.00, format="%.2f")
    K = st.sidebar.number_input("Strike Price (K)", min_value=0.01, value=105.00, step=1.00, format="%.2f")
    T = st.sidebar.number_input("Time to Maturity (T)", min_value=0.01, max_value=10.00, value=0.25, step=0.01, format="%.2f")
    r = st.sidebar.number_input("Risk-free Rate (r)", min_value=0.000, max_value=1.000, value=0.050, step=0.001, format="%.3f")
    sigma = st.sidebar.number_input("Volatility (σ)", min_value=0.010, max_value=2.000, value=0.200, step=0.010, format="%.3f")
    n_steps = st.sidebar.slider("Number of Steps", 1, 10, 5, 1)
    
    option_type = st.sidebar.selectbox("Option Type", ["call", "put"])
    
    # Early exercise threshold fixed at 0 by design
    early_exercise_threshold = 0.0
    
    # Sidebar analysis options section
    st.sidebar.header("📊 Analysis Options")
    show_intermediate = st.sidebar.checkbox("Advanced View", value=False)
    
    try:
        # Model parameters
        model_params = {
            'S0': S0,
            'K': K,
            'T': T,
            'r': r,
            'sigma': sigma,
            'n_steps': n_steps,
            'option_type': option_type
        }
        
        # Always compare European vs American
        european_model, american_model, european_price, american_price = compare_european_american(model_params)
        
        # Display comparison results
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
        
        # Placeholder to keep layout stable whether advanced metrics are shown or not
        adv_placeholder = st.container()
        if show_intermediate:
            with adv_placeholder:
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
            fig_eu, _ = plot_american_options_tree(european_model, coordinates_eu)
            st.pyplot(fig_eu, use_container_width=True)
        
        with col2:
            st.subheader("🇺🇸 American Options Tree")
            coordinates_us = calculate_node_coordinates(american_model)
            fig_us, early_exercise_nodes = plot_american_options_tree(american_model, coordinates_us)
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
        
        # Early exercise explanation
        with st.expander("📚 Early Exercise Theory", expanded=False):
            st.markdown("""
            **American Options Early Exercise:**
            
            American options can be exercised at any time before expiration, unlike European options
            which can only be exercised at expiration.
            
            **Early Exercise Decision:**
            At each node, we compare:
            - **Holding Value**: Expected discounted value of continuing to hold the option
            - **Exercise Value**: Immediate payoff from exercising the option
            
            **Formula:**
            ```
            A_{n,k} = max(E[A_{n+1}], S_{n,k} - K)
            ```
            
            Where:
            - `E[A_{n+1}]` = Expected value of holding the option
            - `S_{n,k} - K` = Exercise value (for calls)
            - `K - S_{n,k}` = Exercise value (for puts)
            
            **Key Insights:**
            - **Call Options**: Rarely exercised early (no dividends)
            - **Put Options**: Often exercised early when deep in-the-money
            - **Early Exercise Premium**: Difference between American and European prices
            """)
        
    except ValueError as e:
        st.error(f"❌ Parameter Error: {e}")
    except Exception as e:
        st.error(f"❌ Unexpected Error: {e}")
        st.exception(e)

if __name__ == "__main__":
    main()
