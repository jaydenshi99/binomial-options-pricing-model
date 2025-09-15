"""
American Options Visualization with Early Exercise Analysis

This creates a visualization showing American options pricing with
early exercise decisions highlighted.
"""

import streamlit as st
import matplotlib.pyplot as plt
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
    Calculate x, y coordinates for each node in the tree for visualization.
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
            y = (y_offset - i) * (max_y_spread / (model.n_steps / 2))  # Scale y-coordinates
            coordinates[(t, node.node_index)] = (x, y)
    
    return coordinates

def plot_american_options_tree(model, coordinates, early_exercise_threshold=0.01):
    """
    Plot the binomial tree with early exercise decisions highlighted.
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Get all nodes for color mapping
    all_nodes = list(model.tree.nodes.values())
    all_option_prices = [node.option_price for node in all_nodes if node.option_price is not None]
    
    if not all_option_prices:
        min_price, max_price = 0, 1
    else:
        min_price, max_price = min(all_option_prices), max(all_option_prices)
    
    # Draw connections first - iterate through all nodes and draw connections to their children
    for (t, i), (x, y) in coordinates.items():
        node = model.tree.nodes[(t, i)]
        
        # Draw connection to up child
        if node.up_child:
            up_child_coords = coordinates[(node.up_child.time_step, node.up_child.node_index)]
            ax.plot([x, up_child_coords[0]], [y, up_child_coords[1]], 
                   'k-', linewidth=2, alpha=0.7, zorder=1)
            
            # Add up movement label
            mid_x = (x + up_child_coords[0]) / 2
            mid_y = (y + up_child_coords[1]) / 2
            ax.text(mid_x, mid_y + 0.1, '↑', ha='center', va='center', 
                   fontsize=12, color='green', weight='bold', zorder=3)
        
        # Draw connection to down child
        if node.down_child:
            down_child_coords = coordinates[(node.down_child.time_step, node.down_child.node_index)]
            ax.plot([x, down_child_coords[0]], [y, down_child_coords[1]], 
                   'k-', linewidth=2, alpha=0.7, zorder=1)
            
            # Add down movement label
            mid_x = (x + down_child_coords[0]) / 2
            mid_y = (y + down_child_coords[1]) / 2
            ax.text(mid_x, mid_y - 0.1, '↓', ha='center', va='center', 
                   fontsize=12, color='red', weight='bold', zorder=3)
    
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
                    min_difference = early_exercise_threshold  # Use the threshold parameter
                    is_early_exercise = (exercise_value > holding_value + tolerance and 
                                       exercise_value - holding_value > min_difference)
                    
                    if is_early_exercise:
                        early_exercise_nodes.append((t, i))
        
        # Color mapping based on option price
        if node.option_price is not None:
            color_norm = (node.option_price - min_price) / (max_price - min_price + 1e-9)
            color = plt.cm.RdYlGn(color_norm)
        else:
            color = 'lightgray'
        
        # Node size and style based on state
        if is_early_exercise:
            # Early exercise node (highlighted with red border)
            circle = plt.Circle((x, y), 0.3, color=color, ec='red', linewidth=4, zorder=5)
            ax.add_patch(circle)
        elif node.is_terminal():
            # Terminal node
            circle = plt.Circle((x, y), 0.25, color=color, ec='black', linewidth=2, zorder=4)
            ax.add_patch(circle)
        else:
            # Regular node
            circle = plt.Circle((x, y), 0.2, color=color, ec='black', linewidth=1, zorder=3)
            ax.add_patch(circle)
        
        # Node labels
        if node.option_price is not None:
            label = f"S: ${node.stock_price:.1f}\nO: ${node.option_price:.2f}"
            if is_early_exercise:
                label += "\n⚡EXERCISE"
        else:
            label = f"S: ${node.stock_price:.1f}\nO: ?"
        
        ax.text(x, y, label, ha='center', va='center', fontsize=8, 
               color='black', weight='bold', zorder=6)
    
    ax.set_title(f"American {model.option_type.title()} Options Pricing Model\n"
                f"Early Exercise Analysis (Red borders = Early Exercise)", 
                fontsize=16, weight='bold', pad=20)
    ax.set_xlabel("Time Steps", fontsize=12)
    ax.set_ylabel("Node Position", fontsize=12)
    ax.set_xticks(range(model.n_steps + 1))
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_aspect('equal')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', 
                  markersize=10, label='Not calculated'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                  markersize=10, label='Calculated'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                  markersize=10, label='Early Exercise', markeredgecolor='red', markeredgewidth=2)
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
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
    
    st.title("🇺🇸 American Options Pricing Model")
    st.markdown("**Visualization of American options with early exercise analysis**")
    
    # Sidebar for parameters
    st.sidebar.header("🎛️ Model Parameters")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        S0 = st.slider("Initial Stock Price (S₀)", 50.0, 200.0, 100.0, 0.5)
        K = st.slider("Strike Price (K)", 50.0, 200.0, 105.0, 0.5)
        T = st.slider("Time to Maturity (T)", 0.1, 2.0, 0.25, 0.05)
    
    with col2:
        r = st.slider("Risk-free Rate (r)", 0.01, 0.10, 0.05, 0.001, format="%.3f")
        sigma = st.slider("Volatility (σ)", 0.05, 0.50, 0.20, 0.01, format="%.2f")
        n_steps = st.slider("Time Steps (n)", 1, 10, 5, 1)
    
    option_type = st.sidebar.selectbox("Option Type", ["call", "put"])
    
    # Analysis options
    st.sidebar.header("📊 Analysis Options")
    show_comparison = st.sidebar.checkbox("Show European vs American Comparison", True)
    show_early_exercise = st.sidebar.checkbox("Highlight Early Exercise Nodes", True)
    
    # Early exercise sensitivity
    early_exercise_threshold = st.sidebar.slider(
        "Early Exercise Threshold ($)", 
        0.001, 0.10, 0.01, 0.001,
        help="Minimum difference required to mark early exercise (higher = less sensitive)"
    )
    
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
        
        if show_comparison:
            # Compare European vs American
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
            
            # Show both trees
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🇪🇺 European Options Tree")
                coordinates_eu = calculate_node_coordinates(european_model)
                fig_eu, _ = plot_american_options_tree(european_model, coordinates_eu, early_exercise_threshold)
                st.pyplot(fig_eu)
            
            with col2:
                st.subheader("🇺🇸 American Options Tree")
                coordinates_us = calculate_node_coordinates(american_model)
                fig_us, early_exercise_nodes = plot_american_options_tree(american_model, coordinates_us, early_exercise_threshold)
                st.pyplot(fig_us)
            
            # Early exercise analysis
            if early_exercise_nodes:
                st.subheader("⚡ Early Exercise Analysis")
                st.write(f"**Nodes where early exercise is optimal:** {len(early_exercise_nodes)}")
                
                for t, i in early_exercise_nodes:
                    node = american_model.tree.nodes[(t, i)]
                    exercise_value = node.get_exercise_value(american_model.K, american_model.option_type)
                    st.write(f"Node ({t},{i}): Stock=${node.stock_price:.2f}, "
                            f"Exercise Value=${exercise_value:.2f}, "
                            f"Option Price=${node.option_price:.2f}")
            else:
                st.subheader("⚡ Early Exercise Analysis")
                st.write("**No early exercise is optimal** - American option behaves like European option")
        
        else:
            # Show only American model
            american_model = BinomialModel(**model_params, option_style='american')
            american_model.build_stock_price_tree()
            american_model.build_option_price_tree()
            
            american_price = american_model.get_option_price()
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("American Option Price", f"${american_price:.4f}")
            
            with col2:
                st.metric("Tree Nodes", len(american_model.tree.nodes))
            
            with col3:
                st.metric("Terminal Nodes", len(american_model.tree.get_terminal_nodes()))
            
            # Show tree
            st.subheader("🇺🇸 American Options Tree")
            coordinates = calculate_node_coordinates(american_model)
            fig, early_exercise_nodes = plot_american_options_tree(american_model, coordinates, early_exercise_threshold)
            st.pyplot(fig)
        
        # Model details
        with st.expander("🔍 Model Details"):
            if show_comparison:
                st.json(american_model.get_model_info())
            else:
                st.json(american_model.get_model_info())
        
        # Early exercise explanation
        with st.expander("📚 Early Exercise Theory"):
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
