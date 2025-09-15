"""
Streamlit visualisation for Binomial Options Pricing Model Tree

This creates an interactive visualisation of the binomial tree structure.
"""

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from binomial_model import BinomialModel

def calculate_node_coordinates(model):
    """
    Calculate x,y coordinates for each node in the tree.
    
    Returns
    -------
    dict: {(time_step, node_index): (x, y)}
    """
    coordinates = {}
    
    for t in range(model.n_steps + 1):
        for i in range(t + 1):
            # X coordinate: time step
            x = t
            
            # Y coordinate: centered around 0, spaced by 2
            # For time step t, we have t+1 nodes, so center them
            y = (i - t/2) * 2
            
            coordinates[(t, i)] = (x, y)
    
    return coordinates

def plot_binomial_tree(model, coordinates, max_nodes_to_show=50):
    """
    Plot the binomial tree using matplotlib.
    
    Parameters
    ----------
    model : BinomialModel
        The binomial model
    coordinates : dict
        Node coordinates
    max_nodes_to_show : int
        Maximum number of nodes to display (for performance)
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Check if we should limit the display
    total_nodes = len(model.tree.nodes)
    if total_nodes > max_nodes_to_show:
        st.warning(f"Tree has {total_nodes} nodes. Showing simplified view (max {max_nodes_to_show} nodes).")
        # Show every nth node
        step = max(1, total_nodes // max_nodes_to_show)
        nodes_to_show = list(model.tree.nodes.keys())[::step]
    else:
        nodes_to_show = list(model.tree.nodes.keys())
    
    # Plot nodes
    for (t, i) in nodes_to_show:
        if (t, i) not in coordinates:
            continue
            
        node = model.tree.nodes[(t, i)]
        x, y = coordinates[(t, i)]
        
        # Color based on option price (if available)
        if node.option_price is not None:
            # Normalize option price for color mapping
            max_option_price = max(n.option_price for n in model.tree.nodes.values() if n.option_price is not None)
            color_intensity = node.option_price / max_option_price if max_option_price > 0 else 0
            color = plt.cm.RdYlBu_r(color_intensity)
        else:
            color = 'lightblue'
        
        # Plot node
        circle = plt.Circle((x, y), 0.3, color=color, alpha=0.7)
        ax.add_patch(circle)
        
        # Add labels
        ax.text(x, y-0.5, f'S: {node.stock_price:.1f}', 
                ha='center', va='top', fontsize=8)
        
        if node.option_price is not None:
            ax.text(x, y+0.5, f'O: {node.option_price:.2f}', 
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Draw connections between nodes
    for (t, i) in nodes_to_show:
        if (t, i) not in coordinates:
            continue
            
        node = model.tree.nodes[(t, i)]
        x, y = coordinates[(t, i)]
        
        # Draw line to up child
        if node.up_child and (node.up_child.time_step, node.up_child.node_index) in coordinates:
            child_x, child_y = coordinates[(node.up_child.time_step, node.up_child.node_index)]
            ax.plot([x, child_x], [y, child_y], 'k-', alpha=0.5, linewidth=1)
        
        # Draw line to down child  
        if node.down_child and (node.down_child.time_step, node.down_child.node_index) in coordinates:
            child_x, child_y = coordinates[(node.down_child.time_step, node.down_child.node_index)]
            ax.plot([x, child_x], [y, child_y], 'k-', alpha=0.5, linewidth=1)
    
    # Customize plot
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Node Position')
    ax.set_title('Binomial Options Pricing Tree')
    ax.grid(True, alpha=0.3)
    
    # Set axis limits
    ax.set_xlim(-0.5, model.n_steps + 0.5)
    
    # Add colorbar if option prices are available
    if any(n.option_price is not None for n in model.tree.nodes.values()):
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r, 
                                 norm=plt.Normalize(vmin=0, vmax=max(n.option_price for n in model.tree.nodes.values() if n.option_price is not None)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Option Theoretical Value')
    
    plt.tight_layout()
    return fig

def main():
    """Main Streamlit application."""
    
    st.set_page_config(
        page_title="Binomial Tree visualisation",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Binomial Options Pricing Tree visualisation")
    st.markdown("Interactive visualisation of the binomial tree structure")
    
    # Sidebar for parameters
    st.sidebar.header("Model Parameters")
    
    # Input parameters
    S0 = st.sidebar.number_input(
        "Initial Stock Price (S₀)",
        min_value=0.01,
        max_value=10000.0,
        value=100.0,
        step=1.0
    )
    
    K = st.sidebar.number_input(
        "Strike Price (K)",
        min_value=0.01,
        max_value=10000.0,
        value=105.0,
        step=1.0
    )
    
    T = st.sidebar.number_input(
        "Time to Maturity (T)",
        min_value=0.01,
        max_value=2.0,
        value=0.25,
        step=0.01
    )
    
    r = st.sidebar.number_input(
        "Risk-free Rate (r)",
        min_value=0.0,
        max_value=1.0,
        value=0.05,
        step=0.001,
        format="%.3f"
    )
    
    sigma = st.sidebar.number_input(
        "Volatility (σ)",
        min_value=0.01,
        max_value=2.0,
        value=0.2,
        step=0.01,
        format="%.3f"
    )
    
    n_steps = st.sidebar.slider(
        "Number of Steps",
        min_value=2,
        max_value=20,  # Limited for visualisation performance
        value=5,
        help="Limited to 20 steps for better visualisation performance"
    )
    
    option_type = st.sidebar.selectbox(
        "Option Type",
        options=['call', 'put']
    )
    
    # Create the model
    try:
        model = BinomialModel(S0, K, T, r, sigma, n_steps, option_type)
        
        # Build the tree
        model.build_stock_price_tree()
        model.build_option_price_tree()
        
        # Calculate coordinates
        coordinates = calculate_node_coordinates(model)
        
        # Display model information
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Stock Price", f"${S0:.2f}")
            st.metric("Strike Price", f"${K:.2f}")
            st.metric("Time to Maturity", f"{T:.2f} years")
        
        with col2:
            st.metric("Risk-free Rate", f"{r:.1%}")
            st.metric("Volatility", f"{sigma:.1%}")
            st.metric("Steps", f"{n_steps}")
        
        with col3:
            st.metric("Up Factor", f"{model.u:.4f}")
            st.metric("Down Factor", f"{model.d:.4f}")
            st.metric("Risk-neutral Prob", f"{model.p:.4f}")
        
        # Calculate and display option price
        option_price = model.get_option_price()
        st.success(f"**Option Theoretical Value: ${option_price:.4f}**")
        
        # Tree statistics
        st.info(f"Tree Statistics: {len(model.tree.nodes)} total nodes, {len(model.tree.get_terminal_nodes())} terminal nodes")
        
        # Plot the tree
        st.header("Binomial Tree Visualisation")
        
        fig = plot_binomial_tree(model, coordinates)
        st.pyplot(fig)
        
        # Show tree data
        if st.checkbox("Show Tree Data"):
            st.subheader("Tree Node Details")
            
            # Create a DataFrame-like display
            tree_data = []
            for (t, i), node in model.tree.nodes.items():
                tree_data.append({
                    'Time': t,
                    'Node': i,
                    'Stock Price': f"{node.stock_price:.2f}",
                    'Option Price': f"{node.option_price:.4f}" if node.option_price is not None else "N/A"
                })
            
            st.dataframe(tree_data, use_container_width=True)
        
    except ValueError as e:
        st.error(f"❌ Invalid parameters: {e}")
        st.info("Please adjust the parameters in the sidebar.")

if __name__ == "__main__":
    main()
