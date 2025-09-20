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

def calculate_pnl_at_node(node, entry_price, num_contracts, commission_per_contract, option_type, position_direction=1):
    """
    Calculate P&L at a given node for a specific trade.
    
    Parameters:
    - position_direction: 1 for long, -1 for short
    - option_type: 'call' or 'put' - determines the payoff structure
    """
    if node.option_price is None:
        return None
    
    # Current option value (positive for long, negative for short)
    current_value = node.option_price * num_contracts * 100 * position_direction
    
    # Entry cost (positive for long, negative for short)
    entry_cost = entry_price * num_contracts * 100
    
    # Commission costs (entry + exit) - always positive
    total_commission = commission_per_contract * num_contracts * 2
    
    # P&L calculation
    # For long: P&L = current_value - entry_cost - commission
    # For short: P&L = current_value - entry_cost - commission
    # (entry_cost is already negative for short positions)
    pnl = current_value - entry_cost - total_commission
    
    return pnl

def plot_american_options_tree(model, coordinates, show_pnl=False, entry_price=None, num_contracts=None, commission_per_contract=None, position_direction=1, option_type_for_pnl=None):
    """
    Plot the binomial tree with early exercise decisions highlighted or P&L analysis.
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
    
    if show_pnl and entry_price is not None and num_contracts is not None and commission_per_contract is not None:
        # P&L mode - calculate P&L for all nodes
        all_pnl_values = []
        for node in all_nodes:
            pnl = calculate_pnl_at_node(node, entry_price, num_contracts, commission_per_contract, option_type_for_pnl or model.option_type, position_direction)
            if pnl is not None:
                all_pnl_values.append(pnl)
        
        if not all_pnl_values:
            min_value, max_value = -1000, 1000
        else:
            min_value, max_value = min(all_pnl_values), max(all_pnl_values)
            # Ensure symmetric range for better visualization
            max_abs = max(abs(min_value), abs(max_value))
            min_value, max_value = -max_abs, max_abs
        
        # P&L color scheme: Red (losses) → Green (profits) with gradients, no yellow/orange
        pnl_colors = [
            '#DC2626',  # dark red for high losses
            '#EF4444',  # medium red for medium losses
            '#F87171',  # light red for small losses
            '#34D399',  # light green for small profits
            '#10B981',  # medium green for medium profits
            '#059669',  # dark green for high profits
        ]
        color_cmap = mcolors.LinearSegmentedColormap.from_list('pnl', pnl_colors, N=256)
    else:
        # Theoretical price mode
        all_option_prices = [node.option_price for node in all_nodes if node.option_price is not None]
        
        if not all_option_prices:
            min_value, max_value = 0, 1
        else:
            min_value, max_value = min(all_option_prices), max(all_option_prices)
        
        # Use the same P&L color scheme for theoretical prices - pure red-green gradient
        pnl_colors = [
            '#DC2626',  # dark red for low values
            '#EF4444',  # medium red for medium-low values
            '#F87171',  # light red for small values
            '#34D399',  # light green for medium-high values
            '#10B981',  # medium green for high values
            '#059669',  # dark green for highest values
        ]
        color_cmap = mcolors.LinearSegmentedColormap.from_list('pnl', pnl_colors, N=256)

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
        
        # Color mapping based on mode
        if show_pnl and entry_price is not None and num_contracts is not None and commission_per_contract is not None:
            # P&L mode
            pnl = calculate_pnl_at_node(node, entry_price, num_contracts, commission_per_contract, option_type_for_pnl or model.option_type, position_direction)
            if pnl is not None:
                # Normalize P&L to 0-1 range, with 0.5 being breakeven
                if max_value == min_value:
                    color_norm = 0.5  # breakeven
                else:
                    color_norm = (pnl - min_value) / (max_value - min_value)
                color = color_cmap(color_norm)
            else:
                color = 'lightgray'
        else:
            # Theoretical price mode
            if node.option_price is not None:
                color_norm = (node.option_price - min_value) / (max_value - min_value + 1e-9)
                color = color_cmap(color_norm)
            else:
                color = 'lightgray'
        
        # Node size and style based on state
        if is_early_exercise:
            # Early exercise node (highlighted with white border, keep same size as regular)
            circle = plt.Circle((x, y), 0.2, color=color, ec='#FFFFFF', linewidth=3, zorder=5)
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
        if show_pnl and entry_price is not None and num_contracts is not None and commission_per_contract is not None:
            # P&L mode labels
            if node.option_price is not None:
                pnl = calculate_pnl_at_node(node, entry_price, num_contracts, commission_per_contract, option_type_for_pnl or model.option_type, position_direction)
                if pnl is not None:
                    label = f"S: ${node.stock_price:.1f}\nO: ${node.option_price:.2f}\nP&L: ${pnl:.0f}"
                else:
                    label = f"S: ${node.stock_price:.1f}\nO: ?\nP&L: ?"
            else:
                label = f"S: ${node.stock_price:.1f}\nO: ?\nP&L: ?"
        else:
            # Theoretical price mode labels
            if node.option_price is not None:
                label = f"S: ${node.stock_price:.1f}\nO: ${node.option_price:.2f}"
                if is_early_exercise:
                    label += "\nEXERCISE"
            else:
                label = f"S: ${node.stock_price:.1f}\nO: ?"
        
        ax.text(x, y, label, ha='center', va='center', fontsize=8, 
               color='#ffffff', weight='bold', zorder=6)
    
    # Set title based on mode
    if show_pnl and entry_price is not None and num_contracts is not None and commission_per_contract is not None:
        ax.set_title(f"{model.option_style.title()} {model.option_type.title()} Options P&L Analysis\n"
                    f"(Red=Losses, Green=Profits)", 
                    fontsize=16, weight='bold', pad=20, color='#e8eaed')
    else:
        ax.set_title(f"{model.option_style.title()} {model.option_type.title()} Options Pricing Model\n"
                    f"(White borders = Early Exercise)", 
                    fontsize=16, weight='bold', pad=20, color='#e8eaed')
    
    ax.set_xlabel("Time Steps", fontsize=12, color='#d0d3d8')
    ax.set_ylabel("Node Position", fontsize=12, color='#d0d3d8')
    ax.set_xticks(range(model.n_steps + 1))
    ax.grid(True, linestyle='--', alpha=0.5, color='#2a2e35')
    ax.set_aspect('equal')
    
    # Add legend based on mode
    if show_pnl and entry_price is not None and num_contracts is not None and commission_per_contract is not None:
        # P&L legend - only red and green
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#DC2626', 
                      markersize=10, label='Losses'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#10B981', 
                      markersize=10, label='Profits')
        ]
    else:
        # Early exercise legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#BBDEFB', 
                      markersize=10, label='Early Exercise', markeredgecolor='#FFFFFF', markeredgewidth=2)
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
    
    # Sidebar analysis options section - moved to top
    st.sidebar.header("📊 Analysis Options")
    
    # Mode toggle
    analysis_mode = st.sidebar.radio(
        "Analysis Mode",
        ["Theoretical Prices", "P&L Analysis"],
        help="Choose between theoretical option pricing or profit/loss analysis"
    )
    
    show_intermediate = st.sidebar.checkbox("Advanced View", value=False)
    
    # Model parameters section
    st.sidebar.header("Model Parameters")
    S0 = st.sidebar.number_input("Initial Stock Price (S₀)", min_value=0.01, value=100.00, step=1.00, format="%.2f")
    K = st.sidebar.number_input("Strike Price (K)", min_value=0.01, value=105.00, step=1.00, format="%.2f")
    T = st.sidebar.number_input("Time to Maturity (T)", min_value=0.01, max_value=10.00, value=0.25, step=0.01, format="%.2f")
    r = st.sidebar.number_input("Risk-free Rate (r)", min_value=0.000, max_value=1.000, value=0.050, step=0.001, format="%.3f")
    sigma = st.sidebar.number_input("Volatility (σ)", min_value=0.010, max_value=2.000, value=0.200, step=0.010, format="%.3f")
    n_steps = st.sidebar.slider("Number of Steps", 1, 10, 4, 1)
    
    # Option type - only show in theoretical mode, hidden in P&L mode
    if analysis_mode == "Theoretical Prices":
        option_type = st.sidebar.selectbox("Option Type", ["call", "put"])
    else:
        # Default to call for P&L mode (will be overridden by position type)
        option_type = "call"
    
    # Early exercise threshold fixed at 0 by design
    early_exercise_threshold = 0.0
    
    
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
        
        # P&L inputs (only show when P&L mode is selected)
        pnl_params = {}
        if analysis_mode == "P&L Analysis":
            st.sidebar.header("💰 Trade Parameters")
            
            # Position type
            position_type = st.sidebar.selectbox(
                "Position Type",
                ["Long Call", "Short Call", "Long Put", "Short Put", "Custom"],
                help="Type of option position"
            )
            
            if position_type == "Custom":
                st.sidebar.markdown("**Custom Position Setup**")
                st.sidebar.markdown("Add multiple option legs for complex strategies")
                # For now, just show a placeholder for custom positions
                st.sidebar.info("Custom combinations coming soon!")
                price_per_option = float(european_price)
                num_options = 5
                position_direction = 1  # Long
            else:
                # Determine position direction and option type
                if "Long" in position_type:
                    position_direction = 1
                    option_type_for_pnl = "call" if "Call" in position_type else "put"
                else:  # Short
                    position_direction = -1
                    option_type_for_pnl = "call" if "Call" in position_type else "put"
                
                price_per_option = st.sidebar.number_input(
                    "Price per Option ($)", 
                    min_value=0.01, 
                    value=float(european_price), 
                    step=0.01, 
                    help="Price you paid per option (positive for long, negative for short)"
                )
                num_options = st.sidebar.number_input(
                    "Number of Options", 
                    min_value=1, 
                    value=5, 
                    step=1,
                    help="Number of individual options"
                )
            
            pnl_params['commission_per_contract'] = st.sidebar.number_input(
                "Commission per Contract ($)", 
                min_value=0.00, 
                value=0.30, 
                step=0.01, 
                help="Commission cost per contract (both entry and exit). 1 contract = 100 options"
            )
            
            # Calculate contracts and entry price
            pnl_params['num_contracts'] = num_options / 100  # Convert options to contracts
            pnl_params['entry_price'] = price_per_option * position_direction
            pnl_params['position_direction'] = position_direction
            pnl_params['option_type_for_pnl'] = option_type_for_pnl if position_type != "Custom" else option_type
            
            # Create P&L-specific models with the correct option type
            pnl_model_params = model_params.copy()
            pnl_model_params['option_type'] = option_type_for_pnl if position_type != "Custom" else option_type
            pnl_european_model, pnl_american_model, _, _ = compare_european_american(pnl_model_params)
            
            # Use P&L-specific models for visualization
            european_model = pnl_european_model
            american_model = pnl_american_model
        
        # Display comparison results
        st.markdown("### Option Pricing")
        col1, col2, col3, col4 = st.columns(4)
        
        if analysis_mode == "P&L Analysis" and pnl_params:
            # In P&L mode, show all 4 option types
            # Create models for both call and put options
            call_model_params = model_params.copy()
            call_model_params['option_type'] = 'call'
            put_model_params = model_params.copy()
            put_model_params['option_type'] = 'put'
            
            # Get all 4 option prices
            eu_call_model, am_call_model, eu_call_price, am_call_price = compare_european_american(call_model_params)
            eu_put_model, am_put_model, eu_put_price, am_put_price = compare_european_american(put_model_params)
            
            with col1:
                st.metric("European Call", f"${eu_call_price:.4f}")
            
            with col2:
                st.metric("European Put", f"${eu_put_price:.4f}")
            
            with col3:
                st.metric("American Call", f"${am_call_price:.4f}")
            
            with col4:
                st.metric("American Put", f"${am_put_price:.4f}")
        else:
            # In Theoretical Prices mode, show the original metrics
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
        
        # P&L Analysis metrics (only show when in P&L mode)
        if analysis_mode == "P&L Analysis" and pnl_params:
            st.markdown("### P&L Analysis")
            pnl_col1, pnl_col2, pnl_col3, pnl_col4 = st.columns(4)
            
            # Calculate P&L metrics
            entry_cost = pnl_params['entry_price'] * pnl_params['num_contracts'] * 100
            total_commission = pnl_params['commission_per_contract'] * pnl_params['num_contracts'] * 2
            
            with pnl_col1:
                st.metric("Entry Cost", f"${entry_cost:.2f}")
            
            with pnl_col2:
                st.metric("Total Commission", f"${total_commission:.2f}")
            
            with pnl_col3:
                # Calculate breakeven price
                breakeven_price = pnl_params['entry_price'] + (total_commission / (pnl_params['num_contracts'] * 100))
                st.metric("Breakeven Price", f"${breakeven_price:.2f}")
            
            with pnl_col4:
                # Calculate max loss (if option expires worthless)
                max_loss = entry_cost + total_commission
                st.metric("Max Loss", f"${max_loss:.2f}")
        
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
            show_pnl = analysis_mode == "P&L Analysis"
            fig_eu, _ = plot_american_options_tree(
                european_model, coordinates_eu, 
                show_pnl=show_pnl,
                entry_price=pnl_params.get('entry_price'),
                num_contracts=pnl_params.get('num_contracts'),
                commission_per_contract=pnl_params.get('commission_per_contract'),
                position_direction=pnl_params.get('position_direction', 1),
                option_type_for_pnl=pnl_params.get('option_type_for_pnl')
            )
            st.pyplot(fig_eu, use_container_width=True)
        
        with col2:
            st.subheader("🇺🇸 American Options Tree")
            coordinates_us = calculate_node_coordinates(american_model)
            fig_us, early_exercise_nodes = plot_american_options_tree(
                american_model, coordinates_us,
                show_pnl=show_pnl,
                entry_price=pnl_params.get('entry_price'),
                num_contracts=pnl_params.get('num_contracts'),
                commission_per_contract=pnl_params.get('commission_per_contract'),
                position_direction=pnl_params.get('position_direction', 1),
                option_type_for_pnl=pnl_params.get('option_type_for_pnl')
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

if __name__ == "__main__":
    main()
