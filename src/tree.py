"""
Simple Tree class for Binomial Tree Management

Minimal implementation for managing nodes and calculations.
"""

from typing import Dict, List, Tuple, Optional
from .node import BinomialNode


class BinomialTree:
    """
    A simple tree structure for the binomial model.
    
    Attributes
    ----------
    model : BinomialModel
        The binomial model instance
    nodes : Dict[Tuple[int, int], BinomialNode]
        Dictionary mapping (time_step, node_index) to nodes
    root : Optional[BinomialNode]
        Root node of the tree
    """
    
    def __init__(self, model):
        """
        Initialize the tree with a binomial model.
        
        Parameters
        ----------
        model : BinomialModel
            The binomial model instance
        """
        self.model = model
        self.nodes: Dict[Tuple[int, int], BinomialNode] = {}
        self.root: Optional[BinomialNode] = None
    
    def build_stock_price_tree(self) -> None:
        """
        Build the stock price tree structure.
        
        Creates all nodes with their stock prices and parent-child relationships.
        """
        # Clear existing nodes
        self.nodes.clear()
        
        # Create root node
        root = BinomialNode(
            time_step=0,
            node_index=0,
            stock_price=self.model.S0
        )
        self.nodes[(0, 0)] = root
        self.root = root
        
        # Build tree level by level
        for t in range(1, self.model.n_steps + 1):
            for i in range(t + 1):
                # Calculate stock price: S0 * u^i * d^(t-i)
                stock_price = self.model.S0 * (self.model.u ** i) * (self.model.d ** (t - i))
                
                # Create node
                node = BinomialNode(
                    time_step=t,
                    node_index=i,
                    stock_price=stock_price
                )
                self.nodes[(t, i)] = node
                
                # Set up parent-child relationships
                if i < t:  # Has up parent
                    up_parent = self.nodes[(t-1, i)]
                    up_parent.up_child = node
                    node.parent = up_parent
                
                if i > 0:  # Has down parent
                    down_parent = self.nodes[(t-1, i-1)]
                    down_parent.down_child = node
                    if node.parent is None:  # Only set if not already set
                        node.parent = down_parent
    
    def build_option_price_tree(self, option_type: str = None) -> None:
        """
        Calculate option prices using backward induction.
        
        Parameters
        ----------
        option_type : str, optional
            Option type ('call' or 'put'). If None, uses model's option_type.
        """
        if option_type is None:
            option_type = self.model.option_type
        
        # Step 1: Calculate terminal payoffs
        terminal_time = self.model.n_steps
        for i in range(terminal_time + 1):
            node = self.nodes[(terminal_time, i)]
            node.option_price = node.get_payoff(self.model.K, option_type)
        
        # Step 2: Backward induction
        for t in range(terminal_time - 1, -1, -1):
            for i in range(t + 1):
                node = self.nodes[(t, i)]
                
                # Get child nodes
                up_child = node.up_child
                down_child = node.down_child
                
                if up_child is None or down_child is None:
                    continue
                
                # Calculate expected value using risk-neutral probability
                expected_value = (
                    self.model.p * up_child.option_price + 
                    self.model.q * down_child.option_price
                )
                
                # Discount to present value using risk-free rate
                import math
                discounted_value = expected_value * math.exp(-self.model.r * self.model.dt)
                
                node.option_price = discounted_value
    
    def get_option_price(self) -> float:
        """
        Get the current option price from the root node.
        
        Returns
        -------
        float
            Option price at time 0
        """
        if self.root is None:
            raise ValueError("Tree not built yet. Call build_stock_price_tree() first.")
        
        if self.root.option_price is None:
            raise ValueError("Option prices not calculated yet. Call build_option_price_tree() first.")
        
        return self.root.option_price
    
    def get_nodes_at_time(self, time_step: int) -> List[BinomialNode]:
        """
        Get all nodes at a specific time step.
        
        Parameters
        ----------
        time_step : int
            Time step
            
        Returns
        -------
        List[BinomialNode]
            List of nodes at the specified time step
        """
        nodes = []
        for i in range(time_step + 1):
            if (time_step, i) in self.nodes:
                nodes.append(self.nodes[(time_step, i)])
        return nodes
    
    def get_terminal_nodes(self) -> List[BinomialNode]:
        """
        Get all terminal nodes (final time step).
        
        Returns
        -------
        List[BinomialNode]
            List of terminal nodes
        """
        return self.get_nodes_at_time(self.model.n_steps)
    
    def __repr__(self) -> str:
        return f"BinomialTree(n_steps={self.model.n_steps}, nodes={len(self.nodes)})"
    
    def __str__(self) -> str:
        return f"Binomial Tree with {len(self.nodes)} nodes"
