"""
Routing Optimization System

A complete last-mile delivery route optimization system that performs:
- Geographic clustering for fair workload distribution
- Route optimization using VRP/TSP
- Business impact calculation
- Visual reports and interactive maps
"""

__version__ = "1.0.0"

from .clustering import PODClusteringSystem, BranchManager
from .data_loader import DataLoader
from .route_optimizer import RouteOptimizer
from .metrics import MetricsCalculator

__all__ = [
    'PODClusteringSystem',
    'BranchManager',
    'DataLoader',
    'RouteOptimizer',
    'MetricsCalculator',
]
