"""
Internet Bandwidth Allocation Optimization System
Backend Module Initialization
"""

__version__ = "1.0.0"
__author__ = "Optimization Team"

# Import all optimization modules
from .core_optimizer import CoreOptimizer, FairnessMetrics
from .multi_objective import MultiObjectiveOptimizer, ParetoAnalyzer
from .robust_optimizer import RobustOptimizer
from .data_generator import DataGenerator
from .visualizer import BandwidthVisualizer
from .benchmark_algorithms import BenchmarkAlgorithms
from .data_generator_enhanced import EnhancedDataGenerator
from .tier_optimizer import TierBasedOptimizer
from .unified_optimizer import UnifiedOptimizer, ConvergenceTracker
from .convergence_visualizer import ConvergenceVisualizer
from .network_topology_optimizer import NetworkTopologyOptimizer, NetworkNode, NetworkLink, TrafficDemand
from .network_visualizer import NetworkVisualizer

__all__ = [
    'CoreOptimizer',
    'FairnessMetrics',
    'MultiObjectiveOptimizer',
    'ParetoAnalyzer',
    'RobustOptimizer',
    'DataGenerator',
    'BandwidthVisualizer',
    'BenchmarkAlgorithms',
    'EnhancedDataGenerator',
    'TierBasedOptimizer',
    'UnifiedOptimizer',
    'ConvergenceTracker',
    'ConvergenceVisualizer',
    'NetworkTopologyOptimizer',
    'NetworkNode',
    'NetworkLink',
    'TrafficDemand',
    'NetworkVisualizer'
]
