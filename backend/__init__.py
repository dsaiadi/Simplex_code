"""
Internet Bandwidth Allocation Optimization System
Backend Module Initialization
"""

__version__ = "1.0.0"
__author__ = "Optimization Team"

# Import all optimization modules
from .core_optimizer import CoreOptimizer, FairnessMetrics
from .multi_objective import MultiObjectiveOptimizer, ParetoAnalyzer
from .time_varying import TimeVaryingOptimizer, TemporalAnalyzer
from .robust_optimizer import RobustOptimizer
from .data_generator import DataGenerator
from .visualizer import BandwidthVisualizer
from .benchmark_algorithms import BenchmarkAlgorithms
from .data_generator_enhanced import EnhancedDataGenerator
from .tier_optimizer import TierBasedOptimizer

__all__ = [
    'CoreOptimizer',
    'FairnessMetrics',
    'MultiObjectiveOptimizer',
    'ParetoAnalyzer',
    'TimeVaryingOptimizer',
    'TemporalAnalyzer',
    'RobustOptimizer',
    'DataGenerator',
    'BandwidthVisualizer',
    'BenchmarkAlgorithms',
    'EnhancedDataGenerator',
    'TierBasedOptimizer'
]
