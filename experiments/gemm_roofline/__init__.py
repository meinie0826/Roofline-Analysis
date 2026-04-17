"""
GEMM Roofline Analysis Package

This package provides tools for analyzing GEMM performance
across different matrix shapes using the Roofline model.
"""

from .benchmark_roofline import (
    benchmark_shape,
    calculate_arithmetic_intensity,
    generate_shapes,
    get_gpu_specs,
    run_benchmark,
    GEMMResult,
)

__all__ = [
    'benchmark_shape',
    'calculate_arithmetic_intensity',
    'generate_shapes',
    'get_gpu_specs',
    'run_benchmark',
    'GEMMResult',
]
