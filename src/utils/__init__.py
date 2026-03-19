"""
Utility Functions for IP Protection Research

This package provides various utility functions for model handling,
file operations, and experiment management.
"""

from .model_utils import (
    setup_model,
    save_encrypted_model,
    load_encrypted_model,
    verify_model_integrity
)

from .imagenet_eval import (
    load_synset_mapping,
    ImageNetValidationDataset,
    validate_model,
    ModelEvaluator,
    evaluate_topk_accuracy,
)

from .metrics import (
    AccuracyMetrics,
    PerformanceTracker,
    calculate_accuracy_drop,
    generate_evaluation_report
)

__all__ = [
    'setup_model',
    'save_encrypted_model',
    'load_encrypted_model',
    'verify_model_integrity',
    'load_config',
    'save_config',
    'merge_configs',
    'validate_config',
    'load_synset_mapping',
    'ImageNetValidationDataset',
    'validate_model',
    'ModelEvaluator',
    'evaluate_topk_accuracy',
    'AccuracyMetrics',
    'PerformanceTracker',
    'calculate_accuracy_drop',
    'generate_evaluation_report'
]
