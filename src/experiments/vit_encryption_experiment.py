#!/usr/bin/env python3
"""
Usage:
    python src/experiments/vit_encryption_experiment.py --model google/vit-base-patch16-224 --model facebook/deit-tiny-patch16-224 --mode basic --strategy top-k --k 6
    uv run src/experiments/vit_encryption_experiment.py --model facebook/deit-tiny-patch16-224 --mode advanced
"""

import argparse
import json
import logging
import random
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.vit_analyzer import VitEncryptionAnalyzer
from src.utils.metrics import generate_evaluation_report


def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Vision Transformer IP Protection Experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model parameters
    parser.add_argument(
        '--model', type=str, default='google/vit-base-patch16-224',
        help='HuggingFace model identifier'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device for computation (e.g., cuda, cpu, cuda:0, cuda:1)'
    )
    parser.add_argument(
        '--local-model-path', type=str, default=None,
        help='Path to local model files'
    )

    # Dataset parameters
    parser.add_argument(
        '--imagenet-path', type=str, default='dataset/imagenet/val',
        help='Path to ImageNet validation dataset'
    )
    parser.add_argument(
        '--batch-size', type=int, default=64,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--num-workers', type=int, default=8,
        help='Number of data loading workers'
    )

    # Encryption parameters
    parser.add_argument(
        '--mode', type=str, default='basic',
        choices=['basic', 'advanced'],
        help='Encryption mode'
    )
    parser.add_argument(
        '--strategy', type=str, default='top-k',
        choices=['top-k', 'random-k', 'last-k'],
        help='Layer selection strategy for basic mode'
    )
    parser.add_argument(
        '--k', type=int, default=1,
        help='Number of layers to encrypt for basic mode'
    )
    parser.add_argument(
        '--arnold-key', type=int, nargs='+', default=None,
        metavar='K',
        help='Arnold Cat Map key parameters [N, a, b, c, d]. If not provided, a random key will be generated.'
    )
    parser.add_argument(
        '--arnold-iterations', type=int, default=None,
        help='Number of iterations for Arnold Cat Map if key is auto-generated. If not provided, sampled from [3, 34].'
    )
    parser.add_argument(
        '--num-permutation-matrices', type=int, default=6,
        help='Number of permutation matrices to generate'
    )

    # Experiment parameters
    parser.add_argument(
        '--output-dir', type=str, default='results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--experiment-name', type=str, default=None,
        help='Name for this experiment (auto-generated if not provided)'
    )
    parser.add_argument(
        '--save-checkpoints', action='store_true', default=True,
        help='Save model checkpoints during encryption'
    )
    parser.add_argument(
        '--no-save-checkpoints', dest='save_checkpoints', action='store_false',
        help='Do not save model checkpoints'
    )
    parser.add_argument(
        '--random-seed', type=int, default=42,
        help='Random seed for reproducibility'
    )

    # Logging
    parser.add_argument(
        '--log-level', type=str, default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress console output (log to file only)'
    )
    
    return parser.parse_args()


def setup_logging(log_level: str = 'INFO', quiet: bool = False) -> None:
    """Setup logging configuration from command line parameters.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        quiet: If True, suppress console output
    """
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add console handler unless quiet mode
    if not quiet:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)


def save_experiment_config(config_dict: dict, experiment_dir: Path) -> None:
    """Save experiment configuration to JSON file.
    
    Args:
        config_dict: Configuration dictionary
        experiment_dir: Directory to save config file
    """
    config_file = experiment_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)


def _model_prefix(model_name: str) -> str:
    """Return 'deit', 'vit', or '' based on model name for directory naming."""
    if not model_name:
        return ""
    name_lower = model_name.lower()
    if "deit" in name_lower:
        return "deit"
    if "vit" in name_lower:
        return "vit"
    return ""


def setup_experiment_directory(output_dir: str, config: dict, experiment_name: str = None) -> Path:
    """Setup experiment directory with mode, strategy, K and timestamp."""
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = config.get('mode', 'basic')
        k = config.get('k', 1)
        strategy = config.get('strategy', 'top-k')
        model_name = config.get('model_name') or config.get('model', '')
        prefix = _model_prefix(model_name)
        prefix_part = f"{prefix}_" if prefix else ""

        if mode == 'advanced':
            experiment_name = f"{prefix_part}advanced_all_{timestamp}"
        else:
            experiment_name = f"{prefix_part}basic_{strategy}_k{k}_{timestamp}"
    
    experiment_dir = Path(output_dir) / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    return experiment_dir


def main():
    """Main experiment function."""
    args = parse_arguments()
    
    # Configuration is now purely from CLI arguments
    config_dict = vars(args)
    
    # Handle local model path - check for local model first, then use provided path
    if not config_dict.get('local_model_path'):
        model_name = config_dict.get('model', 'google/vit-base-patch16-224')
        if '/' in model_name:
            model_identifier = model_name.split('/')[-1]
        else:
            model_identifier = model_name
        local_model_path = Path(f'model/{model_identifier}')
        if local_model_path.exists() and local_model_path.is_dir():
            config_dict['local_model_path'] = str(local_model_path)
    
    # Default values for missing keys that were previously in config
    config_dict.setdefault('model_name', config_dict.get('model', 'google/vit-base-patch16-224'))
    config_dict.setdefault('device', 'cuda')
    config_dict.setdefault('imagenet_path', 'dataset/imagenet/val')
    config_dict.setdefault('batch_size', 64)
    config_dict.setdefault('num_workers', 8)
    config_dict.setdefault('num_permutation_matrices', 6)
    config_dict.setdefault('output_dir', 'results')
    config_dict.setdefault('save_checkpoints', True)
    config_dict.setdefault('log_level', 'INFO')
    config_dict.setdefault('random_seed', 42)
    
    # Setup experiment directory
    experiment_dir = setup_experiment_directory(
        config_dict['output_dir'], 
        config_dict,
        args.experiment_name
    )
    
    # Setup logging from command line parameters
    setup_logging(log_level=args.log_level, quiet=args.quiet)
    logger = logging.getLogger(__name__)
    
    # Add file handler for experiment-specific logging
    log_file = experiment_dir / "experiment.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, args.log_level.upper()))
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(file_handler)
    
    logger.info("Starting Vision Transformer IP Protection Experiment")
    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info(f"Configuration: {config_dict}")
    
    # Set random seeds
    if config_dict.get('random_seed'):
        set_random_seeds(config_dict['random_seed'])
        logger.info(f"Random seed set to: {config_dict['random_seed']}")
    
    # Save experiment configuration
    save_experiment_config(config_dict, experiment_dir)
    
    try:
        # Initialize analyzer
        logger.info("Initializing ViT Encryption Analyzer...")
        
        # Handle Arnold key generation if not provided
        arnold_key = config_dict.get('arnold_key')
        if arnold_key is None:
            from src.encryption.arnold_transform import generate_arnold_key
            # Use hidden size (usually 768 for base) for key generation
            # We'll get the actual size from the model if possible, or use a default
            matrix_size = 768 # Default for ViT-base
            iterations = config_dict.get('arnold_iterations') # Can be None now
            arnold_key = generate_arnold_key(matrix_size, iterations=iterations, seed=config_dict.get('random_seed'))
            logger.info(f"Generated random Arnold key: {arnold_key}")
            config_dict['arnold_key'] = arnold_key

        analyzer = VitEncryptionAnalyzer(
            model_name=config_dict['model_name'],
            batch_size=config_dict['batch_size'],
            num_workers=config_dict['num_workers'],
            num_extra_layers=config_dict.get('num_extra_layers', 4),
            password=None,
            arnold_key=arnold_key,
            device=config_dict['device'],
            imagenet_path=config_dict['imagenet_path'],
            local_model_path=config_dict.get('local_model_path')
        )
        
        logger.info(f"Initial model accuracy: {analyzer.initial_accuracy:.2%}")
        
        # Run encryption experiment
        logger.info(f"Starting model encryption (Mode: {config_dict['mode']}, Strategy: {config_dict['strategy']}, K: {config_dict['k']})...")
        performance_tracker = analyzer.encrypt_model_progressive(
            output_dir=experiment_dir / "checkpoints",
            save_checkpoints=config_dict['save_checkpoints'],
            mode=config_dict['mode'],
            strategy=config_dict['strategy'],
            k=config_dict['k']
        )
        
        # Save final results
        logger.info("Saving final results...")
        performance_tracker.save_to_file(experiment_dir / "final_performance.json")
        
        # Generate evaluation report
        report = generate_evaluation_report(
            performance_tracker,
            output_path=experiment_dir / "evaluation_report.txt"
        )
        
        logger.info("Experiment completed successfully!")
        logger.info(f"Final accuracy: {performance_tracker.get_current_accuracy():.2%}")
        logger.info(f"Total accuracy drop: {performance_tracker.get_total_accuracy_drop():.2%}")
        logger.info(f"Encrypted layers: {len(performance_tracker.encryption_steps)}")
        
        print(f"\nExperiment completed! Results saved to: {experiment_dir}")
        print(f"Final accuracy: {performance_tracker.get_current_accuracy():.2%}")
        print(f"Total accuracy drop: {performance_tracker.get_total_accuracy_drop():.2%}")
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}", exc_info=True)
        print(f"Experiment failed: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
