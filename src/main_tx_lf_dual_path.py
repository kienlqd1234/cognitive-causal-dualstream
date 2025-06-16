#!/usr/local/bin/python
import argparse
import os
import logging
from Executor_tx_lf_dual_path import Executor
from Model_tx_lf_dual_path import Model
from ConfigLoader_tx_lf import logger, config_model

def main():
    parser = argparse.ArgumentParser(description='Enhanced Dual-Path SRL model with noise-aware loss')
    
    # Training and evaluation options
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--dev', action='store_true', help='Run development evaluation')
    parser.add_argument('--test', action='store_true', help='Run test evaluation')
    parser.add_argument('--test_noise_aware', action='store_true', help='Run noise-aware loss test training and evaluation')
    
    # Configuration options
    parser.add_argument('--epochs', type=int, help='Number of epochs to train (overrides config)')
    parser.add_argument('--batch_size', type=int, help='Batch size (overrides config)')
    parser.add_argument('--noise_aware_weight', type=float, help='Weight for noise-aware loss (overrides config)')
    
    args = parser.parse_args()
    
    # Override config with command line arguments if provided
    if args.epochs:
        config_model['n_epochs'] = args.epochs
        logger.info(f"Overriding n_epochs with {args.epochs}")
        
    if args.batch_size:
        config_model['batch_size'] = args.batch_size
        logger.info(f"Overriding batch_size with {args.batch_size}")
        
    if args.noise_aware_weight is not None:
        config_model['noise_aware_weight'] = args.noise_aware_weight
        logger.info(f"Overriding noise_aware_weight with {args.noise_aware_weight}")
    
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Initialize executor
    executor = Executor(Model)
    
    # Run requested operations
    if args.test_noise_aware:
        logger.info("Running noise-aware loss test")
        # Ensure noise-aware loss is enabled in config
        if not config_model.get('use_noise_aware_loss', False):
            logger.warning("Enabling noise_aware_loss for test_noise_aware")
            config_model['use_noise_aware_loss'] = True
        executor.test_noise_aware_loss()
    elif args.train and args.dev:
        logger.info("Running training and development evaluation")
        executor.train_and_dev()
    elif args.test:
        logger.info("Running test evaluation")
        executor.restore_and_test()
    else:
        logger.info("No operation specified. Please use --train, --test, or --test_noise_aware")
        parser.print_help()

if __name__ == "__main__":
    main() 