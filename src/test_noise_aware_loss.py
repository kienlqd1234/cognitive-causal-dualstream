#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the noise-aware loss component in the dual-path SRL model.
This script demonstrates the effect of the noise-aware loss, Enhanced SRL, and Ca-TSU
on the overall model performance.
"""

import os
import sys
import logging
import numpy as np
import yaml
from datetime import datetime
import tensorflow as tf
import json

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import model components
from Model_tx_lf_dual_path import Model as Model_tx_lf_dual_path
from Executor_tx_lf_dual_path import Executor as Executor_tx_lf_dual_path
from ConfigLoader_tx_lf import config, config_model, update_config, logger

def configure_model_for_test():
    """
    Configure the model to use all enhancement modules for testing.
    """
    try:
        # Load and update configuration
        config_path = os.path.join(os.path.dirname(__file__), 'config_tx_lf_dual_path.yml')
        with open(config_path, 'r') as config_file:
            config_data = yaml.load(config_file, Loader=yaml.FullLoader)
        
        # Enable all enhancement modules
        config_data['model']['use_dual_path_srl'] = True
        config_data['model']['use_enhanced_srl'] = True
        config_data['model']['enhanced_srl_weight'] = 0.1
        
        config_data['model']['use_ca_tsu'] = True
        config_data['model']['ca_tsu_weight'] = 0.3
        config_data['model']['causal_loss_lambda'] = 0.3
        
        config_data['model']['use_noise_aware_loss'] = True
        config_data['model']['noise_aware_weight'] = 0.5
        
        # Set test configuration
        if 'test_config' not in config_data:
            config_data['test_config'] = {}
        
        config_data['test_config']['max_test_steps'] = 50
        config_data['test_config']['eval_steps'] = 10
        
        # Save the modified configuration back
        with open(config_path, 'w') as config_file:
            yaml.dump(config_data, config_file, default_flow_style=False)
        
        # Update the global configuration
        update_config(config_path)
        
        # Log configuration
        logger.info("Starting noise-aware loss test for enhanced dual-path SRL...")
        logger.info(f"Configuration - use_dual_path_srl: {config_model['use_dual_path_srl']}")
        logger.info(f"Configuration - use_enhanced_srl: {config_model['use_enhanced_srl']}")
        logger.info(f"Configuration - use_ca_tsu: {config_model['use_ca_tsu']}")
        logger.info(f"Configuration - causal_loss_lambda: {config_model['causal_loss_lambda']}")
        logger.info(f"Configuration - use_noise_aware_loss: {config_model['use_noise_aware_loss']}")
        logger.info(f"Configuration - noise_aware_weight: {config_model['noise_aware_weight']}")
        logger.info(f"Configuration - variant_type: {config_model['variant_type']}")
        
        # Print all model configuration values for debugging
        logger.info("Full model configuration:")
        for key, value in config_model.items():
            logger.info(f"  {key}: {value}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to configure model: {str(e)}")
        return False

def check_model_components(model):
    """
    Check if the model has all the required components for noise-aware loss.
    """
    logger.info("Checking model components for noise-aware loss...")
    
    # Check for basic model attributes
    logger.info(f"Model has 'use_noise_aware_loss': {hasattr(model, 'use_noise_aware_loss')}")
    if hasattr(model, 'use_noise_aware_loss'):
        logger.info(f"use_noise_aware_loss value: {model.use_noise_aware_loss}")
    
    logger.info(f"Model has 'noise_aware_weight': {hasattr(model, 'noise_aware_weight')}")
    if hasattr(model, 'noise_aware_weight'):
        logger.info(f"noise_aware_weight value: {model.noise_aware_weight}")
    
    # Check for loss components
    logger.info(f"Model has 'noise_aware_loss': {hasattr(model, 'noise_aware_loss')}")
    logger.info(f"Model has 'separate_noise_aware_loss': {hasattr(model, 'separate_noise_aware_loss')}")
    logger.info(f"Model has 'noise_aware_loss_raw': {hasattr(model, 'noise_aware_loss_raw')}")
    logger.info(f"Model has 'reliability_weights': {hasattr(model, 'reliability_weights')}")
    
    # Check for SRL components
    logger.info(f"Model has 'use_enhanced_srl': {hasattr(model, 'use_enhanced_srl')}")
    if hasattr(model, 'use_enhanced_srl'):
        logger.info(f"use_enhanced_srl value: {model.use_enhanced_srl}")
    
    logger.info(f"Model has 'enhanced_srl_loss': {hasattr(model, 'enhanced_srl_loss')}")
    logger.info(f"Model has 'enhanced_srl_loss_raw': {hasattr(model, 'enhanced_srl_loss_raw')}")
    
    # Check for Ca-TSU components
    logger.info(f"Model has 'use_ca_tsu': {hasattr(model, 'use_ca_tsu')}")
    if hasattr(model, 'use_ca_tsu'):
        logger.info(f"use_ca_tsu value: {model.use_ca_tsu}")
    
    logger.info(f"Model has 'causal_consistency_loss': {hasattr(model, 'causal_consistency_loss')}")
    logger.info(f"Model has 'causal_consistency_loss_raw': {hasattr(model, 'causal_consistency_loss_raw')}")
    
    # Check for loss components dictionary
    logger.info(f"Model has 'loss_components': {hasattr(model, 'loss_components')}")
    if hasattr(model, 'loss_components'):
        logger.info(f"loss_components keys: {model.loss_components.keys()}")
    
    return True

def manually_activate_components(model):
    """
    Manually activate the loss components in the model if they're not being used.
    """
    logger.info("Manually activating loss components...")
    
    # Set the flags to True
    model.use_enhanced_srl = True
    model.use_ca_tsu = True
    model.use_noise_aware_loss = True
    model.use_dual_path_srl = True
    
    # Set the weights
    model.enhanced_srl_weight = 0.1
    model.ca_tsu_weight = 0.3
    model.causal_loss_lambda = 0.3
    model.noise_aware_weight = 0.5
    
    logger.info("Components manually activated:")
    logger.info(f"use_enhanced_srl: {model.use_enhanced_srl}")
    logger.info(f"use_ca_tsu: {model.use_ca_tsu}")
    logger.info(f"use_noise_aware_loss: {model.use_noise_aware_loss}")
    logger.info(f"use_dual_path_srl: {model.use_dual_path_srl}")
    
    return True

def run_test():
    """
    Run the test with Ca-TSU, enhanced SRL, and noise-aware loss.
    """
    try:
        # Configure the model
        if not configure_model_for_test():
            logger.error("Failed to configure model. Exiting.")
            return False
        
        # Create results directory
        os.makedirs('results/model_test', exist_ok=True)
        
        # Create and initialize the executor with the model
        executor = Executor_tx_lf_dual_path(model_obj=Model_tx_lf_dual_path)
        
        # Verify model configuration before running test
        model = executor.model
        logger.info(f"Verifying model configuration...")
        logger.info(f"Ca-TSU enabled: {getattr(model, 'use_ca_tsu', False)}")
        logger.info(f"Enhanced SRL enabled: {getattr(model, 'use_enhanced_srl', False)}")
        logger.info(f"Noise-Aware Loss enabled: {getattr(model, 'use_noise_aware_loss', False)}")
        
        # Check model components
        check_model_components(model)
        
        # Manually activate components if needed
        manually_activate_components(model)
        
        # Ensure the model has the loss_components attribute
        if not hasattr(model, 'loss_components'):
            logger.warning("Model does not have loss_components dictionary. Creating one...")
            model.loss_components = {}
            
            # Add loss components to the dictionary
            if hasattr(model, 'enhanced_srl_loss'):
                model.loss_components['enhanced_srl_loss'] = model.enhanced_srl_loss
                logger.info("Added enhanced_srl_loss to loss_components")
                
            if hasattr(model, 'causal_consistency_loss'):
                model.loss_components['causal_consistency_loss'] = model.causal_consistency_loss
                logger.info("Added causal_consistency_loss to loss_components")
                
            if hasattr(model, 'noise_aware_loss'):
                model.loss_components['noise_aware_loss'] = model.noise_aware_loss
                logger.info("Added noise_aware_loss to loss_components")
                
            if hasattr(model, 'reliability_weights'):
                model.loss_components['reliability_weights'] = model.reliability_weights
                logger.info("Added reliability_weights to loss_components")
                
            logger.info(f"Created loss_components with keys: {model.loss_components.keys()}")
        
        # Run the test using the executor's test_noise_aware_loss method
        logger.info("Running test_noise_aware_loss...")
        executor.test_noise_aware_loss()
        
        # Save summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = f"results/model_test/enhancement_test_summary_{timestamp}.txt"
        
        with open(summary_file, 'w') as f:
            f.write("Enhancement Modules Test Summary\n")
            f.write("===============================\n\n")
            f.write(f"Test run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("Enabled modules:\n")
            f.write(f"- Ca-TSU: {getattr(model, 'use_ca_tsu', False)}\n")
            f.write(f"- Enhanced SRL: {getattr(model, 'use_enhanced_srl', False)}\n")
            f.write(f"- Noise-Aware Loss: {getattr(model, 'use_noise_aware_loss', False)}\n\n")
            f.write("Module weights:\n")
            f.write(f"- Ca-TSU weight: {getattr(model, 'ca_tsu_weight', 0.0)}\n")
            f.write(f"- Enhanced SRL weight: {getattr(model, 'enhanced_srl_weight', 0.0)}\n")
            f.write(f"- Noise-Aware weight: {getattr(model, 'noise_aware_weight', 0.0)}\n\n")
            f.write("Results are available in the 'results/model_test' directory.\n")
        
        logger.info(f"Test summary saved to {summary_file}")
        logger.info("Test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error running test: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = run_test()
    if success:
        print("Test completed successfully!")
        sys.exit(0)
    else:
        print("Test failed. Check logs for details.")
        sys.exit(1) 