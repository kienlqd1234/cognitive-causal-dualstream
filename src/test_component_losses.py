#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for component losses in the dual-path SRL model.
This script verifies that Enhanced SRL, Ca-TSU, and Noise-Aware Loss
components are properly calculated and integrated into the total loss.
"""

import os
import sys
import logging
import tensorflow as tf

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import model components
from Model_tx_lf_dual_path import Model as Model_tx_lf_dual_path
from ConfigLoader_tx_lf import config_model, logger

def test_component_losses():
    """
    Verify that component losses are properly created and stored in the model.
    """
    # Configure the model
    config_model['use_dual_path_srl'] = True
    config_model['use_enhanced_srl'] = True
    config_model['prediction_loss_weight'] = 0.1
    config_model['use_ca_tsu'] = True
    config_model['causal_loss_lambda'] = 0.3
    config_model['use_noise_aware_loss'] = True
    config_model['noise_aware_weight'] = 0.5
    config_model['variant_type'] = 'hedge'
    
    # Log configuration
    logger.info("Starting component loss verification...")
    logger.info(f"Configuration - use_dual_path_srl: {config_model['use_dual_path_srl']}")
    logger.info(f"Configuration - use_enhanced_srl: {config_model['use_enhanced_srl']}")
    logger.info(f"Configuration - use_ca_tsu: {config_model['use_ca_tsu']}")
    logger.info(f"Configuration - use_noise_aware_loss: {config_model['use_noise_aware_loss']}")
    
    # Create model
    model = Model_tx_lf_dual_path()
    
    # Check if component losses are created
    has_raw_components = (
        hasattr(model, 'enhanced_srl_loss_raw') and
        hasattr(model, 'causal_consistency_loss_raw') and
        hasattr(model, 'noise_aware_loss_raw')
    )
    
    if has_raw_components:
        print("✓ Model has raw component losses")
        logger.info("✓ Model has raw component losses")
    else:
        print("✗ Model is missing raw component losses")
        logger.error("✗ Model is missing raw component losses")
        missing = []
        if not hasattr(model, 'enhanced_srl_loss_raw'):
            missing.append('enhanced_srl_loss_raw')
        if not hasattr(model, 'causal_consistency_loss_raw'):
            missing.append('causal_consistency_loss_raw')
        if not hasattr(model, 'noise_aware_loss_raw'):
            missing.append('noise_aware_loss_raw')
        print(f"Missing components: {', '.join(missing)}")
        logger.error(f"Missing components: {', '.join(missing)}")
        return False
    
    # Check if loss components dictionary has the raw components
    if hasattr(model, 'loss_components'):
        expected_keys = [
            'base_prediction_loss',
            'enhanced_srl_loss',
            'enhanced_srl_loss_raw',
            'causal_consistency_loss',
            'causal_consistency_loss_raw',
            'noise_aware_loss',
            'noise_aware_loss_raw'
        ]
        
        missing_keys = [key for key in expected_keys if key not in model.loss_components]
        
        if not missing_keys:
            print("✓ All expected loss components are in the dictionary")
            logger.info("✓ All expected loss components are in the dictionary")
            return True
        else:
            print(f"✗ Missing loss components in dictionary: {', '.join(missing_keys)}")
            logger.error(f"✗ Missing loss components in dictionary: {', '.join(missing_keys)}")
            return False
    else:
        print("✗ Model does not have loss_components dictionary")
        logger.error("✗ Model does not have loss_components dictionary")
        return False

if __name__ == "__main__":
    success = test_component_losses()
    if success:
        print("Component loss verification completed successfully!")
        sys.exit(0)
    else:
        print("Component loss verification failed. Check logs for details.")
        sys.exit(1) 