#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple test script to verify component losses are working correctly.
This script can be run directly without any command-line parameters.
"""

import os
import sys
import tensorflow as tf
import logging
import numpy as np

# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import model components
from Model_tx_lf_dual_path import Model
from Executor_tx_lf_dual_path import Executor
from ConfigLoader_tx_lf import config_model

def test_component_losses():
    """Simple test to verify component losses are working"""
    
    logger.info("Starting component losses test...")
    
    # Configure model with all enhancement modules enabled
    config_model.update({
        'use_enhanced_srl': True,
        'use_ca_tsu': True,
        'use_noise_aware_loss': True,
        'use_dual_path_srl': True,
        'causal_loss_lambda': 0.3,
        'noise_aware_weight': 0.5,
        'prediction_loss_weight': 0.1,
        'variant_type': 'hedge',  # Non-discriminative for generative ATA
        'batch_size': 4,
        'max_n_days': 5,
        'max_n_msgs': 10,
        'max_n_words': 20,
        'word_embed_size': 50,
        'y_size': 2,
    })
    
    logger.info("Configuration updated with enhancement modules")
    
    # Create model
    tf.reset_default_graph()
    model = Model()
    
    logger.info("Model created, assembling graph...")
    model.assemble_graph()
    logger.info("Graph assembled successfully")
    
    # Check if loss components exist
    components_to_check = [
        'enhanced_srl_loss', 
        'causal_consistency_loss', 
        'noise_aware_loss',
        'separate_noise_aware_loss',
        'weighted_causal_loss'
    ]
    
    logger.info("Checking component losses...")
    found_components = []
    missing_components = []
    
    for component in components_to_check:
        if hasattr(model, component) and getattr(model, component) is not None:
            value = getattr(model, component)
            logger.info(f"✓ {component}: {value.name if hasattr(value, 'name') else value}")
            found_components.append(component)
        else:
            logger.warning(f"✗ {component}: NOT FOUND")
            missing_components.append(component)
    
    # Check loss components dictionary
    if hasattr(model, 'loss_components'):
        logger.info(f"Loss components dictionary: {list(model.loss_components.keys())}")
    else:
        logger.warning("Loss components dictionary not found")
    
    # Test with a simple forward pass
    logger.info("Testing with a simple forward pass...")
    
    try:
        with tf.Session() as sess:
            # Initialize variables
            sess.run(tf.global_variables_initializer())
            
            # Create dummy data
            batch_size = 2
            max_n_days = 3
            max_n_msgs = 5
            max_n_words = 10
            
            dummy_data = {
                model.batch_size: batch_size,
                model.stock_ph: np.zeros(batch_size, dtype=np.int32),
                model.T_ph: np.ones(batch_size, dtype=np.int32) * 2,  # Target day 2
                model.n_words_ph: np.ones((batch_size, max_n_days, max_n_msgs), dtype=np.int32) * 3,
                model.n_msgs_ph: np.ones((batch_size, max_n_days), dtype=np.int32) * 3,
                model.y_ph: np.random.random((batch_size, max_n_days, 2)).astype(np.float32),
                model.price_ph: np.random.random((batch_size, max_n_days, 3)).astype(np.float32),
                model.mv_percent_ph: np.random.randint(0, 3, (batch_size, max_n_days), dtype=np.int32),
                model.word_ph: np.ones((batch_size, max_n_days, max_n_msgs, max_n_words), dtype=np.int32),
                model.ss_index_ph: np.zeros((batch_size, max_n_days, max_n_msgs), dtype=np.int32),
                model.is_training_phase: True
            }
            
            # Add message mask if it exists
            if hasattr(model, 'msg_mask_ph'):
                msg_mask = np.zeros((batch_size, max_n_days, max_n_msgs), dtype=bool)
                for b in range(batch_size):
                    for d in range(max_n_days):
                        msg_mask[b, d, :3] = True  # First 3 messages are valid
                dummy_data[model.msg_mask_ph] = msg_mask
            
            # Run forward pass to get loss values
            fetches = {'total_loss': model.loss}
            
            # Add component losses if they exist
            for component in found_components:
                if hasattr(model, component):
                    fetches[component] = getattr(model, component)
            
            logger.info("Running forward pass...")
            results = sess.run(fetches, feed_dict=dummy_data)
            
            # Log the results
            logger.info("Forward pass results:")
            logger.info(f"Total Loss: {results['total_loss']:.6f}")
            
            for component in found_components:
                if component in results:
                    value = results[component]
                    if np.isscalar(value) or (hasattr(value, 'size') and value.size == 1):
                        logger.info(f"{component}: {float(value):.6f}")
                    else:
                        logger.info(f"{component}: {value} (shape: {value.shape if hasattr(value, 'shape') else 'N/A'})")
            
            # Check if component losses are non-zero
            non_zero_components = []
            zero_components = []
            
            for component in found_components:
                if component in results:
                    value = results[component]
                    if np.isscalar(value) or (hasattr(value, 'size') and value.size == 1):
                        if abs(float(value)) > 1e-8:
                            non_zero_components.append(component)
                        else:
                            zero_components.append(component)
            
            logger.info(f"Non-zero components: {non_zero_components}")
            logger.info(f"Zero components: {zero_components}")
            
            if non_zero_components:
                logger.info("✓ SUCCESS: Some component losses are non-zero!")
            else:
                logger.warning("✗ WARNING: All component losses are zero")
                
    except Exception as e:
        logger.error(f"Error during forward pass: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("COMPONENT LOSSES TEST SUMMARY")
    logger.info("="*60)
    logger.info(f"Found components: {len(found_components)}")
    logger.info(f"Missing components: {len(missing_components)}")
    
    if found_components:
        logger.info("✓ Test completed successfully!")
        return True
    else:
        logger.error("✗ Test failed - no component losses found")
        return False

if __name__ == "__main__":
    success = test_component_losses()
    if success:
        print("\n✓ Component losses test completed successfully!")
        sys.exit(0)
    else:
        print("\n✗ Component losses test failed!")
        sys.exit(1) 