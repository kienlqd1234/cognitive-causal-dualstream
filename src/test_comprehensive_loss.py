#!/usr/bin/env python
# Test script to verify all loss components (prediction, causal consistency, enhanced SRL, noise-aware)
import os
import sys
import tensorflow as tf
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import needed modules
from src.Model_tx_lf_dual_path import Model
from src.Executor_tx_lf_dual_path import Executor
import src.ConfigLoader_tx_lf as config_loader

def test_loss_components():
    """Test that all loss components are correctly integrated"""
    # Override configuration for testing
    config_loader.config_model.update({
        'use_ca_tsu': True,
        'ca_tsu_weight': 0.3,
        'causal_loss_lambda': 0.3,
        'use_enhanced_srl': True,
        'use_noise_aware_loss': True,
        'noise_aware_weight': 0.5,
        'entropy_regularization_lambda': 0.1,
        'use_dual_path_srl': True,
        'variant_type': 'hedge',  # Non-discriminative for generative ATA
    })
    
    # Create model instance with the updated config
    tf.reset_default_graph()
    model = Model()
    
    # Assemble graph
    logger.info("Assembling graph with all loss components enabled...")
    model.assemble_graph()
    
    # Check for all expected loss attributes
    with tf.Session() as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())
        
        # Log all tensors in the graph with 'loss' in their name
        loss_tensors = [op.name for op in tf.get_default_graph().get_operations() 
                       if 'loss' in op.name.lower()]
        logger.info(f"Found {len(loss_tensors)} loss-related tensors in graph:")
        for tensor_name in loss_tensors:
            logger.info(f" - {tensor_name}")
            
        # Check for key loss components
        loss_components = [
            'base_prediction_loss',
            'causal_consistency_loss', 
            'enhanced_srl_loss',
            'noise_aware_loss'
        ]
        
        for component in loss_components:
            if component in model.loss_components:
                logger.info(f"Component '{component}' is correctly included in loss_components")
            else:
                logger.warning(f"Component '{component}' is MISSING from loss_components")
        
        # Check if loss_contributions dictionary is created
        if hasattr(model, 'loss_contributions'):
            logger.info("Loss contributions tracking is set up correctly")
        else:
            logger.error("Loss contributions tracking is NOT set up")
    
    logger.info("Loss components test completed")
    return model

def test_loss_calculation():
    """Create a simplified forward pass to test loss calculation"""
    # First make sure all components are enabled in config
    config_loader.config_model.update({
        'use_ca_tsu': True,
        'ca_tsu_weight': 0.3,
        'causal_loss_lambda': 0.3,
        'use_enhanced_srl': True,
        'use_noise_aware_loss': True,
        'noise_aware_weight': 0.5,
        'entropy_regularization_lambda': 0.1,
        'use_dual_path_srl': True,
        'variant_type': 'hedge',
        'batch_size': 4,
        'max_n_days': 5,
        'max_n_msgs': 10,
        'max_n_words': 20,
        'word_embed_size': 50,
        'y_size': 2,
    })
    
    # Create model instance with the updated config
    tf.reset_default_graph()
    model = Model()
    model.assemble_graph()
    
    # Create executor for the model
    executor = Executor(lambda: model)
    
    # Create dummy data for a forward pass
    batch_size = 4
    max_n_days = 5
    max_n_msgs = 10
    max_n_words = 20
    y_size = 2
    
    dummy_data = {
        'batch_size': batch_size,
        'stock_batch': np.zeros(batch_size, dtype=np.int32),
        'T_batch': np.ones(batch_size, dtype=np.int32) * 3,  # Target day 3
        'n_words_batch': np.ones((batch_size, max_n_days, max_n_msgs), dtype=np.int32) * 5,
        'n_msgs_batch': np.ones((batch_size, max_n_days), dtype=np.int32) * 5,
        'y_batch': np.zeros((batch_size, max_n_days, y_size), dtype=np.float32),
        'price_batch': np.random.random((batch_size, max_n_days, 3)).astype(np.float32),
        'mv_percent_batch': np.zeros((batch_size, max_n_days), dtype=np.int32),
        'word_batch': np.ones((batch_size, max_n_days, max_n_msgs, max_n_words), dtype=np.int32),
        'ss_index_batch': np.zeros((batch_size, max_n_days, max_n_msgs), dtype=np.int32),
    }
    
    # Set y_batch with one-hot encoded values
    for b in range(batch_size):
        for d in range(max_n_days):
            class_idx = np.random.randint(0, y_size)
            dummy_data['y_batch'][b, d, class_idx] = 1.0
            dummy_data['mv_percent_batch'][b, d] = class_idx
    
    # Create message mask
    msg_mask = np.zeros((batch_size, max_n_days, max_n_msgs), dtype=bool)
    for b in range(batch_size):
        for d in range(max_n_days):
            n_msgs = dummy_data['n_msgs_batch'][b, d]
            msg_mask[b, d, :n_msgs] = True
    
    with tf.Session() as sess:
        # Initialize variables
        word_table_init = np.random.random((10000, 50)).astype(np.float32)  # vocab_size x word_embed_size
        feed_dict = {model.word_table_init: word_table_init}
        sess.run(tf.global_variables_initializer(), feed_dict=feed_dict)
        
        # Create feed_dict for forward pass
        feed_dict = {
            model.is_training_phase: True,
            model.batch_size: dummy_data['batch_size'],
            model.stock_ph: dummy_data['stock_batch'],
            model.T_ph: dummy_data['T_batch'],
            model.n_words_ph: dummy_data['n_words_batch'],
            model.n_msgs_ph: dummy_data['n_msgs_batch'],
            model.y_ph: dummy_data['y_batch'],
            model.price_ph: dummy_data['price_batch'],
            model.mv_percent_ph: dummy_data['mv_percent_batch'],
            model.word_ph: dummy_data['word_batch'],
            model.ss_index_ph: dummy_data['ss_index_batch'],
            model.msg_mask_ph: msg_mask,
        }
        
        # Set any annealing parameters if needed
        if hasattr(model, 'anneal_ph'):
            feed_dict[model.anneal_ph] = 0.5  # Set KL annealing to 0.5
        
        # Run forward pass to calculate loss
        try:
            # Get all tensors that represent loss components
            fetches = {
                'total_loss': model.loss,
                'pred_loss': model.pred_loss if hasattr(model, 'pred_loss') else None,
            }
            
            # Add loss components if they exist
            loss_components = ['enhanced_srl_loss', 'causal_consistency_loss', 'noise_aware_loss']
            for component in loss_components:
                if hasattr(model, component):
                    fetches[component] = getattr(model, component)
            
            # Run the graph
            results = sess.run(fetches, feed_dict=feed_dict)
            
            # Log the results
            logger.info("Loss calculation results:")
            for key, value in results.items():
                if value is not None:
                    logger.info(f" - {key}: {value}")
            
            # Check if total loss is greater than 0
            if results['total_loss'] > 0:
                logger.info("Total loss is positive - calculation successful")
            else:
                logger.warning(f"Total loss is non-positive: {results['total_loss']}")
            
            # Check if we have contributions from all components
            for component in loss_components:
                if component in results and results[component] is not None:
                    if results[component] > 0:
                        logger.info(f"Loss component '{component}' has positive contribution")
                    else:
                        logger.warning(f"Loss component '{component}' has zero or negative contribution: {results[component]}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during loss calculation: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

def main():
    """Main test function"""
    logger.info("Starting comprehensive loss function test")
    
    # Test loss components
    model = test_loss_components()
    
    # Test loss calculation
    results = test_loss_calculation()
    
    logger.info("All tests completed")

if __name__ == "__main__":
    main() 