#!/usr/bin/env python
# Test script to verify the dual-path SRL message mask fix
import os
import tensorflow as tf
import numpy as np
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import needed modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.Model_tx_lf_dual_path import Model
from src.Executor_tx_lf_dual_path import Executor

def test_model_initialization():
    """Test that the model can be initialized with the dual-path SRL mask"""
    logger.info("Testing model initialization...")
    
    # Create model instance
    model = Model()
    
    # Check if model was created
    assert model is not None, "Model failed to initialize"
    logger.info("Model initialized successfully")
    
    # Check for critical attributes
    logger.info(f"Model has 'use_dual_path_srl' attribute: {hasattr(model, 'use_dual_path_srl')}")
    if hasattr(model, 'use_dual_path_srl'):
        logger.info(f"use_dual_path_srl value: {model.use_dual_path_srl}")
    
    # Test graph assembly
    logger.info("Assembling graph...")
    model.assemble_graph()
    logger.info("Graph assembled successfully")
    
    # Check for msg_mask_ph
    logger.info(f"Model has 'msg_mask_ph' attribute: {hasattr(model, 'msg_mask_ph')}")
    if hasattr(model, 'msg_mask_ph'):
        logger.info(f"msg_mask_ph name: {model.msg_mask_ph.name}")
        logger.info(f"msg_mask_ph shape: {model.msg_mask_ph.shape}")
    
    logger.info("Model initialization test completed successfully")
    return model

def test_executor_initialization(model_class):
    """Test that the executor can be initialized with the model"""
    logger.info("Testing executor initialization...")
    
    # Create executor instance
    executor = Executor(model_class)
    
    # Check if executor was created
    assert executor is not None, "Executor failed to initialize"
    logger.info("Executor initialized successfully")
    
    # Check model reference
    assert executor.model is not None, "Executor's model reference is None"
    logger.info(f"Executor's model has msg_mask_ph: {hasattr(executor.model, 'msg_mask_ph')}")
    
    logger.info("Executor initialization test completed successfully")
    return executor

def main():
    """Main test function"""
    logger.info("Starting dual-path SRL message mask test")
    
    # Test model initialization
    model = test_model_initialization()
    
    # Test executor initialization
    executor = test_executor_initialization(Model)
    
    logger.info("All tests completed successfully")

if __name__ == "__main__":
    main() 