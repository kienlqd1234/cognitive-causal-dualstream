#!/usr/bin/env python

"""
Main entry point for running a full training and evaluation cycle of the 
Enhanced Dual-Path SRL model.
"""

import os
import logging
from Model_tx_lf_dual_path import Model
from Executor_tx_lf_dual_path import Executor
# Import the config loader to directly modify the configuration for this run
import ConfigLoader_tx_lf_dual_path as config_loader

def main():
    """
    Run a full training and evaluation cycle for the dual-path model.
    This function will train the model and save checkpoints, then restore
    the model to evaluate its performance on the final test set.
    """
    
    logger = config_loader.logger
    logger.info("Starting a full run. Model configuration will be loaded from the YML config file.")

    # Create an executor for the model.
    # The executor's __init__ will instantiate the Model, which will
    # read the configuration from the config_loader.
    model = Model()
    model.assemble_graph()

    silence_step = 0
    skip_step = 20
    
    exe = Executor(model, silence_step=silence_step, skip_step=skip_step)

    # --- Phase 1: Train and Evaluate on Dev Set ---
    print("\n" + "="*60)
    print("STARTING: Training and Development Phase")
    print("="*60)
    exe.train_and_dev() # This calls the train_and_dev() method
    print("\n" + "="*60)
    print("FINISHED: Training and Development Phase")
    print("="*60)

    # --- Phase 2: Evaluate on Final Test Set ---
    print("\n" + "="*60)
    print("STARTING: Final Test Phase")
    print("="*60)
    exe.restore_and_test() # This loads the best checkpoint and runs on test data
    print("\n" + "="*60)
    print("FINISHED: Final Test Phase")
    print("="*60)
    
    print("\nFull run completed.")

if __name__ == "__main__":
    main()

