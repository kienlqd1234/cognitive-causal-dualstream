#!/usr/local/bin/python
import tensorflow as tf
import logging
import os
import numpy as np
import importlib

# Configure logger for the comparison script
comparison_logger = logging.getLogger('ModelComparison')
comparison_logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not comparison_logger.hasHandlers():
    comparison_logger.addHandler(handler)

# Import the load_config functions from both loaders
# For Original Model (Model.py, Executor.py)
from ConfigLoader import load_config_from_file as load_config_original_file, config_model as config_model_original, path_parser as path_parser_original, logger as logger_original
# For TX-LF based Models (Model_tx_lf.py, Model_tx_lf_dual_path.py)
from ConfigLoader_tx_lf import load_config_from_file as load_config_tx_lf_file, config_model as config_model_tx_lf, path_parser as path_parser_tx_lf, logger as logger_tx_lf, load_config as load_config_tx_lf_module_func

# Define paths to the specific configuration files
# CRITICAL: You MUST create these files and configure them appropriately!
# Ensure model_name, paths (log, checkpoints, graphs), and model-specific params
# (like use_dual_path_srl) are correctly set in each.

CONFIG_FOR_ORIGINAL_MODEL = os.path.join(os.path.dirname(__file__), 'config.yml')
CONFIG_FOR_TX_LF_MODEL = os.path.join(os.path.dirname(__file__), 'config_tx_lf.yml')

# --- Import Models and Executors ---
# Original Model (no suffix)
from Model import Model as OriginalModel
from Executor import Executor as OriginalExecutor

# TX-LF Dual Path Model
from Model_tx_lf_dual_path import Model as DualPathModel
from Executor_tx_lf_dual_path import Executor as DualPathExecutor

# TX-LF Base Model (if needed for comparison, uncomment and adjust run_experiment calls)
# from Model_tx_lf import Model as TxLfModel
# from Executor_tx_lf import Executor as TxLfExecutor

# Utility to reset TensorFlow graph and re-initialize components for a fresh run
def reset_tf_graph_and_reinit_config(config_path='config_tx_lf.yml'):
    tf.reset_default_graph()
    # Reload config to ensure fresh state from file for each model type
    # This also re-initializes config_model and path_parser from ConfigLoader_tx_lf
    load_config_tx_lf_module_func(config_path) 
    logger_tx_lf.info(f"TensorFlow graph reset and config reloaded from {config_path}")

def run_experiment(model_class, executor_class, model_name_suffix,
                   config_file_path, config_loader_func, # For loading the config file
                   active_config_module, # To access config_model, path_parser, logger globals
                   config_changes, max_batches):
    
    comparison_logger.info(f"--- Starting Experiment: {model_name_suffix} --- ")
    
    tf.reset_default_graph()
    comparison_logger.info("TensorFlow graph reset.")

    # Load the specified configuration file using the provided loader function
    config_loader_func(config_file_path)
    comparison_logger.info(f"Configuration loaded from {config_file_path} using {config_loader_func.__name__}")

    # Apply any dynamic configuration changes for this specific run
    # These changes are applied to the globally accessible config_model from the *active_config_module*
    # This requires that the config_loader_func has already populated the correct config_model.
    current_config_model = active_config_module.config_model
    for key, value in config_changes.items():
        comparison_logger.info(f"Applying config change for {model_name_suffix}: {key} = {value}")
        current_config_model[key] = value
    
    # Instantiate the model and executor
    # Model __init__ methods usually read from their respective global config_model
    model_instance = model_class() 
    
    # Executor will call model_instance.assemble_graph() if needed, 
    # after config is fully set and model is instantiated.
    executor_instance = executor_class(model_instance, max_batches_override=max_batches)
    
    results = executor_instance.train_and_dev() # This should return a dict of metrics
    
    comparison_logger.info(f"--- Experiment Finished: {model_name_suffix} --- ")
    print(f"\nResults for {model_name_suffix}:")
    print(f"  Batches Tested: {results.get('batches_tested', 'N/A')}")
    print(f"  Average Loss: {results.get('loss', 0.0):.4f}")
    print(f"  Average Accuracy: {results.get('accuracy', 0.0):.4f}")
    print(f"  Average MCC: {results.get('mcc', 0.0):.4f}")
    if 'causal_loss' in results: # Causal loss might not be in original Model.py
        print(f"  Average Causal Consistency Loss: {results.get('causal_loss', 0.0):.4f}")
    if 'sigma_gate_mean' in results:
        print(f"  Average Mean Sigma Gate (Fusion): {results.get('sigma_gate_mean', 0.0):.4f}")
    
    return results

def main():
    max_batches_for_experiment = 20 # Target 20 batches for each experiment

    # --- Experiment 1: Dual-Path Model (using config_tx_lf.yml) ---
    # This model uses ConfigLoader_tx_lf's globals (config_model_tx_lf, etc.)
    dual_path_config_changes = {'use_dual_path_srl': True} 
    # The load_config_tx_lf_module_func is what ConfigLoader_tx_lf.load_config points to.
    # It handles initializing config_model_tx_lf etc. from the file.
    results_dual_path = run_experiment(
        model_class=DualPathModel,
        executor_class=DualPathExecutor,
        model_name_suffix="DualPathModel_tx_lf",
        config_file_path=CONFIG_FOR_TX_LF_MODEL,
        config_loader_func=load_config_tx_lf_module_func, # This sets up ConfigLoader_tx_lf.config_model
        active_config_module=importlib.import_module('ConfigLoader_tx_lf'), # Module whose globals are used
        config_changes=dual_path_config_changes,
        max_batches=max_batches_for_experiment
    )

    # --- Experiment 2: Original Model (Model.py, Executor.py, using config.yml) ---
    # This model uses ConfigLoader.py's globals (config_model_original, etc.)
    original_model_config_changes = {} # No specific dynamic changes, uses config.yml as is.
    # load_config_original_file is ConfigLoader.load_config_from_file
    # It populates ConfigLoader.config_model
    results_original = run_experiment(
        model_class=OriginalModel,
        executor_class=OriginalExecutor,
        model_name_suffix="OriginalModel_Vanilla",
        config_file_path=CONFIG_FOR_ORIGINAL_MODEL,
        config_loader_func=load_config_original_file, # This sets up ConfigLoader.config_model
        active_config_module=importlib.import_module('ConfigLoader'), # Module whose globals are used
        config_changes=original_model_config_changes,
        max_batches=max_batches_for_experiment
    )

    print("\n--- Overall Comparison Summary ---")
    print(f"Tested for up to {max_batches_for_experiment} batches each.")
    
    print("\nDual-Path Model (Model_tx_lf_dual_path) Metrics:")
    print(f"  Accuracy: {results_dual_path.get('accuracy', 0.0):.4f}, MCC: {results_dual_path.get('mcc', 0.0):.4f}")
    if 'causal_loss' in results_dual_path:
        print(f"  Causal Loss: {results_dual_path.get('causal_loss', 0.0):.4f}")
    if 'sigma_gate_mean' in results_dual_path:
        sigma_mean = results_dual_path.get('sigma_gate_mean', 0.0)
        print(f"  Mean Sigma Gate (Fusion Balance): {sigma_mean:.4f}")
        if sigma_mean == 0.0 and not results_dual_path.get('batches_tested', 0) > 0 : # check if it was actually computed
             print("    Sigma gate was not computed (possibly use_dual_path_srl was false or no batches ran).")
        elif sigma_mean > 0.5:
            print("    Interpretation: Dual-Path SRL's fusion gate, on average, gives more weight to the Text-Driven (TD) pathway.")
        elif sigma_mean < 0.5 and sigma_mean !=0.0 : # check against 0.0 if it was not calculated
            print("    Interpretation: Dual-Path SRL's fusion gate, on average, gives more weight to the Price-Driven (PD) pathway.")
        elif sigma_mean == 0.5 :
            print("    Interpretation: Dual-Path SRL's fusion gate, on average, balances TD and PD pathways equally.")
        else: # handles sigma_mean = 0.0 when it was actually computed.
            print("    Interpretation: Sigma gate mean is 0.0. This could mean PD path dominates if 1-sigma is used for PD, or an issue.")


    print("\nOriginal Model (Model.py) Metrics:")
    print(f"  Accuracy: {results_original.get('accuracy', 0.0):.4f}, MCC: {results_original.get('mcc', 0.0):.4f}")
    # Original Model.py does not have causal_consistency_loss unless added.
    if 'causal_loss' in results_original and results_original.get('causal_loss', 0.0) > 0: # Check if it was somehow added and non-zero
        print(f"  Causal Loss: {results_original.get('causal_loss', 0.0):.4f}")
    else:
        print("  Causal Loss: Not applicable or not implemented in this version of Original Model.")


    print("\nInterpretation of Ca-TSU effects (observed via Causal Consistency Loss):")
    if 'causal_loss' in results_dual_path:
        print(f"  Dual-Path Model Causal Loss: {results_dual_path.get('causal_loss', 0.0):.4f}")
    if 'causal_loss' in results_original and results_original.get('causal_loss', 0.0) > 0:
         print(f"  Original Model Causal Loss (if implemented): {results_original.get('causal_loss', 0.0):.4f}")
    print("  A lower causal consistency loss suggests that the two attention mechanisms (alpha_i, omega_i) within Ca-TSU are more aligned.")
    print("  This alignment is a design goal of Ca-TSU for better causal inference in text selection.")
    print("  Differences in this loss between models might indicate how other architectural changes (like Dual-Path SRL) interact with Ca-TSU, or if Ca-TSU is present/active.")

if __name__ == '__main__':
    # Ensure the main config file exists, or provide a default path
    # The ConfigLoader_tx_lf.py should handle loading 'config_tx_lf.yml' by default
    main() 