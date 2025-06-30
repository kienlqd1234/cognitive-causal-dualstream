import optuna
import yaml
import logging
import tensorflow as tf
from typing import Union, Tuple
import os
import datetime
# Import the config module directly to modify its global variables
import ConfigLoader_tx_lf as hps_config 
from Executor_tx_lf_dual_path import Executor
from Model_tx_lf_dual_path import Model

# Configure logging for the tuning process
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Metrics to Optimize ---
# We should optimize for metrics that handle class imbalance well.
# 'mcc' (Matthews Correlation Coefficient) is excellent for binary classification.
# 'acc' (Accuracy) measures overall correctness of predictions.
MULTI_OBJECTIVE = True  # Set to True to optimize for both metrics

# Create result directory if it doesn't exist
os.makedirs("result", exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"result/optuna_tuning_{timestamp}.log"

# Configure file logger
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(file_handler)

def objective(trial: optuna.Trial) -> Union[float, Tuple[float, float]]:
    """
    This is the objective function that Optuna will minimize or maximize.
    It takes a `trial` object, which suggests hyperparameters, runs the training,
    and returns the validation metric(s).
    """
    # Reset the default graph for each trial to avoid variable reuse errors
    tf.reset_default_graph()
    try:
        # 1. Suggest new hyperparameter values for this trial
        causal_loss_lambda = trial.suggest_float('causal_loss_lambda', 0.1, 2.0, log=True)
        path_separation_margin = trial.suggest_float('path_separation_margin', 0.5, 1.0)
        srl_loss_weight = trial.suggest_float('srl_loss_weight', 0.05, 1.0, log=True)

        # 2. Modify the global configuration in the imported module
        # This is the key change: we are "monkey-patching" the config
        # that all other modules will see when they import it.
        
        # Update model hyperparameters
        hps_config.config_model['causal_loss_lambda'] = causal_loss_lambda
        hps_config.config_model['path_separation_margin'] = path_separation_margin
        
        if hps_config.config_model.get('use_noise_aware_loss', False):
            hps_config.config_model['noise_aware_weight'] = srl_loss_weight
        else:
            hps_config.config_model['prediction_loss_weight'] = srl_loss_weight
        
        # --- Use a smaller dataset and fewer epochs for faster tuning ---
        hps_config.dates['train_start_date'] = '2014-01-01'
        hps_config.dates['train_end_date'] = '2014-04-01'  # Using 3 months of data
        hps_config.dates['dev_start_date'] = '2014-04-01'
        hps_config.dates['dev_end_date'] = '2014-05-01'    # Using 1 month for validation
        
        hps_config.config_model['n_epochs'] = 3

        # 3. Instantiate the Executor
        # The executor's __init__ takes the Model class, not a config object.
        # When Model() is called inside the Executor, it will read the modified
        # global variables from the hps_config module.
        executor = Executor(Model)
        
        # 4. Run training and evaluation
        evaluation_results = executor.run()
        
        # 5. Extract the validation metrics to be optimized
        val_mcc = evaluation_results.get('mcc', 0.0)
        val_acc = evaluation_results.get('acc', 0.0)

        # Log trial results to both console and file
        log_msg = f"Trial {trial.number} finished. Validation MCC: {val_mcc:.4f}, Accuracy: {val_acc:.4f}, Parameters: {trial.params}"
        logging.info(log_msg)

        if MULTI_OBJECTIVE:
            # Return both metrics for multi-objective optimization
            return val_mcc, val_acc
        else:
            # Return single metric (default to MCC)
            return val_mcc

    except Exception as e:
        # Handle potential errors during a trial (e.g., model fails to train)
        logging.error(f"Trial {trial.number} failed with error: {e}", exc_info=True)
        # Tell Optuna this was a bad trial by returning a low value
        if MULTI_OBJECTIVE:
            return -1.0, -1.0
        else:
            return -1.0

# Custom callback to log trials as they complete
def log_trial_callback(study, trial):
    """Callback to log trial results as they complete"""
    if MULTI_OBJECTIVE:
        metrics = f"MCC: {trial.values[0]:.4f}, Accuracy: {trial.values[1]:.4f}" if trial.values else "Failed"
    else:
        metrics = f"Value: {trial.value:.4f}" if trial.value is not None else "Failed"
    
    logging.info(f"Trial {trial.number} completed. {metrics}")
    logging.info(f"Current parameters: {trial.params}")
    
    # Log best trial so far
    if MULTI_OBJECTIVE:
        if study.best_trials:
            logging.info(f"Current Pareto front size: {len(study.best_trials)}")
    else:
        logging.info(f"Best value so far: {study.best_value:.4f} (Trial {study.best_trial.number})")


if __name__ == '__main__':
    # Log the start of hyperparameter tuning
    logging.info(f"Starting hyperparameter tuning. Results will be saved to {log_file}")
    
    # --- Create and run the Optuna study ---
    if MULTI_OBJECTIVE:
        # For multi-objective optimization, specify multiple directions
        study = optuna.create_study(
            directions=['maximize', 'maximize'],  # Maximize both MCC and accuracy
            study_name='PEN_Multi_Objective_Tuning'
        )
        metric_names = ['MCC', 'Accuracy']
    else:
        # Single objective (MCC)
        study = optuna.create_study(
            direction='maximize', 
            study_name='PEN_SRL_Loss_Tuning'
        )
        metric_names = ['MCC']

    # Start the optimization. `n_trials` is the number of different hyperparameter
    # combinations Optuna will try. 50 is a good starting point.
    study.optimize(objective, n_trials=50, callbacks=[log_trial_callback])

    # --- Print and log the results ---
    summary = "\n" + "="*40 + "\n"
    summary += "Hyperparameter tuning finished!\n"
    summary += f"Number of finished trials: {len(study.trials)}\n"

    if MULTI_OBJECTIVE:
        # Get the Pareto front for multi-objective optimization
        summary += "\nPareto Front Solutions:\n"
        best_trials = study.best_trials
        
        for i, trial in enumerate(best_trials):
            summary += f"\nSolution {i+1}:\n"
            summary += f"  Values: MCC = {trial.values[0]:.4f}, Accuracy = {trial.values[1]:.4f}\n"
            summary += "  Hyperparameters:\n"
            for key, value in trial.params.items():
                summary += f"    {key}: {value}\n"
    else:
        # Print the best trial for single-objective optimization
        summary += "\nBest trial:\n"
        best_trial = study.best_trial

        summary += f"  Value (Validation {metric_names[0]}): {best_trial.value:.4f}\n"
        summary += "\n  Best Hyperparameters:\n"
        for key, value in best_trial.params.items():
            summary += f"    {key}: {value}\n"
    
    summary += "\n" + "="*40 + "\n"
    summary += "To use these parameters, update your 'src/config_tx_lf.yml' file and run a full training session." 
    
    # Print and log the final summary
    print(summary)
    logging.info(summary)
    
    # Save study results
    results_file = f"result/optuna_best_params_{timestamp}.yml"
    if MULTI_OBJECTIVE:
        # Save all solutions in the Pareto front
        best_params_list = [trial.params for trial in study.best_trials]
        with open(results_file, 'w') as f:
            yaml.dump(best_params_list, f, default_flow_style=False)
    else:
        # Save the best parameters
        with open(results_file, 'w') as f:
            yaml.dump(study.best_params, f, default_flow_style=False)
            
    logging.info(f"Best parameters saved to {results_file}") 