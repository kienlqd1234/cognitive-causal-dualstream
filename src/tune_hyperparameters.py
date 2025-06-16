import optuna
import yaml
import logging
import tensorflow as tf
# Import the config module directly to modify its global variables
import ConfigLoader_tx_lf as hps_config 
from Executor_tx_lf_dual_path import Executor
from Model_tx_lf_dual_path import Model

# Configure logging for the tuning process
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Metric to Optimize ---
# We should optimize for a metric that handles class imbalance well.
# 'mcc' (Matthews Correlation Coefficient) is excellent for binary classification.
# Other options: 'f1' (F1-score), 'acc' (Accuracy).
METRIC_TO_OPTIMIZE = 'mcc' 

def objective(trial: optuna.Trial) -> float:
    """
    This is the objective function that Optuna will minimize or maximize.
    It takes a `trial` object, which suggests hyperparameters, runs the training,
    and returns the validation metric.
    """
    # Reset the default graph for each trial to avoid variable reuse errors
    tf.reset_default_graph()
    try:
        # 1. Suggest new hyperparameter values for this trial
        causal_loss_lambda = trial.suggest_float('causal_loss_lambda', 0.1, 2.0, log=True)
        path_separation_margin = trial.suggest_float('path_separation_margin', 0.5, 2.0)
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
        
        # 5. Extract the validation metric to be optimized
        validation_metric = evaluation_results.get('mcc', 0.0)

        logging.info(f"Trial {trial.number} finished. Validation {METRIC_TO_OPTIMIZE}: {validation_metric:.4f}")

        return validation_metric

    except Exception as e:
        # Handle potential errors during a trial (e.g., model fails to train)
        logging.error(f"Trial {trial.number} failed with error: {e}", exc_info=True)
        # Tell Optuna this was a bad trial by returning a low value
        return -1.0


if __name__ == '__main__':
    # --- Create and run the Optuna study ---
    # We want to MAXIMIZE the MCC or F1-score.
    study = optuna.create_study(direction='maximize', study_name='PEN_SRL_Loss_Tuning')

    # Start the optimization. `n_trials` is the number of different hyperparameter
    # combinations Optuna will try. 50 is a good starting point.
    study.optimize(objective, n_trials=50)

    # --- Print the results ---
    print("\n" + "="*40)
    print("Hyperparameter tuning finished!")
    print(f"Number of finished trials: {len(study.trials)}")

    print("\nBest trial:")
    best_trial = study.best_trial

    print(f"  Value (Validation {METRIC_TO_OPTIMIZE}): {best_trial.value:.4f}")

    print("\n  Best Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    
    print("\n" + "="*40)
    print("To use these parameters, update your 'src/config_tx_lf.yml' file and run a full training session.") 