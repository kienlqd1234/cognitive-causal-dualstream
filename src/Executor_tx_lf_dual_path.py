#!/usr/local/bin/python
import os
import tensorflow as tf
import metrics as metrics
import stat_logger as stat_logger
from DataPipe import DataPipe # Keep DataPipe for now, or create DataPipe_tx_lf if needed
from ConfigLoader_tx_lf import logger # Changed
from Model_tx_lf_dual_path import Model # Changed for dual path
import re # Keep re for now, just in case, but its usage is commented out below
import numpy as np
import matplotlib.pyplot as plt # Added for plotting
import logging
from ConfigLoader_tx_lf import config_model
import json
from datetime import datetime

# Custom imports
#from Model_tx_lf_dual_path import Model_tx_lf_dual_path
#from src.data_pipe import DataPipe
#from src import metrics
#import collections
#import time
#import math
#import random

# Configure logger
# logger = logging.getLogger(__name__) # This overwrites the configured logger from ConfigLoader

class Executor:

    def __init__(self, model_obj, silence_step=200, skip_step=20):
        self.pipe = DataPipe()
        self.model = model_obj() # Will pass Model_tx_lf_dual_path here
        self.model.assemble_graph() # Call assemble_graph to build the model's graph
        
        # print(f">>> EXECUTOR LIFECYCLE (__init__) >>> After assemble_graph on self.model ({id(self.model)}):")
        # print(f">>> EXECUTOR LIFECYCLE (__init__) >>> hasattr(self.model, 'h_t_dual_path'): {hasattr(self.model, 'h_t_dual_path')}")
        # print(f">>> EXECUTOR LIFECYCLE (__init__) >>> hasattr(self.model, 'pd_attention_weights_beta'): {hasattr(self.model, 'pd_attention_weights_beta')}")
        # if hasattr(self.model, 'h_t_dual_path') and self.model.h_t_dual_path is not None:
        #     print(f">>> EXECUTOR LIFECYCLE (__init__) >>> self.model.h_t_dual_path.name: {self.model.h_t_dual_path.name}")
        # if hasattr(self.model, 'pd_attention_weights_beta') and self.model.pd_attention_weights_beta is not None:
        #     print(f">>> EXECUTOR LIFECYCLE (__init__) >>> self.model.pd_attention_weights_beta.name: {self.model.pd_attention_weights_beta.name}")

        self.silence_step = silence_step
        self.skip_step = skip_step

        self.saver = tf.train.Saver()
        self.tf_config = tf.ConfigProto(allow_soft_placement=True)
        self.tf_config.gpu_options.allow_growth = True
        
        # Create results directory if it doesn't exist
        if not os.path.exists('results'):
            os.makedirs('results')

        # Set up progress logger for explicit progress tracking
        self.progress_logger = logging.getLogger('ProgressLogger')
        if not self.progress_logger.handlers:
            self.progress_logger.setLevel(logging.INFO)
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s PROGRESS: %(message)s')
            console_handler.setFormatter(formatter)
            self.progress_logger.addHandler(console_handler)
            self.progress_logger.propagate = False

    def unit_test_train(self):
        with tf.Session() as sess:
            word_table_init = self.pipe.init_word_table()
            feed_table_init = {self.model.word_table_init: word_table_init}
            sess.run(tf.global_variables_initializer(), feed_dict=feed_table_init)
            logger.info('Word table init: done!')

            logger.info('Model: {0}, start a new session!'.format(self.model.model_name))

            n_iter = self.model.global_step.eval()

            # forward
            train_batch_loss_list = list()
            train_epoch_size = 0.0
            train_epoch_n_acc = 0.0
            train_batch_gen = self.pipe.batch_gen(phase='train')
            train_batch_dict = next(train_batch_gen)

            while n_iter < 100:
                feed_dict = {self.model.is_training_phase: True,
                             self.model.batch_size: train_batch_dict['batch_size'],
                             self.model.stock_ph: train_batch_dict['stock_batch'],
                             self.model.T_ph: train_batch_dict['T_batch'],
                             self.model.n_words_ph: train_batch_dict['n_words_batch'],
                             self.model.n_msgs_ph: train_batch_dict['n_msgs_batch'],
                             self.model.y_ph: train_batch_dict['y_batch'],
                             self.model.price_ph: train_batch_dict['price_batch'],
                             self.model.mv_percent_ph: train_batch_dict['mv_percent_batch'],
                             self.model.word_ph: train_batch_dict['word_batch'],
                             self.model.ss_index_ph: train_batch_dict['ss_index_batch'],
                             }

                ops = [self.model.y_T, self.model.y_T_,  self.model.loss, self.model.optimize]
                #pdb.set_trace() # Removed pdb.set_trace()
                train_batch_y, train_batch_y_,  train_batch_loss, _ = sess.run(ops, feed_dict)
                
                # training batch stat
                train_epoch_size += float(train_batch_dict['batch_size'])
                train_batch_loss_list.append(train_batch_loss)
                train_batch_n_acc = sess.run(metrics.n_accurate(y=train_batch_y, y_=train_batch_y_))
                train_epoch_n_acc += float(train_batch_n_acc)

                stat_logger.print_batch_stat(n_iter, train_batch_loss, train_batch_n_acc,
                                             train_batch_dict['batch_size'])
                n_iter += 1

    def generation(self, sess, phase):
        """
        Run the model on the evaluation data and collect the results.
        
        Args:
            sess: TensorFlow session
            phase: Either 'dev' or 'test'
            
        Returns:
            Dictionary containing evaluation results
        """
        # Get generator for batches organized by stocks
        generation_gen = self.pipe.batch_gen_by_stocks(phase)

        # Initialize counters and result lists
        gen_loss_list = list()
        gen_size, gen_n_acc = 0.0, 0.0
        y_list, y_list_ = list(), list()
        
        # Iterate through all batches from the generator
        for gen_batch_dict in generation_gen:
            # Prepare feed dict for this batch
            feed_dict = {self.model.is_training_phase: False,
                         self.model.batch_size: gen_batch_dict['batch_size'],
                         self.model.stock_ph: gen_batch_dict['stock_batch'],
                         self.model.T_ph: gen_batch_dict['T_batch'],
                         self.model.n_words_ph: gen_batch_dict['n_words_batch'],
                         self.model.n_msgs_ph: gen_batch_dict['n_msgs_batch'],
                         self.model.y_ph: gen_batch_dict['y_batch'],
                         self.model.price_ph: gen_batch_dict['price_batch'],
                         self.model.mv_percent_ph: gen_batch_dict['mv_percent_batch'],
                         self.model.word_ph: gen_batch_dict['word_batch'],
                         self.model.ss_index_ph: gen_batch_dict['ss_index_batch'],
                         self.model.dropout_mel_in: 0.0,
                         self.model.dropout_mel: 0.0,
                         self.model.dropout_ce: 0.0,
                         self.model.dropout_vmd_in: 0.0,
                         self.model.dropout_vmd: 0.0,
                         }
            

            # Run the model
            gen_batch_y, gen_batch_y_, gen_batch_loss = sess.run([self.model.y_T, self.model.y_T_, self.model.loss],
                                                                 feed_dict=feed_dict)
            
            # Gather results
            y_list.append(gen_batch_y)
            y_list_.append(gen_batch_y_)
            gen_loss_list.append(gen_batch_loss)  # list of floats

            # Calculate accuracy for this batch
            gen_batch_n_acc = float(sess.run(metrics.n_accurate(y=gen_batch_y, y_=gen_batch_y_)))  # float
            gen_n_acc += gen_batch_n_acc

            # Add batch size to total size
            batch_size = float(gen_batch_dict['batch_size'])
            gen_size += batch_size
            
            #logger.info(f"Evaluated batch for {phase} phase: size={batch_size}, acc={gen_batch_n_acc/batch_size:.4f}")

        # Calculate final evaluation results
        if gen_size > 0:
            results = metrics.eval_res(gen_n_acc, gen_size, gen_loss_list, y_list, y_list_, use_mcc=True)
            logger.info(f"Completed {phase} evaluation: size={gen_size}, acc={gen_n_acc/gen_size:.4f}")
            return results
        else:
            logger.warning(f"No samples were evaluated in {phase} phase")
            # Return empty results
            return {
                'accuracy': 0.0,
                'mcc': 0.0,
                'loss': 0.0
            }
    
    def train_and_dev(self):
        """
        Main training and evaluation entry point. This method trains the model
        for a configured number of epochs, logs detailed metrics, evaluates on the
        development set, and generates diagnostic plots.
        It is structured to be compatible with the Optuna hyperparameter tuning script.
        """
        with tf.Session(config=self.tf_config) as sess:
            # 1. SETUP: Tensorboard writer, variable initialization, and checkpoint restoration
            writer = tf.summary.FileWriter(self.model.tf_graph_path, sess.graph)
            feed_table_init = {self.model.word_table_init: self.pipe.init_word_table()}
            sess.run(tf.global_variables_initializer(), feed_dict=feed_table_init)
            logger.info('Word table initialized.')

            checkpoint = tf.train.get_checkpoint_state(os.path.dirname(self.model.tf_checkpoint_file_path))
            if checkpoint and checkpoint.model_checkpoint_path:
                try:
                    self.saver.restore(sess, checkpoint.model_checkpoint_path)
                    logger.info(f'Model: {self.model.model_name}, session restored from checkpoint.')
                except Exception as e:
                    logger.error(f"Could not restore checkpoint: {e}. Starting new session.")
            else:
                logger.info(f'Model: {self.model.model_name}, starting new session.')

            # 2. CONFIGURATION SUMMARY: Log a clean, readable summary of the model config
            startup_msg = f"""
{'='*60}
STARTING TRAINING & EVALUATION RUN
{'='*60}
MODEL CONFIGURATION:
- Model Name: {self.model.model_name}
- Enhanced SRL: {'[ENABLED]' if getattr(self.model, 'use_enhanced_srl', False) else '[DISABLED]'} (Contrastive Margin: {getattr(self.model, 'path_separation_margin', 'N/A')})
- Ca-TSU: {'[ENABLED]' if getattr(self.model, 'use_ca_tsu', False) else '[DISABLED]'} (Causal Lambda: {getattr(self.model, 'causal_loss_lambda', 'N/A')})
- Noise-Aware Loss: {'[ENABLED]' if getattr(self.model, 'use_noise_aware_loss', False) else '[DISABLED]'}
- SRL Loss Weight: {getattr(self.model, 'noise_aware_weight') if getattr(self.model, 'use_noise_aware_loss', False) else getattr(self.model, 'prediction_loss_weight', 'N/A')}

TRAINING PLAN:
- Epochs: {self.model.n_epochs}
- Batch Size: {self.model.batch_size_for_name}
{'='*60}
"""
            print(startup_msg)
            self.progress_logger.info("Starting model training run.")
            
            # 3. METRIC TRACKING: Initialize lists to store metrics over time
            steps, losses, accuracies, mccs = [], [], [], []
            causal_losses, enhanced_srl_losses, noise_aware_losses = [], [], []
            reliability_weights_avg, volatility_factors, srl_text_weights, srl_price_weights = [], [], [], []

            # 4. TRAINING LOOP
            for epoch in range(self.model.n_epochs):
                self.progress_logger.info(f'Starting Epoch {epoch+1}/{self.model.n_epochs}...')
                train_batch_gen = self.pipe.batch_gen(phase='train')

                for train_batch_dict in train_batch_gen:
                    # A. Prepare feed dictionary
                    feed_dict = {
                        self.model.is_training_phase: True,
                        self.model.batch_size: train_batch_dict['batch_size'],
                        self.model.stock_ph: train_batch_dict['stock_batch'],
                        self.model.T_ph: train_batch_dict['T_batch'],
                        self.model.n_words_ph: train_batch_dict['n_words_batch'],
                        self.model.n_msgs_ph: train_batch_dict['n_msgs_batch'],
                        self.model.y_ph: train_batch_dict['y_batch'],
                        self.model.price_ph: train_batch_dict['price_batch'],
                        self.model.mv_percent_ph: train_batch_dict['mv_percent_batch'],
                        self.model.word_ph: train_batch_dict['word_batch'],
                        self.model.ss_index_ph: train_batch_dict['ss_index_batch'],
                    }

                    if hasattr(self.model, 'msg_mask_ph'):
                        batch_size = train_batch_dict['batch_size']
                        n_msgs_batch = train_batch_dict['n_msgs_batch']
                        msg_mask = np.zeros((batch_size, self.model.max_n_days, self.model.max_n_msgs), dtype=bool)
                        for b in range(batch_size):
                            if b < len(n_msgs_batch):
                                for d in range(len(n_msgs_batch[b])):
                                    n_msgs = n_msgs_batch[b][d]
                                    if n_msgs > 0:
                                        msg_mask[b, d, :n_msgs] = True
                        feed_dict[self.model.msg_mask_ph] = msg_mask

                    # B. Define and run TensorFlow operations
                    ops_to_run = {
                        'optimize': self.model.optimize,
                        'loss': self.model.loss,
                        'global_step': self.model.global_step,
                        'y_T': self.model.y_T,
                        'y_T_': self.model.y_T_,
                    }
                    
                    for tensor_name in [
                        'alpha_i', 'omega_i', 'final_text_weight', 'final_price_weight',
                        'volatility_factor', 'reliability_weights', 'causal_consistency_loss_raw',
                        'enhanced_srl_loss_raw', 'noise_aware_loss_raw'
                    ]:
                        if hasattr(self.model, tensor_name) and getattr(self.model, tensor_name) is not None:
                            ops_to_run[tensor_name] = getattr(self.model, tensor_name)
                    
                    results = sess.run(ops_to_run, feed_dict=feed_dict)
                    
                    # C. Unpack results and calculate metrics
                    batch_loss = results['loss']
                    n_iter = results['global_step']
                    batch_y, batch_y_ = results['y_T'], results['y_T_']
                    
                    batch_n_acc = sess.run(metrics.n_accurate(y=batch_y, y_=batch_y_))
                    batch_accuracy = batch_n_acc / train_batch_dict['batch_size']
                    tp, fp, tn, fn = metrics.create_confusion_matrix(y=batch_y, y_=batch_y_)
                    batch_mcc = metrics.eval_mcc(tp, fp, tn, fn) or 0.0

                    # D. Store metrics for plotting
                    steps.append(n_iter)
                    losses.append(batch_loss)
                    accuracies.append(batch_accuracy)
                    mccs.append(batch_mcc)
                    
                    causal_loss = results.get('causal_consistency_loss_raw', 0.0)
                    srl_loss = results.get('enhanced_srl_loss_raw', 0.0)
                    noise_aware_loss = results.get('noise_aware_loss_raw', 0.0)
                    
                    causal_losses.append(causal_loss)
                    enhanced_srl_losses.append(srl_loss or noise_aware_loss) # Store whichever is active
                    noise_aware_losses.append(noise_aware_loss) # Specifically for noise-aware plot
                    
                    reliability_weights_avg.append(np.mean(results.get('reliability_weights', 0.0)))
                    volatility_factors.append(np.mean(results.get('volatility_factor', 0.0)))
                    srl_text_weights.append(np.mean(results.get('final_text_weight', 0.0)))
                    srl_price_weights.append(np.mean(results.get('final_price_weight', 0.0)))

                    # E. Log progress periodically
                    if n_iter % self.skip_step == 0:
                        self.progress_logger.info(f"Step {n_iter}: Loss={batch_loss:.4f}, Acc={batch_accuracy:.4f}, MCC={batch_mcc:.4f}")
                        stat_logger.print_batch_stat(n_iter, batch_loss, batch_n_acc,
                                                     train_batch_dict['batch_size'])
                        os.makedirs(os.path.dirname(self.model.tf_saver_path), exist_ok=True)  # Ensure the directory exists
                        self.saver.save(sess, self.model.tf_saver_path, n_iter)

                # F. Save checkpoint at the end of each epoch
                #self.saver.save(sess, self.model.tf_saver_path, global_step=self.model.global_step)
                logger.info(f"Epoch {epoch+1} finished. Checkpoint saved.")

            # 5. POST-TRAINING: Generate diagnostic plots
            if steps:
                self._plot_diagnostics(
                    steps, losses, accuracies, mccs,
                    causal_losses, enhanced_srl_losses, noise_aware_losses,
                    srl_text_weights, srl_price_weights, volatility_factors,
                    reliability_weights_avg
                )

            # 6. FINAL EVALUATION: Run on the development set
            self.progress_logger.info('Training finished. Starting final evaluation on dev set...')
            dev_results = self.generation(sess, phase='dev')
            stat_logger.print_eval_res(dev_results, use_mcc=True)
            writer.close()
            
            # 7. Return dev results for Optuna
            return dev_results

    def run(self):
        return self.train_and_dev()

    def _plot_diagnostics(self, steps, losses, accuracies, mccs, 
                          causal_losses, enhanced_srl_losses, noise_aware_losses,
                          srl_text_weights, srl_price_weights, volatility_factors,
                          reliability_weights_avg):
        """
        Generates and saves a set of diagnostic plots for model analysis.
        """
        logger.info("Generating diagnostic plots...")
        results_dir = 'results/model_test/component_plots'
        os.makedirs(results_dir, exist_ok=True)

        # Plot 1: Main Performance Metrics
        plt.figure(figsize=(18, 5))
        plt.subplot(1, 3, 1)
        plt.plot(steps, losses, label='Total Loss', color='black')
        plt.title('Total Training Loss')
        plt.xlabel('Global Step')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 2)
        plt.plot(steps, accuracies, label='Accuracy', color='blue')
        plt.title('Training Accuracy')
        plt.xlabel('Global Step')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 3)
        plt.plot(steps, mccs, label='MCC', color='green')
        plt.title('Training MCC')
        plt.xlabel('Global Step')
        plt.ylabel('MCC')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, '1_performance_metrics.png'), dpi=300)
        plt.close()

        # Plot 2: Loss Components Breakdown
        plt.figure(figsize=(10, 6))
        plt.plot(steps, causal_losses, label='Causal Consistency Loss', linestyle='--')
        plt.plot(steps, enhanced_srl_losses, label='Enhanced SRL Loss', linestyle='--')
        plt.plot(steps, noise_aware_losses, label='Noise-Aware Loss', linestyle='--')
        plt.plot(steps, losses, label='Total Loss', color='black', linewidth=2)
        plt.title('Loss Components vs. Total Loss')
        plt.xlabel('Global Step')
        plt.ylabel('Loss Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(results_dir, '2_loss_components.png'), dpi=300)
        plt.close()

        # Plot 3: Enhanced SRL Pathway Dynamics
        plt.figure(figsize=(10, 6))
        plt.plot(steps, volatility_factors, label='Volatility Factor', color='purple')
        plt.plot(steps, srl_text_weights, label='Text-Driven Weight', color='green', linestyle=':')
        plt.plot(steps, srl_price_weights, label='Price-Driven Weight', color='orange', linestyle=':')
        plt.title('SRL Pathway Dynamics vs. Market Volatility')
        plt.xlabel('Global Step')
        plt.ylabel('Weight / Factor')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(results_dir, '3_srl_dynamics.png'), dpi=300)
        plt.close()

        # Plot 4: Noise-Aware Mechanism
        plt.figure(figsize=(10, 6))
        plt.plot(steps, reliability_weights_avg, label='Avg. Reliability Weight', color='red')
        plt.title('Noise-Aware Mechanism: Average Reliability')
        plt.xlabel('Global Step')
        plt.ylabel('Average Weight')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(results_dir, '4_noise_aware_reliability.png'), dpi=300)
        plt.close()
        
        logger.info(f"Diagnostic plots saved to {results_dir}")


    def _plot_training_metrics_combined(self, steps, losses, accuracies, mccs, 
                                        causal_losses=None, noise_aware_losses=None, 
                                        reliability_weights=None, enhanced_srl_losses=None):
        """
        Plot combined training metrics for the model test.
        
        Args:
            steps: List of training steps
            losses: List of loss values
            accuracies: List of accuracy values
            mccs: List of MCC values
            causal_losses: List of causal consistency loss values (optional)
            noise_aware_losses: List of noise-aware loss values (optional)
            reliability_weights: List of average reliability weights (optional)
            enhanced_srl_losses: List of enhanced SRL loss values (optional)
        """
        # Create results directory
        results_dir = 'results/model_test'
        os.makedirs(results_dir, exist_ok=True)
        
        # Ensure all main arrays have the same length
        min_length = min(len(steps), len(losses), len(accuracies), len(mccs))
        steps = steps[:min_length]
        losses = losses[:min_length]
        accuracies = accuracies[:min_length]
        mccs = mccs[:min_length]
        
        # Trim optional arrays to match the main arrays length
        if causal_losses:
            causal_losses = causal_losses[:min_length] if len(causal_losses) > min_length else causal_losses
        if enhanced_srl_losses:
            enhanced_srl_losses = enhanced_srl_losses[:min_length] if len(enhanced_srl_losses) > min_length else enhanced_srl_losses
        if noise_aware_losses:
            noise_aware_losses = noise_aware_losses[:min_length] if len(noise_aware_losses) > min_length else noise_aware_losses
        if reliability_weights:
            reliability_weights = reliability_weights[:min_length] if len(reliability_weights) > min_length else reliability_weights
        
        logger.info(f"Plotting metrics with {len(steps)} data points")
        
        # Determine number of plots needed
        num_plots = 3  # Always plot loss, accuracy, and MCC
        if causal_losses:
            num_plots += 1
        if enhanced_srl_losses:
            num_plots += 1
        if noise_aware_losses:
            num_plots += 1
        if reliability_weights:
            num_plots += 1
        
        # Determine grid size
        if num_plots <= 3:
            rows, cols = 1, 3
        elif num_plots <= 6:
            rows, cols = 2, 3
        else:
            rows, cols = 3, 3
        
        # Create plot
        plt.figure(figsize=(15, 5 * rows))
        
        # Plot main metrics
        plt.subplot(rows, cols, 1)
        plt.plot(steps, losses)
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(rows, cols, 2)
        plt.plot(steps, accuracies)
        plt.xlabel('Training Steps')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(rows, cols, 3)
        plt.plot(steps, mccs)
        plt.xlabel('Training Steps')
        plt.ylabel('MCC')
        plt.title('Training MCC')
        plt.grid(True, alpha=0.3)
        
        # Plot optional metrics
        plot_idx = 4
        if causal_losses:
            plt.subplot(rows, cols, plot_idx)
            plt.plot(steps, causal_losses)
            plt.xlabel('Training Steps')
            plt.ylabel('Ca-TSU Loss')
            plt.title('Causal Consistency Loss')
            plt.grid(True, alpha=0.3)
            plot_idx += 1
        
        if enhanced_srl_losses:
            plt.subplot(rows, cols, plot_idx)
            plt.plot(steps, enhanced_srl_losses)
            plt.xlabel('Training Steps')
            plt.ylabel('Enhanced SRL Loss')
            plt.title('Enhanced SRL Loss')
            plt.grid(True, alpha=0.3)
            plot_idx += 1
        
        if noise_aware_losses:
            plt.subplot(rows, cols, plot_idx)
            plt.plot(steps, noise_aware_losses)
            plt.xlabel('Training Steps')
            plt.ylabel('Noise-Aware Loss')
            plt.title('Noise-Aware Loss')
            plt.grid(True, alpha=0.3)
            plot_idx += 1
        
        if reliability_weights:
            plt.subplot(rows, cols, plot_idx)
            plt.plot(steps, reliability_weights)
            plt.xlabel('Training Steps')
            plt.ylabel('Avg Reliability')
            plt.title('Average Reliability Weights')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'training_metrics.png'), dpi=300)
        plt.close()
        
        # Create comparison plot for loss components
        if causal_losses or noise_aware_losses or enhanced_srl_losses:
            plt.figure(figsize=(10, 6))
            plt.plot(steps, losses, label='Total Loss')
            
            if causal_losses and len(causal_losses) >= len(steps):
                plt.plot(steps, causal_losses[:len(steps)], label='Ca-TSU Loss')
            elif causal_losses:
                # Extend steps to match causal_losses length if needed
                extended_steps = steps + list(range(len(steps), len(causal_losses)))
                plt.plot(extended_steps, causal_losses, label='Ca-TSU Loss')
            
            if enhanced_srl_losses and len(enhanced_srl_losses) >= len(steps):
                plt.plot(steps, enhanced_srl_losses[:len(steps)], label='Enhanced SRL Loss')
            elif enhanced_srl_losses:
                # Extend steps to match enhanced_srl_losses length if needed
                extended_steps = steps + list(range(len(steps), len(enhanced_srl_losses)))
                plt.plot(extended_steps, enhanced_srl_losses, label='Enhanced SRL Loss')
            
            if noise_aware_losses and len(noise_aware_losses) >= len(steps):
                plt.plot(steps, noise_aware_losses[:len(steps)], label='Noise-Aware Loss')
            elif noise_aware_losses:
                # Extend steps to match noise_aware_losses length if needed
                extended_steps = steps + list(range(len(steps), len(noise_aware_losses)))
                plt.plot(extended_steps, noise_aware_losses, label='Noise-Aware Loss')
            
            plt.xlabel('Training Steps')
            plt.ylabel('Loss')
            plt.title('Comparison of Loss Components')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(results_dir, 'loss_components_comparison.png'), dpi=300)
            plt.close()
            
            # Create component contribution percentage plot
            if all(x is not None for x in [causal_losses, enhanced_srl_losses, noise_aware_losses]):
                plt.figure(figsize=(12, 6))
                
                # Calculate total specialized loss at each step (sum of all components)
                component_totals = []
                for i in range(len(steps)):
                    ca_tsu_val = causal_losses[i] if i < len(causal_losses) else 0
                    srl_val = enhanced_srl_losses[i] if i < len(enhanced_srl_losses) else 0
                    na_val = noise_aware_losses[i] if i < len(noise_aware_losses) else 0
                    component_totals.append(ca_tsu_val + srl_val + na_val)
                
                # Calculate percentages
                ca_tsu_pct = []
                srl_pct = []
                na_pct = []
                
                for i in range(len(steps)):
                    if component_totals[i] > 0:
                        ca_tsu_val = causal_losses[i] if i < len(causal_losses) else 0
                        srl_val = enhanced_srl_losses[i] if i < len(enhanced_srl_losses) else 0
                        na_val = noise_aware_losses[i] if i < len(noise_aware_losses) else 0
                        
                        ca_tsu_pct.append(ca_tsu_val / component_totals[i] * 100)
                        srl_pct.append(srl_val / component_totals[i] * 100)
                        na_pct.append(na_val / component_totals[i] * 100)
                    else:
                        # If total is zero, use equal distribution
                        ca_tsu_pct.append(33.33)
                        srl_pct.append(33.33)
                        na_pct.append(33.33)
                
                # Plot stacked area chart
                plt.stackplot(steps, 
                              ca_tsu_pct, srl_pct, na_pct, 
                              labels=['Ca-TSU', 'Enhanced SRL', 'Noise-Aware'],
                              alpha=0.7)
                
                plt.xlabel('Training Steps')
                plt.ylabel('Percentage Contribution (%)')
                plt.title('Relative Contribution of Loss Components')
                plt.legend(loc='upper right')
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(results_dir, 'component_contributions.png'), dpi=300)
                plt.close()
        
        logger.info(f"Training metrics plots saved to {results_dir}")

    def _save_test_summary(self, test_results, dev_results=None, 
                           avg_loss=None, avg_acc=None, avg_mcc=None,
                           avg_causal_loss=None, avg_noise_aware_loss=None, 
                           avg_enhanced_srl_loss=None, avg_reliability=None):
        """
        Save a summary of the test results to a JSON file.
        
        Args:
            test_results: Dictionary of test results
            dev_results: Dictionary of dev results (optional)
            avg_loss: Average training loss (optional)
            avg_acc: Average training accuracy (optional)
            avg_mcc: Average training MCC (optional)
            avg_causal_loss: Average causal consistency loss (optional)
            avg_noise_aware_loss: Average noise-aware loss (optional)
            avg_enhanced_srl_loss: Average enhanced SRL loss (optional)
            avg_reliability: Average reliability weight (optional)
        """
        # Create results directory
        results_dir = 'results/model_test'
        os.makedirs(results_dir, exist_ok=True)
        
        # Helper function to safely convert values to float
        def safe_float(value):
            if value is None:
                return 0.0
            try:
                # Handle numpy arrays and scalars
                if hasattr(value, 'item'):  # numpy scalar
                    return float(value.item())
                elif hasattr(value, 'numpy'):  # tensorflow tensor
                    return float(value.numpy())
                elif isinstance(value, (np.ndarray, np.generic)):
                    return float(value)
                else:
                    return float(value)
            except (ValueError, TypeError, AttributeError):
                return 0.0
        
        # Create timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize summary dictionary
        summary = {
            'timestamp': timestamp,
            'model_config': {
                'variant_type': self.model.variant_type if hasattr(self.model, 'variant_type') else 'unknown',
                'cell_type': self.model.cell_type if hasattr(self.model, 'cell_type') else 'unknown',
                'dropout': self.model.dropout if hasattr(self.model, 'dropout') else 0.0
            },
            'components': {
                'ca_tsu_enabled': hasattr(self.model, 'use_ca_tsu') and self.model.use_ca_tsu,
                'enhanced_srl_enabled': hasattr(self.model, 'use_enhanced_srl') and self.model.use_enhanced_srl,
                'dual_path_srl_enabled': hasattr(self.model, 'use_dual_path_srl') and self.model.use_dual_path_srl,
                'noise_aware_loss_enabled': hasattr(self.model, 'use_noise_aware_loss') and self.model.use_noise_aware_loss,
                'ca_tsu_weight': safe_float(self.model.ca_tsu_weight if hasattr(self.model, 'ca_tsu_weight') else 0.0),
                'causal_loss_lambda': safe_float(self.model.causal_loss_lambda if hasattr(self.model, 'causal_loss_lambda') else 0.0),
                'enhanced_srl_weight': safe_float(self.model.enhanced_srl_weight if hasattr(self.model, 'enhanced_srl_weight') else 0.0),
                'noise_aware_weight': safe_float(self.model.noise_aware_weight if hasattr(self.model, 'noise_aware_weight') else 0.0)
            },
            'test_results': test_results,
            'training': {}
        }
        
        # Add dev results if available
        if dev_results:
            summary['dev_results'] = dev_results
        
        # Add training metrics if available
        if avg_loss is not None:
            summary['training']['avg_loss'] = safe_float(avg_loss)
        
        if avg_acc is not None:
            summary['training']['avg_accuracy'] = safe_float(avg_acc)
        
        if avg_mcc is not None:
            summary['training']['avg_mcc'] = safe_float(avg_mcc)
        
        if avg_causal_loss is not None:
            summary['training']['avg_causal_loss'] = safe_float(avg_causal_loss)
            summary['components']['ca_tsu_weight'] = safe_float(
                self.model.ca_tsu_weight if hasattr(self.model, 'ca_tsu_weight') else 0.0
            )
            summary['components']['causal_loss_lambda'] = safe_float(
                self.model.causal_loss_lambda if hasattr(self.model, 'causal_loss_lambda') else 0.0
            )
        
        if avg_noise_aware_loss is not None:
            summary['training']['avg_noise_aware_loss'] = safe_float(avg_noise_aware_loss)
            summary['components']['noise_aware_weight'] = safe_float(
                self.model.noise_aware_weight if hasattr(self.model, 'noise_aware_weight') else 0.0
            )
        
        if avg_enhanced_srl_loss is not None:
            summary['training']['avg_enhanced_srl_loss'] = safe_float(avg_enhanced_srl_loss)
            summary['components']['enhanced_srl_weight'] = safe_float(
                self.model.enhanced_srl_weight if hasattr(self.model, 'enhanced_srl_weight') else 0.1
            )
        
        if avg_reliability is not None:
            summary['training']['avg_reliability_weight'] = safe_float(avg_reliability)
        
        # Calculate component contributions if possible
        if all(x is not None for x in [avg_causal_loss, avg_noise_aware_loss, avg_enhanced_srl_loss]):
            total_component_loss = avg_causal_loss + avg_noise_aware_loss + avg_enhanced_srl_loss
            if total_component_loss > 0:
                summary['component_contribution'] = {
                    'ca_tsu_percentage': safe_float(avg_causal_loss / total_component_loss * 100),
                    'enhanced_srl_percentage': safe_float(avg_enhanced_srl_loss / total_component_loss * 100),
                    'noise_aware_loss_percentage': safe_float(avg_noise_aware_loss / total_component_loss * 100)
                }
        
        # Save summary to file
        output_file = os.path.join(results_dir, f'test_summary_{timestamp}.json')
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Test summary saved to {output_file}")
        
        # Create a human-readable summary file
        readable_output_file = os.path.join(results_dir, f'test_summary_{timestamp}.txt')
        with open(readable_output_file, 'w') as f:
            f.write("Enhanced Dual-Path SRL Model Test Summary\n")
            f.write("========================================\n\n")
            f.write(f"Test run at: {timestamp}\n\n")
            
            # Write model configuration
            f.write("Model Configuration:\n")
            f.write(f"- Variant type: {summary['model_config']['variant_type']}\n")
            f.write(f"- Cell type: {summary['model_config']['cell_type']}\n")
            f.write(f"- Dropout: {summary['model_config']['dropout']}\n\n")
            
            # Write component configuration
            f.write("Enhancement Components:\n")
            f.write(f"- Ca-TSU: {'Enabled' if summary['components']['ca_tsu_enabled'] else 'Disabled'}\n")
            if summary['components']['ca_tsu_enabled']:
                f.write(f"  - Weight: {summary['components']['ca_tsu_weight']}\n")
                f.write(f"  - Causal loss lambda: {summary['components']['causal_loss_lambda']}\n")
            
            f.write(f"- Enhanced SRL: {'Enabled' if summary['components']['enhanced_srl_enabled'] else 'Disabled'}\n")
            if summary['components']['enhanced_srl_enabled']:
                f.write(f"  - Weight: {summary['components']['enhanced_srl_weight']}\n")
            
            f.write(f"- Dual-Path SRL: {'Enabled' if summary['components']['dual_path_srl_enabled'] else 'Disabled'}\n")
            
            f.write(f"- Noise-Aware Loss: {'Enabled' if summary['components']['noise_aware_loss_enabled'] else 'Disabled'}\n")
            if summary['components']['noise_aware_loss_enabled']:
                f.write(f"  - Weight: {summary['components']['noise_aware_weight']}\n\n")
            
            # Write training metrics
            if 'training' in summary and summary['training']:
                f.write("Training Metrics:\n")
                if 'avg_loss' in summary['training']:
                    f.write(f"- Average Loss: {summary['training']['avg_loss']:.4f}\n")
                if 'avg_accuracy' in summary['training']:
                    f.write(f"- Average Accuracy: {summary['training']['avg_accuracy']:.4f}\n")
                if 'avg_mcc' in summary['training']:
                    f.write(f"- Average MCC: {summary['training']['avg_mcc']:.4f}\n")
                if 'avg_causal_loss' in summary['training']:
                    f.write(f"- Average Ca-TSU Loss: {summary['training']['avg_causal_loss']:.4f}\n")
                if 'avg_enhanced_srl_loss' in summary['training']:
                    f.write(f"- Average Enhanced SRL Loss: {summary['training']['avg_enhanced_srl_loss']:.4f}\n")
                if 'avg_noise_aware_loss' in summary['training']:
                    f.write(f"- Average Noise-Aware Loss: {summary['training']['avg_noise_aware_loss']:.4f}\n")
                if 'avg_reliability_weight' in summary['training']:
                    f.write(f"- Average Reliability Weight: {summary['training']['avg_reliability_weight']:.4f}\n\n")
            
            # Write component contributions
            if 'component_contribution' in summary:
                f.write("Component Contributions:\n")
                f.write(f"- Ca-TSU: {summary['component_contribution']['ca_tsu_percentage']:.1f}%\n")
                f.write(f"- Enhanced SRL: {summary['component_contribution']['enhanced_srl_percentage']:.1f}%\n")
                f.write(f"- Noise-Aware Loss: {summary['component_contribution']['noise_aware_loss_percentage']:.1f}%\n\n")
            
            # Write test results
            f.write("Test Results:\n")
            f.write(f"- Accuracy: {test_results['accuracy']:.4f}\n")
            f.write(f"- Precision: {test_results['precision']:.4f}\n")
            f.write(f"- Recall: {test_results['recall']:.4f}\n")
            f.write(f"- F1: {test_results['f1']:.4f}\n")
            f.write(f"- MCC: {test_results['mcc']:.4f}\n\n")
            
            # Write dev results if available
            if 'dev_results' in summary:
                f.write("Dev Results:\n")
                f.write(f"- Accuracy: {summary['dev_results']['accuracy']:.4f}\n")
                f.write(f"- Precision: {summary['dev_results']['precision']:.4f}\n")
                f.write(f"- Recall: {summary['dev_results']['recall']:.4f}\n")
                f.write(f"- F1: {summary['dev_results']['f1']:.4f}\n")
                f.write(f"- MCC: {summary['dev_results']['mcc']:.4f}\n")
            
        logger.info(f"Human-readable summary saved to {readable_output_file}")

    def restore_and_test(self):
        with tf.Session(config=self.tf_config) as sess:
            # init all vars with tables
            feed_table_init = {self.model.word_table_init: self.pipe.init_word_table()}
            sess.run(tf.global_variables_initializer(), feed_dict=feed_table_init)
            logger.info('Word table init: done!')

            try:
                # Try to find the latest checkpoint
                checkpoint = tf.train.get_checkpoint_state(os.path.dirname(self.model.tf_saver_path))
                if checkpoint and checkpoint.model_checkpoint_path:
                    self.saver.restore(sess, checkpoint.model_checkpoint_path)
                    logger.info(f'Model: {self.model.model_name}, session restored from latest checkpoint!')
                else:
                    # Try direct path with global step
                    step_value = self.model.global_step.eval(sess)
                    checkpoint_path = f"{self.model.tf_saver_path}-{step_value}"
                    if os.path.exists(f"{checkpoint_path}.index"):
                        self.saver.restore(sess, checkpoint_path)
                        logger.info(f'Model: {self.model.model_name}, session restored from specific step {step_value}!')
                    else:
                        # Try alternative path
                        alt_checkpoint = os.path.join(os.path.dirname(self.model.tf_saver_path), f"model_final-{step_value}")
                        if os.path.exists(f"{alt_checkpoint}.index"):
                            self.saver.restore(sess, alt_checkpoint)
                            logger.info(f'Model: {self.model.model_name}, session restored from alternative checkpoint!')
                        else:
                            logger.warning(f'No checkpoint found for model: {self.model.model_name}, using initialized values')
            except Exception as e:
                logger.error(f"Error restoring checkpoint: {str(e)}")
                logger.warning(f'Using initialized values due to checkpoint restore error')
            
            # test phase
            logger.info('Start test phase...')
            test_results = self.generation(sess, phase='test')
            stat_logger.print_eval_res(test_results)

