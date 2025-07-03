#!/usr/local/bin/python
import os
import tensorflow as tf
import metrics as metrics
import stat_logger as stat_logger
from DataPipe import DataPipe
from ConfigLoader_tx_lf_dual_path import logger
import pdb
import re
import numpy as np



class Executor:

    def __init__(self, model, silence_step=200, skip_step=20):
        self.pipe = DataPipe()
        self.model = model
        self.silence_step = silence_step
        self.skip_step = skip_step

        self.saver = tf.train.Saver()
        self.tf_config = tf.ConfigProto(allow_soft_placement=True)
        
        # Memory optimization configurations
        #self.tf_config.gpu_options.allow_growth = True
        #self.tf_config.gpu_options.per_process_gpu_memory_fraction = 0.95  # Leave some memory for system
        
        # Improve memory usage by setting these options
        ##self.tf_config.graph_options.optimizer_options.opt_level = tf.OptimizerOptions.L1
        
        # TF 1.4 compatible memory optimization options
        #self.tf_config.gpu_options.force_gpu_compatible = True
        
        # Enable operation-level parallelism but keep it reasonable
        #self.tf_config.intra_op_parallelism_threads = 4
        #self.tf_config.inter_op_parallelism_threads = 4

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
                pdb.set_trace()
                train_batch_y, train_batch_y_,  train_batch_loss, _ = sess.run(ops, feed_dict)
                
                # training batch stat
                train_epoch_size += float(train_batch_dict['batch_size'])
                train_batch_loss_list.append(train_batch_loss)
                train_batch_n_acc = sess.run(metrics.n_accurate(y=train_batch_y, y_=train_batch_y_))
                train_epoch_n_acc += float(train_batch_n_acc)

                stat_logger.print_batch_stat(n_iter, train_batch_loss, train_batch_n_acc,
                                             train_batch_dict['batch_size'])
                n_iter += 1

    def generation(self, sess, phase, analyze_attention=False):
        """
        Evaluate the model on a dataset and analyze attention if requested.
        
        Args:
            sess: TensorFlow session
            phase: Dataset phase ('train', 'dev', or 'test')
            analyze_attention: Whether to analyze dual pathway attention
            
        Returns:
            Dictionary containing evaluation results and attention analysis
        """
        generation_gen = self.pipe.batch_gen_by_stocks(phase)

        gen_loss_list = list()
        gen_size, gen_n_acc = 0.0, 0.0
        y_list, y_list_ = list(), list()

        # NEW: Initialize cross-pathway metrics
        cross_pathway_metrics = {
            'correlation': [],
            'causal_influence': [],
            'responsive_influence': []
        }
        
        # Store attention analysis results if requested
        attention_results = {}
        batch_idx = 0
        
        for gen_batch_dict in generation_gen:
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

            # Memory optimization: Always run basic metrics separately
            gen_batch_y, gen_batch_y_, gen_batch_loss = sess.run(
                [self.model.y_T, self.model.y_T_, self.model.loss],
                feed_dict=feed_dict
            )
            
            # Gather results
            y_list.append(gen_batch_y)
            y_list_.append(gen_batch_y_)
            gen_loss_list.append(gen_batch_loss)
            
            gen_batch_n_acc = float(sess.run(metrics.n_accurate(y=gen_batch_y, y_=gen_batch_y_)))
            gen_n_acc += gen_batch_n_acc

            batch_size = float(gen_batch_dict['batch_size'])
            gen_size += batch_size

            # Run cross-pathway metrics calculation as a separate step to reduce memory pressure
            try:
                # NEW: Extract cross-pathway metrics
                correlation, causal_influence, responsive_influence = sess.run(
                    [
                        # Add these new metrics
                        tf.reduce_mean(tf.reduce_sum(tf.multiply(self.model.P_causal, self.model.P_responsive), axis=[1, 2])),
                        tf.reduce_mean(self.model.k_causal),
                        tf.reduce_mean(self.model.k_responsive)
                    ],
                    feed_dict=feed_dict
                )
                
                # Store cross-pathway metrics
                cross_pathway_metrics['correlation'].append(float(correlation))
                cross_pathway_metrics['causal_influence'].append(float(causal_influence))
                cross_pathway_metrics['responsive_influence'].append(float(responsive_influence))
                
                # Log metrics for some batches
                #if batch_idx % 50 == 0:
                #    logger.info(f"{phase} Cross-Pathway Metrics - Correlation: {correlation:.4f}, "
                #           f"Causal Influence: {causal_influence:.4f}, "
                #            f"Responsive Influence: {responsive_influence:.4f}")
            except Exception as e:
                logger.warning(f"Could not extract cross-pathway metrics: {e}")

            # Analyze dual pathway attention if requested, but do this separately and only on small samples
            if analyze_attention and batch_idx < 3:
                try:
                    # Memory optimization: Run this in a separate session run to avoid OOM
                    batch_attention = self.analyze_dual_pathway_attention(sess, feed_dict)
                    attention_results[f'batch_{batch_idx}'] = batch_attention
                    
                    # Log sample of attention results
                    for b in range(min(1, gen_batch_dict['batch_size'])):  # Log just the first example
                        logger.info(f"--- Attention Analysis for {phase} Batch {batch_idx} Example {b} ---")
                        
                        # Get the first time step
                        time_results = batch_attention[f'batch_{b}']['time_0']
                        
                        # Log pathway importances
                        logger.info("Pathway Importance:")
                        for pathway, value in time_results['pathway_importance'].items():
                            logger.info(f"  {pathway}: {value:.4f}")
                        
                        # Log top causal attention weights
                        logger.info("Top Causal Attention:")
                        for i, weight in enumerate(time_results['causal_attention']['weights'][:3]):  # Log just top 3
                            msg_idx = time_results['causal_attention']['indices'][i]
                            logger.info(f"  Message {msg_idx}: {weight:.4f}")
                        
                        # Log top responsive attention weights
                        logger.info("Top Responsive Attention:")
                        for i, weight in enumerate(time_results['responsive_attention']['weights'][:3]):  # Log just top 3
                            msg_idx = time_results['responsive_attention']['indices'][i]
                            logger.info(f"  Message {msg_idx}: {weight:.4f}")
                except Exception as e:
                    logger.error(f"Error analyzing attention: {e}")
                    
                # Instead of resetting the graph (which causes errors), use garbage collection
                # to help free memory
                import gc
                gc.collect()
            
            batch_idx += 1

        # Evaluate metrics
        # Calculate average cross-pathway metrics
        avg_metrics = {}
        for metric, values in cross_pathway_metrics.items():
            if values:
                avg_metrics[metric] = sum(values) / len(values)
        results = metrics.eval_res(gen_n_acc, gen_size, gen_loss_list, y_list, y_list_, use_mcc=True)
        logger.info(f"Completed {phase} evaluation: size={gen_size}, acc={gen_n_acc/gen_size:.4f}")
        results['cross_pathway_metrics'] = avg_metrics
    
        # Log average metrics
        logger.info(f"{phase} Avg Cross-Pathway Metrics - "
                f"Correlation: {avg_metrics.get('correlation', 'N/A'):.4f}, "
                f"Causal Influence: {avg_metrics.get('causal_influence', 'N/A'):.4f}, "
                f"Responsive Influence: {avg_metrics.get('responsive_influence', 'N/A'):.4f}")
        # Add attention results to output if they were requested
        if analyze_attention:
            results['attention_analysis'] = attention_results
        
        return results
    
    def train_and_dev(self):
        with tf.Session(config=self.tf_config) as sess:
            # prep: writer and init
            sanitized_path = re.sub(r'[<>:"/\\|?*;]', '_', self.model.tf_graph_path)
            #os.makedirs(os.path.dirname(sanitized_path), exist_ok=True)
            writer = tf.summary.FileWriter(sanitized_path, sess.graph)

            # init all vars with tables
            feed_table_init = {self.model.word_table_init: self.pipe.init_word_table()}
            sess.run(tf.global_variables_initializer(), feed_dict=feed_table_init)
            logger.info('Word table init: done!')

            # prep: checkpoint
            checkpoint = tf.train.get_checkpoint_state(os.path.dirname(self.model.tf_checkpoint_file_path))
            if checkpoint and checkpoint.model_checkpoint_path:
                # restore partial saved vars
                reader = tf.train.NewCheckpointReader(checkpoint.model_checkpoint_path)
                restore_dict = dict()
                for v in tf.all_variables():
                    tensor_name = v.name.split(':')[0]
                    if reader.has_tensor(tensor_name):
                        print('has tensor: {0}'.format(tensor_name))
                        restore_dict[tensor_name] = v

                checkpoint_saver = tf.train.Saver(restore_dict)
                checkpoint_saver.restore(sess, checkpoint.model_checkpoint_path)
                logger.info('Model: {0}, session restored!'.format(self.model.model_name))
            else:
                logger.info('Model: {0}, start a new session!'.format(self.model.model_name))

            # Add a list to track metrics over epochs
            cross_pathway_history = []

            for epoch in range(self.model.n_epochs):
                logger.info('Epoch: {0}/{1} start'.format(epoch+1, self.model.n_epochs))

                # training phase
                train_batch_loss_list = list()
                epoch_size, epoch_n_acc = 0.0, 0.0
                # NEW: Track epoch metrics
                epoch_cross_metrics = {
                    'correlation': [],
                    'causal_influence': [],
                    'responsive_influence': []
                }

                train_batch_gen = self.pipe.batch_gen(phase='train')  # a new gen for a new epoch

                for train_batch_dict in train_batch_gen:

                    # logger.info('train: batch_size: {0}'.format(train_batch_dict['batch_size']))

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

                    #ops = [self.model.y_T, self.model.y_T_,  self.model.loss, self.model.optimize,
                    #       self.model.global_step]
                    #train_batch_y, train_batch_y_,  train_batch_loss, _, n_iter = sess.run(ops, feed_dict)
                    # NEW: Extract cross-pathway metrics during training
                    try:
                        ops = [self.model.y_T, self.model.y_T_, self.model.loss, self.model.optimize,
                            self.model.global_step,
                            # Add these new metrics
                            tf.reduce_mean(tf.reduce_sum(tf.multiply(self.model.P_causal, self.model.P_responsive), axis=[1, 2])),
                            tf.reduce_mean(self.model.k_causal),
                            tf.reduce_mean(self.model.k_responsive)]
                            
                        train_batch_y, train_batch_y_, train_batch_loss, _, n_iter, corr, c_infl, r_infl = sess.run(ops, feed_dict)
                        
                        # Store metrics
                        epoch_cross_metrics['correlation'].append(float(corr))
                        epoch_cross_metrics['causal_influence'].append(float(c_infl))
                        epoch_cross_metrics['responsive_influence'].append(float(r_infl))
                    except Exception as e:
                        logger.warning(f"Could not extract cross-pathway metrics: {e}")
                        ops = [self.model.y_T, self.model.y_T_, self.model.loss, self.model.optimize,
                            self.model.global_step]
                        train_batch_y, train_batch_y_, train_batch_loss, _, n_iter = sess.run(ops, feed_dict)

                    # training batch stat
                    epoch_size += float(train_batch_dict['batch_size'])
                    train_batch_loss_list.append(train_batch_loss)  # list of floats
                    train_batch_n_acc = sess.run(metrics.n_accurate(y=train_batch_y, y_=train_batch_y_))  # float
                    epoch_n_acc += float(train_batch_n_acc)

                    # save model and generation
                    if n_iter >= self.silence_step and n_iter % self.skip_step == 0:
                        stat_logger.print_batch_stat(n_iter, train_batch_loss, train_batch_n_acc,
                                                     train_batch_dict['batch_size'])
                        
                        os.makedirs(os.path.dirname(self.model.tf_saver_path), exist_ok=True)  # Ensure the directory exists
                        self.saver.save(sess, self.model.tf_saver_path, n_iter)
                        
                        # Run validation with attention analysis at regular intervals
                        analyze_attention = (n_iter % (self.skip_step * 5) == 0)  # Analyze every 5 save points
                        if analyze_attention:
                            logger.info("Running validation with dual-pathway attention analysis")
                        
                        # NEW: Log cross-pathway metrics
                        if epoch_cross_metrics['correlation']:
                            recent_corr = sum(epoch_cross_metrics['correlation'][-10:]) / min(len(epoch_cross_metrics['correlation']), 10)
                            recent_c_infl = sum(epoch_cross_metrics['causal_influence'][-10:]) / min(len(epoch_cross_metrics['causal_influence']), 10)
                            recent_r_infl = sum(epoch_cross_metrics['responsive_influence'][-10:]) / min(len(epoch_cross_metrics['responsive_influence']), 10)
                            
                            logger.info(f"Recent Cross-Pathway Metrics - Correlation: {recent_corr:.4f}, "
                                    f"Causal Influence: {recent_c_infl:.4f}, "
                                    f"Responsive Influence: {recent_r_infl:.4f}")
                            
                        res = self.generation(sess, phase='dev', analyze_attention=analyze_attention)
                        stat_logger.print_eval_res(res, use_mcc=True)

                # print training epoch stat
                epoch_loss, epoch_acc = metrics.basic_train_stat(train_batch_loss_list, epoch_n_acc, epoch_size)
                stat_logger.print_epoch_stat(epoch_loss=epoch_loss, epoch_acc=epoch_acc)

                # Calculate epoch average metrics
                epoch_avg_metrics = {}
                for metric, values in epoch_cross_metrics.items():
                    if values:
                        epoch_avg_metrics[metric] = sum(values) / len(values)
                
                # Add to history
                cross_pathway_history.append({
                    'epoch': epoch + 1,
                    'metrics': epoch_avg_metrics
                })
                
                # Log epoch metrics
                logger.info(f"Epoch {epoch+1} Avg Cross-Pathway Metrics - "
                        f"Correlation: {epoch_avg_metrics.get('correlation', 'N/A'):.4f}, "
                        f"Causal Influence: {epoch_avg_metrics.get('causal_influence', 'N/A'):.4f}, "
                        f"Responsive Influence: {epoch_avg_metrics.get('responsive_influence', 'N/A'):.4f}")
                
                # Save metrics to file
                self._save_cross_pathway_metrics(cross_pathway_history)

        writer.close()

    def restore_and_test(self):
        with tf.Session(config=self.tf_config) as sess:
            checkpoint = tf.train.get_checkpoint_state(os.path.dirname(self.model.tf_checkpoint_file_path))
            if checkpoint and checkpoint.model_checkpoint_path:
                logger.info('Model: {0}, session restored!'.format(self.model.model_name))
                self.saver.restore(sess, checkpoint.model_checkpoint_path)
            else:
                logger.info('Model: {0}: NOT found!'.format(self.model.model_name))
                raise IOError

            # Run test with dual-pathway attention analysis
            logger.info("Running test with dual-pathway attention analysis")
            res = self.generation(sess, phase='test', analyze_attention=True)
            stat_logger.print_eval_res(res, use_mcc=True)
            
            # Save attention analysis to file if present
            if 'attention_analysis' in res:
                import json
                import datetime
                
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                attention_file = f"attention_analysis_{timestamp}.json"
                attention_path = os.path.join(os.path.dirname(self.model.tf_saver_path), attention_file)
                
                try:
                    # Convert attention results to serializable format (if needed)
                    serializable_results = {}
                    for batch_key, batch_data in res['attention_analysis'].items():
                        serializable_results[batch_key] = {}
                        for example_key, example_data in batch_data.items():
                            serializable_results[batch_key][example_key] = {}
                            for time_key, time_data in example_data.items():
                                serializable_results[batch_key][example_key][time_key] = time_data
                    
                    with open(attention_path, 'w') as f:
                        json.dump(serializable_results, f, indent=2)
                    logger.info(f"Saved attention analysis to {attention_path}")
                except Exception as e:
                    logger.error(f"Error saving attention analysis: {e}")

            # Save cross-pathway metrics from test results
            if 'cross_pathway_metrics' in res:
                import json
                import datetime
                import os
                
                # Create metrics directory
                metrics_dir = os.path.join(os.path.dirname(self.model.tf_checkpoints_path), 'metrics')
                os.makedirs(metrics_dir, exist_ok=True)
                
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                metrics_file = f"test_cross_pathway_metrics_{timestamp}.json"
                metrics_path = os.path.join(metrics_dir, metrics_file)
                
                try:
                    with open(metrics_path, 'w') as f:
                        json.dump(res['cross_pathway_metrics'], f, indent=2)
                    logger.info(f"Saved test cross-pathway metrics to {metrics_path}")
                except Exception as e:
                    logger.error(f"Error saving test metrics: {e}")

    def _save_cross_pathway_metrics(self, metrics_history):
        """
        Save cross-pathway communication metrics to a file
        
        Args:
            metrics_history: List of dictionaries containing epoch metrics
        """
        import json
        import datetime
        import os
        
        # Create a more sanitized path
        sanitized_name = re.sub(r'[<>:"/\\|?*;]', '_', self.model.model_name)
        metrics_dir = os.path.join(os.path.dirname(self.model.tf_checkpoints_path), 'metrics')
        os.makedirs(metrics_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        metrics_file = f"cross_pathway_metrics_{timestamp}.json"
        metrics_path = os.path.join(metrics_dir, metrics_file)
        
        try:
            # Convert to serializable format
            serializable_metrics = []
            for epoch_data in metrics_history:
                epoch_metrics = {
                    'epoch': epoch_data['epoch'],
                    'correlation': float(epoch_data['metrics'].get('correlation', 0)),
                    'causal_influence': float(epoch_data['metrics'].get('causal_influence', 0)),
                    'responsive_influence': float(epoch_data['metrics'].get('responsive_influence', 0))
                }
                serializable_metrics.append(epoch_metrics)
            
            with open(metrics_path, 'w') as f:
                json.dump(serializable_metrics, f, indent=2)
            logger.info(f"Saved cross-pathway metrics to {metrics_path}")
        except Exception as e:
            logger.error(f"Error saving cross-pathway metrics: {e}")

    def analyze_dual_pathway_attention(self, sess, feed_dict, messages=None):
        """
        Analyze the dual-pathway attention to provide explainable results.
        
        Args:
            sess: TensorFlow session
            feed_dict: Feed dictionary for the session run
            messages: List of message texts (optional)
            
        Returns:
            Dictionary containing analysis results
        """
        #outputs = sess.run(
        #    [self.model.P_causal, self.model.P_responsive, self.model.k_price, 
        #     self.model.k_causal, self.model.k_responsive], 
        #    feed_dict=feed_dict
        #)
        #P_causal_val, P_responsive_val, k_price_val, k_causal_val, k_responsive_val = outputs
        outputs = sess.run(
        [self.model.P_causal, self.model.P_responsive, self.model.k_price, 
         self.model.k_causal, self.model.k_responsive,
         # Add correlation metric
         tf.reduce_sum(tf.multiply(self.model.P_causal, self.model.P_responsive), axis=[2])], 
                        feed_dict=feed_dict
        )
        P_causal_val, P_responsive_val, k_price_val, k_causal_val, k_responsive_val, correlation_val = outputs
        
        # Get top messages by attention weight
        top_k = 5  # Number of top messages to return
        batch_size = P_causal_val.shape[0]
        time_steps = P_causal_val.shape[1]
        
        results = {}
        
        for b in range(batch_size):
            batch_results = {}

            batch_results['batch_metrics'] = {
            'avg_correlation': float(np.mean(correlation_val[b])),
            'avg_causal_influence': float(np.mean(k_causal_val[b])),
            'avg_responsive_influence': float(np.mean(k_responsive_val[b])),
            }
            
            for t in range(time_steps):
                # Get attention weights for specific time step
                causal_weights = P_causal_val[b, t]
                responsive_weights = P_responsive_val[b, t]
                
                # Get gate values
                k_price = float(k_price_val[b, t, 0])
                k_causal = float(k_causal_val[b, t, 0])
                k_responsive = float(k_responsive_val[b, t, 0])
                
                # Find top messages by attention weight
                causal_top_idxs = np.argsort(-causal_weights)[:top_k]
                responsive_top_idxs = np.argsort(-responsive_weights)[:top_k]
                
                # Create analysis results
                time_results = {
                    'pathway_importance': {
                        'price_pathway': k_price,
                        'causal_pathway': k_causal,
                        'responsive_pathway': k_responsive
                    },
                    'causal_attention': {
                        'weights': [float(causal_weights[i]) for i in causal_top_idxs],
                        'indices': [int(i) for i in causal_top_idxs]
                    },
                    'responsive_attention': {
                        'weights': [float(responsive_weights[i]) for i in responsive_top_idxs],
                        'indices': [int(i) for i in responsive_top_idxs]
                    }
                }
                
                # Add message texts if provided
                if messages is not None:
                    time_results['causal_attention']['messages'] = [
                        messages[b][t][i] if i < len(messages[b][t]) else "" 
                        for i in causal_top_idxs
                    ]
                    time_results['responsive_attention']['messages'] = [
                        messages[b][t][i] if i < len(messages[b][t]) else "" 
                        for i in responsive_top_idxs
                    ]
                
                
                # Add correlation at this time step
                time_results['cross_pathway_metrics'] = {
                    'correlation': float(correlation_val[b, t]),
                    'causal_influence': float(k_causal_val[b, t, 0]),
                    'responsive_influence': float(k_responsive_val[b, t, 0]),
                }
                
                batch_results[f'time_{t}'] = time_results
            
            results[f'batch_{b}'] = batch_results
        
        logger.info("Dual-pathway attention analysis completed")
        return results

            
            