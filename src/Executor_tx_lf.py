#!/usr/local/bin/python
import os
import tensorflow as tf
import metrics as metrics
import stat_logger as stat_logger
from DataPipe import DataPipe # Keep DataPipe for now, or create DataPipe_tx_lf if needed
from ConfigLoader_tx_lf import logger # Changed
from Model_tx_lf import Model # Changed
import re # Keep re for now, just in case, but its usage is commented out below
import numpy as np
import matplotlib.pyplot as plt # Added for plotting


class Executor:

    def __init__(self, model_obj, silence_step=200, skip_step=20):
        self.pipe = DataPipe()
        self.model = model_obj() # Will pass Model_tx_lf here
        self.model.assemble_graph() # Call assemble_graph to build the model's graph
        self.silence_step = silence_step
        self.skip_step = skip_step

        self.saver = tf.train.Saver()
        self.tf_config = tf.ConfigProto(allow_soft_placement=True)
        self.tf_config.gpu_options.allow_growth = True
        
        # Create results directory if it doesn't exist
        if not os.path.exists('results'):
            os.makedirs('results')

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
        generation_gen = self.pipe.batch_gen_by_stocks(phase)

        gen_loss_list = list()
        gen_size, gen_n_acc = 0.0, 0.0
        y_list, y_list_ = list(), list()

        
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

            gen_batch_y, gen_batch_y_, gen_batch_loss= sess.run([self.model.y_T, self.model.y_T_, self.model.loss],
                                                                 feed_dict=feed_dict)

            # gather
            y_list.append(gen_batch_y)
            y_list_.append(gen_batch_y_)
            gen_loss_list.append(gen_batch_loss)  # list of floats


            

            gen_batch_n_acc = float(sess.run(metrics.n_accurate(y=gen_batch_y, y_=gen_batch_y_)))  # float
            gen_n_acc += gen_batch_n_acc

            batch_size = float(gen_batch_dict['batch_size'])
            gen_size += batch_size

        results = metrics.eval_res(gen_n_acc, gen_size, gen_loss_list, y_list, y_list_,use_mcc =True)
        return results
    
    def train_and_dev(self):
        with tf.Session(config=self.tf_config) as sess:
            # prep: writer and init
            #sanitized_path = re.sub(r'[<>:"/\\|?*;]', '_', self.model.tf_graph_path) # Removed re import and usage
            #os.makedirs(os.path.dirname(sanitized_path), exist_ok=True)
            writer = tf.summary.FileWriter(self.model.tf_graph_path, sess.graph) # Use original path

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
                for v in tf.global_variables(): # Changed for TF1.x compatibility
                    tensor_name = v.name.split(':')[0]
                    if reader.has_tensor(tensor_name):
                        print('has tensor: {0}'.format(tensor_name))
                        restore_dict[tensor_name] = v

                checkpoint_saver = tf.train.Saver(restore_dict) # Changed for TF1.x compatibility
                checkpoint_saver.restore(sess, checkpoint.model_checkpoint_path)
                logger.info('Model: {0}, session restored!'.format(self.model.model_name))
            else:
                logger.info('Model: {0}, start a new session!'.format(self.model.model_name))

            # MODIFIED FOR TESTING: Run for a few batches only
            max_batches_to_test = 20 # Test for 20 batches
            batches_tested = 0

            # Lists to store metrics for plotting
            all_batch_losses = []
            all_batch_accuracies = []
            all_batch_mccs = []
            all_causal_losses = [] # Added to track causal consistency loss


            for epoch in range(self.model.n_epochs): # Loop for the number of epochs specified in config
                logger.info(f'Starting Epoch {epoch+1}/{self.model.n_epochs}...')

                # training phase
                train_batch_loss_list = list()
                epoch_size, epoch_n_acc = 0.0, 0.0

                train_batch_gen = self.pipe.batch_gen(phase='train')

                for train_batch_dict in train_batch_gen:
                    if batches_tested >= max_batches_to_test: # Test condition
                        logger.info(f"Reached max_batches_to_test ({max_batches_to_test}). Stopping training for test.")
                        break

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

                    ops_to_run = [
                        self.model.loss,
                        self.model.optimize,
                        self.model.global_step,
                        self.model.y_T,
                        self.model.y_T_,
                        self.model.causal_consistency_loss, # Added for Ca-TSU
                        self.model.alpha_i, # Added for Ca-TSU inspection
                        self.model.omega_i # Added for Ca-TSU inspection
                    ]
                    if hasattr(self.model, 'msin_cross_attention_weights') and self.model.msin_cross_attention_weights is not None:
                        ops_to_run.append(self.model.msin_cross_attention_weights)
                        batch_loss, _, n_iter, batch_y_T, batch_y_T_, causal_loss_val, alpha_i_val, omega_i_val, attention_weights_val = sess.run(ops_to_run, feed_dict)
                    else:
                        batch_loss, _, n_iter, batch_y_T, batch_y_T_, causal_loss_val, alpha_i_val, omega_i_val = sess.run(ops_to_run[:-1], feed_dict) 
                        attention_weights_val = "Not fetched (msin_cross_attention_weights is None)"
                    
                    current_batch_size = train_batch_dict['batch_size']
                    batch_n_accurate = metrics.n_accurate(y=batch_y_T, y_=batch_y_T_) 
                    batch_n_accurate_val = sess.run(batch_n_accurate) 
                    
                    batch_accuracy = metrics.eval_acc(n_acc=batch_n_accurate_val, total=current_batch_size)
                    
                    tp, fp, tn, fn = metrics.create_confusion_matrix(y=batch_y_T, y_=batch_y_T_)
                    batch_mcc = metrics.eval_mcc(tp, fp, tn, fn)
                    if batch_mcc is None: 
                        batch_mcc = 0.0 

                    # Store metrics
                    all_batch_losses.append(batch_loss)
                    all_batch_accuracies.append(batch_accuracy)
                    all_batch_mccs.append(batch_mcc)
                    all_causal_losses.append(causal_loss_val) # Track causal loss

                    if n_iter % 1 == 0: # Print every batch for detailed testing
                        logger.info(f"--- Epoch {epoch+1}, Batch {n_iter} (Overall batch {batches_tested + 1}) ---")
                        logger.info(f"Total Loss: {batch_loss}")
                        logger.info(f"Causal Consistency Loss: {causal_loss_val}") # Log Ca-TSU loss
                        logger.info(f"Accuracy: {batch_accuracy:.4f}")
                        logger.info(f"MCC: {batch_mcc:.4f}")
                        # Log alpha_i and omega_i for the first sample, first day, first 5 messages
                        if alpha_i_val.ndim >= 3 and omega_i_val.ndim >=3:
                            logger.info(f"Alpha_i (sample 0, day 0, first 5 msgs): {alpha_i_val[0, 0, :5]}")
                            logger.info(f"Omega_i (sample 0, day 0, first 5 msgs): {omega_i_val[0, 0, :5]}")
                        # logger.info(f"MSIN Cross Attention Weights (first item in batch, first 5 messages if available):\\n{attention_weights_val[0, :, :5] if isinstance(attention_weights_val, np.ndarray) else attention_weights_val}")
                    
                    train_batch_loss_list.append(batch_loss)
                    epoch_size += float(current_batch_size)
                    epoch_n_acc += float(batch_n_accurate_val)
                    
                    batches_tested += 1 # Increment batches_tested
                
                if batches_tested >= max_batches_to_test: # Also break outer epoch loop if max_batches_to_test is reached
                    break

                # Epoch summary
                epoch_loss, epoch_acc = metrics.basic_train_stat(train_batch_loss_list, epoch_n_acc, epoch_size)
                stat_logger.print_epoch_stat(epoch_loss=epoch_loss, epoch_acc=epoch_acc)
                logger.info(f"--- Epoch {epoch+1} Completed ---")

                # Save model checkpoint periodically
                if n_iter >= self.silence_step and n_iter % self.skip_step == 0:
                    os.makedirs(os.path.dirname(self.model.tf_saver_path), exist_ok=True)
                    self.saver.save(sess, self.model.tf_saver_path, global_step=n_iter)
                    logger.info(f"Model saved at global step {n_iter}")


            logger.info(f"--- Training completed for {self.model.n_epochs} epochs ---")

            # Save final metrics
            final_avg_loss = np.mean(all_batch_losses)
            final_avg_accuracy = np.mean(all_batch_accuracies)
            final_avg_mcc = np.mean(all_batch_mccs)
            final_avg_causal_loss = np.mean(all_causal_losses) # Added

            logger.info(f"--- Final Average Metrics ---")
            logger.info(f"Average Total Loss: {final_avg_loss}")
            logger.info(f"Average Causal Consistency Loss: {final_avg_causal_loss}") # Added
            logger.info(f"Average Accuracy: {final_avg_accuracy:.4f}")
            logger.info(f"Average MCC: {final_avg_mcc:.4f}")

            with open("results/final_metrics.txt", "w") as f:
                f.write(f"Final Average Total Loss: {final_avg_loss}\n")
                f.write(f"Final Average Causal Consistency Loss: {final_avg_causal_loss}\n") # Added
                f.write(f"Final Average Accuracy: {final_avg_accuracy:.4f}\n")
                f.write(f"Final Average MCC: {final_avg_mcc:.4f}\n")

            # Plotting
            batches = range(len(all_batch_losses))

            plt.figure(figsize=(10, 5))
            plt.plot(batches, all_batch_losses, label='Total Loss')
            plt.xlabel('Batch Number')
            plt.ylabel('Loss')
            plt.title('Total Loss vs. Batches')
            plt.legend()
            plt.savefig('results/total_loss_plot.png')
            plt.close()
            logger.info("Total Loss plot saved to results/total_loss_plot.png")

            plt.figure(figsize=(10, 5))
            plt.plot(batches, all_batch_accuracies, label='Accuracy')
            plt.xlabel('Batch Number')
            plt.ylabel('Accuracy')
            plt.title('Accuracy vs. Batches')
            plt.legend()
            plt.savefig('results/accuracy_plot.png')
            plt.close()
            logger.info("Accuracy plot saved to results/accuracy_plot.png")
            
            plt.figure(figsize=(10, 5))
            plt.plot(batches, all_batch_mccs, label='MCC')
            plt.xlabel('Batch Number')
            plt.ylabel('MCC')
            plt.title('MCC vs. Batches')
            plt.legend()
            plt.savefig('results/mcc_plot.png')
            plt.close()
            logger.info("MCC plot saved to results/mcc_plot.png")

            plt.figure(figsize=(10, 5))
            plt.plot(batches, all_causal_losses, label='Causal Consistency Loss')
            plt.xlabel('Batch Number')
            plt.ylabel('Causal Loss')
            plt.title('Causal Consistency Loss vs. Batches')
            plt.legend()
            plt.savefig('results/causal_loss_plot.png')
            plt.close()
            logger.info("Causal Loss plot saved to results/causal_loss_plot.png")

            # Save final checkpoint
            self.saver.save(sess, self.model.tf_saver_path, global_step=n_iter) # Use n_iter from training
            logger.info(f'Model {self.model.model_name} saved at step {n_iter}')

            # Dev phase (generation)
            logger.info('Start dev phase...')
            dev_results = self.generation(sess, phase='dev') # Assuming self.generation exists and works similarly
            stat_logger.print_eval_res(dev_results) # Assuming stat_logger.print_eval_res exists
            writer.close()
            return dev_results # Added return

        #writer.close()

    def restore_and_test(self):
        with tf.Session(config=self.tf_config) as sess:
            checkpoint = tf.train.get_checkpoint_state(os.path.dirname(self.model.tf_checkpoint_file_path))
            if checkpoint and checkpoint.model_checkpoint_path:
                logger.info('Model: {0}, session restored!'.format(self.model.model_name))
                self.saver.restore(sess, checkpoint.model_checkpoint_path)
            else:
                logger.info('Model: {0}: NOT found!'.format(self.model.model_name))
                raise IOError

            res= self.generation(sess, phase='test')
            stat_logger.print_eval_res(res,use_mcc =True) 