#!/usr/local/bin/python
import os
import tensorflow as tf
import metrics as metrics_module # Renamed to avoid conflict
import stat_logger as stat_logger_module # Renamed to avoid conflict
from DataPipe_filter import DataPipe # Use the filtered version
from ConfigLoader import logger
import pdb
import re
# Removed Model import from here, it will be passed as an object

class Executor:

    def __init__(self, model_object_creator, silence_step=200, skip_step=20): # Renamed model_obj to model_object_creator
        self.pipe = DataPipe() # Uses DataPipe_filter
        # model_object_creator is a function/class that returns a new model instance, e.g., Model_filter
        self.model = model_object_creator() # Creates an instance of Model_filter
        self.silence_step = silence_step
        self.skip_step = skip_step

        self.saver = tf.train.Saver()
        self.tf_config = tf.ConfigProto(allow_soft_placement=True)
        self.tf_config.gpu_options.allow_growth = True

    def unit_test_train(self):
        with tf.Session() as sess:
            word_table_init_val = self.pipe.init_word_table() # Renamed word_table_init
            feed_table_init = {self.model.word_table_init: word_table_init_val}
            sess.run(tf.global_variables_initializer(), feed_dict=feed_table_init)
            logger.info('Word table init: done!')

            logger.info('Model: {0}, start a new session!'.format(self.model.model_name))

            n_iter = self.model.global_step.eval()

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

                ops = [self.model.y_T, self.model.y_T_gt,  self.model.loss, self.model.optimize, self.model.increment_global_step] # y_T_ renamed to y_T_gt in Model
                # pdb.set_trace() # Commented out debugger
                train_batch_y, train_batch_y_gt,  train_batch_loss, _, _ = sess.run(ops, feed_dict) # Adjusted for increment_global_step
                
                train_epoch_size += float(train_batch_dict['batch_size'])
                train_batch_loss_list.append(train_batch_loss)
                train_batch_n_acc = sess.run(metrics_module.n_accurate(y=train_batch_y, y_=train_batch_y_gt))
                train_epoch_n_acc += float(train_batch_n_acc)

                stat_logger_module.print_batch_stat(n_iter, train_batch_loss, train_batch_n_acc,
                                             train_batch_dict['batch_size'])
                n_iter = self.model.global_step.eval() # Re-evaluate n_iter after increment

    def generation(self, sess, phase):
        generation_gen = self.pipe.batch_gen_by_stocks(phase)

        gen_loss_list = list()
        gen_size, gen_n_acc = 0.0, 0.0
        y_list, y_list_gt = list(), list() # Renamed y_list_

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

            gen_batch_y, gen_batch_y_gt, gen_batch_loss= sess.run([self.model.y_T, self.model.y_T_gt, self.model.loss],
                                                                 feed_dict=feed_dict)
            y_list.append(gen_batch_y)
            y_list_gt.append(gen_batch_y_gt)
            gen_loss_list.append(gen_batch_loss)

            gen_batch_n_acc = float(sess.run(metrics_module.n_accurate(y=gen_batch_y, y_=gen_batch_y_gt)))
            gen_n_acc += gen_batch_n_acc
            batch_size_val = float(gen_batch_dict['batch_size']) # Renamed batch_size
            gen_size += batch_size_val

        results = metrics_module.eval_res(gen_n_acc, gen_size, gen_loss_list, y_list, y_list_gt, use_mcc =True)
        return results
    
    def train_and_dev(self):
        with tf.Session(config=self.tf_config) as sess:
            sanitized_path = re.sub(r'[<>:"/\\|?*;]', '_', self.model.tf_graph_path)
            # Ensure directory exists, alternative to exist_ok=True for older Pythons if needed
            if not os.path.exists(os.path.dirname(sanitized_path)):
                 os.makedirs(os.path.dirname(sanitized_path))
            writer = tf.summary.FileWriter(sanitized_path, sess.graph)

            feed_table_init = {self.model.word_table_init: self.pipe.init_word_table()}
            sess.run(tf.global_variables_initializer(), feed_dict=feed_table_init)
            logger.info('Word table init: done!')

            checkpoint = tf.train.get_checkpoint_state(os.path.dirname(self.model.tf_checkpoint_file_path))
            if checkpoint and checkpoint.model_checkpoint_path:
                reader = tf.train.NewCheckpointReader(checkpoint.model_checkpoint_path)
                restore_dict = dict()
                for var_to_restore in tf.global_variables(): # Iterate over global_variables for restoration
                    tensor_name = var_to_restore.name.split(':')[0]
                    if reader.has_tensor(tensor_name):
                        # print('has tensor: {0}'.format(tensor_name)) # Optional print
                        restore_dict[tensor_name] = var_to_restore
                
                if restore_dict: # Only create and restore if there are variables to restore
                    checkpoint_saver = tf.train.Saver(restore_dict)
                    checkpoint_saver.restore(sess, checkpoint.model_checkpoint_path)
                    logger.info('Model: {0}, session restored from checkpoint!'.format(self.model.model_name))
                else:
                    logger.info('Model: {0}, checkpoint found but no matching variables to restore. Starting new session.'.format(self.model.model_name))
            else:
                logger.info('Model: {0}, start a new session! No checkpoint found.'.format(self.model.model_name))

            for epoch in range(self.model.n_epochs):
                logger.info('Epoch: {0}/{1} start'.format(epoch+1, self.model.n_epochs))
                train_batch_loss_list = list()
                epoch_size, epoch_n_acc = 0.0, 0.0
                train_batch_gen = self.pipe.batch_gen(phase='train')

                for train_batch_dict in train_batch_gen:
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

                    ops = [self.model.y_T, self.model.y_T_gt,  self.model.loss, self.model.optimize, self.model.increment_global_step, self.model.global_step]
                    train_batch_y, train_batch_y_gt, train_batch_loss, _, _, n_iter_val = sess.run(ops, feed_dict) # Renamed n_iter
                    
                    epoch_size += float(train_batch_dict['batch_size'])
                    train_batch_loss_list.append(train_batch_loss)
                    train_batch_n_acc = sess.run(metrics_module.n_accurate(y=train_batch_y, y_=train_batch_y_gt))
                    epoch_n_acc += float(train_batch_n_acc)

                    if n_iter_val >= self.silence_step and n_iter_val % self.skip_step == 0:
                        stat_logger_module.print_batch_stat(n_iter_val, train_batch_loss, train_batch_n_acc,
                                                     train_batch_dict['batch_size'])
                        # Ensure directory for saver path exists
                        saver_dir = os.path.dirname(self.model.tf_saver_path)
                        if not os.path.exists(saver_dir):
                            os.makedirs(saver_dir)
                        self.saver.save(sess, self.model.tf_saver_path, n_iter_val)
                        # dev_res = self.generation(sess, phase='dev') # Renamed res to dev_res
                        # stat_logger_module.print_eval_res(dev_res)

                epoch_loss, epoch_acc = metrics_module.basic_train_stat(train_batch_loss_list, epoch_n_acc, epoch_size)
                stat_logger_module.print_epoch_stat(epoch_loss=epoch_loss, epoch_acc=epoch_acc)
        writer.close()

    def restore_and_test(self):
        with tf.Session(config=self.tf_config) as sess:
            checkpoint = tf.train.get_checkpoint_state(os.path.dirname(self.model.tf_checkpoint_file_path))
            if checkpoint and checkpoint.model_checkpoint_path:
                logger.info('Model: {0}, session restored for testing!'.format(self.model.model_name))
                self.saver.restore(sess, checkpoint.model_checkpoint_path)
            else:
                logger.info('Model checkpoint: {0} NOT found for testing!'.format(self.model.tf_checkpoint_file_path))
                raise IOError("Model checkpoint not found for testing.")

            test_res = self.generation(sess, phase='test') # Renamed res
            stat_logger_module.print_eval_res(test_res, use_mcc =True) 