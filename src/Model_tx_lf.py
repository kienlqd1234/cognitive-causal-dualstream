#!/usr/local/bin/python
from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
import neural as neural
#from MSINModule import MSINCell, MSIN, MSINStateTuple
from MSINModule_MHA import MSINCell, MSIN, MSINStateTuple
import tensorflow.contrib.distributions as ds
from tensorflow.contrib.layers import batch_norm
from ConfigLoader_tx_lf import logger, ss_size, vocab_size, config_model, path_parser






class Model:

    def __init__(self):
        logger.info('INIT: #stock: {0}, #vocab+1: {1}'.format(ss_size, vocab_size))

        # model config
        self.mode = config_model['mode']
        self.opt = config_model['opt']
        self.lr = config_model['lr']
        self.decay_step = config_model['decay_step']
        self.decay_rate = config_model['decay_rate']
        self.momentum = config_model['momentum']

        self.kl_lambda_anneal_rate = config_model['kl_lambda_anneal_rate']
        self.kl_lambda_start_step = config_model['kl_lambda_start_step']
        self.use_constant_kl_lambda = config_model['use_constant_kl_lambda']
        self.constant_kl_lambda = config_model['constant_kl_lambda']

        self.daily_att = config_model['daily_att']
        self.alpha = config_model['alpha']

        self.entropy_regularization_lambda = config_model.get('entropy_regularization_lambda', 0.0)
        self.causal_loss_lambda = config_model.get('causal_loss_lambda', 1.0)

        self.clip = config_model['clip']
        self.n_epochs = config_model['n_epochs']
        self.batch_size_for_name = config_model['batch_size']

        self.max_n_days = config_model['max_n_days']
        self.max_n_msgs = config_model['max_n_msgs']
        self.max_n_words = config_model['max_n_words']

        self.weight_init = config_model['weight_init']
        uniform = True if self.weight_init == 'xavier-uniform' else False
        self.initializer = tf.contrib.layers.xavier_initializer(uniform=uniform)
        self.bias_initializer = tf.constant_initializer(0.0, dtype=tf.float32)

        self.word_embed_type = config_model['word_embed_type']

        self.y_size = config_model['y_size']
        self.word_embed_size = config_model['word_embed_size']
        self.stock_embed_size = config_model['stock_embed_size']
        self.price_embed_size = config_model['word_embed_size']

        self.mel_cell_type = config_model['mel_cell_type']
        self.variant_type = config_model['variant_type']
        self.vmd_cell_type = config_model['vmd_cell_type']

        self.vmd_rec = config_model['vmd_rec']

        self.msin_h_size =  config_model['msin_h_size']
        self.mel_h_size = config_model['mel_h_size']
        self.msg_embed_size = config_model['mel_h_size']
        self.corpus_embed_size = config_model['mel_h_size']

        self.h_size = config_model['h_size']
        self.z_size = config_model['h_size']
        self.g_size = config_model['g_size']
        self.use_in_bn= config_model['use_in_bn']
        self.use_o_bn = config_model['use_o_bn']
        self.use_g_bn = config_model['use_g_bn']

        self.dropout_train_mel_in = config_model['dropout_mel_in']
        self.dropout_train_mel = config_model['dropout_mel']
        self.dropout_train_ce = config_model['dropout_ce']
        self.dropout_train_vmd_in = config_model['dropout_vmd_in']
        self.dropout_train_vmd = config_model['dropout_vmd']

        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        # model name
        name_pattern_max_n = 'days-{0}.msgs-{1}-words-{2}'
        name_max_n = name_pattern_max_n.format(self.max_n_days, self.max_n_msgs, self.max_n_words)

        name_pattern_input_type = 'word_embed-{0}.vmd_in-{1}'
        name_input_type = name_pattern_input_type.format(self.word_embed_type, self.variant_type)

        name_pattern_key = 'alpha-{0}.anneal-{1}.rec-{2}'
        name_key = name_pattern_key.format(self.alpha, self.kl_lambda_anneal_rate, self.vmd_rec)

        name_pattern_train = 'batch-{0}.opt-{1}.lr-{2}-drop-{3}-cell-{4}-tmp'
        name_train = name_pattern_train.format(self.batch_size_for_name, self.opt, self.lr, self.dropout_train_mel_in, self.mel_cell_type)

        name_tuple = (self.mode, name_max_n, name_input_type, name_key, name_train)
        self.model_name = '_'.join(name_tuple)

        # paths
        self.tf_graph_path = os.path.join(path_parser.graphs, self.model_name)  # summary
        self.tf_checkpoints_path = os.path.join(path_parser.checkpoints, self.model_name)  # checkpoints
        self.tf_checkpoint_file_path = os.path.join(self.tf_checkpoints_path, 'checkpoint')  # for restore
        self.tf_saver_path = os.path.join(self.tf_checkpoints_path, 'sess')  # for save

        # verification
        assert self.opt in ('sgd', 'adam')
        assert self.mel_cell_type in ('ln-lstm', 'gru', 'basic')
        assert self.vmd_cell_type in ('ln-lstm', 'gru')
        assert self.variant_type in ('hedge', 'fund', 'tech', 'discriminative')
        assert self.vmd_rec in ('zh', 'h')
        assert self.weight_init in ('xavier-uniform', 'xavier-normal')

    def _build_placeholders(self):
        with tf.name_scope('placeholder'):
            self.is_training_phase = tf.placeholder(dtype=tf.bool, shape=())
            self.batch_size = tf.placeholder(dtype=tf.int32, shape=())

            # init
            self.word_table_init = tf.placeholder(dtype=tf.float32, shape=[vocab_size, self.word_embed_size])

            # model
            self.stock_ph = tf.placeholder(dtype=tf.int32, shape=[None])
            self.T_ph = tf.placeholder(dtype=tf.int32, shape=[None, ])
            self.n_words_ph = tf.placeholder(dtype=tf.int32, shape=[None, self.max_n_days, self.max_n_msgs])
            self.n_msgs_ph = tf.placeholder(dtype=tf.int32, shape=[None, self.max_n_days])
            self.y_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.max_n_days, self.y_size])  # 2-d vectorised movement
            self.mv_percent_ph = tf.placeholder(dtype=tf.int32, shape=[None, self.max_n_days])  # movement percent
            self.price_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.max_n_days, 3])  # high, low, close
            self.word_ph = tf.placeholder(dtype=tf.int32, shape=[None, self.max_n_days, self.max_n_msgs, self.max_n_words])
            self.ss_index_ph = tf.placeholder(dtype=tf.int32, shape=[None, self.max_n_days, self.max_n_msgs])

            # dropout
            self.dropout_mel_in = tf.placeholder_with_default(self.dropout_train_mel_in, shape=())
            self.dropout_mel = tf.placeholder_with_default(self.dropout_train_mel, shape=())
            self.dropout_ce = tf.placeholder_with_default(self.dropout_train_ce, shape=())
            self.dropout_vmd_in = tf.placeholder_with_default(self.dropout_train_vmd_in, shape=())
            self.dropout_vmd = tf.placeholder_with_default(self.dropout_train_vmd, shape=())

    def _build_embeds(self):
        with tf.name_scope('embeds'):
            with tf.variable_scope('embeds'):
                word_table = tf.get_variable('word_table', initializer=self.word_table_init, trainable=False)
                self.word_embed = tf.nn.embedding_lookup(word_table, self.word_ph, name='word_embed')

    def _create_msg_embed_layer_in(self):
        """
            acquire the inputs for MEL.

            Input:
                word_embed: batch_size * max_n_days * max_n_msgs * max_n_words * word_embed_size

            Output:
                mel_in: same as word_embed
        """
        with tf.name_scope('mel_in'):
            with tf.variable_scope('mel_in'):
                mel_in = self.word_embed
                if self.use_in_bn:
                    mel_in = neural.bn(mel_in, self.is_training_phase, bn_scope='bn-mel_inputs')
                self.mel_in = tf.nn.dropout(mel_in, keep_prob=1-self.dropout_mel_in)

    def _create_msg_embed_layer(self):
        """
            Input:
                mel_in: same as word_embed

            Output:
                msg_embed: batch_size * max_n_days * max_n_msgs * msg_embed_size
        """

        def _for_one_trading_day(daily_in, daily_ss_index_vec, daily_mask):
            """
                daily_in: max_n_msgs * max_n_words * word_embed_size
            """
            current_batch_size = tf.shape(daily_in)[0] # This is max_n_msgs for the current day

            # Create initial states for this specific call, matching the "batch size" of messages for the day
            mel_init_f_daily = mel_cell_f.zero_state(current_batch_size, dtype=tf.float32)
            mel_init_b_daily = mel_cell_b.zero_state(current_batch_size, dtype=tf.float32)

            # The daily_mask (n_words_ph for the day) serves as the sequence_length for the RNN
            _, (final_state_fw, final_state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=mel_cell_f,
                cell_bw=mel_cell_b,
                inputs=daily_in,
                sequence_length=daily_mask, # n_words for each message
                initial_state_fw=mel_init_f_daily,
                initial_state_bw=mel_init_b_daily,
                dtype=tf.float32
            )

            if self.mel_cell_type in ('ln-lstm', 'basic'): # LSTM cells return LSTMStateTuple
                msg_embed_fw = final_state_fw.h
                msg_embed_bw = final_state_bw.h
            else: # GRU cells return the state directly
                msg_embed_fw = final_state_fw
                msg_embed_bw = final_state_bw
            
            msg_embed = (msg_embed_fw + msg_embed_bw) / 2.0
            return msg_embed

        def _for_one_sample(sample, sample_ss_index, sample_mask):
            # sample_mask here is n_words_ph for that sample: [max_n_days, max_n_msgs]
            # sample_ss_index here is ss_index_ph for that sample: [max_n_days, max_n_msgs]
            # We need to pass the correct mask to _for_one_trading_day, which is n_msgs for each day
            # The current sample_mask for neural.iter is n_words_ph, which is [max_n_days, max_n_msgs]
            # _for_one_trading_day expects daily_mask to be the n_words for each of the max_n_msgs messages.
            return neural.iter(size=self.max_n_days, func=_for_one_trading_day,
                               iter_arg=sample,             # daily_in: [max_n_msgs, max_n_words, word_embed_size]
                               iter_arg2=sample_ss_index,   # daily_ss_index_vec (unused)
                               iter_arg3=sample_mask)      # daily_mask: [max_n_msgs] (n_words for each msg)

        def _for_one_batch():
            # self.n_words_ph has shape [batch_size, max_n_days, max_n_msgs]
            return neural.iter(size=self.batch_size, func=_for_one_sample,
                               iter_arg=self.mel_in,       # word_embed for the batch
                               iter_arg2=self.ss_index_ph, # ss_index_ph for the batch
                               iter_arg3=self.n_words_ph)  # n_words_ph for the batch

        with tf.name_scope('mel'):
            with tf.variable_scope('mel_iter', reuse=tf.AUTO_REUSE): # Removed reuse=True
                if self.mel_cell_type == 'ln-lstm':
                    mel_cell_f = tf.contrib.rnn.LayerNormBasicLSTMCell(self.mel_h_size, layer_norm=True,
                                                                       dropout_keep_prob=1-self.dropout_mel) # Removed reuse=True
                    mel_cell_b = tf.contrib.rnn.LayerNormBasicLSTMCell(self.mel_h_size, layer_norm=True,
                                                                       dropout_keep_prob=1-self.dropout_mel) # Removed reuse=True
                elif self.mel_cell_type == 'gru':
                    mel_cell_f = tf.contrib.rnn.GRUCell(self.mel_h_size, kernel_initializer=self.initializer,
                                                        bias_initializer=self.bias_initializer) # Removed reuse=True
                    mel_cell_b = tf.contrib.rnn.GRUCell(self.mel_h_size, kernel_initializer=self.initializer,
                                                        bias_initializer=self.bias_initializer) # Removed reuse=True
                else:  # basic lstm
                    mel_cell_f = tf.contrib.rnn.BasicLSTMCell(self.mel_h_size) # Removed reuse=True
                    mel_cell_b = tf.contrib.rnn.BasicLSTMCell(self.mel_h_size) # Removed reuse=True

                # These initial states are for the outer loops in neural.iter, not directly for the RNN.
                # The RNN initial states are created inside _for_one_trading_day.
                # However, the original code had these. Let's keep them for now, but they seem unused by the new _for_one_trading_day logic.
                # mel_init_f = mel_cell_f.zero_state(self.batch_size * self.max_n_days, dtype=tf.float32)
                # mel_init_b = mel_cell_b.zero_state(self.batch_size * self.max_n_days, dtype=tf.float32)

                msg_embed_shape = (self.batch_size, self.max_n_days, self.max_n_msgs, self.msg_embed_size)
                msg_embed_tensor = _for_one_batch()
                # Ensure the output of _for_one_batch is correctly shaped.
                # The output of nested neural.iter should match the structure based on iteration counts.
                # If neural.iter correctly stacks results, msg_embed_tensor should already be
                # [batch_size, max_n_days, max_n_msgs, msg_embed_size]
                self.msg_embed = tf.reshape(msg_embed_tensor, shape=msg_embed_shape, name='msg_embed_reshaped')
                self.msg_embed = tf.nn.dropout(self.msg_embed, keep_prob=1-self.dropout_mel, name='msg_embed')

    def _create_corpus_embed(self):
        """
            msg_embed: batch_size * max_n_days * max_n_msgs * msg_embed_size
            This is the TSU (Text Selection Unit)
            => corpus_embed: batch_size * max_n_days * corpus_embed_size
            Update: This is now Ca-TSU (Causal-Aware Text Selection Unit)
        """
        
        
        with tf.name_scope('corpus_embed'): # This scope acts as Ca-TSU
            # The Ca-TSU now operates on msg_embed which should already be filtered by DataPipe
            with tf.variable_scope('u_t'): # Original TSU weights
                proj_u = self._linear(self.msg_embed, self.msg_embed_size, 'tanh', use_bias=False)
                w_u = tf.get_variable('w_u_original', shape=(self.msg_embed_size, 1), initializer=self.initializer)
            u_scores = tf.reduce_mean(tf.tensordot(proj_u, w_u, axes=1), axis=-1)  # scores: batch_size * max_n_days * max_n_msgs

            # Mask for padding based on actual number of messages (n_msgs_ph now reflects relevant messages)
            mask_msgs_padding = tf.sequence_mask(self.n_msgs_ph, maxlen=self.max_n_msgs, dtype=tf.bool, name='mask_msgs_padding')
            
            ninf = tf.fill(tf.shape(u_scores), -np.inf) # Use -np.inf for softmax
            # Apply the padding mask: if it's a padding message, set score to -infinity
            masked_u_scores = tf.where(mask_msgs_padding, u_scores, ninf)
            
            # Calculate original attention weights omega_i (alpha_t in some papers)
            # self.omega_i = tf.nn.softmax(masked_u_scores, axis=-1) # Softmax over messages for each day (TF1.4 compatible)
            exp_masked_u_scores = tf.exp(masked_u_scores - tf.reduce_max(masked_u_scores, axis=[-1], keep_dims=True))
            self.omega_i = exp_masked_u_scores / tf.reduce_sum(exp_masked_u_scores, axis=[-1], keep_dims=True)
            # Replace NaN with 0.0 (e.g., if all messages on a day are masked out by padding)
            self.omega_i = tf.where(tf.is_nan(self.omega_i), tf.zeros_like(self.omega_i), self.omega_i)

            # Ca-TSU specific calculations for alpha_i
            with tf.variable_scope('ca_tsu'):
                # ei is self.msg_embed: batch_size * max_n_days * max_n_msgs * msg_embed_size
                # W_Q and W_K should project ei to a new dimension, or the same dimension if desired.
                # For simplicity, let's project to the same dimension: self.msg_embed_size
                W_Q = tf.get_variable("W_Q", shape=(self.msg_embed_size, self.msg_embed_size), initializer=self.initializer)
                W_K = tf.get_variable("W_K", shape=(self.msg_embed_size, self.msg_embed_size), initializer=self.initializer)
                b_causal = tf.get_variable("b_causal", shape=(self.msg_embed_size), initializer=self.bias_initializer)

                # q = WQ * ei (element-wise or tensordot? Paper implies WQ is a matrix, ei is a vector)
                # Assuming WQ and WK are dense layers applied to each e_i
                # msg_embed shape: (batch, days, msgs, embed_size)
                q_causal = tf.tensordot(self.msg_embed, W_Q, axes=[[3], [0]]) # (batch, days, msgs, embed_size)
                k_causal = tf.tensordot(self.msg_embed, W_K, axes=[[3], [0]]) # (batch, days, msgs, embed_size)

                # alpha_i = softmax(q^T tanh(k_i + b))
                # q^T * tanh(k_i + b) needs to result in a score per message
                # q_causal is (batch, days, msgs, embed_size), k_causal is (batch, days, msgs, embed_size)
                # tanh(k_i + b) : (batch, days, msgs, embed_size)
                tanh_k_b = tf.nn.tanh(k_causal + b_causal) # b_causal broadcasts

                # Element-wise product of q_causal and tanh_k_b, then sum over the last dimension to get scores
                # This is equivalent to a dot product for each message's q and transformed k
                alpha_scores_unnormalized = tf.reduce_sum(q_causal * tanh_k_b, axis=-1) # (batch, days, msgs)

                masked_alpha_scores = tf.where(mask_msgs_padding, alpha_scores_unnormalized, ninf)
                # self.alpha_i = tf.nn.softmax(masked_alpha_scores, axis=-1) # (batch, days, msgs) (TF1.4 compatible)
                exp_masked_alpha_scores = tf.exp(masked_alpha_scores - tf.reduce_max(masked_alpha_scores, axis=[-1], keep_dims=True))
                self.alpha_i = exp_masked_alpha_scores / tf.reduce_sum(exp_masked_alpha_scores, axis=[-1], keep_dims=True)
                self.alpha_i = tf.where(tf.is_nan(self.alpha_i), tf.zeros_like(self.alpha_i), self.alpha_i)

            # Calculate omega_i_prime = omega_i * alpha_i
            omega_i_prime = self.omega_i * self.alpha_i
            # Normalize omega_i_prime so that it sums to 1 across messages for each day
            # This is important if omega_i_prime is to be used as attention weights
            sum_omega_i_prime = tf.reduce_sum(omega_i_prime, axis=[-1], keep_dims=True) # (batch, days, 1)
            # Add a small epsilon to prevent division by zero if all omega_i_prime are zero (e.g., all messages masked)
            u_weights = omega_i_prime / (sum_omega_i_prime + 1e-8) # (batch, days, msgs)
            # Replace NaN with 0.0 (e.g., if sum_omega_i_prime was 0)
            u_weights = tf.where(tf.is_nan(u_weights), tf.zeros_like(u_weights), u_weights)
            self.ca_tsu_weights = u_weights # Store for potential analysis

            u_weights_expanded = tf.expand_dims(u_weights, axis=-2)  # batch_size * max_n_days * 1 * max_n_msgs
            # Weighted sum of message embeddings to get daily corpus embedding
            corpus_embed_G = tf.matmul(u_weights_expanded, self.msg_embed)  # batch_size * max_n_days * 1 * msg_embed_size
            self.daily_text_embed_ca_tsu = tf.squeeze(corpus_embed_G, axis=-2) # Squeeze to [batch_size, max_n_days, corpus_embed_size]
            
            # Optional: re-add dropout if it was originally here and is still desired
            self.daily_text_embed_ca_tsu = tf.nn.dropout(self.daily_text_embed_ca_tsu, keep_prob=1-self.dropout_ce, name='daily_text_embed_ca_tsu_dropout')

    def _build_mie(self):
        """
            Message Impact Extractor (MIE)
            Represent all messages within a trading day by a single vector

            Input:
                msg_embed: batch_size * max_n_days * max_n_msgs * msg_embed_size (from _create_msg_embed_layer)
                n_msgs_ph: batch_size * max_n_days (actual number of messages for each day)
                price_ph: batch_size * max_n_days * 3 (market data: high, low, close)
            Output:
                msg_embeds: batch_size * max_n_days * msg_embed_size (aggregated message embedding per day)
                P: batch_size * max_n_days * max_n_msgs (attention weights over messages within each day) - these are the cross_weights
        """
        with tf.name_scope('mie'):
            with tf.variable_scope('mie'):
                msin_cell = MSINCell(input_size=3,  # Based on price_ph: high, low, close
                                     num_units=self.msg_embed_size,  # h_size for MSINCell
                                     v_size=self.msg_embed_size,    # v_size for MSINCell (memory for text)
                                     max_n_msgs=self.max_n_msgs)    # To get P in correct shape

                msin = MSIN()
                
                # Assuming self.msg_embed is [batch, max_n_days, max_n_msgs, embed_size]
                # Assuming self.price_ph is [batch, max_n_days, 3]
                # Assuming self.T_ph is [batch] (number of actual days)
                
                # The dynamic_msin in MSINModule_MHA is designed to iterate over the time dimension (days here)
                # and call the cell for each day.
                # inputs to dynamic_msin:
                #   cell: the MSINCell instance
                #   inputs (market data): self.price_ph [batch, max_n_days, 3]
                #   s_inputs (text data): self.msg_embed [batch, max_n_days, max_n_msgs, embed_size]
                #   sequence_length (number of days): self.T_ph [batch]
                
                msin_batch_size = tf.shape(self.msg_embed)[0]
                initial_state = msin_cell.zero_state(msin_batch_size, tf.float32)

                msg_embeds_, P_, _ = msin.dynamic_msin(
                    cell=msin_cell,
                    inputs=self.price_ph, 
                    s_inputs=self.msg_embed,
                    sequence_length=self.T_ph, 
                    initial_state=initial_state,
                    dtype=tf.float32
                )
                # Expected shapes after dynamic_msin based on its internal loop:
                # msg_embeds_ (stacked H from cell): [batch_size, max_n_days_padded, self.msg_embed_size]
                # P_ (stacked cross_weights from cell): [batch_size, max_n_days_padded, self.max_n_msgs]

                self.msg_embeds = msg_embeds_
                self.P = P_
                self.msin_cross_attention_weights = P_ # Assign P_ (cross_weights from MSIN) here

        with tf.name_scope('corpus_embed_layer'): 
            with tf.variable_scope('corpus_embed_layer'):
                # average msg_embeds over trading days to get corpus_embed
                mask_processed_days = tf.sequence_mask(self.T_ph, maxlen=tf.shape(self.msg_embeds)[1], dtype=tf.float32)
                # Using tf.shape(self.msg_embeds)[1] for maxlen to handle potentially padded day dimension from dynamic_msin
                mask_processed_days = tf.expand_dims(mask_processed_days, axis=-1)  # batch_size * max_n_days * 1
                masked_msg_embeds = tf.multiply(self.msg_embeds, mask_processed_days)  # batch_size * max_n_days * msg_embed_size
                sum_msg_embeds = tf.reduce_sum(masked_msg_embeds, axis=1)  # batch_size * msg_embed_size
                self.corpus_embed = tf.divide(sum_msg_embeds, tf.expand_dims(tf.cast(self.T_ph, tf.float32), axis=1) + 1e-8) # batch_size * msg_embed_size, add epsilon for stability

                if self.use_o_bn:
                    self.corpus_embed = neural.bn(self.corpus_embed, self.is_training_phase, bn_scope='bn-corpus_embed')
                self.corpus_embed = tf.nn.dropout(self.corpus_embed, keep_prob=1-self.dropout_ce, name='corpus_embed')

    def _create_vmd_with_h_rec(self):
        with tf.name_scope('vmd'):
            with tf.variable_scope('vmd_h_rec'):
                x = tf.nn.dropout(self.x, keep_prob=1-self.dropout_vmd_in)
                x = tf.transpose(x, [1, 0, 2])  # max_n_days * batch_size * x_size
                y_ = tf.transpose(self.y_ph, [1, 0, 2])  # max_n_days * batch_size * y_size

                self.mask_aux_trading_days = tf.sequence_mask(self.T_ph - 1, self.max_n_days, dtype=tf.bool,
                                                              name='mask_aux_trading_days')

                def _loop_body(t, ta_h_s, ta_z_prior, ta_z_post, ta_kl):

                    with tf.variable_scope('iter_body', reuse=tf.AUTO_REUSE):

                        def _init():
                            h_s_init = tf.nn.tanh(tf.random_normal(shape=[self.batch_size, self.h_size]))
                            h_z_init = tf.nn.tanh(tf.random_normal(shape=[self.batch_size, self.z_size]))

                            z_init, _ = self._z(arg=h_z_init, is_prior=False)

                            return h_s_init, z_init

                        def _subsequent():
                            h_s_t_1 = tf.reshape(ta_h_s.read(t-1), [self.batch_size, self.h_size])
                            z_t_1 = tf.reshape(ta_z_post.read(t-1), [self.batch_size, self.z_size])

                            return h_s_t_1, z_t_1

                        h_s_t_1, z_t_1 = tf.cond(t >= 1, _subsequent, _init)

                        gate_args = [x[t], h_s_t_1, z_t_1]

                        with tf.variable_scope('gru_r'):
                            r = self._linear(gate_args, self.h_size, 'sigmoid')
                        with tf.variable_scope('gru_u'):
                            u = self._linear(gate_args, self.h_size, 'sigmoid')

                        h_args = [x[t], tf.multiply(r, h_s_t_1), z_t_1]

                        with tf.variable_scope('gru_h'):
                            h_tilde = self._linear(h_args, self.h_size, 'tanh')

                        h_s_t = tf.multiply(1 - u, h_s_t_1) + tf.multiply(u, h_tilde)

                        with tf.variable_scope('h_z_prior'):
                            h_z_prior_t = self._linear([x[t], h_s_t], self.z_size, 'tanh')
                        with tf.variable_scope('z_prior'):
                            z_prior_t, z_prior_t_pdf = self._z(h_z_prior_t, is_prior=True)

                        with tf.variable_scope('h_z_post'):
                            h_z_post_t = self._linear([x[t], h_s_t, y_[t]], self.z_size, 'tanh')
                        with tf.variable_scope('z_post'):
                            z_post_t, z_post_t_pdf = self._z(h_z_post_t, is_prior=False)

                    kl_t = ds.kl_divergence(z_post_t_pdf, z_prior_t_pdf)

                    # write
                    ta_h_s = ta_h_s.write(t, h_s_t)
                    ta_z_prior = ta_z_prior.write(t, z_prior_t)  # write: batch_size * z_size
                    ta_z_post = ta_z_post.write(t, z_post_t)  # write: batch_size * z_size
                    ta_kl = ta_kl.write(t, kl_t)  # write: batch_size * 1

                    return t + 1, ta_h_s, ta_z_prior, ta_z_post, ta_kl

                ta_h_s_init = tf.TensorArray(tf.float32, size=self.max_n_days, clear_after_read=False)
                ta_z_prior_init = tf.TensorArray(tf.float32, size=self.max_n_days)
                ta_z_post_init = tf.TensorArray(tf.float32, size=self.max_n_days, clear_after_read=False)
                ta_kl_init = tf.TensorArray(tf.float32, size=self.max_n_days)

                loop_init = (0, ta_h_s_init, ta_z_prior_init, ta_z_post_init, ta_kl_init)
                loop_cond = lambda t, *args: t < self.max_n_days
                _, ta_h_s, ta_z_prior, ta_z_post, ta_kl = tf.while_loop(loop_cond, _loop_body, loop_init)

                h_s = tf.reshape(ta_h_s.stack(), shape=(self.max_n_days, self.batch_size, self.h_size))
                z_shape = (self.max_n_days, self.batch_size, self.z_size)
                z_prior = tf.reshape(ta_z_prior.stack(), shape=z_shape)
                z_post = tf.reshape(ta_z_post.stack(), shape=z_shape)
                kl = tf.reshape(ta_kl.stack(), shape=z_shape)

                x = tf.transpose(x, [1, 0, 2])  # batch_size * max_n_days * x_size
                h_s = tf.transpose(h_s, [1, 0, 2])  # batch_size * max_n_days * vmd_h_size
                z_prior = tf.transpose(z_prior, [1, 0, 2])  # batch_size * max_n_days * z_size
                z_post = tf.transpose(z_post, [1, 0, 2])  # batch_size * max_n_days * z_size
                self.kl = tf.reduce_sum(tf.transpose(kl, [1, 0, 2]), axis=2)  # batch_size * max_n_days

                with tf.variable_scope('g'):
                    self.g = self._linear([x, h_s, z_post], self.g_size, 'tanh', use_bn=False)

                with tf.variable_scope('y'):
                    self.y = self._linear(self.g, self.y_size, 'softmax')

                sample_index = tf.reshape(tf.range(self.batch_size), (self.batch_size, 1), name='sample_index')
                self.indexed_T = tf.concat([sample_index, tf.reshape(self.T_ph-1, (self.batch_size, 1))], axis=1)

                def _infer_func():
                    g_T = tf.gather_nd(params=self.g, indices=self.indexed_T)  # batch_size * g_size

                    if not self.daily_att:
                        y_T = tf.gather_nd(params=self.y, indices=self.indexed_T)  # batch_size * y_size
                        return g_T, y_T

                    return g_T

                def _gen_func():
                    # use prior for g
                    z_prior_T = tf.gather_nd(params=z_prior, indices=self.indexed_T)  # batch_size * z_size
                    h_s_T = tf.gather_nd(params=h_s, indices=self.indexed_T)
                    x_T = tf.gather_nd(params=x, indices=self.indexed_T)

                    with tf.variable_scope('g', reuse=tf.AUTO_REUSE):
                        g_T = self._linear([x_T, h_s_T, z_prior_T], self.g_size, 'tanh', use_bn=False)

                    if not self.daily_att:
                        with tf.variable_scope('y', reuse=tf.AUTO_REUSE):
                            y_T = self._linear(g_T, self.y_size, 'softmax')
                        return g_T, y_T

                    return g_T

                if not self.daily_att:
                    self.g_T, self.y_T = tf.cond(tf.equal(self.is_training_phase, True), _infer_func, _gen_func)
                else:
                    self.g_T = tf.cond(tf.equal(self.is_training_phase, True), _infer_func, _gen_func)

    def _create_vmd_with_zh_rec(self):
        """
            Create a variational movement decoder.

            x: batch_size * max_n_days * vmd_in_size
            => vmd_h: batch_size * max_n_days * vmd_h_size
            => z: batch_size * max_n_days * vmd_z_size
            => y: batch_size * max_n_days * 2
        """
        with tf.name_scope('vmd'):
            with tf.variable_scope('vmd_zh_rec', reuse=tf.AUTO_REUSE):
                x = tf.nn.dropout(self.x, keep_prob=1-self.dropout_vmd_in)

                self.mask_aux_trading_days = tf.sequence_mask(self.T_ph - 1, self.max_n_days, dtype=tf.bool,
                                                              name='mask_aux_trading_days')

                if self.vmd_cell_type == 'ln-lstm':
                    cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.h_size)
                else:
                    cell = tf.contrib.rnn.GRUCell(self.h_size)
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0-self.dropout_vmd)

                init_state = None
                # calculate vmd_h, batch_size * max_n_days * vmd_h_size
                h_s, _ = tf.nn.dynamic_rnn(cell, x, sequence_length=self.T_ph, initial_state=init_state, dtype=tf.float32)

                # forward max_n_days
                x = tf.transpose(x, [1, 0, 2])  # max_n_days * batch_size * x_size
                h_s = tf.transpose(h_s, [1, 0, 2])  # max_n_days * batch_size * vmd_h_size
                y_ = tf.transpose(self.y_ph, [1, 0, 2])  # max_n_days * batch_size * y_size

                def _loop_body(t, ta_z_prior, ta_z_post, ta_kl):
                    """
                        iter body. iter over trading days.
                    """
                    with tf.variable_scope('iter_body', reuse=tf.AUTO_REUSE):

                        init = lambda: tf.random_normal(shape=[self.batch_size, self.z_size], name='z_post_t_1')
                        subsequent = lambda: tf.reshape(ta_z_post.read(t-1), [self.batch_size, self.z_size])

                        z_post_t_1 = tf.cond(t >= 1, subsequent, init)

                        with tf.variable_scope('h_z_prior'):
                            h_z_prior_t = self._linear([x[t], h_s[t], z_post_t_1], self.z_size, 'tanh')
                        with tf.variable_scope('z_prior'):
                            z_prior_t, z_prior_t_pdf = self._z(h_z_prior_t, is_prior=True)

                        with tf.variable_scope('h_z_post'):
                            h_z_post_t = self._linear([x[t], h_s[t], y_[t], z_post_t_1], self.z_size, 'tanh')
                        with tf.variable_scope('z_post'):
                            z_post_t, z_post_t_pdf = self._z(h_z_post_t, is_prior=False)

                    kl_t = ds.kl_divergence(z_post_t_pdf, z_prior_t_pdf)  # batch_size * z_size

                    ta_z_prior = ta_z_prior.write(t, z_prior_t)  # write: batch_size * z_size
                    ta_z_post = ta_z_post.write(t, z_post_t)  # write: batch_size * z_size
                    ta_kl = ta_kl.write(t, kl_t)  # write: batch_size * 1

                    return t + 1, ta_z_prior, ta_z_post, ta_kl

                # loop_init
                ta_z_prior_init = tf.TensorArray(tf.float32, size=self.max_n_days)
                ta_z_post_init = tf.TensorArray(tf.float32, size=self.max_n_days, clear_after_read=False)
                ta_kl_init = tf.TensorArray(tf.float32, size=self.max_n_days)

                loop_init = (0, ta_z_prior_init, ta_z_post_init, ta_kl_init)
                cond = lambda t, *args: t < self.max_n_days

                _, ta_z_prior, ta_z_post, ta_kl = tf.while_loop(cond, _loop_body, loop_init)

                z_shape = (self.max_n_days, self.batch_size, self.z_size)
                z_prior = tf.reshape(ta_z_prior.stack(), shape=z_shape)
                z_post = tf.reshape(ta_z_post.stack(), shape=z_shape)
                kl = tf.reshape(ta_kl.stack(), shape=z_shape)

                h_s = tf.transpose(h_s, [1, 0, 2])  # batch_size * max_n_days * vmd_h_size
                z_prior = tf.transpose(z_prior, [1, 0, 2])  # batch_size * max_n_days * z_size
                z_post = tf.transpose(z_post, [1, 0, 2])  # batch_size * max_n_days * z_size
                self.kl = tf.reduce_sum(tf.transpose(kl, [1, 0, 2]), axis=2)  # batch_size * max_n_days

                with tf.variable_scope('g'):
                    self.g = self._linear([h_s, z_post], self.g_size, 'tanh')  # batch_size * max_n_days * g_size

                with tf.variable_scope('y'):
                    self.y = self._linear(self.g, self.y_size, 'softmax')  # batch_size * max_n_days * y_size

                sample_index = tf.reshape(tf.range(self.batch_size), (self.batch_size, 1), name='sample_index')

                self.indexed_T = tf.concat([sample_index, tf.reshape(self.T_ph-1, (self.batch_size, 1))], axis=1)

                def _infer_func():
                    g_T = tf.gather_nd(params=self.g, indices=self.indexed_T)  # batch_size * g_size

                    if not self.daily_att:
                        y_T = tf.gather_nd(params=self.y, indices=self.indexed_T)  # batch_size * y_size
                        return g_T, y_T

                    return g_T

                def _gen_func():
                    # use prior for g & y
                    z_prior_T = tf.gather_nd(params=z_prior, indices=self.indexed_T)  # batch_size * z_size
                    h_s_T = tf.gather_nd(params=h_s, indices=self.indexed_T)

                    with tf.variable_scope('g', reuse=tf.AUTO_REUSE):
                        g_T = self._linear([h_s_T, z_prior_T], self.g_size, 'tanh', use_bn=False)

                    if not self.daily_att:
                        with tf.variable_scope('y', reuse=tf.AUTO_REUSE):
                            y_T = self._linear(g_T, self.y_size, 'softmax')
                        return g_T, y_T

                    return g_T

                if not self.daily_att:
                    self.g_T, self.y_T = tf.cond(tf.equal(self.is_training_phase, True), _infer_func, _gen_func)
                else:
                    self.g_T = tf.cond(tf.equal(self.is_training_phase, True), _infer_func, _gen_func)

    def _create_discriminative_vmd(self):
        """
            Create a discriminative movement decoder.

            x: batch_size * max_n_days * vmd_in_size
            => vmd_h: batch_size * max_n_days * vmd_h_size
            => z: batch_size * max_n_days * vmd_z_size
            => y: batch_size * max_n_days * 2
        """
        with tf.name_scope('vmd'):
            with tf.variable_scope('vmd_zh_rec', reuse=tf.AUTO_REUSE):
                x = tf.nn.dropout(self.x, keep_prob=1-self.dropout_vmd_in)

                self.mask_aux_trading_days = tf.sequence_mask(self.T_ph - 1, self.max_n_days, dtype=tf.bool,
                                                              name='mask_aux_trading_days')

                if self.vmd_cell_type == 'ln-lstm':
                    cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.h_size)
                else:
                    cell = tf.contrib.rnn.GRUCell(self.h_size)
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0-self.dropout_vmd)

                init_state = None
                h_s, _ = tf.nn.dynamic_rnn(cell, x, sequence_length=self.T_ph, initial_state=init_state, dtype=tf.float32)

                # forward max_n_days
                x = tf.transpose(x, [1, 0, 2])  # max_n_days * batch_size * x_size
                h_s = tf.transpose(h_s, [1, 0, 2])  # max_n_days * batch_size * vmd_h_size

                def _loop_body(t, ta_z):
                    """
                        iter body. iter over trading days.
                    """
                    with tf.variable_scope('iter_body', reuse=tf.AUTO_REUSE):

                        init = lambda: tf.random_normal(shape=[self.batch_size, self.z_size], name='z_post_t_1')
                        subsequent = lambda: tf.reshape(ta_z.read(t-1), [self.batch_size, self.z_size])

                        z_t_1 = tf.cond(t >= 1, subsequent, init)

                        with tf.variable_scope('h_z'):
                            h_z_t = self._linear([x[t], h_s[t], z_t_1], self.z_size, 'tanh')
                        with tf.variable_scope('z'):
                            z_t = self._linear(h_z_t, self.z_size, 'tanh')

                    ta_z = ta_z.write(t, z_t)  # write: batch_size * z_size
                    return t + 1, ta_z

                # loop_init
                ta_z_init = tf.TensorArray(tf.float32, size=self.max_n_days, clear_after_read=False)

                loop_init = (0, ta_z_init)
                cond = lambda t, *args: t < self.max_n_days

                _, ta_z_init = tf.while_loop(cond, _loop_body, loop_init)

                z_shape = (self.max_n_days, self.batch_size, self.z_size)
                z = tf.reshape(ta_z_init.stack(), shape=z_shape)

                h_s = tf.transpose(h_s, [1, 0, 2])  # batch_size * max_n_days * vmd_h_size
                z = tf.transpose(z, [1, 0, 2])  # batch_size * max_n_days * z_size

                with tf.variable_scope('g'):
                    self.g = self._linear([h_s, z], self.g_size, 'tanh')  # batch_size * max_n_days * g_size

                with tf.variable_scope('y'):
                    self.y = self._linear(self.g, self.y_size, 'softmax')  # batch_size * max_n_days * y_size

                # get g_T
                sample_index = tf.reshape(tf.range(self.batch_size), (self.batch_size, 1), name='sample_index')
                self.indexed_T = tf.concat([sample_index, tf.reshape(self.T_ph-1, (self.batch_size, 1))], axis=1)
                self.g_T = tf.gather_nd(params=self.g, indices=self.indexed_T)

    def _build_vmd(self):
        if self.variant_type == 'discriminative':
            self._create_discriminative_vmd()
        else:
            if self.vmd_rec == 'h':
                self._create_vmd_with_h_rec()
            else:
                self._create_vmd_with_zh_rec()

    def _build_temporal_att(self):
        """
            g: batch_size * max_n_days * g_size
            g_T: batch_size * g_size
        """
        with tf.name_scope('tda'):
            with tf.variable_scope('tda'):
                with tf.variable_scope('v_i'):
                    proj_i = self._linear([self.g], self.g_size, 'tanh', use_bias=False)
                    w_i = tf.get_variable('w_i', shape=(self.g_size, 1), initializer=self.initializer)
                v_i = tf.reduce_sum(tf.tensordot(proj_i, w_i, axes=1), axis=-1)  # batch_size * max_n_days

                with tf.variable_scope('v_d'):
                    proj_d = self._linear([self.g], self.g_size, 'tanh', use_bias=False)
                g_T = tf.expand_dims(self.g_T, axis=-1)  # batch_size * g_size * 1
                v_d = tf.reduce_sum(tf.matmul(proj_d, g_T), axis=-1)  # batch_size * max_n_days

                aux_score = tf.multiply(v_i, v_d, name='v_stared')
                ninf = tf.fill(tf.shape(aux_score), np.NINF)
                masked_aux_score = tf.where(self.mask_aux_trading_days, aux_score, ninf)
                v_stared = tf.nn.softmax(masked_aux_score)

                # v_stared: batch_size * max_n_days
                self.v_stared = tf.where(tf.is_nan(v_stared), tf.zeros_like(v_stared), v_stared)

                if self.daily_att == 'y':
                    context = tf.transpose(self.y, [0, 2, 1])  # batch_size * y_size * max_n_days
                else:
                    context = tf.transpose(self.g, [0, 2, 1])  # batch_size * g_size * max_n_days

                v_stared = tf.expand_dims(self.v_stared, -1)  # batch_size * max_n_days * 1
                att_c = tf.reduce_sum(tf.matmul(context, v_stared), axis=-1)  # batch_size * g_size / y_size
                with tf.variable_scope('y_T'):
                    self.y_T = self._linear([att_c, self.g_T], self.y_size, 'softmax')

    def _create_generative_ata(self):
        """
            calculate loss.

            g: batch_size * max_n_days * g_size
            y: batch_size * max_n_days * y_size
            kl_loss: batch_size * max_n_days
            self.msin_cross_attention_weights: batch_size * max_n_days_padded * max_n_msgs
            => loss: batch_size
        """
        with tf.name_scope('ata'):
            with tf.variable_scope('ata_generative'): # Keep the updated scope name

                v_aux = self.alpha * self.v_stared  # batch_size * max_n_days

                minor = 0.0 # 0.0, 1e-7* # Reverted to original minor for generative
                likelihood_aux = tf.reduce_sum(tf.multiply(self.y_ph, tf.log(self.y + minor)), axis=2)  # batch_size * max_n_days

                kl_lambda = self._kl_lambda() # Reverted: Add back kl_lambda
                obj_aux = likelihood_aux - kl_lambda * self.kl  # batch_size * max_n_days # Reverted

                # deal with T specially, likelihood_T: batch_size, 1
                self.y_T_ = tf.gather_nd(params=self.y_ph, indices=self.indexed_T)  # batch_size * y_size
                # self.y_T = tf.gather_nd(params=self.y, indices=self.indexed_T) # This was from the incorrect edit, self.y_T is defined later by _build_temporal_att
                likelihood_T = tf.reduce_sum(tf.multiply(self.y_T_, tf.log(self.y_T + minor)), axis=1, keep_dims=True) # Reverted: Use self.y_T (defined by _build_temporal_att, not self.y here)

                kl_T = tf.reshape(tf.gather_nd(params=self.kl, indices=self.indexed_T), shape=[self.batch_size, 1]) # Reverted
                obj_T = likelihood_T - kl_lambda * kl_T # Reverted

                obj = obj_T + tf.reduce_sum(tf.multiply(obj_aux, v_aux), axis=1, keep_dims=True)  # batch_size * 1 # Reverted
                self.loss = tf.reduce_mean(-obj, axis=[0, 1]) # Reverted
                # The P_obj term was from the incorrect merge with discriminative, remove for generative original
                # self.loss = self.loss +  tf.reduce_mean(-P_obj) 


                # Causal Consistency Loss DKL(alpha || omega)
                # self.alpha_i and self.omega_i have shape [batch, max_n_days, max_n_msgs]
                epsilon_kl = 1e-8 
                day_mask_for_causal_loss = tf.sequence_mask(self.T_ph, self.max_n_days, dtype=tf.float32) 
                msg_mask_for_causal_loss = tf.sequence_mask(self.n_msgs_ph, self.max_n_msgs, dtype=tf.float32)
                kl_elements = self.alpha_i * tf.log((self.alpha_i + epsilon_kl) / (self.omega_i + epsilon_kl) + epsilon_kl)
                masked_kl_elements = kl_elements * msg_mask_for_causal_loss 
                kl_per_day = tf.reduce_sum(masked_kl_elements, axis=2) 
                masked_kl_per_day = kl_per_day * day_mask_for_causal_loss 
                total_batch_kl_causal = tf.reduce_sum(masked_kl_per_day)
                num_actual_days_in_batch_causal = tf.reduce_sum(day_mask_for_causal_loss) 
                self.causal_consistency_loss = total_batch_kl_causal / (num_actual_days_in_batch_causal + epsilon_kl)
                self.loss += self.causal_loss_lambda * self.causal_consistency_loss 

                # Add entropy regularization for MSIN cross-attention weights
                if self.entropy_regularization_lambda > 0.0:
                    if hasattr(self, 'msin_cross_attention_weights') and self.msin_cross_attention_weights is not None:
                        epsilon = 1e-8
                        # Mask for actual number of days
                        # self.T_ph has shape [batch_size]
                        day_mask = tf.sequence_mask(self.T_ph, self.max_n_days, dtype=tf.float32) # [batch_size, max_n_days]

                        # Mask for actual number of messages per day
                        # self.n_msgs_ph has shape [batch_size, max_n_days]
                        # self.msin_cross_attention_weights has shape [batch_size, max_n_days, max_n_msgs]
                        msg_mask_for_entropy = tf.sequence_mask(self.n_msgs_ph, self.max_n_msgs, dtype=tf.float32) # [batch_size, max_n_days, max_n_msgs]

                        # Apply message mask to attentions and add epsilon for numerical stability
                        attentions_for_entropy = self.msin_cross_attention_weights * msg_mask_for_entropy
                        
                        # Calculate entropy: -sum(p * log(p))
                        # Add epsilon to prevent log(0)
                        entropy_per_msg_element = -attentions_for_entropy * tf.log(attentions_for_entropy + epsilon)
                        
                        # Sum entropy contributions over messages for each day
                        entropy_per_day = tf.reduce_sum(entropy_per_msg_element, axis=2) # [batch_size, max_n_days]
                        
                        # Apply day mask to zero out entropy for padded days
                        masked_entropy_per_day = entropy_per_day * day_mask # [batch_size, max_n_days]
                        
                        # Sum entropy across all active days in the batch
                        total_batch_entropy = tf.reduce_sum(masked_entropy_per_day) 
                        
                        # Count the number of actual days in the batch to normalize
                        num_actual_days_in_batch = tf.reduce_sum(day_mask) # Sum of all 1s in day_mask
                        
                        # Average entropy per active day
                        average_entropy = total_batch_entropy / (num_actual_days_in_batch + epsilon)
                        
                        explainability_loss = self.entropy_regularization_lambda * average_entropy
                        self.loss = self.loss + explainability_loss
                        self.explainability_loss = explainability_loss # Store for potential logging
                    else:
                        logger.warn("entropy_regularization_lambda > 0 but msin_cross_attention_weights not found or None.")

    def _create_discriminative_ata(self):
        """
             calculate discriminative loss.

             g: batch_size * max_n_days * g_size
             y: batch_size * max_n_days * y_size
             self.msin_cross_attention_weights: batch_size * max_n_days_padded * max_n_msgs
             => loss: batch_size
        """
        with tf.name_scope('ata'):
            with tf.variable_scope('ata_discriminative'): # Keep the updated scope name
                v_aux = self.alpha * self.v_stared  # batch_size * max_n_days

                minor = 1e-7  # 0.0, 1e-7* # This is correct for discriminative
                likelihood_aux = tf.reduce_sum(tf.multiply(self.y_ph, tf.log(self.y + minor)), axis=2)  # batch_size * max_n_days

                # deal with T specially, likelihood_T: batch_size, 1
                self.y_T_ = tf.gather_nd(params=self.y_ph, indices=self.indexed_T)  # batch_size * y_size
                # For discriminative, self.y_T (model's prediction for day T) is used, which is correct.
                # The original code used self.y_T which is defined in _build_temporal_att
                likelihood_T = tf.reduce_sum(tf.multiply(self.y_T_, tf.log(self.y_T + minor)), axis=1, keep_dims=True)


                obj = likelihood_T + tf.reduce_sum(tf.multiply(likelihood_aux, v_aux), axis=1, keep_dims=True)  # batch_size * 1
                
                # The P_obj term is part of the original discriminative loss
                new_P = tf.clip_by_value(self.P, 1e-8, 1)
                P_obj = tf.reduce_sum(tf.multiply(self.P, tf.log(new_P)), axis=-1)
                
                self.loss = tf.reduce_mean(-obj, axis=[0, 1])
                self.loss = self.loss +  tf.reduce_mean(-P_obj)

                # Causal Consistency Loss DKL(alpha || omega)
                epsilon_kl = 1e-8 
                day_mask_for_causal_loss = tf.sequence_mask(self.T_ph, self.max_n_days, dtype=tf.float32)
                msg_mask_for_causal_loss = tf.sequence_mask(self.n_msgs_ph, self.max_n_msgs, dtype=tf.float32)
                kl_elements = self.alpha_i * tf.log((self.alpha_i + epsilon_kl) / (self.omega_i + epsilon_kl) + epsilon_kl)
                masked_kl_elements = kl_elements * msg_mask_for_causal_loss
                kl_per_day = tf.reduce_sum(masked_kl_elements, axis=2)
                masked_kl_per_day = kl_per_day * day_mask_for_causal_loss
                total_batch_kl_causal = tf.reduce_sum(masked_kl_per_day)
                num_actual_days_in_batch_causal = tf.reduce_sum(day_mask_for_causal_loss)
                self.causal_consistency_loss = total_batch_kl_causal / (num_actual_days_in_batch_causal + epsilon_kl)
                self.loss += self.causal_loss_lambda * self.causal_consistency_loss

                # Add entropy regularization for MSIN cross-attention weights
                if self.entropy_regularization_lambda > 0.0:
                    if hasattr(self, 'msin_cross_attention_weights') and self.msin_cross_attention_weights is not None:
                        epsilon = 1e-8
                        # Mask for actual number of days
                        # self.T_ph has shape [batch_size]
                        day_mask = tf.sequence_mask(self.T_ph, self.max_n_days, dtype=tf.float32) # [batch_size, max_n_days]

                        # Mask for actual number of messages per day
                        # self.n_msgs_ph has shape [batch_size, max_n_days]
                        # self.msin_cross_attention_weights has shape [batch_size, max_n_days, max_n_msgs]
                        msg_mask_for_entropy = tf.sequence_mask(self.n_msgs_ph, self.max_n_msgs, dtype=tf.float32) # [batch_size, max_n_days, max_n_msgs]

                        # Apply message mask to attentions and add epsilon for numerical stability
                        attentions_for_entropy = self.msin_cross_attention_weights * msg_mask_for_entropy
                        
                        # Calculate entropy: -sum(p * log(p))
                        # Add epsilon to prevent log(0)
                        entropy_per_msg_element = -attentions_for_entropy * tf.log(attentions_for_entropy + epsilon)
                        
                        # Sum entropy contributions over messages for each day
                        entropy_per_day = tf.reduce_sum(entropy_per_msg_element, axis=2) # [batch_size, max_n_days]
                        
                        # Apply day mask to zero out entropy for padded days
                        masked_entropy_per_day = entropy_per_day * day_mask # [batch_size, max_n_days]
                        
                        # Sum entropy across all active days in the batch
                        total_batch_entropy = tf.reduce_sum(masked_entropy_per_day) 
                        
                        # Count the number of actual days in the batch to normalize
                        num_actual_days_in_batch = tf.reduce_sum(day_mask) # Sum of all 1s in day_mask
                        
                        # Average entropy per active day
                        average_entropy = total_batch_entropy / (num_actual_days_in_batch + epsilon)
                        
                        explainability_loss = self.entropy_regularization_lambda * average_entropy
                        self.loss = self.loss + explainability_loss
                        self.explainability_loss = explainability_loss # Store for potential logging
                    else:
                        logger.warn("entropy_regularization_lambda > 0 but msin_cross_attention_weights not found or None.")

    def _build_ata(self):
        if self.variant_type == 'discriminative':
            self._create_discriminative_ata()
        else:
            self._create_generative_ata()

    def _create_optimizer(self):
        with tf.name_scope('optimizer'):
            if self.opt == 'sgd':
                decayed_lr = tf.train.exponential_decay(learning_rate=self.lr, global_step=self.global_step,
                                                        decay_steps=self.decay_step, decay_rate=self.decay_rate)
                optimizer = tf.train.MomentumOptimizer(learning_rate=decayed_lr, momentum=self.momentum)
            else:
                optimizer = tf.train.AdamOptimizer(self.lr)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, self.clip)
            self.optimize = optimizer.apply_gradients(zip(gradients, variables))
            self.global_step = tf.assign_add(self.global_step, 1)

    def assemble_graph(self):
        logger.info('Start graph assembling...')
        with tf.device('/device:GPU:0'):
            self._build_placeholders()
            self._build_embeds()
            self._create_msg_embed_layer_in()
            self._create_msg_embed_layer()      # Defines self.msg_embed
            
            self._create_corpus_embed()         # Ca-TSU: Defines self.daily_text_embed_ca_tsu, self.alpha_i, self.omega_i
            
            self._build_mie()                   # Defines self.msg_embeds (daily from MSIN), self.P, and self.corpus_embed (global average)

            # Define self.x as input to VMD: using text embeddings from Ca-TSU
            self.x = tf.concat([self.daily_text_embed_ca_tsu, self.price_ph], axis=-1, name='vmd_input_x')

            self._build_vmd()
            self._build_temporal_att()
            self._build_ata()
            self._create_optimizer()
            

    def _kl_lambda(self):
        def _nonzero_kl_lambda():
            if self.use_constant_kl_lambda:
                return self.constant_kl_lambda
            else:
                return tf.minimum(self.kl_lambda_anneal_rate * global_step, 1.0)

        global_step = tf.cast(self.global_step, tf.float32)

        return tf.cond(global_step < self.kl_lambda_start_step, lambda: 0.0, _nonzero_kl_lambda)

    def _linear(self, args, output_size, activation=None, use_bias=True, use_bn=False):
        if type(args) not in (list, tuple):
            args = [args]

        shape = [a if a else -1 for a in args[0].get_shape().as_list()[:-1]]
        shape.append(output_size)

        sizes = [a.get_shape()[-1].value for a in args]
        total_arg_size = sum(sizes)
        scope = tf.get_variable_scope()
        x = args[0] if len(args) == 1 else tf.concat(args, -1)

        with tf.variable_scope(scope):
            weight = tf.get_variable('weight', [total_arg_size, output_size], dtype=tf.float32, initializer=self.initializer)
            res = tf.tensordot(x, weight, axes=1)
            if use_bias:
                bias = tf.get_variable('bias', [output_size], dtype=tf.float32, initializer=self.bias_initializer)
                res = tf.nn.bias_add(res, bias)

        res = tf.reshape(res, shape)

        if use_bn:
            res = batch_norm(res, center=True, scale=True, decay=0.99, updates_collections=None,
                             is_training=self.is_training_phase, scope=scope)

        if activation == 'tanh':
            res = tf.nn.tanh(res)
        elif activation == 'sigmoid':
            res = tf.nn.sigmoid(res)
        elif activation == 'relu':
            res = tf.nn.relu(res)
        elif activation == 'softmax':
            res = tf.nn.softmax(res)

        return res

    def _z(self, arg, is_prior):
        mean = self._linear(arg, self.z_size)
        stddev = self._linear(arg, self.z_size)
        stddev = tf.sqrt(tf.exp(stddev))
        
        epsilon = tf.random_normal(shape=[self.batch_size, self.z_size])

        z = mean if is_prior else mean + tf.multiply(stddev, epsilon)
        pdf_z = ds.Normal(loc=mean, scale=stddev)

        return z, pdf_z 