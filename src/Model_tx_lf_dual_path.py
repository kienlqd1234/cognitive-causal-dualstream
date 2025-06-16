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
import logging # Add this import
import hashlib

















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
        self.use_dual_path_srl = config_model.get('use_dual_path_srl', True)  # Store the dual-path SRL configuration
        
        # Ca-TSU and noise-aware loss configuration
        self.use_ca_tsu = config_model.get('use_ca_tsu', False)
        self.ca_tsu_weight = config_model.get('ca_tsu_weight', 0.3)
        self.use_noise_aware_loss = config_model.get('use_noise_aware_loss', False)
        self.noise_aware_weight = config_model.get('noise_aware_weight', 0.5)
        
        # Enhanced dual-path SRL config
        self.use_enhanced_srl = config_model.get('use_enhanced_srl', False)
        self.td_temporal_attention = config_model.get('td_temporal_attention', True)
        self.volatility_adaptation_factor = config_model.get('volatility_adaptation_factor', 0.3)
        self.residual_connection_weight = config_model.get('residual_connection_weight', 0.1)
        self.pathway_competition_type = config_model.get('pathway_competition_type', 'dynamic')
        
        # Enhanced SRL loss weights
        self.prediction_loss_weight = config_model.get('prediction_loss_weight', 0.1)
        self.consistency_loss_weight = config_model.get('consistency_loss_weight', 0.05)
        self.volatility_reg_weight = config_model.get('volatility_reg_weight', 0.05)

        # New margin parameter for the contrastive pathway separation loss
        self.path_separation_margin = config_model.get('path_separation_margin', 1.0)

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

        # model name (shortened to avoid Windows path length issues)
        base_name = f"{self.mode}-{self.variant_type}-{self.mel_cell_type}"
        
        # Create a hash of the full config to ensure uniqueness for different hyperparameters
        config_string = str(sorted(config_model.items())) # Use sorted items for consistency
        config_hash = hashlib.md5(config_string.encode('utf-8')).hexdigest()[:8] # short hash

        self.model_name = f"{base_name}_{config_hash}"


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
            # Add message mask placeholder
            self.msg_mask_ph = tf.placeholder_with_default(
                tf.ones([tf.shape(self.n_msgs_ph)[0], self.max_n_days, self.max_n_msgs], dtype=tf.bool),
                shape=[None, self.max_n_days, self.max_n_msgs],
                name='msg_mask_ph'
            )

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
        '''
            acquire the inputs for MEL.

            Input:
                word_embed: batch_size * max_n_days * max_n_msgs * max_n_words * word_embed_size

            Output:
                mel_in: same as word_embed
        '''
        with tf.name_scope('mel_in'):
            with tf.variable_scope('mel_in'):
                mel_in = self.word_embed
                if self.use_in_bn:
                    mel_in = neural.bn(mel_in, self.is_training_phase, bn_scope='bn-mel_inputs')
                self.mel_in = tf.nn.dropout(mel_in, keep_prob=1-self.dropout_mel_in)

    def _create_msg_embed_layer(self):
        '''
            Input:
                mel_in: same as word_embed

            Output:
                msg_embed: batch_size * max_n_days * max_n_msgs * msg_embed_size
        '''

        def _for_one_trading_day(daily_in, daily_ss_index_vec, daily_mask):
            '''
                daily_in: max_n_msgs * max_n_words * word_embed_size
            '''
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
            with tf.variable_scope('mel_iter', reuse=tf.AUTO_REUSE):
                if self.mel_cell_type == 'ln-lstm':
                    mel_cell_f = tf.contrib.rnn.LayerNormBasicLSTMCell(self.mel_h_size, layer_norm=True,
                                                                       dropout_keep_prob=1-self.dropout_mel)
                    mel_cell_b = tf.contrib.rnn.LayerNormBasicLSTMCell(self.mel_h_size, layer_norm=True,
                                                                       dropout_keep_prob=1-self.dropout_mel)
                elif self.mel_cell_type == 'gru':
                    mel_cell_f = tf.contrib.rnn.GRUCell(self.mel_h_size, kernel_initializer=self.initializer,
                                                        bias_initializer=self.bias_initializer)
                    mel_cell_b = tf.contrib.rnn.GRUCell(self.mel_h_size, kernel_initializer=self.initializer,
                                                        bias_initializer=self.bias_initializer)
                else:  # basic lstm
                    mel_cell_f = tf.contrib.rnn.BasicLSTMCell(self.mel_h_size)
                    mel_cell_b = tf.contrib.rnn.BasicLSTMCell(self.mel_h_size)

                msg_embed_shape = (self.batch_size, self.max_n_days, self.max_n_msgs, self.msg_embed_size)
                msg_embed_tensor = _for_one_batch()
                self.msg_embed = tf.reshape(msg_embed_tensor, shape=msg_embed_shape, name='msg_embed_reshaped')
                self.msg_embed = tf.nn.dropout(self.msg_embed, keep_prob=1-self.dropout_mel, name='msg_embed')

    def _create_corpus_embed(self):
        '''
            msg_embed: batch_size * max_n_days * max_n_msgs * msg_embed_size
            This is the TSU (Text Selection Unit)
            => corpus_embed: batch_size * max_n_days * corpus_embed_size
            Update: This is now Ca-TSU (Causal-Aware Text Selection Unit)
        '''
        
        
        with tf.name_scope('corpus_embed'): # This scope acts as Ca-TSU
            # The Ca-TSU now operates on msg_embed which should already be filtered by DataPipe
            with tf.variable_scope('u_t'): # Original TSU weights
                # Reshape self.msg_embed from [batch_size, max_n_days, max_n_msgs, msg_embed_size]
                # to [batch_size * max_n_days * max_n_msgs, msg_embed_size]
                batch_size = tf.shape(self.msg_embed)[0]
                msg_embed_flat = tf.reshape(self.msg_embed, [-1, self.msg_embed_size])
                
                # Apply linear transformation to the flattened tensor
                proj_u_flat = self._linear(msg_embed_flat, self.msg_embed_size, 'tanh', use_bias=False)
                
                # Reshape back to 4D
                proj_u = tf.reshape(proj_u_flat, [batch_size, self.max_n_days, self.max_n_msgs, self.msg_embed_size])
                
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
                alpha_scores_unnormalized = tf.reduce_sum(q_causal * tanh_k_b, axis=-1)

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
            self.daily_text_embed_ca_tsu = tf.squeeze(corpus_embed_G, axis=-2)
            
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
        # This method uses MSINCell, which is a more complex attention/GRU-like mechanism.
        # The original PEN paper's SRL doesn't seem to map directly to this MSINCell structure.
        # The PEN paper's TSU produces a daily text embedding.
        # The PEN paper's TMU preserves useful historical text (implicitly done by VMD's recurrence).
        # The PEN paper's IFU fuses text and price.
        #
        # Current `_create_corpus_embed` (Ca-TSU) already produces `self.daily_text_embed_ca_tsu`.
        # This `self.daily_text_embed_ca_tsu` is the `i_t` (or a representation derived from it) for our Dual-Path SRL.
        # The `self.price_ph` is our `p_t`.
        #
        # We will replace the simple concatenation of `self.daily_text_embed_ca_tsu` and `self.price_ph`
        # (which currently forms `self.x` for VMD) with the output of the Dual-Path SRL.
        #
        # The MSINCell logic inside `_build_mie` might be an alternative way the original codebase
        # implemented text-price fusion or daily text aggregation.
        # For the Dual-Path SRL, we'll implement the TD and PD pathways and Fusion module
        # directly, likely in a new method, and then use its output to define `self.x`.
        # We might not need to modify `_build_mie` itself if Ca-TSU's output is sufficient.

        with tf.name_scope('mie'):
            self.price = self.price_ph
            self.price_size = 3

            if self.variant_type == 'tech':
                # If only technical data, x is just price. Dual-Path SRL might not be applicable or needs adaptation.
                # For now, we assume 'hedge' or 'fund' where text is used.
                self.x = self.price
                self.x_size = self.price_size
                self.msg_embeds = None # No message embeddings if only tech
                self.P = None # No attention weights if only tech
                self.corpus_embed = None # No corpus embed if only tech
                return # Skip the rest of MIE if only tech data

            # For variants using text ('hedge', 'fund'):
            # The MSINCell is used here. It processes messages for each day.
            # MSINCell processes `s_inputs` (daily message embeddings) and `x_inputs` (daily price data)
            # `s_inputs` should be `self.msg_embed` (batch_size * max_n_days * max_n_msgs * msg_embed_size)
            # `x_inputs` should be `self.price_ph` (batch_size * max_n_days * price_size)

            # Reshape msg_embed and price_ph for MSINCell processing if necessary.
            # MSIN expects inputs per day.
            
            # The original MSINCell processes individual messages with attention,
            # and then incorporates price. This is different from the PEN paper's SRL order.
            # However, `Model_tx_lf.py` already has Ca-TSU in `_create_corpus_embed`
            # which produces `self.daily_text_embed_ca_tsu`. This is a daily summary.

            # Let's assume for now that `_build_mie` with MSIN is the original model's way of getting daily representations.
            # The output `self.msg_embeds` from MSIN is a daily aggregated embedding.
            # It also produces `self.P` (attention weights).

            # If we are to implement the Dual-Path SRL as described,
            # it should operate on the output of Ca-TSU (`self.daily_text_embed_ca_tsu`) and `self.price_ph`.

            # For now, let this function define `self.msg_embeds` and `self.P` as it did.
            # We will later replace how `self.x` (input to VMD) is formed.

            msin_cell = MSINCell(
                input_size=self.price_size, # x_inputs (price)
                num_units=self.msin_h_size, # Hidden size of MSIN
                v_size=self.msg_embed_size, # s_inputs (message embeddings)
                max_n_msgs=self.max_n_msgs
            )

            # MSIN processes day by day. We need to iterate through days.
            # Inputs to MSINCell for each day:
            # X: price data for the day (batch_size * price_size)
            # S: message embeddings for the day (batch_size * max_n_msgs * msg_embed_size)
            # State: previous state (h, v)

            # Prepare inputs for iterating over days
            # price_ph: (batch_size, max_n_days, price_size)
            # msg_embed: (batch_size, max_n_days, max_n_msgs, msg_embed_size)
            
            price_unstacked = tf.unstack(self.price_ph, axis=1) # List of (batch_size, price_size), len=max_n_days
            msg_embed_unstacked = tf.unstack(self.msg_embed, axis=1) # List of (batch_size, max_n_msgs, msg_embed_size), len=max_n_days
            
            outputs_h = []
            outputs_P = [] # Attention weights from MSINCell
            
            # Initial state for MSIN for the whole batch
            # The state needs to be per sample in the batch.
            # MSINCell.zero_state expects batch_size.
            state = msin_cell.zero_state(self.batch_size, dtype=tf.float32)

            with tf.variable_scope("MSIN_RNN", reuse=tf.AUTO_REUSE):
                for day_idx in range(self.max_n_days):
                    daily_price_input = price_unstacked[day_idx]
                    daily_msg_input = msg_embed_unstacked[day_idx]
                    
                    # The MSINCell call might update its own variables, so ensure proper scoping if called in a loop.
                    # The original MSINCell might not be designed for explicit reuse like this in a Python loop.
                    # Typically, tf.nn.dynamic_rnn or similar handles the temporal iteration.
                    # However, the provided MSINCell has a 'call' method.
                    
                    # We need to handle masking for days that have no messages or are padding.
                    # n_msgs_ph: (batch_size, max_n_days) tells actual messages per day.
                    # If n_msgs_ph[batch_idx, day_idx] == 0, then that day's text input is effectively padding.
                    # MSINCell itself uses max_n_msgs, but the attention inside should handle varying numbers of actual messages
                    # if the input S is masked or sequence lengths are provided.
                    # The current MSINCell takes max_n_msgs and expects S to be padded up to that.
                    # Its internal attention P should reflect actual messages if inputs are zero-padded for non-messages.

                    # For now, assume MSINCell handles varying numbers of messages internally via its attention.
                    # Or, the inputs (self.msg_embed) are already masked/zeroed out for padding messages by DataPipe.
                    
                    h_day, P_day, state = msin_cell.call(X=daily_price_input, S=daily_msg_input, state=state)
                    # h_day: (batch_size, msin_h_size)
                    # P_day: (batch_size, max_n_msgs) - these are cross_weights in MSIN MHA
                    outputs_h.append(h_day)
                    outputs_P.append(P_day)

            self.msg_embeds = tf.stack(outputs_h, axis=1) # (batch_size, max_n_days, msin_h_size)
            self.P = tf.stack(outputs_P, axis=1) # (batch_size, max_n_days, max_n_msgs) - these are VoS in the paper.
            self.msin_cross_attention_weights = self.P # For potential inspection

            # The original code also had a global self.corpus_embed, which was an average of daily embeddings.
            # This might be different from the Ca-TSU output.
            # For now, let's assume self.daily_text_embed_ca_tsu is the primary text input for Dual-Path SRL.
            # The `corpus_embed` here is defined as the output of MSINCell, which is `self.msg_embeds`.
            # Let's keep this naming for now, but be mindful of its origin (MSIN output).
            self.corpus_embed = self.msg_embeds # (batch_size, max_n_days, msin_h_size)

            # We need to decide what self.corpus_embed_size refers to.
            # If it's the output of Ca-TSU, then it's self.mel_h_size.
            # If it's the output of MSIN (self.msg_embeds), then it's self.msin_h_size.
            # The config has self.corpus_embed_size = self.mel_h_size.
            # This implies that the corpus_embed intended for VMD input might be the Ca-TSU output.
            
            # The original concatenation for self.x was:
            # self.x = tf.concat([self.daily_text_embed_ca_tsu, self.price_ph], axis=-1, name='vmd_input_x')
            # self.x_size = self.corpus_embed_size + self.price_size
            # Here, corpus_embed_size is mel_h_size (from Ca-TSU).
            
            # If Dual-Path SRL replaces this, then self.x will be h_t from the fusion module.
            # The size of h_t will be determined by the Dual-Path SRL architecture.

    def _create_dual_path_srl(self):
        logger.info('Creating Dual-Path SRL module...')
        with tf.variable_scope('dual_path_srl'):
            # Inputs to SRL:
            # i_t: self.daily_text_embed_ca_tsu (output of Ca-TSU, shape: batch, days, corpus_embed_size)
            # p_t: self.price_ph (raw price features, shape: batch, days, price_feat_dim)
            # e_t: self.msg_embed (BiGRU over words for each message, shape: batch, days, msgs, mel_h_size)

            srl_path_hidden_size = self.corpus_embed_size # Let's assume the output dim for each path is this.
            
            # --- Temporal Dependency (TD) Pathway ---
            # Uses i_t (text info) and p_t (price info)
            # For simplicity, let's say it processes a concatenation of i_t and p_t through a GRU.
            td_input_dim = self.daily_text_embed_ca_tsu.shape[-1].value + self.price_ph.shape[-1].value
            td_hidden_size = srl_path_hidden_size # Example size

            # --- Price-Driven (PD) Pathway ---
            # Uses e_t (detailed message embeddings) and p_t (price info)
            # This pathway uses the stance-aware attention.
            pd_input_dim_text = self.msg_embed.shape[-1].value # mel_h_size
            pd_input_dim_price = self.price_ph.shape[-1].value
            pd_hidden_size = srl_path_hidden_size # Example size
            
            hTD_t_list = []
            hPD_t_list = []
            pd_beta_list_all_days = [] # Initialize list for beta attentions

            # Process each day in the sequence
            for day_idx in range(self.max_n_days):
                with tf.variable_scope('daily_srl_processing', reuse=tf.AUTO_REUSE):
                    current_i_t_day = self.daily_text_embed_ca_tsu[:, day_idx, :] # (batch, corpus_embed_size)
                    current_p_t_day = self.price_ph[:, day_idx, :]               # (batch, price_feat_dim)
                    current_e_t_day = self.msg_embed[:, day_idx, :, :]           # (batch, msgs, mel_h_size)

                    # TD Pathway processing for current day
                    with tf.variable_scope('td_pathway'):
                        # Remove the 'name' parameter which is causing the error
                        td_gru_cell = tf.nn.rnn_cell.GRUCell(td_hidden_size)
                        # Simplified: just use current day's input, no explicit state passing here for now
                        # This is a simplification; a real TD path would likely have recurrent state.
                        td_input_concat = tf.concat([current_i_t_day, current_p_t_day], axis=-1)
                        # output_td, _ = td_gru_cell(td_input_concat, initial_state_for_td_if_any)
                        # For now, let's assume a dense layer for h_TD(t)
                        h_TD_t = tf.layers.dense(td_input_concat, td_hidden_size, activation=tf.nn.relu, name='h_TD_t_dense')
                        hTD_t_list.append(h_TD_t)

                    # PD Pathway processing for current day
                    with tf.variable_scope('pd_pathway'):
                        # Stance-aware attention for c_text(t)
                        # Query: current_p_t_day (price info)
                        # Keys/Values: current_e_t_day (message embeddings)

                        # Project query (price) to match key/value dimension if needed, or use for attention scores directly
                        # W_q_pd * p_t
                        attention_query_pd = tf.layers.dense(current_p_t_day, units=pd_input_dim_text, activation=None, name='pd_price_query_projection') # (batch, mel_h_size)
                        attention_query_pd_expanded = tf.expand_dims(attention_query_pd, axis=1) # (batch, 1, mel_h_size)

                        # Keys and Values are current_e_t_day
                        # W_k_pd * e_t  (assuming W_k_pd is identity or part of dense layer in e_t creation)
                        attention_keys_pd = current_e_t_day # (batch, msgs, mel_h_size)
                        attention_values_pd = current_e_t_day # (batch, msgs, mel_h_size)

                        # Scaled Dot-Product Attention
                        # attention_scores = Q * K^T / sqrt(d_k)
                        attention_scores_pd = tf.matmul(attention_query_pd_expanded, attention_keys_pd, transpose_b=True) # (batch, 1, msgs)
                        attention_scores_pd = tf.squeeze(attention_scores_pd, axis=1) # (batch, msgs)

                        # Apply mask (e.g., for padding if messages are padded)
                        # Check if msg_mask_ph exists, otherwise create a dummy mask that doesn't mask anything
                        if hasattr(self, 'msg_mask_ph'):
                            current_msg_mask_day = self.msg_mask_ph[:, day_idx, :] # (batch, msgs)
                        else:
                            # Create a placeholder mask that doesn't mask anything (all True)
                            # This ensures the code will work even if msg_mask_ph hasn't been defined
                            logger.info('msg_mask_ph not found, creating dummy mask for PD pathway')
                            current_msg_mask_day = tf.ones_like(attention_scores_pd, dtype=tf.bool)
                            
                        adder = (1.0 - tf.cast(current_msg_mask_day, tf.float32)) * -10000.0 # Large negative for masked positions
                        masked_attention_scores_pd = attention_scores_pd + adder
                        
                        # TensorFlow 1.x doesn't support axis in softmax
                        # Instead of: beta = tf.nn.softmax(masked_attention_scores_pd, axis=-1)
                        # Use exp and sum for manual softmax
                        exp_masked_scores = tf.exp(masked_attention_scores_pd - tf.reduce_max(masked_attention_scores_pd, keep_dims=True))
                        beta = exp_masked_scores / (tf.reduce_sum(exp_masked_scores, keep_dims=True) + 1e-8)
                        # Replace NaN with 0 (e.g., if all messages were masked)
                        beta = tf.where(tf.is_nan(beta), tf.zeros_like(beta), beta)
                        
                        # self.pd_attention_weights_beta = beta # Store for inspection for one day (will be overwritten)
                        pd_beta_list_all_days.append(beta) # Append current day's beta

                        # c_text = sum(beta_i * e_i)
                        context_vector_pd = tf.matmul(tf.expand_dims(beta, axis=1), attention_values_pd) # (batch, 1, mel_h_size)
                        context_vector_pd = tf.squeeze(context_vector_pd, axis=1) # (batch, mel_h_size)

                        # h_PD(t) = GRU(c_text(t), h_PD(t-1)) or Dense(c_text(t), p_t(t))
                        # Simplified: Dense layer over context_vector_pd and current_p_t_day
                        pd_input_concat = tf.concat([context_vector_pd, current_p_t_day], axis=-1)
                        h_PD_t = tf.layers.dense(pd_input_concat, pd_hidden_size, activation=tf.nn.relu, name='h_PD_t_dense')
                        hPD_t_list.append(h_PD_t)

            hTD_t_series = tf.stack(hTD_t_list, axis=1) # (batch_size, max_n_days, td_hidden_size)
            hPD_t_series = tf.stack(hPD_t_list, axis=1) # (batch_size, max_n_days, pd_hidden_size)

            if pd_beta_list_all_days:
                self.pd_attention_weights_beta = tf.stack(pd_beta_list_all_days, axis=1, name="pd_attention_weights_beta_stacked")
                # print(f"*** MODEL DEBUG (_create_dual_path_srl) *** self.pd_attention_weights_beta created. Name: {self.pd_attention_weights_beta.name if hasattr(self.pd_attention_weights_beta, 'name') else 'N/A'}")
            else:
                self.pd_attention_weights_beta = None
                # print("*** MODEL DEBUG (_create_dual_path_srl) *** pd_beta_list_all_days was empty. self.pd_attention_weights_beta set to None.")

            # --- Fusion Module ---
            with tf.variable_scope('fusion_module'):
                # Assuming hTD_t_series and hPD_t_series might have different dimensions if td_hidden_size != pd_hidden_size
                # If they need to be same for simple addition/gating, ensure srl_path_hidden_size is used for both.
                # For now, assuming they are both srl_path_hidden_size
                
                # Align dimensions if necessary (e.g. if one path is considered primary)
                # Here, assume they are already aligned to srl_path_hidden_size.
                hPD_t_series_aligned = hPD_t_series # Placeholder if alignment needed

                # Gating mechanism
                gate_input = tf.concat([hTD_t_series, hPD_t_series_aligned], axis=-1)
                gate_values = tf.layers.dense(gate_input, units=srl_path_hidden_size, activation=tf.nn.sigmoid, name='fusion_gate_sigma')
                self.sigma_gate = gate_values # This is sigma(...)

                # h_t = sigma_gate * hTD_t_series + (1 - sigma_gate) * hPD_t_series_aligned
                self.h_t_dual_path = self.sigma_gate * hTD_t_series + (1 - self.sigma_gate) * hPD_t_series_aligned
                # print(f"*** MODEL DEBUG (_create_dual_path_srl) *** self.h_t_dual_path created. Name: {self.h_t_dual_path.name if hasattr(self.h_t_dual_path, 'name') else 'N/A'}")
                
                # The dimension of self.h_t_dual_path is (batch_size, max_n_days, srl_path_hidden_size)
                # This will be self.x for the VMD module.
        logger.info('Dual-Pathway SRL module created.')

    def _create_vmd_with_h_rec(self):
        """
            z_t: batch_size * h_size
            y_t: batch_size * y_size
            g_t: batch_size * g_size
        """
        # VMD (Variational Macro Dynamics) - This is equivalent to DRG (Deep Recurrent Generation)
        # The input self.x to this module will now be self.h_t_dual_path (if dual path is used)
        # or the original concatenation of text and price.

        # We need to ensure self.x is correctly defined before this function is called.
        # And self.x_size reflects the dimension of self.x.

        with tf.name_scope('vmd_h_rec'):
            with tf.variable_scope('vmd_h_rec_iter', reuse=tf.AUTO_REUSE): # Removed reuse=True
                if self.vmd_cell_type == 'ln-lstm':
                    # lstm_h = tf.contrib.rnn.LayerNormBasicLSTMCell(self.h_size, layer_norm=True,
                    #                                                dropout_keep_prob=1-self.dropout_vmd, reuse=True) # ERROR: reuse cannot be True at cell construction
                    lstm_h = tf.contrib.rnn.LayerNormBasicLSTMCell(self.h_size, layer_norm=True,
                                                                   dropout_keep_prob=1-self.dropout_vmd) 
                else:  # gru
                    # lstm_h = tf.contrib.rnn.GRUCell(self.h_size, kernel_initializer=self.initializer,
                    #                                 bias_initializer=self.bias_initializer, reuse=True)
                    lstm_h = tf.contrib.rnn.GRUCell(self.h_size, kernel_initializer=self.initializer,
                                                     bias_initializer=self.bias_initializer)


                def _loop_body(t, ta_h_s, ta_z_prior, ta_z_post, ta_kl):
                    # ta_h_s accumulates h_s from all previous steps
                    # ta_z_prior accumulates (mu_prior, sigma_prior) from all previous steps
                    # ta_z_post accumulates (mu_post, sigma_post) from all previous steps
                    # ta_kl accumulates kl_t from all previous steps

                    def _init():
                        # initial state for lstm_h
                        h_s_prev = lstm_h.zero_state(self.batch_size, dtype=tf.float32) # h_s_0
                        y_prev = tf.zeros(shape=(self.batch_size, self.y_size), dtype=tf.float32)  # y_0

                        # prior_0
                        z_prior_mu_t, z_prior_sigma_t = self._z(h_s_prev, is_prior=True) # mu_0^prior, sigma_0^prior

                        # post_0
                        z_post_mu_t, z_post_sigma_t = self._z([h_s_prev, self.y_ph[:, t, :]], is_prior=False) # mu_0^post, sigma_0^post

                        return h_s_prev, y_prev, z_prior_mu_t, z_prior_sigma_t, z_post_mu_t, z_post_sigma_t

                    def _subsequent():
                        h_s_prev = ta_h_s.read(t - 1)
                        y_prev = self.y_ph[:, t - 1, :]

                        # prior_t
                        z_prior_mu_t, z_prior_sigma_t = self._z(h_s_prev, is_prior=True)

                        # post_t
                        z_post_mu_t, z_post_sigma_t = self._z([h_s_prev, self.y_ph[:, t, :]], is_prior=False) # y_t from data

                        return h_s_prev, y_prev, z_prior_mu_t, z_prior_sigma_t, z_post_mu_t, z_post_sigma_t

                    h_s_prev, y_prev, z_prior_mu_t, z_prior_sigma_t, z_post_mu_t, z_post_sigma_t = \
                        tf.cond(tf.equal(t, 0), true_fn=_init, false_fn=_subsequent)

                    # sample z_t from posterior (training) or prior (testing/generation)
                    dist_prior = ds.Normal(loc=z_prior_mu_t, scale=z_prior_sigma_t)
                    dist_post = ds.Normal(loc=z_post_mu_t, scale=z_post_sigma_t)
                    z_t = tf.cond(self.is_training_phase,
                                  true_fn=lambda: dist_post.sample(),
                                  false_fn=lambda: dist_prior.sample()
                                  )
                    
                    # x_t is input for this day. Shape (batch_size, self.x_size)
                    # self.x has shape (batch_size, max_n_days, self.x_size)
                    x_t = self.x[:, t, :]


                    # lstm_in_t = [x_t, y_{t-1}, z_t] for VMD
                    # Original paper SRL->DRG->TAP. DRG input is h_t (from SRL).
                    # Here, self.x is h_t from Dual-Path SRL (if enabled).
                    # The y_{t-1} and z_t are standard VAE/sequential VAE components.
                    lstm_in_t = tf.concat([x_t, y_prev, z_t], axis=-1, name='vmd_lstm_input_t')
                    
                    # Dropout on VMD input
                    lstm_in_t_dropped = tf.nn.dropout(lstm_in_t, keep_prob=1-self.dropout_vmd_in)

                    h_s_t, _ = lstm_h(lstm_in_t_dropped, h_s_prev) # h_s_t

                    # write results
                    ta_h_s = ta_h_s.write(t, h_s_t)
                    ta_z_prior = ta_z_prior.write(t, tf.stack([z_prior_mu_t, z_prior_sigma_t], axis=-1)) # Store mu and sigma
                    ta_z_post = ta_z_post.write(t, tf.stack([z_post_mu_t, z_post_sigma_t], axis=-1))   # Store mu and sigma

                    # calculate KL divergence for this step
                    kl_t = ds.kl_divergence(dist_post, dist_prior) # KL(q || p)
                    ta_kl = ta_kl.write(t, kl_t)

                    return t + 1, ta_h_s, ta_z_prior, ta_z_post, ta_kl

                # TensorArrays to store results from each time step
                ta_h_s = tf.TensorArray(dtype=tf.float32, size=self.max_n_days, name='ta_h_s')
                ta_z_prior = tf.TensorArray(dtype=tf.float32, size=self.max_n_days, name='ta_z_prior_params', element_shape=[None, self.z_size, 2])
                ta_z_post = tf.TensorArray(dtype=tf.float32, size=self.max_n_days, name='ta_z_post_params', element_shape=[None, self.z_size, 2])
                ta_kl = tf.TensorArray(dtype=tf.float32, size=self.max_n_days, name='ta_kl')

                # Loop through time
                t_final, ta_h_s_final, ta_z_prior_final, ta_z_post_final, ta_kl_final = \
                    tf.while_loop(cond=lambda t, *_: t < self.max_n_days,
                                  body=_loop_body,
                                  loop_vars=(0, ta_h_s, ta_z_prior, ta_z_post, ta_kl)
                                  )

                self.h_s = tf.transpose(ta_h_s_final.stack(), perm=[1, 0, 2], name='h_s_stacked') # batch_size * max_n_days * h_size
                self.z_prior_params = tf.transpose(ta_z_prior_final.stack(), perm=[1, 0, 2, 3], name='z_prior_params_stacked') # batch * days * z_size * 2
                self.z_post_params = tf.transpose(ta_z_post_final.stack(), perm=[1, 0, 2, 3], name='z_post_params_stacked')   # batch * days * z_size * 2
                self.kl_t = tf.transpose(ta_kl_final.stack(), perm=[1, 0, 2], name='kl_t_stacked') # batch_size * max_n_days * z_size
                
                self.kl = tf.reduce_sum(self.kl_t, axis=2) # Sum KL over z_dimensions for each day -> shape [batch_size, max_n_days]


                # g_t = f(h_s_t), y_t = f(h_s_t) or f(g_t)
                # Original model used g = h_s, and y = linear(g)
                # This is effectively the DRG part of PEN, where g is the latent representation from VAE.
                self.g = self.h_s # batch_size * max_n_days * g_size (here g_size = h_size)
                if self.use_g_bn:
                     self.g = neural.bn(self.g, self.is_training_phase, bn_scope='bn-g')


                # Prediction layer for y_t (movement) from g_t
                # This output is y_hat in some notations, or the parameters of the distribution for y.
                # For classification (price up/down), y_size is 2 (for logits).
                self.y_logits = self._linear(self.g, self.y_size, scope='g_to_y_logits')
                # Replace: self.y = tf.nn.softmax(self.y_logits, axis=-1, name='y_pred_softmax')
                # Manual softmax implementation compatible with TF1.x
                y_exp = tf.exp(self.y_logits - tf.reduce_max(self.y_logits, axis=-1, keep_dims=True))
                self.y = tf.divide(y_exp, tf.reduce_sum(y_exp, axis=-1, keep_dims=True), name='y_pred_softmax')
                
                # For inference (decode), we need g_T and y_T
                # g_T: g at the prediction time T (from placeholder T_ph)
                # y_T: predicted y at time T

                def _infer_func():
                    # This function is usually for getting g_T and y_T at the specific T_ph day.
                    # The self.g and self.y are already computed for all max_n_days.
                    # We just need to gather the values at T_ph.
                    sample_index = tf.reshape(tf.range(self.batch_size), (self.batch_size, 1), name='sample_index_infer')
                    # T_ph is 1-indexed day number, convert to 0-indexed for gathering
                    day_index = tf.reshape(self.T_ph - 1, (self.batch_size, 1), name='day_index_infer')
                    indexed_T = tf.concat([sample_index, day_index], axis=1)
                    
                    g_T = tf.gather_nd(params=self.g, indices=indexed_T) # batch_size * g_size
                    y_T = tf.gather_nd(params=self.y, indices=indexed_T) # batch_size * y_size
                    return g_T, y_T

                def _gen_func():
                    # use prior for g
                    # This is for generation mode, not typically used for prediction task like this.
                    # For now, let's assume it's similar to inference for prediction.
                    sample_index = tf.reshape(tf.range(self.batch_size), (self.batch_size, 1), name='sample_index_gen')
                    day_index = tf.reshape(self.T_ph - 1, (self.batch_size, 1), name='day_index_gen')
                    indexed_T = tf.concat([sample_index, day_index], axis=1)
                    
                    g_T = tf.gather_nd(params=self.g, indices=indexed_T) 
                    y_T = tf.gather_nd(params=self.y, indices=indexed_T)
                    return g_T, y_T

                self.g_T, self.y_T = tf.cond(self.is_training_phase, true_fn=_infer_func, false_fn=_gen_func)
                if self.use_o_bn:
                     self.y_T = neural.bn(self.y_T, self.is_training_phase, bn_scope='bn-y_T')


    def _create_vmd_with_zh_rec(self):
        """
            z_t: batch_size * z_size
            y_t: batch_size * y_size
            g_t: batch_size * g_size (not explicitly h_s like in h_rec)
        """
        # This is another variant of VMD/DRG.
        # We need to ensure self.x and self.x_size are set correctly before this.
        with tf.name_scope('vmd_zh_rec'):
            with tf.variable_scope('vmd_zh_rec_iter', reuse=tf.AUTO_REUSE): # Removed reuse=True
                if self.vmd_cell_type == 'ln-lstm':
                    # lstm_g = tf.contrib.rnn.LayerNormBasicLSTMCell(self.g_size, layer_norm=True,
                    #                                                dropout_keep_prob=1-self.dropout_vmd, reuse=True)
                    lstm_g = tf.contrib.rnn.LayerNormBasicLSTMCell(self.g_size, layer_norm=True,
                                                                    dropout_keep_prob=1-self.dropout_vmd)
                else:  # gru
                    # lstm_g = tf.contrib.rnn.GRUCell(self.g_size, kernel_initializer=self.initializer,
                    #                                 bias_initializer=self.bias_initializer, reuse=True)
                    lstm_g = tf.contrib.rnn.GRUCell(self.g_size, kernel_initializer=self.initializer,
                                                     bias_initializer=self.bias_initializer)

                # initial state for lstm_g
                g_init = lstm_g.zero_state(self.batch_size, dtype=tf.float32)

                def _loop_body(t, g_prev, ta_z_prior, ta_z_post, ta_kl):
                    def _init():
                        # prior_0
                        z_prior_mu_t, z_prior_sigma_t = self._z(g_prev, is_prior=True)

                        # post_0
                        # y_ph[:, t, :] is y_0
                        z_post_mu_t, z_post_sigma_t = self._z([g_prev, self.y_ph[:, t, :]], is_prior=False)
                        y_prev = tf.zeros(shape=(self.batch_size, self.y_size), dtype=tf.float32)

                        return y_prev, z_prior_mu_t, z_prior_sigma_t, z_post_mu_t, z_post_sigma_t

                    def _subsequent():
                        # y_ph[:, t-1, :] is y_{t-1}
                        y_prev = self.y_ph[:, t - 1, :]

                        # prior_t
                        z_prior_mu_t, z_prior_sigma_t = self._z(g_prev, is_prior=True)

                        # post_t
                        # y_ph[:, t, :] is y_t
                        z_post_mu_t, z_post_sigma_t = self._z([g_prev, self.y_ph[:, t, :]], is_prior=False)

                        return y_prev, z_prior_mu_t, z_prior_sigma_t, z_post_mu_t, z_post_sigma_t

                    y_prev, z_prior_mu_t, z_prior_sigma_t, z_post_mu_t, z_post_sigma_t = \
                        tf.cond(tf.equal(t, 0), true_fn=_init, false_fn=_subsequent)

                    # sample z_t
                    dist_prior = ds.Normal(loc=z_prior_mu_t, scale=z_prior_sigma_t)
                    dist_post = ds.Normal(loc=z_post_mu_t, scale=z_post_sigma_t)
                    z_t = tf.cond(self.is_training_phase,
                                  true_fn=lambda: dist_post.sample(),
                                  false_fn=lambda: dist_prior.sample()
                                  )
                    
                    x_t = self.x[:, t, :] # Input h_t from Dual-Path SRL (if enabled)

                    # lstm_in_t = [x_t, y_{t-1}, z_t]
                    lstm_in_t = tf.concat([x_t, y_prev, z_t], axis=-1, name='vmd_zh_lstm_input_t')
                    lstm_in_t_dropped = tf.nn.dropout(lstm_in_t, keep_prob=1-self.dropout_vmd_in)
                    
                    g_t, _ = lstm_g(lstm_in_t_dropped, g_prev)

                    # write results
                    ta_z_prior = ta_z_prior.write(t, tf.stack([z_prior_mu_t, z_prior_sigma_t], axis=-1))
                    ta_z_post = ta_z_post.write(t, tf.stack([z_post_mu_t, z_post_sigma_t], axis=-1))

                    dist_prior = ds.Normal(loc=z_prior_mu_t, scale=z_prior_sigma_t)
                    dist_post = ds.Normal(loc=z_post_mu_t, scale=z_post_sigma_t)
                    kl_t = ds.kl_divergence(dist_post, dist_prior)
                    ta_kl = ta_kl.write(t, kl_t)
                    
                    return t + 1, g_t, ta_z_prior, ta_z_post, ta_kl


                ta_g = tf.TensorArray(dtype=tf.float32, size=self.max_n_days, name='ta_g') # To store g_t at each step
                ta_z_prior = tf.TensorArray(dtype=tf.float32, size=self.max_n_days, name='ta_z_prior_params_zh', element_shape=[None, self.z_size, 2])
                ta_z_post = tf.TensorArray(dtype=tf.float32, size=self.max_n_days, name='ta_z_post_params_zh', element_shape=[None, self.z_size, 2])
                ta_kl = tf.TensorArray(dtype=tf.float32, size=self.max_n_days, name='ta_kl_zh')

                # Need to prime g_prev for the loop.
                # The loop body itself writes g_t to g_prev for the next iteration.
                # The initial g_prev is g_init.
                # We also need to store g_t in its own TensorArray.
                
                # Revised loop to store g_t
                def _loop_body_store_g(t, g_prev, ta_g_main, ta_z_prior_main, ta_z_post_main, ta_kl_main):
                    # ... (same logic as _loop_body to compute z_prior_mu_t, z_prior_sigma_t, z_post_mu_t, z_post_sigma_t, z_t, x_t)
                    y_prev, z_prior_mu_t, z_prior_sigma_t, z_post_mu_t, z_post_sigma_t = \
                        tf.cond(tf.equal(t, 0), 
                                lambda: _init_for_loop_body(g_prev), # _init needs g_prev
                                lambda: _subsequent_for_loop_body(g_prev, t)) # _subsequent needs g_prev and t

                    dist_prior = ds.Normal(loc=z_prior_mu_t, scale=z_prior_sigma_t)
                    dist_post = ds.Normal(loc=z_post_mu_t, scale=z_post_sigma_t)
                    z_t = tf.cond(self.is_training_phase,
                                  true_fn=lambda: dist_post.sample(),
                                  false_fn=lambda: dist_prior.sample()
                                  )
                    x_t = self.x[:, t, :]
                    lstm_in_t = tf.concat([x_t, y_prev, z_t], axis=-1)
                    lstm_in_t_dropped = tf.nn.dropout(lstm_in_t, keep_prob=1-self.dropout_vmd_in)
                    
                    g_t, _ = lstm_g(lstm_in_t_dropped, g_prev) # g_t computed
                    
                    ta_g_main = ta_g_main.write(t, g_t) # Store g_t
                    
                    ta_z_prior_main = ta_z_prior_main.write(t, tf.stack([z_prior_mu_t, z_prior_sigma_t], axis=-1))
                    ta_z_post_main = ta_z_post_main.write(t, tf.stack([z_post_mu_t, z_post_sigma_t], axis=-1))

                    dist_prior = ds.Normal(loc=z_prior_mu_t, scale=z_prior_sigma_t)
                    dist_post = ds.Normal(loc=z_post_mu_t, scale=z_post_sigma_t)
                    kl_t = ds.kl_divergence(dist_post, dist_prior)
                    ta_kl_main = ta_kl_main.write(t, kl_t)

                    return t + 1, g_t, ta_g_main, ta_z_prior_main, ta_z_post_main, ta_kl_main

                # Helper functions for _loop_body_store_g to avoid redefining _init and _subsequent if they capture loop vars
                def _init_for_loop_body(g_prev_arg):
                    z_prior_mu_t, z_prior_sigma_t = self._z(g_prev_arg, is_prior=True)
                    z_post_mu_t, z_post_sigma_t = self._z([g_prev_arg, self.y_ph[:, 0, :]], is_prior=False) # t=0 for y_ph
                    y_prev = tf.zeros(shape=(self.batch_size, self.y_size), dtype=tf.float32)
                    return y_prev, z_prior_mu_t, z_prior_sigma_t, z_post_mu_t, z_post_sigma_t

                def _subsequent_for_loop_body(g_prev_arg, t_arg):
                    y_prev = self.y_ph[:, t_arg - 1, :]
                    z_prior_mu_t, z_prior_sigma_t = self._z(g_prev_arg, is_prior=True)
                    z_post_mu_t, z_post_sigma_t = self._z([g_prev_arg, self.y_ph[:, t_arg, :]], is_prior=False)
                    return y_prev, z_prior_mu_t, z_prior_sigma_t, z_post_mu_t, z_post_sigma_t


                _, _, ta_g_final, ta_z_prior_final, ta_z_post_final, ta_kl_final = \
                    tf.while_loop(cond=lambda t, *_: t < self.max_n_days,
                                  body=_loop_body_store_g,
                                  loop_vars=(0, g_init, ta_g, ta_z_prior, ta_z_post, ta_kl)
                                  )
                
                self.g = tf.transpose(ta_g_final.stack(), perm=[1, 0, 2], name='g_stacked_zh') # batch_size * max_n_days * g_size
                if self.use_g_bn:
                     self.g = neural.bn(self.g, self.is_training_phase, bn_scope='bn-g_zh')

                self.z_prior_params = tf.transpose(ta_z_prior_final.stack(), perm=[1, 0, 2, 3], name='z_prior_params_stacked_zh')
                self.z_post_params = tf.transpose(ta_z_post_final.stack(), perm=[1, 0, 2, 3], name='z_post_params_stacked_zh')
                
                # kl_t inside the loop is [batch, z_size], so stacking makes it [days, batch, z_size]
                kl_t_stacked = ta_kl_final.stack()
                self.kl_t = tf.transpose(kl_t_stacked, perm=[1, 0, 2], name='kl_t_stacked_zh') # shape: [batch, days, z_size]
                # Sum over the z_size dimension to get the total KL for each day
                self.kl = tf.reduce_sum(self.kl_t, axis=2, name='kl_daily_zh') # shape: [batch, days]


                self.y_logits = self._linear(self.g, self.y_size, scope='g_to_y_logits_zh')
                # Replace: self.y = tf.nn.softmax(self.y_logits, axis=-1, name='y_pred_softmax_zh')
                # Manual softmax implementation compatible with TF1.x
                y_exp = tf.exp(self.y_logits - tf.reduce_max(self.y_logits, axis=-1, keep_dims=True))
                self.y = tf.divide(y_exp, tf.reduce_sum(y_exp, axis=-1, keep_dims=True), name='y_pred_softmax_zh')


                def _infer_func():
                    sample_index = tf.reshape(tf.range(self.batch_size), (self.batch_size, 1))
                    day_index = tf.reshape(self.T_ph - 1, (self.batch_size, 1))
                    indexed_T = tf.concat([sample_index, day_index], axis=1)
                    
                    g_T = tf.gather_nd(params=self.g, indices=indexed_T)
                    y_T = tf.gather_nd(params=self.y, indices=indexed_T)
                    return g_T, y_T

                def _gen_func():
                    # use prior for g & y
                    sample_index = tf.reshape(tf.range(self.batch_size), (self.batch_size, 1))
                    day_index = tf.reshape(self.T_ph - 1, (self.batch_size, 1))
                    indexed_T = tf.concat([sample_index, day_index], axis=1)
                    
                    g_T = tf.gather_nd(params=self.g, indices=indexed_T) # Should ideally sample g_T from prior if truly generative
                    y_T = tf.gather_nd(params=self.y, indices=indexed_T) # Then sample y_T from p(y|g_T)
                    return g_T, y_T
                
                self.g_T, self.y_T = tf.cond(self.is_training_phase, true_fn=_infer_func, false_fn=_gen_func)
                if self.use_o_bn:
                    self.y_T = neural.bn(self.y_T, self.is_training_phase, bn_scope='bn-y_T_zh')


    def _create_discriminative_vmd(self):
        """
            No z (latent variable), direct mapping from x to g (deterministic).
            g_t = RNN_cell(x_t, g_{t-1})
        """
        # This is a purely discriminative RNN, not a VAE/DRG.
        # Input self.x will be h_t from Dual-Path SRL (if enabled).
        with tf.name_scope('discriminative_vmd'):
            with tf.variable_scope('discriminative_vmd_iter', reuse=tf.AUTO_REUSE):
                if self.vmd_cell_type == 'ln-lstm':
                    rnn_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.g_size, layer_norm=True,
                                                                    dropout_keep_prob=1-self.dropout_vmd)
                else:  # gru
                    rnn_cell = tf.contrib.rnn.GRUCell(self.g_size, kernel_initializer=self.initializer,
                                                     bias_initializer=self.bias_initializer)
                
                # Initial state for RNN
                initial_state = rnn_cell.zero_state(self.batch_size, dtype=tf.float32)
                
                # self.x has shape (batch_size, max_n_days, self.x_size)
                # We need to provide sequence_length to dynamic_rnn if days are padded.
                # T_ph contains the actual number of days for each sample IF it's per sample.
                # However, T_ph is usually the prediction day.
                # Assuming all samples are processed for max_n_days, and masking handled by T_ph later.
                # Or, if we have a placeholder for sequence length of days per batch item.
                # For now, assume fixed max_n_days and T_ph selects the relevant output.

                # Dropout on input to RNN for VMD
                x_dropped = tf.nn.dropout(self.x, keep_prob=1-self.dropout_vmd_in)

                outputs, final_state = tf.nn.dynamic_rnn(
                    cell=rnn_cell,
                    inputs=x_dropped, # self.x is (batch, max_n_days, x_size)
                    initial_state=initial_state,
                    dtype=tf.float32,
                    scope="discriminative_rnn"
                )
                
                self.g = outputs # (batch_size, max_n_days, g_size)
                if self.use_g_bn:
                     self.g = neural.bn(self.g, self.is_training_phase, bn_scope='bn-g_disc')
                
                # No KL divergence term for discriminative model
                self.kl = tf.constant(0.0, dtype=tf.float32, name="kl_discriminative_zero")
                self.kl_t = tf.zeros_like(self.g) # Placeholder for compatibility if anything expects kl_t

                # Get g_T at the prediction day T_ph
                sample_index = tf.reshape(tf.range(self.batch_size), (self.batch_size, 1))
                day_index = tf.reshape(self.T_ph - 1, (self.batch_size, 1)) # T_ph is 1-indexed
                indexed_T = tf.concat([sample_index, day_index], axis=1)
                
                self.g_T = tf.gather_nd(params=self.g, indices=indexed_T) # (batch_size, g_size)
                
                # Prediction from g_T
                self.y_logits_T = self._linear(self.g_T, self.y_size, scope='gT_to_y_logits_disc')
                # Replace: self.y_T = tf.nn.softmax(self.y_logits_T, axis=-1, name='y_T_pred_softmax_disc')
                # Manual softmax implementation compatible with TF1.x
                y_T_exp = tf.exp(self.y_logits_T - tf.reduce_max(self.y_logits_T, axis=-1, keep_dims=True))
                self.y_T = tf.divide(y_T_exp, tf.reduce_sum(y_T_exp, axis=-1, keep_dims=True), name='y_T_pred_softmax_disc')
                
                if self.use_o_bn:
                    self.y_T = neural.bn(self.y_T, self.is_training_phase, bn_scope='bn-y_T_disc')

                # For loss calculation, we also need y (predictions for all days) if temporal attention is not used,
                # or if loss is computed over all days.
                # If only y_T is used for loss, then this is sufficient.
                # The _build_temporal_att might redefine y_T if daily_att=True
                # Let's also define self.y for all days for consistency, though it might not be directly used in loss if daily_att=True.
                self.y_logits = self._linear(self.g, self.y_size, scope='g_to_y_logits_disc_all_days')
                # Replace: self.y = tf.nn.softmax(self.y_logits, axis=-1, name='y_pred_softmax_disc_all_days')
                # Manual softmax implementation compatible with TF1.x
                y_exp = tf.exp(self.y_logits - tf.reduce_max(self.y_logits, axis=-1, keep_dims=True))
                self.y = tf.divide(y_exp, tf.reduce_sum(y_exp, axis=-1, keep_dims=True), name='y_pred_softmax_disc_all_days')


    def _build_vmd(self):
        # This is the DRG module from the paper.
        # It will use self.x as its input.
        # self.x will be set by the Dual-Path SRL if that path is active.
        if self.variant_type == 'discriminative':
            self._create_discriminative_vmd()
        else:
            if self.vmd_rec == 'h':
                self._create_vmd_with_h_rec()
            else: # 'zh'
                self._create_vmd_with_zh_rec()


    def _build_temporal_att(self):
        """
            g: batch_size * max_n_days * g_size
            g_T: batch_size * g_size
        """
        # This is the TAP (Temporal Attention Prediction) module from the paper.
        # It operates on the output of VMD/DRG (self.g or self.g_T).
        # If daily_att is true, it recomputes y_T using attention over daily g states.
        # Otherwise, y_T from VMD is used directly.

        with tf.name_scope('tda'): # Temporal Destination Awareness (Original name) / Temporal Attention
            with tf.variable_scope('tda', reuse=tf.AUTO_REUSE): # Removed reuse=True
                # y_T_ is the ground truth label for the prediction day T
                sample_index = tf.reshape(tf.range(self.batch_size), (self.batch_size, 1), name='sample_index_tda')
                self.indexed_T = tf.concat([sample_index, tf.reshape(self.T_ph-1, (self.batch_size, 1))], axis=1) # T_ph is 1-indexed
                self.y_T_ = tf.gather_nd(params=self.y_ph, indices=self.indexed_T)  # batch_size * y_size (ground truth)

                if self.daily_att:
                    logger.info("Using Temporal Attention (TAP)")
                    # Attention over daily g states (self.g from VMD)
                    # self.g: batch_size * max_n_days * g_size
                    # self.g_T: batch_size * g_size (from VMD, g at day T_ph before attention)

                    # v_i = w_i^T * tanh(W_g * g_i + b_g) (Simplified Bahdanau-style attention score)
                    # Or, use g_T as query: v_i = q_T^T * W_att * g_i
                    with tf.variable_scope('v_i'): # Scores for each day's g
                        # Project self.g, then dot with a temporal context vector w_i
                        # proj_i = self._linear([self.g], self.g_size, 'tanh', use_bias=False, scope='g_projection_att')
                        proj_i = tf.layers.dense(self.g, self.g_size, activation=tf.nn.tanh, use_bias=False, name='g_projection_att',
                                                 kernel_initializer=self.initializer)
                        w_i = tf.get_variable('w_i_temporal_att', shape=(self.g_size, 1), initializer=self.initializer)
                        v_i_scores = tf.squeeze(tf.tensordot(proj_i, w_i, axes=[[2], [0]]), axis=-1) # batch_size * max_n_days

                    # Mask for actual sequence length (T_ph should ideally be sequence length if it varies per sample)
                    # For now, assume T_ph is just the prediction day, and attention is over all max_n_days prior.
                    # If sequence lengths vary, use tf.sequence_mask(sequence_lengths_ph, maxlen=self.max_n_days)
                    # Let's assume for now we attend over all max_n_days
                    # No explicit masking here, softmax will handle. If some days are padding in self.g (e.g. zero vectors),
                    # their contribution after projection might be small or handled by training.

                    # v_i scores calculated above: batch_size * max_n_days
                    # Replace: self.v_stared = tf.nn.softmax(v_i_scores, axis=-1, name='temporal_attention_weights')
                    # Manual softmax implementation compatible with TF1.x
                    v_i_exp = tf.exp(v_i_scores - tf.reduce_max(v_i_scores, axis=-1, keep_dims=True))
                    self.v_stared = tf.divide(v_i_exp, tf.reduce_sum(v_i_exp, axis=-1, keep_dims=True), name='temporal_attention_weights')
                    v_stared_expanded = tf.expand_dims(self.v_stared, axis=-1) # batch_size * max_n_days * 1
                    
                    # Context vector c_t = sum(v_stared_i * g_i)
                    # A common practice is to use the g_T from VMD as part of the attention or final prediction
                    # Here, the attended context c_t will form the new g_T_att.
                    g_T_att = tf.reduce_sum(v_stared_expanded * self.g, axis=1) # batch_size * g_size
                    self.g_T_raw_vmd = self.g_T # Save g_T from VMD before TAP overrides it
                    self.g_T = g_T_att # Update g_T with attended version
                    
                    # Final prediction from attended g_T
                    y_T_logits_att = tf.layers.dense(self.g_T, self.y_size, name='pred_from_gT_att_logits',
                                                   kernel_initializer=self.initializer, bias_initializer=self.bias_initializer)
                    
                    # Replace: self.y_T = tf.nn.softmax(y_T_logits_att, axis=-1, name='y_T_pred_softmax_att')
                    # Manual softmax implementation compatible with TF1.x
                    y_T_att_exp = tf.exp(y_T_logits_att - tf.reduce_max(y_T_logits_att, axis=-1, keep_dims=True))
                    self.y_T = tf.divide(y_T_att_exp, tf.reduce_sum(y_T_att_exp, axis=-1, keep_dims=True), name='y_T_pred_softmax_att')
                    
                    if self.use_o_bn: # Batch norm on final prediction
                        self.y_T = neural.bn(self.y_T, self.is_training_phase, bn_scope='bn-y_T_att')
                else:
                    logger.info("Not using Temporal Attention (TAP), y_T from VMD is final.")
                    # self.y_T is already defined by _build_vmd if daily_att is False.
                    # self.y_T_ (ground truth) is also defined above.
                    pass # y_T from VMD is used directly.

    def _create_generative_ata(self):
        """
            Calculates loss for generative models based on the original PEN model's objective.
            The final objective to be maximized is:
            Objective = Likelihood_T + weighted_sum(Likelihood_aux - KL_aux)
            The loss is the negative of this objective.
        """
        with tf.name_scope('generative_ata_loss'):
            # This implementation mirrors the original PEN model's loss function.
            # self.y contains predictions for all days: [batch, max_n_days, y_size]
            # self.y_ph is the ground truth for all days: [batch, max_n_days, y_size]
            # self.y_T is the prediction for the final day T: [batch, y_size]
            # self.y_T_ is the ground truth for the final day T: [batch, y_size]
            # self.kl is the KL divergence for all days: [batch, max_n_days]
            # self.v_stared is the temporal attention weights from TAP: [batch, max_n_days]
            
            minor = 1e-7  # To prevent log(0)

            # 1. Calculate likelihood for auxiliary days (all days)
            # This is the log P(y_t | ...) for t = 1 to max_n_days
            likelihood_aux = tf.reduce_sum(tf.multiply(self.y_ph, tf.log(self.y + minor)), axis=2, name='likelihood_aux')  # Shape: [batch_size, max_n_days]

            # 2. Get the KL divergence lambda (annealing factor)
            kl_lambda = self._kl_lambda()
            
            # 3. Combine likelihood and KL divergence for auxiliary days
            # This is ( log P(y_t|...) - lambda * KL_t ) for each day
            obj_aux = likelihood_aux - kl_lambda * self.kl  # Shape: [batch_size, max_n_days]

            # 4. Calculate likelihood for the final prediction day T
            likelihood_T = tf.reduce_sum(tf.multiply(self.y_T_, tf.log(self.y_T + minor)), axis=1, keep_dims=True, name='likelihood_T') # Shape: [batch_size, 1]

            # 5. Get KL divergence for the final day T
            sample_index = tf.range(self.batch_size)
            day_index = self.T_ph - 1
            indexed_T_kl = tf.stack([sample_index, day_index], axis=1)
            kl_T = tf.gather_nd(self.kl, indexed_T_kl) # Shape: [batch_size]
            kl_T = tf.reshape(kl_T, [self.batch_size, 1]) # Shape: [batch_size, 1]
            
            # 6. Combine likelihood and KL for the final day T
            obj_T = likelihood_T - kl_lambda * kl_T # Shape: [batch_size, 1]

            # 7. Weight the auxiliary objective by temporal attention
            # self.v_stared should have been computed in _build_temporal_att
            if not hasattr(self, 'v_stared'):
                 logger.warning("v_stared not found for generative loss. Assuming uniform temporal attention.")
                 self.v_stared = tf.ones([self.batch_size, self.max_n_days], dtype=tf.float32) / tf.cast(self.max_n_days, tf.float32)

            # Note: The original paper uses `alpha` as a hyperparameter to scale temporal attention.
            v_aux = self.alpha * self.v_stared  # Shape: [batch_size, max_n_days]

            # 8. Combine final day objective with weighted auxiliary objective
            generative_objective = obj_T + tf.reduce_sum(tf.multiply(obj_aux, v_aux), axis=1, keep_dims=True)  # Shape: [batch_size, 1]
            
            # 9. The final generative loss is the negative mean of the objective (maximization -> minimization)
            self.generative_base_loss = tf.reduce_mean(-generative_objective)
            self.loss = self.generative_base_loss # Set self.loss to this new base loss


    def _create_discriminative_ata(self):
        """
            Calculates loss for discriminative models.
            
            Note: Causal Consistency Loss, Enhanced SRL Loss, and Noise-Aware Loss 
            are now handled in the central _build_ata method.
        """
        with tf.name_scope('discriminative_ata_loss'):
            # self.y_T is softmax output (batch_size * y_size)
            # self.y_T_ is ground truth (batch_size * y_size)
            # self.y_logits_T is logits for y_T (batch_size * y_size) - from _create_discriminative_vmd

            # Use softmax cross-entropy with logits for numerical stability
            reconstruction_loss_y = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.y_logits_T,
                labels=self.y_T_
            )
            self.log_likelihood_y = tf.reduce_mean(reconstruction_loss_y, name='log_likelihood_y')
            self.pred_loss = self.log_likelihood_y  # Store as pred_loss for consistency with generative
            self.loss = self.log_likelihood_y
            
            # Explainability loss for enhancing model interpretability
            if hasattr(self, 'ca_tsu_weights') and self.ca_tsu_weights is not None and self.entropy_regularization_lambda > 0:
                with tf.name_scope('explainability_loss_L2_disc'):
                    # Calculate entropy of attention weights (higher entropy = more uniform attention)
                    entropy_vos = -tf.reduce_sum(self.ca_tsu_weights * tf.log(self.ca_tsu_weights + 1e-9), axis=-1) # (batch, days)
                    mean_entropy_vos = tf.reduce_mean(entropy_vos)
                    
                    # Negative entropy to encourage more uniform attention (maximize entropy)
                    self.explain_loss_L2_disc = -mean_entropy_vos
                    
                    # Add to loss with entropy regularization lambda
                    explainability_loss = self.entropy_regularization_lambda * self.explain_loss_L2_disc
                    self.loss = tf.add(self.loss, explainability_loss, name='add_explainability_loss')
                
            # KL divergence is zero for discriminative models
            self.kl_divergence = tf.constant(0.0, dtype=tf.float32, name="kl_divergence_disc_zero")


    def _build_ata(self):
        """
        Build the adaptive temporal attention (ATA) module and define the final loss.
        
        This method defines the final objective function by:
        1. Calculating a 'base loss'. For generative models, this is the original PEN
           objective (Likelihood + KL-divergence). For discriminative, it's cross-entropy.
        2. Adding the specialized, weighted auxiliary losses from the enhancement modules.
        """
        logger.info("Building Adaptive Temporal Attention (ATA) module and Final Loss...")
        
        # Log the features that are enabled
        logger.info(f"Ca-TSU enabled: {getattr(self, 'use_ca_tsu', False)}")
        logger.info(f"Enhanced SRL enabled: {getattr(self, 'use_enhanced_srl', False)}")
        logger.info(f"Noise-Aware Loss enabled: {getattr(self, 'use_noise_aware_loss', False)}")
        
        # Dictionary to track all loss components
        self.loss_components = {}
        
        # 1. CALCULATE THE BASE LOSS
        if self.variant_type == 'discriminative':
            # For discriminative models, base loss is simple cross-entropy on the final prediction
            self._create_discriminative_ata()  # This sets self.loss
            logger.info(f"Using Discriminative Base Loss (Cross-Entropy): {self.loss}")
            self.loss_components['base_prediction_loss'] = self.loss
        else:
            # For generative models, use the original PEN objective (Likelihood + KL)
            self._create_generative_ata()  # This now sets self.loss to the generative objective
            logger.info(f"Using Generative Base Loss (Likelihood + KL Objective): {self.loss}")
            self.loss_components['base_generative_objective'] = self.loss
        
        # Initialize total_loss with the calculated base loss
        total_loss = self.loss
        
        # 2. ADD AUXILIARY ENHANCEMENT LOSSES
        logger.info("Adding auxiliary enhancement losses...")
        
        # =====================================================================
        # Add Enhanced SRL and/or Noise-Aware Loss if enabled
        # =====================================================================
        srl_loss_condition = (
            getattr(self, 'use_enhanced_srl', False) and 
            hasattr(self, 'h_TD') and hasattr(self, 'h_PD') and
            self.h_TD is not None and self.h_PD is not None
        )
        if srl_loss_condition:
            with tf.name_scope('enhanced_srl_contrastive_loss'):
                # Per user request, replace predictive loss with a contrastive margin loss
                # to enforce pathway specialization.
                # L_path = max(0, δ − ∥hTD − hPD∥)
                
                # Calculate the Euclidean distance for each day
                euclidean_distance = tf.norm(self.h_TD - self.h_PD, ord='euclidean', axis=-1) # Shape: [batch, max_n_days]
                
                # Calculate the daily margin loss
                daily_margin_loss = tf.maximum(0.0, self.path_separation_margin - euclidean_distance)

                # If noise-aware is enabled, it REPLACES the standard SRL loss.
                if getattr(self, 'use_noise_aware_loss', False):
                    logger.info("Creating Noise-Aware Reliability Weights for Contrastive Loss...")
                    with tf.name_scope('noise_aware_component'):
                        # Reliability is estimated from the final fused representation
                        h_t_dual_path = self.h_enhanced_dual_path
                        h_t_shape = tf.shape(h_t_dual_path)
                        batch_size, time_steps = h_t_shape[0], h_t_shape[1]
                        static_h_size = h_t_dual_path.shape[-1].value
                        h_t_flat = tf.reshape(h_t_dual_path, [-1, static_h_size])
                        
                        W_reliability = tf.get_variable('W_reliability', [static_h_size, 1], initializer=tf.contrib.layers.xavier_initializer())
                        b_reliability = tf.get_variable('b_reliability', [1], initializer=tf.zeros_initializer())
                        reliability_logits = tf.matmul(h_t_flat, W_reliability) + b_reliability
                        self.reliability_weights = tf.sigmoid(reliability_logits)
                        
                        reliability_weights_reshaped = tf.reshape(self.reliability_weights, [batch_size, time_steps])
                        
                        # Apply reliability weights to the margin loss
                        weighted_losses = daily_margin_loss * reliability_weights_reshaped
                        srl_component_loss = tf.reduce_mean(weighted_losses)
                        srl_component_weight = getattr(self, 'noise_aware_weight', 0.5)
                        
                        self.noise_aware_loss_raw = srl_component_loss
                        self.loss_components['noise_aware_loss_raw'] = srl_component_loss
                        logger.info(f"Using Noise-Aware Contrastive Loss as SRL component with weight: {srl_component_weight}")

                else: # Otherwise, use the standard (non-weighted) SRL loss
                    logger.info("Creating standard Contrastive Loss component...")
                    srl_component_loss = tf.reduce_mean(daily_margin_loss)
                    srl_component_weight = getattr(self, 'prediction_loss_weight', 0.1)
                    
                    self.enhanced_srl_loss_raw = srl_component_loss
                    self.loss_components['enhanced_srl_loss_raw'] = srl_component_loss
                    logger.info(f"Using standard Contrastive Loss as SRL component with weight: {srl_component_weight}")

                # Add the chosen SRL component loss to the total loss
                weighted_srl_loss = tf.multiply(srl_component_loss, srl_component_weight)
                total_loss = tf.add(total_loss, weighted_srl_loss, name='add_srl_contrastive_loss')

        # =====================================================================
        # Add Causal Consistency Loss if enabled
        # =====================================================================
        causal_loss_condition = (
            getattr(self, 'use_ca_tsu', False) and
            hasattr(self, 'alpha_i') and self.alpha_i is not None and
            hasattr(self, 'omega_i') and self.omega_i is not None
        )

        if causal_loss_condition:
            logger.info("Creating Causal Consistency Loss component (KL Divergence)...")
            with tf.name_scope('causal_consistency_component_kl'):
                # Per user request, Causal Loss is DKL(α || ω)
                # α (P): ground-truth distribution -> self.alpha_i from Ca-TSU
                # ω (Q): model's prediction to regularize -> self.omega_i from Ca-TSU
                
                P = self.alpha_i
                Q = self.omega_i
                
                logger.info(f"Causal Loss KL Divergence: P(alpha_i) shape={P.shape}, Q(omega_i) shape={Q.shape}")
                
                epsilon = 1e-8 # To prevent log(0)
                
                # DKL(P || Q) = ∑ P(x) * log(P(x) / Q(x))
                #             = ∑ P(x) * (log(P(x)) - log(Q(x)))
                kl_divergence = P * (tf.log(P + epsilon) - tf.log(Q + epsilon))
                
                # Sum over the message dimension for each day
                daily_kl_loss = tf.reduce_sum(kl_divergence, axis=-1) # Shape: [batch, max_n_days]
                
                # Create a mask for days within the actual sequence length
                day_mask = tf.sequence_mask(
                    self.T_ph, 
                    maxlen=self.max_n_days, 
                    dtype=tf.float32
                ) # Shape: [batch, max_n_days]
                
                masked_daily_kl = daily_kl_loss * day_mask
                
                # Average the loss over valid days for each sample in the batch
                per_sample_loss = tf.reduce_sum(masked_daily_kl, axis=1) / (tf.cast(self.T_ph, tf.float32) + epsilon)
                causal_loss_raw = tf.reduce_mean(per_sample_loss)

                self.causal_consistency_loss_raw = causal_loss_raw
                self.loss_components['causal_consistency_loss_raw'] = causal_loss_raw
                
                causal_weight = getattr(self, 'causal_loss_lambda', 0.3)
                weighted_causal_loss = tf.multiply(causal_loss_raw, causal_weight)
                total_loss = tf.add(total_loss, weighted_causal_loss, name='add_causal_loss_kl')
                logger.info(f"Using KL-based Causal Consistency Loss with weight: {causal_weight}")
        
        # Set the final loss for the optimizer
        self.loss = total_loss
        
        # Log final verification
        logger.info(f"Final loss after all components: {self.loss}")
        logger.info(f"Final loss components being optimized: {list(self.loss_components.keys())}")


    def _create_optimizer(self):
        with tf.name_scope('optimizer'):
            if self.opt == 'sgd':
                # Learning rate decay
                self.lr_decayed = tf.train.exponential_decay(self.lr, self.global_step,
                                                              self.decay_step, self.decay_rate, staircase=True)
                self.optimizer = tf.train.MomentumOptimizer(self.lr_decayed, self.momentum)
            elif self.opt == 'adam':
                # Adam doesn't typically use manual decay like this, its adaptive lr handles it.
                # However, if specified, can still apply. For now, use fixed LR with Adam.
                self.lr_decayed = self.lr # No decay for Adam here, or use tf.train.cosine_decay etc. if needed
                self.optimizer = tf.train.AdamOptimizer(self.lr_decayed)

            # Gradient clipping
            if self.clip > 0:
                grads, vs = zip(*self.optimizer.compute_gradients(self.loss))
                grads, gnorm = tf.clip_by_global_norm(grads, self.clip)
                self.optimize = self.optimizer.apply_gradients(zip(grads, vs), global_step=self.global_step)
            else:
                self.optimize = self.optimizer.minimize(self.loss, global_step=self.global_step)


    def assemble_graph(self):
        logger.info('Start graph assembling...')
        
        with tf.device('/device:GPU:0'):
            self._build_placeholders()
            self._build_embeds()  # Defines self.word_embed

            # TEL: Text Embedding Layer
            self._create_msg_embed_layer_in()  # Defines self.mel_in (from word_embed)
            self._create_msg_embed_layer()     # Defines self.msg_embed (BiGRU over words for each message)
            
            # Ca-TSU (already part of _create_corpus_embed)
            self._create_corpus_embed()        # Defines self.daily_text_embed_ca_tsu (i_t for Dual-Path)
                                              # also self.ca_tsu_weights (VoS for L2 loss)
            
            # Create message mask placeholder for dual-path SRL
            # This is used by the Enhanced SRL to identify valid messages
            # NOTE: The msg_mask_ph is already defined in _build_placeholders(), so we don't need to redefine it here
            # Commenting out the duplicate definition to avoid TF errors
            # with tf.name_scope('dual_path_placeholders'):
            #     self.msg_mask_ph = tf.placeholder(
            #         tf.bool, 
            #         shape=[None, self.max_n_days, self.max_n_msgs],
            #         name='msg_mask_ph'
            #     )
            #     logger.info(f"Created msg_mask_ph placeholder with shape: {self.msg_mask_ph.shape}")

            # --- Original MIE (Message Impact Extractor using MSIN) ---
            # This was part of the original complex model.
            if self.variant_type != 'tech':  # MIE is usually for text-inclusive variants
                self._build_mie()

            # --- Dual-Pathway SRL ---
            # Explicitly log the configurations for debugging
            logger.info(f"use_dual_path_srl: {getattr(self, 'use_dual_path_srl', False)}")
            logger.info(f"variant_type: {self.variant_type}")
            logger.info(f"use_enhanced_srl: {getattr(self, 'use_enhanced_srl', False)}")
            
            # Force enable the enhanced SRL if it's specified in tests
            if config_model.get('use_enhanced_srl', False):
                self.use_enhanced_srl = True
                logger.info("Enhanced SRL enabled from config")
            
            # Ensure dual-path is enabled whenever enhanced SRL is requested
            if getattr(self, 'use_enhanced_srl', False):
                self.use_dual_path_srl = True
                logger.info("Dual-path SRL automatically enabled for Enhanced SRL")

            effective_use_dual_path_srl = getattr(self, 'use_dual_path_srl', False)
            effective_variant_type = self.variant_type
            use_enhanced_srl = getattr(self, 'use_enhanced_srl', False)

            srl_condition = effective_use_dual_path_srl and effective_variant_type != 'tech'
            logger.info(f"SRL condition: {srl_condition} = {effective_use_dual_path_srl} and {effective_variant_type != 'tech'}")

            if srl_condition:
                if use_enhanced_srl:
                    logger.info('Using enhanced dual-path SRL architecture')
                    self._create_enhanced_dual_path_srl()
                    if hasattr(self, 'h_enhanced_dual_path') and self.h_enhanced_dual_path is not None:
                        logger.info("Enhanced dual-path output found - using h_enhanced_dual_path")
                        self.x = self.h_enhanced_dual_path
                        self.x_size = self.h_enhanced_dual_path.shape[-1].value
                        # Store for debugging
                        self.h_t_dual_path = self.h_enhanced_dual_path
                    else:
                        logger.warning("h_enhanced_dual_path not found, falling back to default SRL")
                        self.x = tf.concat([self.daily_text_embed_ca_tsu, self.price_ph], axis=-1, name='vmd_input_x_fallback_srl')
                        self.x_size = self.daily_text_embed_ca_tsu.shape[-1].value + self.price_ph.shape[-1].value
                else:
                    logger.info('Using original dual-path SRL architecture')
                    self._create_dual_path_srl()
                    if hasattr(self, 'h_t_dual_path') and self.h_t_dual_path is not None:
                        logger.info("Original dual-path output found - using h_t_dual_path")
                        self.x = self.h_t_dual_path
                        self.x_size = self.h_t_dual_path.shape[-1].value
                    else:
                        logger.warning("h_t_dual_path not found, falling back to default SRL")
                        self.x = tf.concat([self.daily_text_embed_ca_tsu, self.price_ph], axis=-1, name='vmd_input_x_fallback_srl')
                        self.x_size = self.daily_text_embed_ca_tsu.shape[-1].value + self.price_ph.shape[-1].value
            elif effective_variant_type == 'tech':
                logger.info('Using price-only (tech variant) for VMD input')
                self.x = self.price_ph 
                self.x_size = self.price_ph.shape[-1].value if hasattr(self.price_ph, 'shape') else 3
            else: 
                logger.info('Using Ca-TSU output and Price concatenation for VMD input (default SRL)')
                self.x = tf.concat([self.daily_text_embed_ca_tsu, self.price_ph], axis=-1, name='vmd_input_x_original_srl')
                self.x_size = self.daily_text_embed_ca_tsu.shape[-1].value + self.price_ph.shape[-1].value

            # Log the final x tensor for debugging
            logger.info(f"Final VMD input tensor: {self.x.name} with shape {self.x.shape}")

            # DRG: Deep Recurrent Generation (implemented by VMD)
            self._build_vmd() 

            # TAP: Temporal Attention Prediction
            self._build_temporal_att() 

            # Loss Calculation (ATA in original naming)
            self._build_ata() 
            
            # Optimizer
            self._create_optimizer() 
            
            logger.info('Graph assembled successfully.')

    def _kl_lambda(self):
        def _nonzero_kl_lambda():
            if self.use_constant_kl_lambda:
                return self.constant_kl_lambda
            else:
                # Anneal lambda: min(rate * step, 1.0) -- but should be max_lambda, not 1.0
                # Assuming max lambda is 1.0 as per typical VAE.
                # The annealing rate should be small enough to reach max_lambda over many steps.
                # Example: if max_lambda is 1.0, rate is 1/20000, it reaches 1.0 at step 20000.
                return tf.minimum(self.kl_lambda_anneal_rate * global_step_float, 1.0) # Max lambda is 1.0

        global_step_float = tf.cast(self.global_step, tf.float32)

        return tf.cond(global_step_float < self.kl_lambda_start_step, lambda: 0.0, _nonzero_kl_lambda)

    def _linear(self, args, output_size, activation=None, use_bias=True, use_bn=False, scope=None, reuse=None):
        """
        Linear layer or MLP layer.
        Args:
            args: a tensor or a list of tensors. If list, tensors are concatenated.
            output_size: integer, dimension of the output.
            activation: string or tf activation function.
            use_bias: boolean, whether to use bias.
            use_bn: boolean, whether to use batch normalization.
            scope: string, variable scope.
            reuse: boolean, whether to reuse variables.
        Returns:
            Tensor with shape [..., output_size].
        """
        if not isinstance(args, (list, tuple)):
            args = [args]

        # Concatenate input tensors if multiple are provided
        inputs = tf.concat(args, axis=-1)
        
        # Get input dimension
        input_size = inputs.get_shape()[-1].value
        
        # Store original shape for reshaping output later
        original_shape = tf.shape(inputs)
        input_rank = len(inputs.get_shape().as_list())
        
        # For inputs with rank > 2, reshape to 2D for matmul
        if input_rank > 2:
            # Flatten all dimensions except the last one
            reshaped_inputs = tf.reshape(inputs, [-1, input_size])
        else:
            reshaped_inputs = inputs

        with tf.variable_scope(scope or "linear_layer", reuse=reuse):
            W = tf.get_variable("W", shape=[input_size, output_size], initializer=self.initializer)
            output = tf.matmul(reshaped_inputs, W)

            if use_bias:
                b = tf.get_variable("b", shape=[output_size], initializer=self.bias_initializer)
                output = output + b
            
            if activation:
                if isinstance(activation, str):
                    if activation == 'tanh':
                        output = tf.nn.tanh(output)
                    elif activation == 'sigmoid':
                        output = tf.nn.sigmoid(output)
                    elif activation == 'relu':
                        output = tf.nn.relu(output)
                    else:
                        raise ValueError(f"Unsupported string activation: {activation}")
                else: # Assume it's a tf activation function
                    output = activation(output)

            if use_bn:
                # Ensure bn_scope is unique if this linear layer is called multiple times in different contexts
                # within the same outer scope without explicit sub-scoping for bn.
                # A simple way is to append a unique suffix to bn_scope based on 'scope' or a counter.
                bn_scope_name = f"bn_{scope}" if scope else "bn_linear"
                output = neural.bn(output, self.is_training_phase, bn_scope=bn_scope_name) # neural.bn handles reuse correctly based on scope.
            
            # Reshape output back to original shape if input was higher rank
            if input_rank > 2:
                # Create new shape: original shape except last dimension is now output_size
                new_shape = tf.concat([original_shape[:-1], [output_size]], axis=0)
                output = tf.reshape(output, new_shape)
        
        return output

    def _z(self, arg, is_prior):
        """
        Calculates parameters (mu, sigma) for latent variable z.
        Args:
            arg: input tensor (e.g., h_s_prev or [h_s_prev, y_target_t])
            is_prior: boolean, True if for prior, False for posterior.
        Returns:
            mu, sigma for z.
        """
        scope_name = 'z_prior' if is_prior else 'z_posterior'
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE): # Important: reuse for prior/posterior if called multiple times (e.g. in loop)
            # Linear layer to project input to 2 * z_size (for mu and log_sigma_sq)
            # then split into mu and log_sigma_sq
            # sigma = exp(0.5 * log_sigma_sq)
            
            # Original code might have separate linear layers for mu and sigma.
            # Let's follow a common pattern: one linear to 2*z_size, then split.
            
            # If arg is a list, _linear handles concatenation.
            hidden = self._linear(arg, self.z_size * 2, activation='tanh', scope='hidden_for_z')
            
            mu = self._linear(hidden, self.z_size, activation=None, scope='z_mu')
            log_sigma_sq = self._linear(hidden, self.z_size, activation=None, scope='z_log_sigma_sq') # Output log_variance

            # Ensure sigma is positive and stable
            # sigma = tf.nn.softplus(log_sigma_sq) # Softplus ensures positivity. Or exp(0.5 * log_var)
            sigma = tf.exp(0.5 * log_sigma_sq, name="z_sigma") # sigma = sqrt(variance)
            
            # Add small epsilon for numerical stability if sigma can be zero
            sigma = sigma + 1e-7

        return mu, sigma

    def _create_enhanced_dual_path_srl(self):
        """
        Enhanced Dual-path SRL with separate pathways for:
        - Text-driven branch: Identifies influential text that drives price movements
        - Price-driven branch: Identifies text that reflects/follows price patterns
        
        This implementation distinguishes between "influential text" and "price-driven text"
        using a dual pathway architecture with adaptive fusion based on market conditions.
        """
        logger.info('Creating Enhanced Dual-Path SRL module...')
        
        with tf.variable_scope('enhanced_dual_path_srl'):
            # Input dimensions
            text_embed_dim = self.daily_text_embed_ca_tsu.shape[-1].value  # From Ca-TSU
            price_dim = self.price_ph.shape[-1].value                      # Raw price features
            msg_embed_dim = self.msg_embed.shape[-1].value                 # Individual message embeddings
            
            # Shared parameters
            hidden_dim = self.corpus_embed_size  # Output dimension
            attention_dim = hidden_dim // 2      # Dimension for attention mechanism
            
            # Get configuration parameters for enhanced SRL
            use_td_temporal_attention = config_model.get('td_temporal_attention', True)
            pathway_competition_type = config_model.get('pathway_competition_type', 'dynamic')
            volatility_adaptation_factor = config_model.get('volatility_adaptation_factor', 0.3)
            residual_connection_weight = config_model.get('residual_connection_weight', 0.1)
            
            # === 1. TEXT-DRIVEN PATHWAY ===
            # Identifies "influential text" - text that predicts future price movements
            with tf.variable_scope('text_driven_pathway'):
                # Process text embeddings with temporal attention if enabled
                if use_td_temporal_attention:
                    text_enhanced = self._apply_temporal_attention(
                        self.daily_text_embed_ca_tsu, 
                        dim=text_embed_dim
                    )
                else:
                    text_enhanced = self.daily_text_embed_ca_tsu
                
                # Text-to-price predictor (predict next day's price movement)
                predicted_price_features = self._text_to_price_predictor(
                    text_enhanced, 
                    target_dim=price_dim
                )
                
                # Compute text influence scores based on prediction accuracy
                # For days [0, 1, ..., max_n_days-2], compare predicted price with actual price on days [1, 2, ..., max_n_days-1]
                actual_future_prices = self.price_ph[:, 1:, :]  # [batch, max_n_days-1, price_dim]
                predicted_current_prices = predicted_price_features[:, :-1, :]  # [batch, max_n_days-1, price_dim]
                
                # Calculate prediction error (MSE)
                prediction_errors = tf.reduce_mean(
                    tf.squared_difference(predicted_current_prices, actual_future_prices),
                    axis=-1  # Average across price dimensions
                )  # [batch, max_n_days-1]
                
                # Convert error to influence score (lower error = higher influence)
                # Using sigmoid to map errors to [0,1] range where higher means more influential
                text_influence_scores = tf.nn.sigmoid(-prediction_errors + 1.0)  # [batch, max_n_days-1]
                
                # Pad last day with neutral value since we can't evaluate its prediction
                text_influence_scores = tf.pad(
                    text_influence_scores, 
                    [[0, 0], [0, 1]],  # Pad only the time dimension
                    constant_values=0.5  # Neutral value for last day
                )  # [batch, max_n_days]
                
                # Store influence scores for analysis
                self.text_influence_scores = text_influence_scores
                
                # Apply influence scores to text features
                influence_expanded = tf.expand_dims(text_influence_scores, -1)  # [batch, max_n_days, 1]
                weighted_text_features = text_enhanced * influence_expanded
                
                # Combine weighted text with predicted prices for richer representation
                td_combined = tf.concat([weighted_text_features, predicted_price_features], axis=-1)
                
                # Final text-driven representation
                h_TD = tf.layers.dense(
                    td_combined,
                    units=hidden_dim,
                    activation=tf.nn.tanh,
                    name='td_representation'
                )  # [batch, max_n_days, hidden_dim]
                self.h_TD = h_TD # Store for the contrastive loss calculation
            
            # === 2. PRICE-DRIVEN PATHWAY ===
            # Identifies "price-driven text" - text that reflects current price patterns
            with tf.variable_scope('price_driven_pathway'):
                h_PD_list = []
                price_attn_weights_list = []
                
                # Process each day separately
                for day_idx in range(self.max_n_days):
                    with tf.variable_scope('daily_pd_processing', reuse=tf.AUTO_REUSE):
                        # Get current day's price and messages
                        current_price = self.price_ph[:, day_idx, :]         # [batch, price_dim]
                        current_messages = self.msg_embed[:, day_idx, :, :]  # [batch, max_n_msgs, msg_embed_dim]
                        
                        # Price-guided attention
                        # 1. Transform price to create query vector
                        price_query = tf.layers.dense(
                            current_price, 
                            units=attention_dim, 
                            activation=tf.nn.tanh,
                            name='price_query'
                        )  # [batch, attention_dim]
                        
                        # 2. Transform messages to create key vectors
                        message_keys = tf.layers.dense(
                            current_messages,
                            units=attention_dim,
                            activation=None,
                            name='message_keys'
                        )  # [batch, max_n_msgs, attention_dim]
                        
                        # 3. Compute attention scores using scaled dot-product
                        # Reshape query for broadcasting: [batch, 1, attention_dim]
                        price_query_expanded = tf.expand_dims(price_query, 1)
                        
                        # Compute scaled dot-product: [batch, 1, max_n_msgs]
                        # (manual implementation for TF1.x compatibility)
                        attention_logits = tf.matmul(
                            price_query_expanded,
                            message_keys,
                            transpose_b=True  # [batch, 1, attention_dim] x [batch, attention_dim, max_n_msgs]
                        ) / tf.sqrt(tf.cast(attention_dim, tf.float32))
                        
                        # Remove extra dimension: [batch, max_n_msgs]
                        attention_logits = tf.squeeze(attention_logits, axis=1)
                        
                        # 4. Apply mask for padding
                        # Get message mask for current day
                        if hasattr(self, 'msg_mask_ph'):
                            current_mask = self.msg_mask_ph[:, day_idx, :]  # [batch, max_n_msgs]
                        else:
                            # Create dummy mask (all True)
                            current_mask = tf.ones_like(attention_logits, dtype=tf.bool)
                        
                        # Apply large negative value to masked positions
                        masked_logits = attention_logits + (1.0 - tf.cast(current_mask, tf.float32)) * (-1e9)
                        
                        # 5. Apply softmax to get attention weights
                        # Manual softmax implementation for TF1.x
                        exp_logits = tf.exp(masked_logits - tf.reduce_max(masked_logits, keep_dims=True))
                        attention_weights = exp_logits / (tf.reduce_sum(exp_logits, keep_dims=True) + 1e-8)
                        
                        # Handle potential NaN values
                        attention_weights = tf.where(
                            tf.is_nan(attention_weights),
                            tf.zeros_like(attention_weights),
                            attention_weights
                        )
                        
                        # Store attention weights for this day
                        price_attn_weights_list.append(attention_weights)
                        
                        # 6. Apply attention to get context vector
                        # [batch, 1, max_n_msgs] x [batch, max_n_msgs, msg_embed_dim]
                        context_vector = tf.matmul(
                            tf.expand_dims(attention_weights, 1),
                            current_messages
                        )  # [batch, 1, msg_embed_dim]
                        
                        # Remove extra dimension
                        context_vector = tf.squeeze(context_vector, axis=1)  # [batch, msg_embed_dim]
                        
                        # 7. Combine with price information
                        combined_price_text = tf.concat([context_vector, current_price], axis=-1)
                        
                        # 8. Generate price-driven representation for current day
                        h_PD_day = tf.layers.dense(
                            combined_price_text,
                            units=hidden_dim,
                            activation=tf.nn.tanh,
                            name='pd_representation'
                        )  # [batch, hidden_dim]
                        
                        # Add to list
                        h_PD_list.append(h_PD_day)
                
                # Stack daily representations
                h_PD = tf.stack(h_PD_list, axis=1)  # [batch, max_n_days, hidden_dim]
                self.h_PD = h_PD # Store for the contrastive loss calculation
                
                # Stack attention weights for all days
                self.price_attention_weights = tf.stack(price_attn_weights_list, axis=1)  # [batch, max_n_days, max_n_msgs]
            
            # === 3. COMPUTE PRICE VOLATILITY ===
            # Detect market volatility for adaptive fusion
            with tf.variable_scope('volatility_detection'):
                # Compute price differences between consecutive days
                price_diffs = self.price_ph[:, 1:, :] - self.price_ph[:, :-1, :]  # [batch, max_n_days-1, price_dim]
                
                # Compute volatility as mean absolute change
                volatility = tf.reduce_mean(tf.abs(price_diffs), axis=-1)  # [batch, max_n_days-1]
                
                # Pad first day (no prior day to compute change)
                volatility = tf.pad(volatility, [[0, 0], [1, 0]], constant_values=0.0)  # [batch, max_n_days]
                
                # Normalize to [0,1] range with sigmoid
                volatility_factor = tf.nn.sigmoid(volatility * 5.0)  # Scaling factor of 5.0 for better sigmoid range
                
                # Store for analysis
                self.volatility_factor = volatility_factor  # [batch, max_n_days]
            
            # === 4. PATHWAY COMPETITION AND FUSION ===
            with tf.variable_scope('pathway_fusion'):
                # Dynamic pathway importance
                if pathway_competition_type == 'dynamic':
                    # Compute pathway importance based on features from both pathways
                    pathway_features = tf.concat([h_TD, h_PD], axis=-1)  # [batch, max_n_days, 2*hidden_dim]
                    
                    # Project to logits for 2 pathways
                    pathway_logits = tf.layers.dense(
                        pathway_features,
                        units=2,  # Two pathways
                        activation=None,
                        name='pathway_logits'
                    )  # [batch, max_n_days, 2]
                    
                    # Apply softmax to get pathway weights
                    # Manual softmax implementation for TF1.x
                    exp_logits = tf.exp(pathway_logits - tf.reduce_max(pathway_logits, axis=-1, keep_dims=True))
                    pathway_weights = exp_logits / (tf.reduce_sum(exp_logits, axis=-1, keep_dims=True) + 1e-8)
                    
                    # Extract individual pathway weights
                    text_weight = pathway_weights[:, :, 0:1]  # [batch, max_n_days, 1]
                    price_weight = pathway_weights[:, :, 1:2]  # [batch, max_n_days, 1]
                else:
                    # Static equal weighting
                    text_weight = tf.ones_like(h_TD[:, :, 0:1]) * 0.5  # [batch, max_n_days, 1]
                    price_weight = tf.ones_like(h_PD[:, :, 0:1]) * 0.5  # [batch, max_n_days, 1]
                
                # Store original weights for analysis
                self.pathway_text_weight = text_weight
                self.pathway_price_weight = price_weight
                
                # Volatility-aware fusion
                # Adjust weights based on volatility factor
                volatility_factor_expanded = tf.expand_dims(volatility_factor, -1)  # [batch, max_n_days, 1]
                
                # During high volatility, increase weight of price-driven pathway
                volatility_adjustment = volatility_adaptation_factor * volatility_factor_expanded
                
                # Apply adjustment
                adjusted_text_weight = text_weight * (1.0 - volatility_adjustment)
                adjusted_price_weight = price_weight * (1.0 + volatility_adjustment)
                
                # Renormalize weights
                total_weight = adjusted_text_weight + adjusted_price_weight
                final_text_weight = adjusted_text_weight / (total_weight + 1e-8)
                final_price_weight = adjusted_price_weight / (total_weight + 1e-8)
                
                # Store final weights for analysis
                self.final_text_weight = final_text_weight
                self.final_price_weight = final_price_weight
                
                # Apply weights to create fused representation
                h_fused = final_text_weight * h_TD + final_price_weight * h_PD  # [batch, max_n_days, hidden_dim]
                
                # Add residual connection to original Ca-TSU output
                if text_embed_dim == hidden_dim:
                    # Same dimensions - direct addition
                    h_with_residual = h_fused + residual_connection_weight * self.daily_text_embed_ca_tsu
                else:
                    # Different dimensions - need projection
                    ca_tsu_projected = tf.layers.dense(
                        self.daily_text_embed_ca_tsu,
                        units=hidden_dim,
                        activation=None,
                        name='ca_tsu_projection'
                    )
                    h_with_residual = h_fused + residual_connection_weight * ca_tsu_projected
                
                # Final output
                self.h_enhanced_dual_path = h_with_residual  # [batch, max_n_days, hidden_dim]
        
        logger.info('Enhanced Dual-Path SRL module created successfully.')

    def _apply_temporal_attention(self, inputs, dim):
        """
        Apply temporal attention over days to enhance text features with temporal context.
        
        Args:
            inputs: Tensor of shape [batch, max_n_days, dim]
            dim: Input feature dimension
            
        Returns:
            Enhanced features of same shape as inputs
        """
        with tf.variable_scope('temporal_attention'):
            batch_size = tf.shape(inputs)[0]
            max_n_days = tf.shape(inputs)[1]
            
            # Create query, key, value projections
            queries = tf.layers.dense(inputs, dim, activation=None, name='queries')
            keys = tf.layers.dense(inputs, dim, activation=None, name='keys')
            values = inputs  # Use original inputs as values
            
            # Compute attention scores: [batch, max_n_days, max_n_days]
            scores = tf.matmul(queries, keys, transpose_b=True)
            
            # Scale scores by sqrt(dim)
            scores = scores / tf.sqrt(tf.cast(dim, tf.float32))
            
            # Apply softmax to get attention weights
            # Manual softmax implementation for TF1.x
            exp_scores = tf.exp(scores - tf.reduce_max(scores, axis=-1, keep_dims=True))
            attention_weights = exp_scores / (tf.reduce_sum(exp_scores, axis=-1, keep_dims=True) + 1e-8)
            
            # Apply attention to values
            attended_values = tf.matmul(attention_weights, values)  # [batch, max_n_days, dim]
            
            # Add residual connection (original input + attended values)
            return inputs + attended_values

    def _text_to_price_predictor(self, inputs, target_dim, reuse=None):
        """
        Creates a neural network that predicts price movements from text representations.
        
        Args:
            inputs: Text representation tensor
            target_dim: Dimension of the price prediction output
            reuse: Whether to reuse variables (None, True, or tf.AUTO_REUSE)
            
        Returns:
            predictions: Price movement predictions
        """
        # Create a unique variable scope to avoid conflicts
        with tf.variable_scope('text_to_price_predictor', reuse=reuse):
            # Get input shape and log it for debugging
            input_shape = inputs.get_shape().as_list()
            logger.info(f"Text to price predictor input shape: {input_shape}, target_dim: {target_dim}")
            
            # Get the batch size and sequence length dynamically
            batch_size = tf.shape(inputs)[0]
            
            # Handle different input dimensions:
            # If input is 3D [batch, time, features], process as is
            # If input is 2D [batch, features], add time dimension temporarily
            if len(input_shape) == 3:
                # 3D input: [batch, time, features]
                seq_length = tf.shape(inputs)[1]
                input_features = input_shape[2]
                
                # Reshape to 2D for dense layers: [batch*time, features]
                flat_inputs = tf.reshape(inputs, [-1, input_features])
                logger.info(f"3D input reshaped to: [batch*time, features] = [-1, {input_features}]")
                
                # Process through dense layers
                hidden1 = tf.layers.dense(
                    flat_inputs, 
                    64, 
                    activation=tf.nn.relu, 
                    name='hidden1',
                    reuse=reuse
                )
                
                hidden2 = tf.layers.dense(
                    hidden1, 
                    32, 
                    activation=tf.nn.relu, 
                    name='hidden2',
                    reuse=reuse
                )
                
                # Output layer
                flat_predictions = tf.layers.dense(
                    hidden2, 
                    target_dim, 
                    activation=None, 
                    name='price_predictions',
                    reuse=reuse
                )
                
                # Reshape back to 3D: [batch, time, target_dim]
                predictions = tf.reshape(flat_predictions, [batch_size, seq_length, target_dim])
                logger.info(f"Predictions reshaped to: [batch, time, target_dim] = [{batch_size}, seq_length, {target_dim}]")
            
            else:
                # 2D input: [batch, features]
                # Process directly through dense layers
                hidden1 = tf.layers.dense(
                    inputs, 
                    64, 
                    activation=tf.nn.relu, 
                    name='hidden1',
                    reuse=reuse
                )
                
                hidden2 = tf.layers.dense(
                    hidden1, 
                    32, 
                    activation=tf.nn.relu, 
                    name='hidden2',
                    reuse=reuse
                )
                
                # Output layer
                predictions = tf.layers.dense(
                    hidden2, 
                    target_dim, 
                    activation=None, 
                    name='price_predictions',
                    reuse=reuse
                )
                logger.info(f"2D input processed directly: predictions shape = [batch, {target_dim}]")
            
        return predictions

    def _create_enhanced_srl_loss(self):
        """
        Creates the enhanced SRL loss which ensures text representations
        contain meaningful information about price movements.
        
        Returns:
            enhanced_srl_loss: The computed loss
        """
        logger.info("Creating enhanced SRL loss...")
        
        # Use a descriptive variable scope for the enhanced SRL loss
        with tf.variable_scope('enhanced_srl_loss', reuse=tf.AUTO_REUSE):
            
            # Get text representations from the dual-path SRL module
            if hasattr(self, 'h_t_dual_path') and self.h_t_dual_path is not None:
                text_repr = self.h_t_dual_path
                logger.info(f"Using h_t_dual_path for enhanced SRL loss: {self.h_t_dual_path.name} with shape {self.h_t_dual_path.shape}")
            elif hasattr(self, 'h_enhanced_dual_path') and self.h_enhanced_dual_path is not None:
                text_repr = self.h_enhanced_dual_path
                logger.info(f"Using h_enhanced_dual_path for enhanced SRL loss: {self.h_enhanced_dual_path.name} with shape {self.h_enhanced_dual_path.shape}")
            else:
                logger.warning("Neither h_t_dual_path nor h_enhanced_dual_path found, using x as text representation")
                text_repr = self.x
                logger.info(f"Using x for enhanced SRL loss: {self.x.name} with shape {self.x.shape}")
            
            # Store the actual text representation used for debugging
            self.srl_text_repr_used = text_repr
            
            # Log tensor shapes for debugging
            text_repr_shape = text_repr.get_shape().as_list()
            price_shape = self.price_ph.get_shape().as_list()
            logger.info(f"Enhanced SRL - text_repr shape: {text_repr_shape}, price_ph shape: {price_shape}")
            
            # Create prediction loss
            with tf.variable_scope('prediction_loss'):
                # Ensure text_repr has proper 3D shape [batch, time, features]
                if len(text_repr_shape) != 3:
                    logger.warning(f"Text representation doesn't have expected 3D shape. Current shape: {text_repr_shape}")
                    # If it's 2D [batch, features], reshape to 3D [batch, 1, features]
                    if len(text_repr_shape) == 2:
                        text_repr = tf.expand_dims(text_repr, axis=1)
                        logger.info(f"Reshaped 2D text_repr to 3D: {text_repr.get_shape().as_list()}")
                
                # Predict price movements from text representations
                try:
                    # Try to match the time dimension of price_ph
                    price_predictions = self._text_to_price_predictor(
                        inputs=text_repr,
                        target_dim=self.price_ph.shape[-1].value,
                        reuse=tf.AUTO_REUSE
                    )
                    
                    # Log the predicted vs actual price shapes
                    logger.info(f"Price predictions shape: {price_predictions.shape}, Actual price shape: {self.price_ph.shape}")
                    
                    # Ensure shapes match for loss calculation
                    pred_shape = price_predictions.get_shape().as_list()
                    price_shape = self.price_ph.get_shape().as_list()
                    
                    if pred_shape[1] != price_shape[1]:
                        logger.warning(f"Time dimension mismatch between predictions ({pred_shape[1]}) and prices ({price_shape[1]})")
                        # Handle time dimension mismatch
                        if pred_shape[1] > price_shape[1]:
                            # Trim predictions to match price time dimension
                            price_predictions = price_predictions[:, :price_shape[1], :]
                            logger.info(f"Trimmed predictions to match price shape: {price_predictions.shape}")
                        else:
                            # Pad predictions to match price time dimension
                            padding = [[0, 0], [0, price_shape[1] - pred_shape[1]], [0, 0]]
                            price_predictions = tf.pad(price_predictions, padding)
                            logger.info(f"Padded predictions to match price shape: {price_predictions.shape}")
                    
                    # Calculate mean squared error between predictions and actual prices
                    prediction_loss = tf.reduce_mean(
                        tf.square(price_predictions - self.price_ph)
                    )
                    logger.info(f"Base prediction loss created: {prediction_loss.name}")
                
                except Exception as e:
                    # Fallback to simpler loss calculation in case of tensor shape issues
                    logger.error(f"Error in price prediction: {str(e)}")
                    prediction_loss = tf.constant(0.1, dtype=tf.float32)
                    logger.info("Using fallback constant prediction loss")
                
                # If noise-aware loss is enabled, apply reliability weighting
                if hasattr(self, 'use_noise_aware_loss') and self.use_noise_aware_loss:
                    logger.info("Creating reliability weights for noise-aware loss")
                    try:
                        # Create reliability weights based on text quality
                        with tf.variable_scope('reliability_weights'):
                            # Ensure text_repr is properly shaped for dense layer
                            if len(text_repr.get_shape().as_list()) == 3:
                                # Reshape from [batch, time, features] to [batch*time, features]
                                batch_size = tf.shape(text_repr)[0]
                                time_steps = tf.shape(text_repr)[1]
                                feature_dim = text_repr.get_shape().as_list()[2]
                                
                                flat_text_repr = tf.reshape(text_repr, [-1, feature_dim])
                                logger.info(f"Reshaped text_repr for reliability estimation: {flat_text_repr.shape}")
                                
                                # First, estimate text influence on price
                                flat_text_influence = tf.layers.dense(
                                    flat_text_repr, 
                                    1, 
                                    activation=tf.nn.sigmoid,
                                    name='text_influence_estimator'
                                )
                                
                                # Reshape back to [batch, time, 1]
                                text_influence = tf.reshape(flat_text_influence, [batch_size, time_steps, 1])
                                self.text_influence_scores = text_influence
                                logger.info(f"Created text_influence_scores: {self.text_influence_scores.shape}")
                                
                                # Compute reliability weights (how much to trust each text)
                                flat_combined = tf.concat([flat_text_repr, tf.reshape(flat_text_influence, [-1, 1])], axis=-1)
                                flat_reliability_weights = tf.layers.dense(
                                    flat_combined,
                                    1,
                                    activation=tf.nn.sigmoid,
                                    name='reliability_estimator'
                                )
                                
                                # Reshape back to [batch, time, 1]
                                reliability_weights = tf.reshape(flat_reliability_weights, [batch_size, time_steps, 1])
                            else:
                                # Handle 2D case [batch, features]
                                text_influence = tf.layers.dense(
                                    text_repr, 
                                    1, 
                                    activation=tf.nn.sigmoid,
                                    name='text_influence_estimator'
                                )
                                self.text_influence_scores = text_influence
                                
                                reliability_weights = tf.layers.dense(
                                    tf.concat([text_repr, text_influence], axis=-1),
                                    1,
                                    activation=tf.nn.sigmoid,
                                    name='reliability_estimator'
                                )
                            
                            # Store weights for analysis
                            self.reliability_weights = reliability_weights
                            logger.info(f"Created reliability_weights: {self.reliability_weights.shape}")
                            
                            # Apply weights to the prediction loss - use proper broadcasting
                            weighted_squared_diff = reliability_weights * tf.square(price_predictions - self.price_ph)
                            
                            # Sum over all dimensions and normalize
                            weighted_loss = tf.reduce_sum(weighted_squared_diff) / (tf.reduce_sum(reliability_weights) + 1e-8)
                            
                            # Store the noise-aware loss component
                            self.noise_aware_loss = weighted_loss
                            logger.info(f"Created noise_aware_loss: {self.noise_aware_loss.name}")
                            
                            # Combine with regular prediction loss
                            noise_aware_weight = getattr(self, 'noise_aware_weight', 0.5)
                            prediction_loss = (1.0 - noise_aware_weight) * prediction_loss + \
                                            noise_aware_weight * weighted_loss
                            logger.info(f"Combined prediction loss with noise-aware loss using weight: {noise_aware_weight}")
                    
                    except Exception as e:
                        logger.error(f"Error in noise-aware loss calculation: {str(e)}")
                        # Continue with just the basic prediction loss
                        logger.info("Continuing with basic prediction loss without noise-aware component")
                
                # Scale the loss
                prediction_loss_weight = getattr(self, 'prediction_loss_weight', 0.1)
                enhanced_srl_loss = prediction_loss_weight * prediction_loss
                logger.info(f"Final enhanced SRL loss created with weight {prediction_loss_weight}: {enhanced_srl_loss.name}")
                
                # Log the enhanced SRL loss
                tf.summary.scalar('enhanced_srl_loss', enhanced_srl_loss)
                
                # Return the loss to be added to the total loss in _build_ata
                return enhanced_srl_loss

    def _add_enhanced_srl_summaries(self):
        """
        Add TensorFlow summaries for enhanced SRL components
        """
        with tf.name_scope('enhanced_srl_summaries'):
            if hasattr(self, 'text_influence_scores'):
                tf.summary.histogram('text_influence_scores', self.text_influence_scores)
            
            if hasattr(self, 'reliability_weights'):
                tf.summary.histogram('reliability_weights', self.reliability_weights)
                
            if hasattr(self, 'noise_aware_loss'):
                tf.summary.scalar('noise_aware_loss', self.noise_aware_loss)

    def _create_noise_aware_loss(self):
        """
        Create a noise-aware loss component for the enhanced dual-path SRL.
        This loss function is designed to make the model more robust to noisy tweets
        by adjusting the weight of each tweet based on its estimated reliability.
        """
        with tf.variable_scope('noise_aware_loss'):
            # Only add this loss if enhanced SRL is active
            if hasattr(self, 'h_enhanced_dual_path'):
                # Parameters for noise detection
                noise_threshold = 0.3  # Tweets with influence score below this are considered noisy
                max_penalty = 0.8      # Maximum reduction in weight for noisy tweets
                
                # Get text influence scores (already calculated in enhanced SRL)
                # Higher influence = more reliable, lower influence = potentially noisy
                if hasattr(self, 'text_influence_scores'):
                    influence_scores = self.text_influence_scores  # [batch, max_n_days]
                    
                    # Calculate reliability weight for each day based on influence scores
                    # Days with higher influence scores (more reliable) get weight closer to 1.0
                    # Days with lower influence scores (more noisy) get reduced weight
                    reliability_weights = tf.minimum(
                        1.0,
                        influence_scores / noise_threshold
                    )
                    
                    # Apply a floor to ensure even noisy days have some influence
                    reliability_weights = (1.0 - max_penalty) + (max_penalty * reliability_weights)
                    
                    # Store reliability weights for analysis
                    self.reliability_weights = reliability_weights
                    
                    # Create a mask for days within the actual sequence length
                    day_mask = tf.sequence_mask(
                        self.T_ph,
                        maxlen=self.max_n_days,
                        dtype=tf.float32
                    )  # [batch, max_n_days]
                    
                    # Calculate price prediction loss for each day
                    with tf.variable_scope('text_to_price_predictor', reuse=True):
                        predicted_prices = self._text_to_price_predictor(
                            self.daily_text_embed_ca_tsu,
                            target_dim=self.price_ph.shape[-1].value
                        )
                    
                    # Calculate MSE per day
                    daily_mse = tf.reduce_mean(
                        tf.squared_difference(
                            predicted_prices,
                            self.price_ph
                        ),
                        axis=-1  # Average across price dimensions
                    )  # [batch, max_n_days]
                    
                    # Apply reliability weights to the loss
                    weighted_mse = daily_mse * reliability_weights * day_mask
                    
                    # Normalize by the sum of weights to ensure balanced loss
                    normalization_factor = tf.reduce_sum(reliability_weights * day_mask, axis=1, keep_dims=True) + 1e-8
                    normalized_weighted_mse = tf.reduce_sum(weighted_mse, axis=1) / tf.squeeze(normalization_factor)
                    
                    # Final noise-aware loss
                    noise_aware_loss = tf.reduce_mean(normalized_weighted_mse)
                    
                    # Add summary for monitoring
                    tf.summary.scalar('noise_aware_loss', noise_aware_loss)
                    tf.summary.histogram('reliability_weights', reliability_weights)
                    
                    return noise_aware_loss
            
            # If conditions not met, return zero loss
            return tf.constant(0.0, dtype=tf.float32)

