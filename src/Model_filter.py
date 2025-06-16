#!/usr/local/bin/python
from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
import neural as neural
#from MSINModule import MSINCell, MSIN, MSINStateTuple
from MSINModule_MHA import MSINCell, MSIN, MSINStateTuple # Assuming this is the intended MSIN module
import tensorflow.contrib.distributions as ds
from tensorflow.contrib.layers import batch_norm
from ConfigLoader import logger, ss_size, vocab_size, config_model, path_parser


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

        self.clip = config_model['clip']
        self.n_epochs = config_model['n_epochs']
        self.batch_size_for_name = config_model['batch_size']

        self.max_n_days = config_model['max_n_days']
        self.max_n_msgs = config_model['max_n_msgs'] # This now refers to max relevant messages
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

        # Add _filter suffix to model name if this is the filtered version
        name_tuple = (self.mode + "_filter", name_max_n, name_input_type, name_key, name_train)
        self.model_name = '_'.join(name_tuple)

        # paths
        self.tf_graph_path = os.path.join(path_parser.graphs, self.model_name)
        self.tf_checkpoints_path = os.path.join(path_parser.checkpoints, self.model_name)
        self.tf_checkpoint_file_path = os.path.join(self.tf_checkpoints_path, 'checkpoint')
        self.tf_saver_path = os.path.join(self.tf_checkpoints_path, 'sess')

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

            self.word_table_init = tf.placeholder(dtype=tf.float32, shape=[vocab_size, self.word_embed_size])

            self.stock_ph = tf.placeholder(dtype=tf.int32, shape=[None])
            self.T_ph = tf.placeholder(dtype=tf.int32, shape=[None, ])
            self.n_words_ph = tf.placeholder(dtype=tf.int32, shape=[None, self.max_n_days, self.max_n_msgs])
            self.n_msgs_ph = tf.placeholder(dtype=tf.int32, shape=[None, self.max_n_days]) # Count of relevant msgs
            self.y_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.max_n_days, self.y_size])
            self.mv_percent_ph = tf.placeholder(dtype=tf.int32, shape=[None, self.max_n_days])
            self.price_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.max_n_days, 3])
            self.word_ph = tf.placeholder(dtype=tf.int32, shape=[None, self.max_n_days, self.max_n_msgs, self.max_n_words])
            self.ss_index_ph = tf.placeholder(dtype=tf.int32, shape=[None, self.max_n_days, self.max_n_msgs])

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
        with tf.name_scope('mel_in'):
            with tf.variable_scope('mel_in'):
                mel_in = self.word_embed
                if self.use_in_bn:
                    mel_in = neural.bn(mel_in, self.is_training_phase, bn_scope='bn-mel_inputs')
                self.mel_in = tf.nn.dropout(mel_in, keep_prob=1-self.dropout_mel_in)

    def _create_msg_embed_layer(self):
        def _for_one_trading_day(daily_in, daily_ss_index_vec, daily_mask):
            out, _ = tf.nn.bidirectional_dynamic_rnn(mel_cell_f, mel_cell_b, daily_in, daily_mask,
                                                     mel_init_f, mel_init_b, dtype=tf.float32)
            out_f, out_b = out
            ss_indices = tf.reshape(daily_ss_index_vec, [-1, 1])
            msg_ids = tf.constant(list(range(0, self.max_n_msgs)), dtype=tf.int32, shape=[self.max_n_msgs, 1])
            out_id = tf.concat([msg_ids, ss_indices], axis=1)
            mel_h_f, mel_h_b = tf.gather_nd(out_f, out_id), tf.gather_nd(out_b, out_id)
            msg_embed = (mel_h_f + mel_h_b) / 2
            return msg_embed

        def _for_one_sample(sample, sample_ss_index, sample_mask):
            return neural.iter(size=self.max_n_days, func=_for_one_trading_day,
                               iter_arg=sample, iter_arg2=sample_ss_index, iter_arg3=sample_mask)

        def _for_one_batch():
            return neural.iter(size=self.batch_size, func=_for_one_sample,
                               iter_arg=self.mel_in, iter_arg2=self.ss_index_ph, iter_arg3=self.n_words_ph)

        with tf.name_scope('mel'):
            with tf.variable_scope('mel_iter', reuse=tf.AUTO_REUSE):
                if self.mel_cell_type == 'ln-lstm':
                    mel_cell_f = tf.contrib.rnn.LayerNormBasicLSTMCell(self.mel_h_size)
                    mel_cell_b = tf.contrib.rnn.LayerNormBasicLSTMCell(self.mel_h_size)
                elif self.mel_cell_type == 'gru':
                    mel_cell_f = tf.contrib.rnn.GRUCell(self.mel_h_size)
                    mel_cell_b = tf.contrib.rnn.GRUCell(self.mel_h_size)
                else:
                    mel_cell_f = tf.contrib.rnn.BasicRNNCell(self.mel_h_size)
                    mel_cell_b = tf.contrib.rnn.BasicRNNCell(self.mel_h_size)

                mel_cell_f = tf.contrib.rnn.DropoutWrapper(mel_cell_f, output_keep_prob=1.0-self.dropout_mel)
                mel_cell_b = tf.contrib.rnn.DropoutWrapper(mel_cell_b, output_keep_prob=1.0-self.dropout_mel)
                mel_init_f = mel_cell_f.zero_state([self.max_n_msgs], tf.float32)
                mel_init_b = mel_cell_f.zero_state([self.max_n_msgs], tf.float32)

                msg_embed_shape = (self.batch_size, self.max_n_days, self.max_n_msgs, self.msg_embed_size)
                msg_embed_res = tf.reshape(_for_one_batch(), shape=msg_embed_shape) # Renamed msg_embed to avoid conflict
                self.msg_embed = tf.nn.dropout(msg_embed_res, keep_prob=1-self.dropout_mel, name='msg_embed')

    def _create_corpus_embed(self):
        with tf.name_scope('corpus_embed'):
            with tf.variable_scope('u_t'): 
                proj_u = self._linear(self.msg_embed, self.msg_embed_size, 'tanh', use_bias=False)
                w_u = tf.get_variable('w_u', shape=(self.msg_embed_size, 1), initializer=self.initializer)
            u_scores = tf.reduce_mean(tf.tensordot(proj_u, w_u, axes=1), axis=-1)

            mask_msgs_padding = tf.sequence_mask(self.n_msgs_ph, maxlen=self.max_n_msgs, dtype=tf.bool, name='mask_msgs_padding')
            
            ninf = tf.fill(tf.shape(u_scores), -np.inf)
            masked_score = tf.where(mask_msgs_padding, u_scores, ninf)
            
            u_weights = tf.nn.softmax(masked_score, axis=-1)
            u_weights = tf.where(tf.is_nan(u_weights), tf.zeros_like(u_weights), u_weights)

            u_weights_expanded = tf.expand_dims(u_weights, axis=-2)
            corpus_embed_G = tf.matmul(u_weights_expanded, self.msg_embed)
            self.corpus_embed = tf.squeeze(corpus_embed_G, axis=-2)
            self.corpus_embed = tf.nn.dropout(self.corpus_embed, keep_prob=1-self.dropout_ce, name='corpus_embed_dropout')

    def _build_mie(self):
        with tf.name_scope('mie'):
            self.price = self.price_ph
            self.price_size = 3

            if self.variant_type == 'tech':
                self.x = self.price
                self.x_size = self.price_size
            else:
                self._create_msg_embed_layer_in()
                self._create_msg_embed_layer()
                # The following MSINCell part implies TSU (_create_corpus_embed) output is not directly self.x
                # If MSINCell directly uses self.msg_embed (embeddings of individual messages)
                # then _create_corpus_embed might not be on the main path to self.x for this variant.
                # However, the paper describes TSU as part of SRL which feeds into IFU.
                # For variants 'hedge', 'fund' (if not 'tech'), text data is used.
                # Original code had a commented out section that used self.corpus_embed here.
                # We need to ensure the correct text representation flows into MSIN or VMD.

                # Assuming SRL (including TSU) produces self.corpus_embed and this should be fused.
                # The original code used MSINCell with self.msg_embed for s_inputs.
                # If _create_corpus_embed() is called before _build_mie() (as per assemble_graph order), 
                # self.corpus_embed would be available.
                # Let's assume the structure from assemble_graph where _create_corpus_embed IS called.
                self._create_corpus_embed() # Ensure corpus_embed is created

                if self.variant_type == 'fund':
                    self.x = self.corpus_embed # Only text based for 'fund'
                    self.x_size = self.corpus_embed_size
                elif self.variant_type in ('hedge', 'discriminative'): # 'discriminative' also implies text+price
                    # Concatenate corpus (text summary) and price for 'hedge' and other non-'tech' variants
                    self.x = tf.concat([self.corpus_embed, self.price], axis=2)
                    self.x_size = self.corpus_embed_size + self.price_size
                else:
                    # Fallback or error for unexpected variant_type if it should use text
                    logger.error(f"Unexpected variant_type '{self.variant_type}' in _build_mie regarding text data handling.")
                    # Defaulting to price only to avoid crash, but this needs review.
                    self.x = self.price 
                    self.x_size = self.price_size

    def _create_vmd_with_h_rec(self):
        with tf.name_scope('vmd'):
            with tf.variable_scope('vmd_h_rec'):
                x_input = tf.nn.dropout(self.x, keep_prob=1-self.dropout_vmd_in) # Renamed x to x_input
                x_transposed = tf.transpose(x_input, [1, 0, 2])
                y_transposed = tf.transpose(self.y_ph, [1, 0, 2])

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
                        gate_args = [x_transposed[t], h_s_t_1, z_t_1]

                        with tf.variable_scope('gru_r'):
                            r = self._linear(gate_args, self.h_size, 'sigmoid')
                        with tf.variable_scope('gru_u'):
                            u = self._linear(gate_args, self.h_size, 'sigmoid')

                        h_args = [x_transposed[t], tf.multiply(r, h_s_t_1), z_t_1]
                        with tf.variable_scope('gru_h'):
                            h_tilde = self._linear(h_args, self.h_size, 'tanh')

                        h_s_t = tf.multiply(1 - u, h_s_t_1) + tf.multiply(u, h_tilde)

                        with tf.variable_scope('h_z_prior'):
                            h_z_prior_t = self._linear([x_transposed[t], h_s_t], self.z_size, 'tanh')
                        with tf.variable_scope('z_prior'):
                            z_prior_t, z_prior_t_pdf = self._z(h_z_prior_t, is_prior=True)

                        with tf.variable_scope('h_z_post'):
                            h_z_post_t = self._linear([x_transposed[t], h_s_t, y_transposed[t]], self.z_size, 'tanh')
                        with tf.variable_scope('z_post'):
                            z_post_t, z_post_t_pdf = self._z(h_z_post_t, is_prior=False)

                    kl_t = ds.kl_divergence(z_post_t_pdf, z_prior_t_pdf)
                    ta_h_s = ta_h_s.write(t, h_s_t)
                    ta_z_prior = ta_z_prior.write(t, z_prior_t)
                    ta_z_post = ta_z_post.write(t, z_post_t)
                    ta_kl = ta_kl.write(t, kl_t)
                    return t + 1, ta_h_s, ta_z_prior, ta_z_post, ta_kl

                ta_h_s_init = tf.TensorArray(tf.float32, size=self.max_n_days, clear_after_read=False)
                ta_z_prior_init = tf.TensorArray(tf.float32, size=self.max_n_days)
                ta_z_post_init = tf.TensorArray(tf.float32, size=self.max_n_days, clear_after_read=False)
                ta_kl_init = tf.TensorArray(tf.float32, size=self.max_n_days)

                loop_init_vars = (0, ta_h_s_init, ta_z_prior_init, ta_z_post_init, ta_kl_init) # Renamed loop_init
                loop_cond = lambda t, *args: t < self.max_n_days
                _, ta_h_s, ta_z_prior, ta_z_post, ta_kl = tf.while_loop(loop_cond, _loop_body, loop_init_vars)

                h_s_stacked = tf.reshape(ta_h_s.stack(), shape=(self.max_n_days, self.batch_size, self.h_size)) # Renamed h_s
                z_shape = (self.max_n_days, self.batch_size, self.z_size)
                z_prior_stacked = tf.reshape(ta_z_prior.stack(), shape=z_shape) # Renamed z_prior
                z_post_stacked = tf.reshape(ta_z_post.stack(), shape=z_shape) # Renamed z_post
                kl_stacked = tf.reshape(ta_kl.stack(), shape=z_shape) # Renamed kl

                x_input_transposed_back = tf.transpose(x_transposed, [1, 0, 2]) # Renamed x
                h_s_final = tf.transpose(h_s_stacked, [1, 0, 2]) # Renamed h_s
                z_prior_final = tf.transpose(z_prior_stacked, [1, 0, 2]) # Renamed z_prior
                z_post_final = tf.transpose(z_post_stacked, [1, 0, 2]) # Renamed z_post
                self.kl = tf.reduce_sum(tf.transpose(kl_stacked, [1, 0, 2]), axis=2)

                with tf.variable_scope('g'):
                    self.g = self._linear([x_input_transposed_back, h_s_final, z_post_final], self.g_size, 'tanh', use_bn=False)

                with tf.variable_scope('y'):
                    self.y = self._linear(self.g, self.y_size, 'softmax')

                sample_index = tf.reshape(tf.range(self.batch_size), (self.batch_size, 1), name='sample_index')
                self.indexed_T = tf.concat([sample_index, tf.reshape(self.T_ph-1, (self.batch_size, 1))], axis=1)

                def _infer_func():
                    g_T_val = tf.gather_nd(params=self.g, indices=self.indexed_T) # Renamed g_T
                    if not self.daily_att:
                        y_T_val = tf.gather_nd(params=self.y, indices=self.indexed_T) # Renamed y_T
                        return g_T_val, y_T_val
                    return g_T_val

                def _gen_func():
                    z_prior_T_val = tf.gather_nd(params=z_prior_final, indices=self.indexed_T) # Renamed z_prior_T
                    h_s_T_val = tf.gather_nd(params=h_s_final, indices=self.indexed_T) # Renamed h_s_T
                    x_T_val = tf.gather_nd(params=x_input_transposed_back, indices=self.indexed_T) # Renamed x_T

                    with tf.variable_scope('g', reuse=tf.AUTO_REUSE):
                        g_T_val = self._linear([x_T_val, h_s_T_val, z_prior_T_val], self.g_size, 'tanh', use_bn=False)
                    if not self.daily_att:
                        with tf.variable_scope('y', reuse=tf.AUTO_REUSE):
                            y_T_val = self._linear(g_T_val, self.y_size, 'softmax')
                        return g_T_val, y_T_val
                    return g_T_val

                if not self.daily_att:
                    self.g_T, self.y_T = tf.cond(tf.equal(self.is_training_phase, True), _infer_func, _gen_func)
                else:
                    self.g_T = tf.cond(tf.equal(self.is_training_phase, True), _infer_func, _gen_func)

    def _create_vmd_with_zh_rec(self):
        with tf.name_scope('vmd'):
            with tf.variable_scope('vmd_zh_rec', reuse=tf.AUTO_REUSE):
                x_input = tf.nn.dropout(self.x, keep_prob=1-self.dropout_vmd_in) # Renamed x
                self.mask_aux_trading_days = tf.sequence_mask(self.T_ph - 1, self.max_n_days, dtype=tf.bool,
                                                              name='mask_aux_trading_days')

                if self.vmd_cell_type == 'ln-lstm':
                    cell_vmd = tf.contrib.rnn.LayerNormBasicLSTMCell(self.h_size) # Renamed cell
                else:
                    cell_vmd = tf.contrib.rnn.GRUCell(self.h_size)
                cell_vmd = tf.contrib.rnn.DropoutWrapper(cell_vmd, output_keep_prob=1.0-self.dropout_vmd)

                init_state_vmd = None # Renamed init_state
                h_s_dyn, _ = tf.nn.dynamic_rnn(cell_vmd, x_input, sequence_length=self.T_ph, initial_state=init_state_vmd, dtype=tf.float32) # Renamed h_s

                x_transposed = tf.transpose(x_input, [1, 0, 2])
                h_s_transposed = tf.transpose(h_s_dyn, [1, 0, 2]) # Renamed h_s
                y_transposed = tf.transpose(self.y_ph, [1, 0, 2])

                def _loop_body(t, ta_z_prior, ta_z_post, ta_kl):
                    with tf.variable_scope('iter_body', reuse=tf.AUTO_REUSE):
                        init_fn = lambda: tf.random_normal(shape=[self.batch_size, self.z_size], name='z_post_t_1') # Renamed init
                        subsequent_fn = lambda: tf.reshape(ta_z_post.read(t-1), [self.batch_size, self.z_size]) # Renamed subsequent
                        z_post_t_1 = tf.cond(t >= 1, subsequent_fn, init_fn)

                        with tf.variable_scope('h_z_prior'):
                            h_z_prior_t = self._linear([x_transposed[t], h_s_transposed[t], z_post_t_1], self.z_size, 'tanh')
                        with tf.variable_scope('z_prior'):
                            z_prior_t, z_prior_t_pdf = self._z(h_z_prior_t, is_prior=True)

                        with tf.variable_scope('h_z_post'):
                            h_z_post_t = self._linear([x_transposed[t], h_s_transposed[t], y_transposed[t], z_post_t_1], self.z_size, 'tanh')
                        with tf.variable_scope('z_post'):
                            z_post_t, z_post_t_pdf = self._z(h_z_post_t, is_prior=False)

                    kl_t = ds.kl_divergence(z_post_t_pdf, z_prior_t_pdf)
                    ta_z_prior = ta_z_prior.write(t, z_prior_t)
                    ta_z_post = ta_z_post.write(t, z_post_t)
                    ta_kl = ta_kl.write(t, kl_t)
                    return t + 1, ta_z_prior, ta_z_post, ta_kl

                ta_z_prior_init = tf.TensorArray(tf.float32, size=self.max_n_days)
                ta_z_post_init = tf.TensorArray(tf.float32, size=self.max_n_days, clear_after_read=False)
                ta_kl_init = tf.TensorArray(tf.float32, size=self.max_n_days)

                loop_init_vars = (0, ta_z_prior_init, ta_z_post_init, ta_kl_init) # Renamed loop_init
                cond_fn = lambda t, *args: t < self.max_n_days # Renamed cond
                _, ta_z_prior, ta_z_post, ta_kl = tf.while_loop(cond_fn, _loop_body, loop_init_vars)

                z_shape = (self.max_n_days, self.batch_size, self.z_size)
                z_prior_stacked = tf.reshape(ta_z_prior.stack(), shape=z_shape) # Renamed z_prior
                z_post_stacked = tf.reshape(ta_z_post.stack(), shape=z_shape) # Renamed z_post
                kl_stacked = tf.reshape(ta_kl.stack(), shape=z_shape) # Renamed kl

                h_s_final = tf.transpose(h_s_transposed, [1, 0, 2]) # Renamed h_s
                z_prior_final = tf.transpose(z_prior_stacked, [1, 0, 2]) # Renamed z_prior
                z_post_final = tf.transpose(z_post_stacked, [1, 0, 2]) # Renamed z_post
                self.kl = tf.reduce_sum(tf.transpose(kl_stacked, [1, 0, 2]), axis=2)

                with tf.variable_scope('g'):
                    self.g = self._linear([h_s_final, z_post_final], self.g_size, 'tanh')
                with tf.variable_scope('y'):
                    self.y = self._linear(self.g, self.y_size, 'softmax')

                sample_index = tf.reshape(tf.range(self.batch_size), (self.batch_size, 1), name='sample_index')
                self.indexed_T = tf.concat([sample_index, tf.reshape(self.T_ph-1, (self.batch_size, 1))], axis=1)

                def _infer_func():
                    g_T_val = tf.gather_nd(params=self.g, indices=self.indexed_T)
                    if not self.daily_att:
                        y_T_val = tf.gather_nd(params=self.y, indices=self.indexed_T)
                        return g_T_val, y_T_val
                    return g_T_val

                def _gen_func():
                    z_prior_T_val = tf.gather_nd(params=z_prior_final, indices=self.indexed_T)
                    h_s_T_val = tf.gather_nd(params=h_s_final, indices=self.indexed_T)
                    with tf.variable_scope('g', reuse=tf.AUTO_REUSE):
                        g_T_val = self._linear([h_s_T_val, z_prior_T_val], self.g_size, 'tanh', use_bn=False)
                    if not self.daily_att:
                        with tf.variable_scope('y', reuse=tf.AUTO_REUSE):
                            y_T_val = self._linear(g_T_val, self.y_size, 'softmax')
                        return g_T_val, y_T_val
                    return g_T_val

                if not self.daily_att:
                    self.g_T, self.y_T = tf.cond(tf.equal(self.is_training_phase, True), _infer_func, _gen_func)
                else:
                    self.g_T = tf.cond(tf.equal(self.is_training_phase, True), _infer_func, _gen_func)

    def _create_discriminative_vmd(self):
        with tf.name_scope('vmd'):
            with tf.variable_scope('vmd_zh_rec', reuse=tf.AUTO_REUSE):
                x_input = tf.nn.dropout(self.x, keep_prob=1-self.dropout_vmd_in) # Renamed x
                self.mask_aux_trading_days = tf.sequence_mask(self.T_ph - 1, self.max_n_days, dtype=tf.bool,
                                                              name='mask_aux_trading_days')

                if self.vmd_cell_type == 'ln-lstm':
                    cell_vmd = tf.contrib.rnn.LayerNormBasicLSTMCell(self.h_size)
                else:
                    cell_vmd = tf.contrib.rnn.GRUCell(self.h_size)
                cell_vmd = tf.contrib.rnn.DropoutWrapper(cell_vmd, output_keep_prob=1.0-self.dropout_vmd)

                init_state_vmd = None
                h_s_dyn, _ = tf.nn.dynamic_rnn(cell_vmd, x_input, sequence_length=self.T_ph, initial_state=init_state_vmd, dtype=tf.float32)

                x_transposed = tf.transpose(x_input, [1, 0, 2])
                h_s_transposed = tf.transpose(h_s_dyn, [1, 0, 2])

                def _loop_body(t, ta_z):
                    with tf.variable_scope('iter_body', reuse=tf.AUTO_REUSE):
                        init_fn = lambda: tf.random_normal(shape=[self.batch_size, self.z_size], name='z_post_t_1')
                        subsequent_fn = lambda: tf.reshape(ta_z.read(t-1), [self.batch_size, self.z_size])
                        z_t_1 = tf.cond(t >= 1, subsequent_fn, init_fn)

                        with tf.variable_scope('h_z'):
                            h_z_t = self._linear([x_transposed[t], h_s_transposed[t], z_t_1], self.z_size, 'tanh')
                        with tf.variable_scope('z'):
                            z_t = self._linear(h_z_t, self.z_size, 'tanh')

                    ta_z_res = ta_z.write(t, z_t) # Renamed ta_z to ta_z_res
                    return t + 1, ta_z_res

                ta_z_init_arr = tf.TensorArray(tf.float32, size=self.max_n_days, clear_after_read=False) # Renamed ta_z_init
                loop_init_vars = (0, ta_z_init_arr)
                cond_fn = lambda t, *args: t < self.max_n_days
                _, ta_z_final = tf.while_loop(cond_fn, _loop_body, loop_init_vars) # Renamed ta_z_init

                z_shape = (self.max_n_days, self.batch_size, self.z_size)
                z_stacked = tf.reshape(ta_z_final.stack(), shape=z_shape) # Renamed z

                h_s_final = tf.transpose(h_s_transposed, [1, 0, 2])
                z_final = tf.transpose(z_stacked, [1, 0, 2]) # Renamed z

                with tf.variable_scope('g'):
                    self.g = self._linear([h_s_final, z_final], self.g_size, 'tanh')
                with tf.variable_scope('y'):
                    self.y = self._linear(self.g, self.y_size, 'softmax')

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
        with tf.name_scope('tda'):
            with tf.variable_scope('tda'):
                with tf.variable_scope('v_i'):
                    proj_i = self._linear([self.g], self.g_size, 'tanh', use_bias=False)
                    w_i = tf.get_variable('w_i', shape=(self.g_size, 1), initializer=self.initializer)
                v_i = tf.reduce_sum(tf.tensordot(proj_i, w_i, axes=1), axis=-1)

                with tf.variable_scope('v_d'):
                    proj_d = self._linear([self.g], self.g_size, 'tanh', use_bias=False)
                g_T_expanded = tf.expand_dims(self.g_T, axis=-1) # Renamed g_T
                v_d = tf.reduce_sum(tf.matmul(proj_d, g_T_expanded), axis=-1)

                aux_score = tf.multiply(v_i, v_d, name='v_stared')
                ninf_tda = tf.fill(tf.shape(aux_score), -np.inf) # Renamed ninf
                masked_aux_score = tf.where(self.mask_aux_trading_days, aux_score, ninf_tda)
                v_stared_softmax = tf.nn.softmax(masked_aux_score) # Renamed v_stared

                self.v_stared = tf.where(tf.is_nan(v_stared_softmax), tf.zeros_like(v_stared_softmax), v_stared_softmax)

                if self.daily_att == 'y':
                    context_tensor = tf.transpose(self.y, [0, 2, 1]) # Renamed context
                else:
                    context_tensor = tf.transpose(self.g, [0, 2, 1])

                v_stared_expanded = tf.expand_dims(self.v_stared, -1) # Renamed v_stared
                att_c = tf.reduce_sum(tf.matmul(context_tensor, v_stared_expanded), axis=-1)
                with tf.variable_scope('y_T'):
                    self.y_T = self._linear([att_c, self.g_T], self.y_size, 'softmax')

    def _create_generative_ata(self):
        with tf.name_scope('ata'):
            with tf.variable_scope('ata'):
                v_aux = self.alpha * self.v_stared
                minor_val = 1e-7 # Renamed minor
                likelihood_aux = tf.reduce_sum(tf.multiply(self.y_ph, tf.log(self.y + minor_val)), axis=2)

                kl_lambda_val = self._kl_lambda() # Renamed kl_lambda
                obj_aux = likelihood_aux - kl_lambda_val * self.kl

                self.y_T_gt = tf.gather_nd(params=self.y_ph, indices=self.indexed_T) # Renamed y_T_
                likelihood_T = tf.reduce_sum(tf.multiply(self.y_T_gt, tf.log(self.y_T + minor_val)), axis=1, keep_dims=True)

                kl_T_val = tf.reshape(tf.gather_nd(params=self.kl, indices=self.indexed_T), shape=[self.batch_size, 1]) # Renamed kl_T
                obj_T = likelihood_T - kl_lambda_val * kl_T_val

                obj = obj_T + tf.reduce_sum(tf.multiply(obj_aux, v_aux), axis=1, keep_dims=True)
                self.loss = tf.reduce_mean(-obj, axis=[0, 1])
        
    def _create_discriminative_ata(self):
        with tf.name_scope('ata'):
            with tf.variable_scope('ata'):
                v_aux = self.alpha * self.v_stared
                minor_val = 1e-7
                likelihood_aux = tf.reduce_sum(tf.multiply(self.y_ph, tf.log(self.y + minor_val)), axis=2)

                self.y_T_gt = tf.gather_nd(params=self.y_ph, indices=self.indexed_T) # Renamed y_T_
                likelihood_T = tf.reduce_sum(tf.multiply(self.y_T_gt, tf.log(self.y_T + minor_val)), axis=1, keep_dims=True)

                obj = likelihood_T + tf.reduce_sum(tf.multiply(likelihood_aux, v_aux), axis=1, keep_dims=True)
                # The P_obj part seems to be related to MSIN model, might not be standard PEN loss.
                # If self.P is not defined or relevant for standard PEN, this part can be an issue or might be specific to a variant.
                # For safety, checking if self.P exists if it comes from an optional component (MSIN).
                if hasattr(self, 'P') and self.P is not None:
                    new_P = tf.clip_by_value(self.P, 1e-8, 1.0) # Ensure P is not zero for log
                    P_obj = tf.reduce_sum(tf.multiply(self.P, tf.log(new_P)), axis=-1) # Ensure axis is correct for P's shape
                    self.loss = tf.reduce_mean(-obj, axis=[0, 1]) + tf.reduce_mean(-P_obj, axis=[0,1]) # Added reduction for P_obj
                else:
                    self.loss = tf.reduce_mean(-obj, axis=[0, 1])


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
                optimizer_obj = tf.train.MomentumOptimizer(learning_rate=decayed_lr, momentum=self.momentum) # Renamed optimizer
            else:
                optimizer_obj = tf.train.AdamOptimizer(self.lr)
            gradients, variables = zip(*optimizer_obj.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, self.clip)
            self.optimize = optimizer_obj.apply_gradients(zip(gradients, variables))
            # self.global_step should be incremented by apply_gradients if it's a tf.Variable, or manually.
            # optimizer.apply_gradients returns an op, doesn't directly return incremented global_step.
            # Let's ensure global_step is correctly incremented after the op.
            with tf.control_dependencies([self.optimize]):
                 self.increment_global_step = tf.assign_add(self.global_step, 1)

    def assemble_graph(self):
        logger.info('Start graph assembling...')
        # Corrected order: _create_corpus_embed should be called by _build_mie if text is used.
        # _build_mie decides if it needs text embeddings, and if so, it should call the necessary text processing pipeline.
        # The original paper suggests TEL -> SRL (TSU, TMU, IFU) -> DRG -> TAP.
        # _build_embeds -> (_create_msg_embed_layer_in -> _create_msg_embed_layer) -> _create_corpus_embed (TSU)
        # This corpus_embed is then part of input x to VMD/DRG.

        # Current call order in _build_mie for non-'tech' variants:
        # _create_msg_embed_layer_in()
        # _create_msg_embed_layer()
        # _create_corpus_embed() -- this ensures self.corpus_embed is ready
        # Then self.x is constructed using self.corpus_embed and/or self.price
        
        # So, the graph assembly order should be:
        self._build_placeholders()
        self._build_embeds() # Creates self.word_embed (raw word embeddings)
        # _build_mie will internally call the text processing pipeline if needed:
        #  _create_msg_embed_layer_in -> _create_msg_embed_layer -> _create_corpus_embed
        self._build_mie()      # Creates self.x (input to VMD)
        self._build_vmd()      # Creates self.g, self.y, self.g_T, self.y_T (if no daily_att)
        self._build_temporal_att() # Creates final self.y_T if daily_att is used
        self._build_ata()      # Calculates self.loss
        self._create_optimizer() # Creates self.optimize and self.increment_global_step
            
    def _kl_lambda(self):
        def _nonzero_kl_lambda():
            if self.use_constant_kl_lambda:
                return self.constant_kl_lambda
            else:
                # global_step_float is already defined before this function call in the original code
                # Need to ensure it's passed or accessible if this is a standalone helper.
                # Assuming self.global_step is the tf.Variable for global step.
                current_global_step_float = tf.cast(self.global_step, tf.float32)
                return tf.minimum(self.kl_lambda_anneal_rate * current_global_step_float, 1.0)

        current_global_step_float = tf.cast(self.global_step, tf.float32)
        return tf.cond(current_global_step_float < self.kl_lambda_start_step, lambda: 0.0, _nonzero_kl_lambda)

    def _linear(self, args, output_size, activation=None, use_bias=True, use_bn=False):
        if type(args) not in (list, tuple):
            args = [args]

        # Dynamically determine shape to avoid issues with None dimensions from placeholders
        input_shape = tf.shape(args[0])
        leading_dims = input_shape[:-1]
        output_shape_dynamic = tf.concat([leading_dims, [output_size]], axis=0)

        sizes = [a.get_shape()[-1].value for a in args]
        total_arg_size = sum(sizes)
        current_scope = tf.get_variable_scope() # Renamed scope
        x_concat = args[0] if len(args) == 1 else tf.concat(args, -1) # Renamed x

        with tf.variable_scope(current_scope):
            weight = tf.get_variable('weight', [total_arg_size, output_size], dtype=tf.float32, initializer=self.initializer)
            res = tf.tensordot(x_concat, weight, axes=1)
            if use_bias:
                bias = tf.get_variable('bias', [output_size], dtype=tf.float32, initializer=self.bias_initializer)
                res = tf.nn.bias_add(res, bias)
        
        # Set static shape for rank, but use dynamic shape for dimensions that might be None
        # This helps with downstream shape inference where possible but relies on dynamic for runtime.
        static_shape = args[0].get_shape().as_list()[:-1] + [output_size]
        res.set_shape(static_shape)
        # res = tf.reshape(res, output_shape_dynamic) # Reshape with dynamic shape if needed, but tensordot might already give correct shape

        if use_bn:
            res = batch_norm(res, center=True, scale=True, decay=0.99, updates_collections=None,
                             is_training=self.is_training_phase, scope=current_scope) # Use current_scope for bn

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
        stddev_raw = self._linear(arg, self.z_size) # Renamed stddev to stddev_raw
        stddev_processed = tf.sqrt(tf.exp(stddev_raw)) # Renamed stddev
        
        epsilon = tf.random_normal(tf.shape(mean)) # Use shape of mean for epsilon

        z_sample = mean if is_prior else mean + tf.multiply(stddev_processed, epsilon) # Renamed z
        pdf_z_dist = ds.Normal(loc=mean, scale=stddev_processed) # Renamed pdf_z

        return z_sample, pdf_z_dist

