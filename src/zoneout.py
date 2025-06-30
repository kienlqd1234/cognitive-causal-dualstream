import tensorflow as tf
from tensorflow.python.ops.rnn_cell import RNNCell

class ZoneoutWrapper(RNNCell):
    """RNNWrapper that applies zoneout.
    This is a vanilla implementation of zoneout of
    https://arxiv.org/abs/1606.01305
    """

    def __init__(self, cell, zoneout_prob, is_training=True, seed=None):
        """Create a cell with zoneout.
        Args:
            cell: an RNNCell to wrap.
            zoneout_prob: a Tensor or float or tuple of floats between 0 and 1 specifying the zoneout
                probability. If a tuple, it must be for an LSTM cell and is (zoneout_cell, zoneout_hidden).
            is_training: a bool Tensor, specifying if we are in training mode.
            seed: (optional) integer, the randomness seed.
        Raises:
            TypeError: if cell is not an RNNCell.
            ValueError: if zoneout_prob is not between 0 and 1.
        """
        if not isinstance(cell, RNNCell):
            raise TypeError("The parameter cell is not an RNNCell.")
        
        if isinstance(zoneout_prob, tuple):
            if len(zoneout_prob) != 2:
                 raise ValueError("zoneout_prob must be a tuple of length 2.")
            if not (zoneout_prob[0] >= 0.0 and zoneout_prob[0] <= 1.0 and \
                    zoneout_prob[1] >= 0.0 and zoneout_prob[1] <= 1.0):
                raise ValueError("zoneout_prob probabilities must be between 0 and 1.")
        elif isinstance(zoneout_prob, float) and not (zoneout_prob >= 0.0 and zoneout_prob <= 1.0):
            raise ValueError("Parameter zoneout_prob must be between 0 and 1: %f" % zoneout_prob)
        
        self._cell = cell
        self._zoneout_prob = zoneout_prob
        self._is_training = is_training
        self._seed = seed

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        output, new_state = self._cell(inputs, state, scope)

        is_lstm = isinstance(self.state_size, tuple)
        
        if is_lstm:
            # This is for LSTM cells, which have a state tuple (c, h)
            if not isinstance(self._zoneout_prob, tuple):
                raise TypeError("zoneout_prob must be a tuple for LSTMCells.")
            new_c, new_h = new_state
            old_c, old_h = state
            zoneout_prob_c, zoneout_prob_h = self._zoneout_prob

            final_c = self._apply_zoneout(old_c, new_c, zoneout_prob_c)
            final_h = self._apply_zoneout(old_h, new_h, zoneout_prob_h)
            
            final_state = tf.nn.rnn_cell.LSTMStateTuple(final_c, final_h)

        else:
            # This is for GRU cells, which have a single state tensor
            if not isinstance(self._zoneout_prob, float):
                raise TypeError("zoneout_prob must be a float for GRUCells.")
            final_state = self._apply_zoneout(state, new_state, self._zoneout_prob)

        return output, final_state

    def _apply_zoneout(self, old_state, new_state, zoneout_prob):
        """Apply zoneout to a single state tensor."""
        
        def zoned_out_state():
            # The random tensor is of the same shape as the state.
            random_tensor = 1 - zoneout_prob
            random_tensor += tf.random_uniform(
                tf.shape(old_state), seed=self._seed, dtype=random_tensor.dtype)
            # 0. if random_tensor < zoneout_prob (zoneout)
            # 1. if random_tensor >= zoneout_prob (no zoneout)
            binary_tensor = tf.floor(random_tensor)
            
            zoned_state = (new_state - old_state) * binary_tensor + old_state
            return zoned_state

        def passthrough_state():
            return (1 - zoneout_prob) * new_state + zoneout_prob * old_state

        # We apply zoneout only during training.
        return tf.cond(self._is_training, zoned_out_state, passthrough_state) 