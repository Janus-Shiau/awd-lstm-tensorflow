
'''
Copyright (c) 2019 [Jia-Yau Shiau]
Code work by Jia-Yau (jiayau.shiau@gmail.com).
--------------------------------------------------
Weight-dropped Long short-term memory unit (AWD-LSTM) recurrent network cell.
The implementation is based on:

    https://arxiv.org/abs/1708.02182
        
"Regularizing and Optimizing LSTM Language Models,"
Stephen Merity, Nitish Shirish Keskar, Richard Socher.

The code is modified from tensorflow source code:
    tf.nn.rnn_cell.LSTMCell
'''
import tensorflow as tf
from tensorflow.nn import dropout
from tensorflow.python.ops import array_ops, clip_ops, math_ops, nn_ops
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple

from variational_dropout import VariationalDropout

try:
    from tensorflow.nn.rnn_cell import LSTMCell
except:
    from tf.keras.layers import LSTMCell



class WeightDropLSTMCell(LSTMCell):
    """ Weight-dropped Long short-term memory unit (AWD-LSTM) recurrent network cell.
        The weight-drop implementation is based on:

            https://arxiv.org/abs/1708.02182
        
        "Regularizing and Optimizing LSTM Language Models,"
        Stephen Merity, Nitish Shirish Keskar, Richard Socher.

        The non-peephole implementation is based on:

            http://www.bioinf.jku.at/publications/older/2604.pdf

        S. Hochreiter and J. Schmidhuber.
        "Long Short-Term Memory". Neural Computation, 9(8):1735-1780, 1997.

        The peephole implementation is based on:

            https://research.google.com/pubs/archive/43905.pdf

        Hasim Sak, Andrew Senior, and Francoise Beaufays.
        "Long short-term memory recurrent neural network architectures for
        large scale acoustic modeling." INTERSPEECH, 2014.

        The code is modified from tensorflow source code:
            tf.nn.rnn_cell.LSTMCell
    """

    def __init__(self, 
                 num_units,
                 use_peepholes=False, 
                 cell_clip=None,
                 initializer=None, 
                 num_proj=None, 
                 proj_clip=None,
                 num_unit_shards=None, 
                 num_proj_shards=None,
                 forget_bias=1.0, 
                 state_is_tuple=True,
                 weight_drop_kr=1.0, 
                 use_vd=False, 
                 input_size=None, 
                 activation=None, 
                 reuse=None, 
                 name=None, 
                 dtype=None, 
                 **kwargs):
        """ Initialize the parameters for an LSTM cell.

            Args:
                num_units: int, The number of units in the LSTM cell.
                use_peepholes: bool, set True to enable diagonal/peephole connections.
                cell_clip: (optional) A float value, if provided the cell state is clipped
                    by this value prior to the cell output activation.
                initializer: (optional) The initializer to use for the weight and
                    projection matrices.
                num_proj: (optional) int, The output dimensionality for the projection
                    matrices.  If None, no projection is performed.
                proj_clip: (optional) A float value.  If `num_proj > 0` and `proj_clip` is
                    provided, then the projected values are clipped elementwise to within
                    `[-proj_clip, proj_clip]`.
                num_unit_shards: Deprecated, will be removed by Jan. 2017.
                    Use a variable_scope partitioner instead.
                num_proj_shards: Deprecated, will be removed by Jan. 2017.
                    Use a variable_scope partitioner instead.
                forget_bias: Biases of the forget gate are initialized by default to 1
                    in order to reduce the scale of forgetting at the beginning of
                    the training. Must set it manually to `0.0` when restoring from
                    CudnnLSTM trained checkpoints.
                state_is_tuple: If True, accepted and returned states are 2-tuples of
                    the `c_state` and `m_state`.  If False, they are concatenated
                    along the column axis.  This latter behavior will soon be deprecated.
                weight_drop_kr: The keep rate of weight drop-connect.
                use_vd: If true, using variational dropout on weight drop-connect, 
                    standard dropout otherwise.
                input_size: If use_vd is True, input_size (dimension of last channel) 
                    should be provided.
                activation: Activation function of the inner states.  Default: `tanh`. It
                    could also be string that is within Keras activation function names.
                reuse: (optional) Python boolean describing whether to reuse variables
                    in an existing scope.  If not `True`, and the existing scope already has
                    the given variables, an error is raised.
                name: String, the name of the layer. Layers with the same name will
                    share weights, but to avoid mistakes we require reuse=True in such
                    cases.
                dtype: Default dtype of the layer (default of `None` means use the type
                    of the first input). Required when `build` is called before `call`.
                **kwargs: Dict, keyword named properties for common layer attributes, like
                    `trainable` etc when constructing the cell from configs of get_config().

        """
        super(WeightDropLSTMCell, self).__init__(num_units=num_units, **kwargs)

        if use_vd and input_size is None:
            raise KeyError("input_size should be provided if use_vd is True!")

        self._weight_drop_kr   = weight_drop_kr
        self._use_vd           = use_vd
        self._input_size       = input_size

        if self._use_vd and not self._weight_drop_kr == 1.0:
            h_depth      = self._num_units if self._num_proj is None else self._num_proj
            kernel_shape = [input_size + h_depth, 4 * self._num_units]
            self.vd = VariationalDropout(
                input_shape=kernel_shape,
                keep_prob=self._weight_drop_kr)
        

    def call(self, inputs, state):
        """Run one step of LSTM.
            Args:
                inputs: input Tensor, 2D, `[batch, num_units].
                state: if `state_is_tuple` is False, this must be a state Tensor,
                `2-D, [batch, state_size]`.  If `state_is_tuple` is True, this must be a
                tuple of state Tensors, both `2-D`, with column sizes `c_state` and
                `m_state`.

            Returns:
                A tuple containing:

                - A `2-D, [batch, output_dim]`, Tensor representing the output of the
                LSTM after reading `inputs` when previous state was `state`.
                Here output_dim is:
                    num_proj if num_proj was set,
                    num_units otherwise.
                - Tensor(s) representing the new state of LSTM after reading `inputs` when
                the previous state was `state`.  Same type and shape(s) as `state`.

            Raises:
                ValueError: If input size cannot be inferred from inputs via
                static shape inference.
        """

        num_proj = self._num_units if self._num_proj is None else self._num_proj
        sigmoid = math_ops.sigmoid

        if self._state_is_tuple:
            (c_prev, m_prev) = state
        else:
            c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
            m_prev = array_ops.slice(state, [0, self._num_units], [-1, num_proj])

        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        if not self._weight_drop_kr == 1.0:
            if self._use_vd:
                with tf.variable_scope('var_weight_drop_connect'):
                    drop_kernel = self.vd(self._kernel)

            else:
                with tf.variable_scope('weight_drop_connect'):
                    w1, w2 = self._kernel.get_shape().as_list()
                    drop_kernel = tf.reshape(self._kernel, [-1])
                    drop_kernel = dropout(drop_kernel, keep_prob=self._weight_drop_kr)
                    drop_kernel = tf.reshape(drop_kernel, [w1, w2])

            lstm_matrix = math_ops.matmul(
                array_ops.concat([inputs, m_prev], 1), drop_kernel)
        else:
            lstm_matrix = math_ops.matmul(
                array_ops.concat([inputs, m_prev], 1), self._kernel)

        lstm_matrix = nn_ops.bias_add(lstm_matrix, self._bias)

        i, j, f, o = array_ops.split(
            value=lstm_matrix, num_or_size_splits=4, axis=1)

        if self._use_peepholes:
            c = (sigmoid(f + self._forget_bias + self._w_f_diag * c_prev) * c_prev +
                sigmoid(i + self._w_i_diag * c_prev) * self._activation(j))
        else:
            c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) *
                self._activation(j))

        if self._cell_clip is not None:
            c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)
        if self._use_peepholes:
            m = sigmoid(o + self._w_o_diag * c) * self._activation(c)
        else:
            m = sigmoid(o) * self._activation(c)

        if self._num_proj is not None:
            m = math_ops.matmul(m, self._proj_kernel)

        if self._proj_clip is not None:
            m = clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)

        new_state = (LSTMStateTuple(c, m) if self._state_is_tuple else
                    array_ops.concat([c, m], 1))

        return m, new_state

    
    def get_vd_update_op(self):
        if self._use_vd and not self._weight_drop_kr == 1.0:
            return self.vd.get_update_mask_op()
        else:
            print("Variational dropout is not used!!!")
            return []


    def get_config(self):
        config = {
            "weight_drop_kr": self._weight_drop_kr,
            "use_vd": self._use_vd,
            "input_size": self._input_size
        }
        base_config = super(WeightDropLSTMCell, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))



