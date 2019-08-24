'''
Copyright (c) 2019 [Jia-Yau Shiau]
Code work by Jia-Yau (jiayau.shiau@gmail.com).
--------------------------------------------------
Simple example for awd-lstm.
'''
import numpy as np
import tensorflow as tf

from weight_drop_lstm import WeightDropLSTMCell

INPUT_SIZE = 10
BATCH_SIZE = 1
CELL_NUM = 5

WEIGHT_DP_KR = 0.9


def test_with_sess_run(lstm_cell, init_states, x):
    """ Test awd-lstm with sess.run() """
    x_out, _ = lstm_cell(x, init_states)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        
        for i in range(10):
            if np.mod(i, 5) == 0:
                sess.run(lstm_cell.get_vd_update_op())

            print(sess.run(x_out))


def test_with_control_dependencies(lstm_cell, init_states, x):
    """ Test awd-lstm with control_dependencies and dynamic_rnn. """
    x = tf.broadcast_to(x, [BATCH_SIZE, 5, INPUT_SIZE])

    with tf.control_dependencies(lstm_cell.get_vd_update_op()):
        outputs, _ = tf.nn.dynamic_rnn(lstm_cell, x, initial_state=init_states)

    """ You can validate the variational dropout kernel by:
        1. Goto variational_dropout.py and find get_update_mask_op in VariationalDropout.
        2. Add "binary_tensor = tf.Print(binary_tensor, [binary_tensor])" 
           right after "binary_tensor = self._get_binary_mask(self.input_shape, self.dtype)".
    """

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        
        for i in range(5):
            print("running {}/5".format(i+1))
            results = sess.run(outputs)

            print (results)



if __name__ == "__main__":
    x  = np.arange(BATCH_SIZE*INPUT_SIZE, dtype=np.float32).reshape(BATCH_SIZE,INPUT_SIZE)
    x  = tf.convert_to_tensor(x)

    lstm_cell = WeightDropLSTMCell(CELL_NUM, 
        weight_drop_kr=WEIGHT_DP_KR, use_vd=True, input_size=INPUT_SIZE)
    init_states = lstm_cell.zero_state(BATCH_SIZE, dtype=tf.float32)


    # Use awd-lstm with sess.run()
    test_with_sess_run(lstm_cell, init_states, x)

    # Use awd-lstm with tf.control_dependencies and dynamic_rnn
    test_with_control_dependencies(lstm_cell, init_states, x)
    