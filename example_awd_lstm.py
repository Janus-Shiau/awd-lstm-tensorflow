'''
Copyright (c) 2019 [Jia-Yau Shiau]
Code work by Jia-Yau (jiayau.shiau@gmail.com).
--------------------------------------------------
Simple example for awd-lstm.
'''
import os
import numpy as np
import tensorflow as tf

from weight_drop_lstm import WeightDropLSTMCell

INPUT_SIZE = 10
BATCH_SIZE = 1
CELL_NUM = 5

WEIGHT_DP_KR = 0.9




def write_graph(sess, log_dir):
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    
    writer = tf.summary.FileWriter(log_dir, sess.graph)
    writer.flush()

    print ('saving graph to {}...'.format(log_dir))


def test_with_simple_awd(x):
    """ Test awd-lstm with simple drop connect """  
    tf.reset_default_graph()
    lstm_cell = WeightDropLSTMCell(CELL_NUM, 
        weight_drop_kr=WEIGHT_DP_KR, use_vd=False)
    init_states = lstm_cell.zero_state(BATCH_SIZE, dtype=tf.float32)

    x = tf.convert_to_tensor(x)
    x_out, _ = lstm_cell(x, init_states)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        
        for i in range(10):
            print(sess.run(x_out))
        
        write_graph(sess=sess, log_dir="./logs/simple_dropout")


def test_with_sess_run(x):
    """ Test awd-lstm with sess.run() """
    tf.reset_default_graph()
    lstm_cell = WeightDropLSTMCell(CELL_NUM, 
        weight_drop_kr=WEIGHT_DP_KR, use_vd=True, input_size=INPUT_SIZE)
    init_states = lstm_cell.zero_state(BATCH_SIZE, dtype=tf.float32)

    x  = tf.convert_to_tensor(x)
    x_out, _ = lstm_cell(x, init_states)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        
        for i in range(10):
            if np.mod(i, 5) == 0:
                sess.run(lstm_cell.get_vd_update_op())

            print(sess.run(x_out))
        
        write_graph(sess=sess, log_dir="./logs/variational_dp")



def test_with_control_dependencies(x):
    """ Test awd-lstm with control_dependencies and dynamic_rnn. """
    tf.reset_default_graph()
    lstm_cell = WeightDropLSTMCell(CELL_NUM, 
            weight_drop_kr=WEIGHT_DP_KR, use_vd=True, input_size=INPUT_SIZE)
    init_states = lstm_cell.zero_state(BATCH_SIZE, dtype=tf.float32)

    x  = tf.convert_to_tensor(x)
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
        
        write_graph(sess=sess, log_dir="./logs/control_dependencies")



def test_with_integral_quantization(x, is_train=True):
    """ Test fully integral quantization AWD-LSTM. """
    tf.reset_default_graph()
    lstm_cell = WeightDropLSTMCell(CELL_NUM, 
        weight_drop_kr=WEIGHT_DP_KR, use_vd=False,
        is_quant=True, is_train=is_train)
    init_states = lstm_cell.zero_state(BATCH_SIZE, dtype=tf.float32)

    x  = tf.convert_to_tensor(x)
    x = tf.broadcast_to(x, [BATCH_SIZE, 5, INPUT_SIZE])
    outputs, _ = tf.nn.dynamic_rnn(lstm_cell, x, initial_state=init_states)

    with tf.Session() as sess:
        if is_train:
            """
                Call the training rewrite which rewrites the graph in-place with
                FakeQuantization nodes and folds batchnorm for training. It is often
                needed to finetune a floating point model for quantization with this
                training tool. When training from scratch, quant_delay can be used to
                activate quantization after training to convergence with the float
                graph, effectively finetuning the model.
            """
            tf.contrib.quantize.create_training_graph(sess.graph, quant_delay=0)
        else:
            """
                Call the eval rewrite which rewrites the graph in-place with
                FakeQuantization nodes and fold batchnorm for eval.
            """
        tf.contrib.quantize.create_eval_graph(sess.graph)

        tf.global_variables_initializer().run()
        
        for i in range(5):
            print("running {}/5".format(i+1))
            results = sess.run(outputs)

            print (results)

        write_graph(sess=sess, log_dir="./logs/quantization")
        
        


if __name__ == "__main__":
    x = np.arange(BATCH_SIZE*INPUT_SIZE, dtype=np.float32).reshape(BATCH_SIZE,INPUT_SIZE)
    
    ### Run with simple drop-connect LSTM ###
    test_with_simple_awd(x)

    
    ### Run with variational drop-connect LSTM ###
    # Use awd-lstm with sess.run()
    test_with_sess_run(x)

    # Use awd-lstm with tf.control_dependencies and dynamic_rnn
    test_with_control_dependencies(x)
    

    ### Run with fully integral quantization LSTM ###
    test_with_integral_quantization(x)