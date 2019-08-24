'''
Copyright (c) 2019 [Jia-Yau Shiau]
Code work by Jia-Yau (jiayau.shiau@gmail.com).
--------------------------------------------------
Simple example for variational dropout.
'''
import numpy as np
import tensorflow as tf

from variational_dropout import VariationalDropout

def test_with_sess_run(vd, x):
    """ Test variational dropout with sess.run() """
    vd_call = vd(x)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        
        print("---------------Running with sess.run---------------")
        for i in range(10):
            if np.mod(i, 5) == 0:
                sess.run(vd.get_update_mask_op())

            print(sess.run(vd_call))


def test_with_control_dependencies(vd, x):
    """ Test variational dropout with control_dependencies. """
    step = tf.constant(0, dtype=tf.int32)
    results_array = tf.TensorArray(dtype=tf.float32, size=5)

    def main_loop(step, array):
        vd_call = vd(x)
        array = array.write(step, vd_call)

        return step+1, array
    
    with tf.control_dependencies(vd.get_update_mask_op()):
        step, results_array = tf.while_loop(
            cond=lambda step, _: step < 5,
            body=main_loop,
            loop_vars=(step, results_array))

    results = results_array.stack()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        print ("---------------Running with tf.control_dependencies---------------")
        for i in range(5):
            print ("running {}/5".format(i+1))
            print (sess.run([results]))


if __name__ == '__main__':
    x  = np.arange(5, dtype=np.float32)
    x  = tf.convert_to_tensor(x)

    vd = VariationalDropout(input_shape=[5], keep_prob=0.5)

    # Use variational dropout with sess.run()
    test_with_sess_run(vd, x)

    # Use variational dropout with tf.control_dependencies
    test_with_control_dependencies(vd, x)


    """ Some failure case with while_loop. Check it out if needed.

        step = tf.constant(0, dtype=tf.int32)
        results_array = tf.TensorArray(dtype=tf.float32, size=10)

        def main_loop(step, array):
            vd_call = vd(x)
            array = array.write(step, vd_call)
            step = tf.Print(step, [step])

            return step+1, array
        
        with tf.control_dependencies(update_op):
            step, results_array = tf.while_loop(
                cond=lambda step, _: step < 5,
                body=main_loop,
                loop_vars=(step, results_array))



        with tf.control_dependencies(update_op):
            step, results_array2 = tf.while_loop(
                cond=lambda step, _: step < 10,
                body=main_loop,
                loop_vars=(step, results_array))

        results = results_array2.stack()

        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            print (sess.run([results]))
    """
