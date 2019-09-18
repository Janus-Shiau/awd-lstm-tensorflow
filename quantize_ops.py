
'''
Copyright (c) 2019 [Jia-Yau Shiau]
Code work by 
1. Jia-Yau (jiayau.shiau@gmail.com).
2. Peter Huang (peter124574@gmail.com)
--------------------------------------------------
Quantization operations and fully integral calculation for Weight-Drop LSTM cell.
This implementation is based on:

    https://arxiv.org/pdf/1712.05877.pdf

"Quantization and Training of Neural Networks for 
 Efficient Integer-Arithmetic-Only Inference"
Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, 
Matthew Tang, Andrew Howard, Hartwig Adam, Dmitry Kalenichenko

The code is modified from tensorflow source code:
    tf.quantization.quantize
'''

import re

import tensorflow as tf
from tensorflow.contrib import graph_editor
from tensorflow.contrib.quantize.python import common, quant_ops
from tensorflow.python.framework import dtypes, ops
from tensorflow.python.ops import clip_ops, control_flow_ops, math_ops
from tensorflow.python.platform import tf_logging as logging


def hard_sigmoid(x, name="hard_sigmoid"):
    with ops.name_scope(name, "HardSigmoid", [x]) as name:
        x = tf.add (tf.multiply(0.2, x), 0.5)
        x = clip_ops.clip_by_value(x, 0.0, 1.0)

    return x

def insert_quant_ops(ops_dict, 
                     graph=None,
                     is_train=True, 
                     weight_bits=8, 
                     activation_bits=8, 
                     ema_decay=0.999, 
                     quant_delay=0, 
                     vars_collection=None, 
                     scope=None):
    if graph is None:
        graph = tf.get_default_graph()
    if vars_collection is None:
        vars_collection = tf.GraphKeys.GLOBAL_VARIABLES

    matrix_context = _GetContextFromOp(ops_dict['lstm_matrix'])
    producer = graph.get_operation_by_name(matrix_context + '/weights')
    consumers = producer.outputs[0].consumers()
    InsertQuantOp(context=matrix_context,
                    name='weights_quant',
                    producer=producer,
                    consumers=consumers,
                    is_training=is_train,
                    moving_avg=False,
                    ema_decay=ema_decay,
                    quant_delay=quant_delay,
                    narrow_range=True,
                    vars_collection=vars_collection,
                    bits=weight_bits,
                    consumer_scope=matrix_context)
    
    producer = graph.get_operation_by_name(matrix_context + '/BiasAdd')
    consumers = producer.outputs[0].consumers()
    InsertQuantOp(context=matrix_context,
                    name='act_quant',
                    producer=producer,
                    consumers=consumers,
                    is_training=is_train,
                    moving_avg=True,
                    ema_decay=ema_decay,
                    quant_delay=quant_delay,
                    vars_collection=vars_collection,
                    bits=activation_bits,
                    init_min=0.0,
                    producer_scope=matrix_context)
        
    post_activation_bypass_context = _GetContextFromOp(ops_dict['i'])
    producer_list = [output.consumers()[0] for output in consumers[0].outputs]

    try:
        matrix_context = _GetContextFromOp(ops_dict['proj_kernel'])
        producer = graph.get_operation_by_name(matrix_context + '/weights')
        consumers = producer.outputs[0].consumers()
        InsertQuantOp(context=matrix_context,
                    name='weights_quant',
                    producer=producer,
                    consumers=consumers,
                    is_training=is_train,
                    moving_avg=False,
                    ema_decay=ema_decay,
                    quant_delay=quant_delay,
                    narrow_range=True,
                    vars_collection=vars_collection,
                    bits=weight_bits,
                    consumer_scope=matrix_context)
    except KeyError:
        pass

    stop_list = [post_activation_bypass_context + '/end_m',
                post_activation_bypass_context + '/end_c']

    while(producer_list != []):
        producer_list_new = []
        for producer in producer_list:
            consumers = producer.outputs[0].consumers()


            cond1 = ('hard_sigmoid' in producer.name) and ( not producer.name.endswith('clip_by_value'))
            cond2 = ('Relu' in producer.name.split('/')[-1])
            cond3 = ('hard_sigmoid' not in producer.name) and \
                    (producer.name.endswith('clip_by_value/Minimum'))
            cond4 = producer.name.endswith('add') or producer.name.endswith('add_1')
            if not (cond1 or cond2 or cond3 or cond4):
                InsertQuantOp(context=post_activation_bypass_context,
                            name='post_activation',
                            producer=producer,
                            consumers=consumers,
                            is_training=is_train,
                            moving_avg=True,
                            ema_decay=ema_decay,
                            quant_delay=quant_delay,
                            vars_collection=vars_collection,
                            bits=activation_bits,
                            producer_scope=scope)
                stop_list.append(producer.name)
                
            for consumer in consumers:
                if consumer not in producer_list_new and consumer.name not in stop_list:
                    producer_list_new.append(consumer)

        producer_list = producer_list_new



def InsertQuantOp(context,
                  name,
                  producer,
                  consumers,
                  is_training,
                  moving_avg=True,
                  init_min=-6.0,
                  init_max=6.0,
                  bits=8,
                  ema_decay=0.999,
                  quant_delay=None,
                  vars_collection=ops.GraphKeys.GLOBAL_VARIABLES,
                  narrow_range=False,
                  producer_scope=None,
                  consumer_scope=None):
    """Inserts a quant op between a producer op and (multiple) consumer ops.
        Args:
        context: Context where producer and consumer operations are nested.
        name: Name for the new quantization op within the context.
        producer: Producer operation of the pairs where quantization will be
            inserted.
        consumers: Consumer operations of the pairs.
        is_training: Whether quantizing training graph or eval graph.
        moving_avg: Specifies whether to use exponential moving average or just
            the last value seen.
        init_min: Starting minimum value for the new quantization op.
        init_max: Starting maximum value for the new quantization op.
        bits: Number of bits to use for quantization, must be between 2 and 8.
        ema_decay: (Optional) Float, EMA decay parameter.  EMA is used to update
            quantization intervals for quantizing activations (see here about EMA:
            https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average).
        quant_delay: (Optional, default None) Int, count of global steps for which
            to delay quantization.  This helps weights stabilize at the start of
            training.
        vars_collection: (Optional) Collection where to store the variables for
            quantization interval ends.
        narrow_range: Whether to use the narrow quantization range
            [1; 2^bits - 1] or wide range [0; 2^bits - 1].
        producer_scope: The restriction of producer scope. If not None, the new op
            will be inserted only when the producer is in this scope.
        consumer_scope: The restriction of producer scope. If not None, the new op
            will be inserted only when all the consumers are in this scope.
        Raises:
        ValueError: When producer operation is not directly connected to the
            consumer operation.
    """
    if producer_scope and not producer.name.startswith(producer_scope):
        logging.info(
            'InsertQuantOp ignores context="%s" name="%s" '
            'because producer "%s" is not in scope "%s"',
            context, name, producer.name, producer_scope)
        return

    if consumer_scope:
        consumers_in_scope = []
        for consumer in consumers:
            if consumer.name.startswith(consumer_scope):
                consumers_in_scope.append(consumer)
            else:
                logging.info(
                    'InsertQuantOp context="%s" name="%s" ignores '
                    'consumer "%s" because it is not in scope "%s"',
                    context, name, consumer.name, consumer_scope)
                return
        consumers = consumers_in_scope

    name_prefix = _AddContextToName(context, name)
    # This is needed on TPU where name_scope == 'TPUReplicate/loop', and
    # name_prefix starts with 'TPUReplicate/loop/'; without dropping it
    # variables are created as TPUReplicate/loop/TPUReplicate/loop/..., which
    # breaks things later.
    name_scope = ops.get_name_scope()
    if name_scope:
        name_prefix = common.DropStringPrefix(name_prefix, name_scope + '/')

    inputs = producer.outputs[0]
    # Prevent ops from being quantized multiple times. Bypass ops can sometimes
    # overlap between multiple matches, so we need to ensure that we don't
    # add duplicate FakeQuant operations.
    fake_quant_ops = set([
        'FakeQuantWithMinMaxVars',
        'FakeQuantWithMinMaxArgs'
    ])
    if fake_quant_ops.intersection(set([c.type for c in inputs.consumers()])):
        return

    if moving_avg:
        quant = (
            quant_ops.MovingAvgQuantize(
                inputs,
                init_min=init_min,
                init_max=init_max,
                ema_decay=ema_decay,
                is_training=is_training,
                num_bits=bits,
                narrow_range=narrow_range,
                vars_collection=vars_collection,
                name_prefix=name_prefix))
    else:
        quant = (
            quant_ops.LastValueQuantize(
                inputs,
                init_min=init_min,
                init_max=init_max,
                is_training=is_training,
                num_bits=bits,
                narrow_range=narrow_range,
                vars_collection=vars_collection,
                name_prefix=name_prefix))

    if quant_delay and quant_delay > 0:
        # activate_quant = math_ops.greater_equal(
        #     common.CreateOrGetQuantizationStep(),
        #     quant_delay,
        #     name=name_prefix + '/activate_quant')
        activate_quant = math_ops.greater_equal(
            tf.get_default_graph().get_tensor_by_name('global_step:0'),
            quant_delay,
            name=name_prefix + '/activate_quant')
        quant = control_flow_ops.cond(
            activate_quant,
            lambda: quant,
            lambda: inputs,
            name=name_prefix + '/delayed_quant')

    if consumers:
        tensors_modified_count = graph_editor.reroute_ts(
            [quant], [inputs], can_modify=consumers)
        # Some operations can have multiple output tensors going to the same
        # consumer. Since consumers is a set, we need to ensure that
        # tensors_modified_count is greater than or equal to the length of the set
        # of consumers.
        if tensors_modified_count < len(consumers):
            raise ValueError('No inputs quantized for ops: [%s]' % ', '.join(
                [consumer.name for consumer in consumers]))


def _GetContextFromOp(op):
    """Gets the root context name from the op name."""
    context_re = re.search(r'^(.*)/([^/]+)', op.name)
    if context_re:
        return context_re.group(1)
        
    return ''


def _AddContextToName(context, name):
    """Adds the context to the name if it exists."""
    if not context:
        return name
    return context + '/' + name
