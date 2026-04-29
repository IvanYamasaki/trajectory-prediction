import typing
import tensorflow as tf
from .shape_checker import ShapeChecker

class AttentionAggregator(tf.keras.layers.Layer):
    def __init__(self, units):
        super(AttentionAggregator, self).__init__()
        self.units = units
        self.W1 = tf.keras.layers.Dense(units, use_bias=False)
        self.W2 = tf.keras.layers.Dense(units, use_bias=False)
        # O vetor V é o que permite ao modelo aprender a focar em frames específicos
        self.V = tf.keras.layers.Dense(1, use_bias=False)

    def call(self, query, value):
        pass

    def continue_call(self, w1_query, value, shape_checker):
        # w1_query: (batch, 1, units), value: (batch, 30, 256)
        w2_key = self.W2(value) # (batch, 30, units)
        
        # Score de Bahdanau: V * tanh(W1*q + W2*k)
        # O expand_dims garante que o cálculo seja feito para todos os 30 frames
        score = self.V(tf.nn.tanh(w1_query + w2_key)) # (batch, 30, 1)
        
        # Softmax no eixo 1 (frames do passado) para somar 1.0
        attention_weights = tf.nn.softmax(score, axis=1) # (batch, 30, 1)
        
        # Vetor de contexto: soma ponderada
        context_vector = tf.reduce_sum(attention_weights * value, axis=1, keepdims=True)

        return context_vector, attention_weights

class AttentionAggregator2D(AttentionAggregator):
    def __init__(self, units):
        super(AttentionAggregator2D, self).__init__(units)

    def call(self, query, value):
        shape_checker = ShapeChecker()
        w1_query = self.W1(query)
        w1_query = tf.keras.layers.Reshape((1, self.units))(w1_query)
        return self.continue_call(w1_query, value, shape_checker)

class AttentionAggregator3D(AttentionAggregator):
    def __init__(self, units):
        super(AttentionAggregator3D, self).__init__(units)

    def call(self, query, value):
        shape_checker = ShapeChecker()
        w1_query = self.W1(query)
        return self.continue_call(w1_query, value, shape_checker)

class BallAggregatorInputs(typing.NamedTuple):
    robot_seq: typing.Any
    ball_seq: typing.Any

class BallAggregator(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BallAggregator, self).__init__()
        self.attention = AttentionAggregator3D(units)
        self.dim = int(units*2/15)
        self.W1 = tf.keras.layers.Dense(self.dim)

    def call(self, inputs: BallAggregatorInputs, **kwargs):
        shape_checker = ShapeChecker()
        shape_checker(inputs.robot_seq, ('batch', 't', 'robot_dims'))
        context_vector, attention_weights = self.attention(
            query=inputs.ball_seq, value=inputs.robot_seq,
        )
        pos = self.W1(context_vector)
        pos = tf.keras.layers.Reshape((1, -1))(pos)
        return pos