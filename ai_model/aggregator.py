import typing

import tensorflow as tf
from .shape_checker import ShapeChecker


class AttentionAggregator(tf.keras.layers.Layer):
    def __init__(self, units):
        super(AttentionAggregator, self).__init__()
        self.units = units

        self.W1 = tf.keras.layers.Dense(units, use_bias=False)
        self.W2 = tf.keras.layers.Dense(units, use_bias=False)
        self.attention = tf.keras.layers.AdditiveAttention()

    def build(self, input_shape):
         super().build(input_shape)

    def call(self, query, value):

         raise NotImplementedError("Subclasses must implement the call method.")


    def continue_call(self, w1_query, value, shape_checker):

        w2_key = self.W2(value)
        # shape_checker(w2_key, ('batch', 's', 'attn_units'))

        context_vector, attention_weights = self.attention(
            inputs=[w1_query, value, w2_key],
            return_attention_scores=True,
        )

        # shape_checker(context_vector, ('batch', 't', 'value_units'))
        # shape_checker(attention_weights, ('batch', 't', 's'))

        return context_vector, attention_weights


class AttentionAggregator2D(AttentionAggregator):
    def __init__(self, units):
        super(AttentionAggregator2D, self).__init__(units)

    def build(self, input_shape):
         super().build(input_shape)

    def call(self, query, value):
        shape_checker = ShapeChecker()
        # shape_checker(query, ('batch', 'query_units'))

        w1_query = self.W1(query)
        w1_query = tf.keras.layers.Reshape((1, self.units))(w1_query)
        # shape_checker(w1_query, ('batch', 't', 'attn_units'))

        return self.continue_call(w1_query, value, shape_checker)


class AttentionAggregator3D(AttentionAggregator):
    def __init__(self, units):
        super(AttentionAggregator3D, self).__init__(units)

    def build(self, input_shape):
         super().build(input_shape)

    def call(self, query, value):
        shape_checker = ShapeChecker()
        # shape_checker(query, ('batch', 't', 'query_units'))

        w1_query = self.W1(query)
        return self.continue_call(w1_query, value, shape_checker)


class BallAggregatorInputs(typing.NamedTuple):
    robot_seq: typing.Any
    ball_seq: typing.Any


class BallAggregator(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BallAggregator, self).__init__()
        self.attention = AttentionAggregator3D(units)
        self.output_dim = 2 # Dimensão de saída fixa
        self.W1 = tf.keras.layers.Dense(self.output_dim)
        self.reshape = tf.keras.layers.Reshape((1, self.output_dim))
        # Adicionar uma camada de redução temporal (GlobalAveragePooling)
        self.pool = tf.keras.layers.GlobalAveragePooling1D()


    def build(self, input_shape):
         super().build(input_shape)

    def call(self, inputs: BallAggregatorInputs, **kwargs):
        shape_checker = ShapeChecker()
        
        # O AttentionAggregator3D retorna (batch, look_back, value_units)
        context_vector, attention_weights = self.attention(
            query=inputs.ball_seq, value=inputs.robot_seq,
        )

        # 1. Aplicar redução temporal para obter (batch, value_units)
        # Isso resolve a inconsistência de forma que o reshape estava encontrando
        reduced_context = self.pool(context_vector)

        # 2. Aplicar Dense para obter (batch, 2)
        pos = self.W1(reduced_context) # Forma: (batch, 2)

        # 3. Aplicar Reshape (batch, 1, 2)
        pos = self.reshape(pos)

        return pos