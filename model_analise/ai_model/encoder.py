import typing
import tensorflow as tf
from .shape_checker import ShapeChecker

class EncoderOutput(typing.NamedTuple):
    sequence: typing.Any
    state_h: typing.Any
    state_c: typing.Any
    state_hf: typing.Any
    state_cf: typing.Any
    state_hb: typing.Any
    state_cb: typing.Any

class Encoder(tf.keras.layers.Layer):
    def __init__(self, units):
        super(Encoder, self).__init__()
        self.units = units
        self.lstm = tf.keras.layers.LSTM(
            units,
            return_sequences=True,
            return_state=True
        )        
        self.built = True

    def call(self, sequence, state=None):
        shape_checker = ShapeChecker()
        shape_checker(sequence, ('batch', 'look_back', 'coordinates'))

        output, h, c = self.lstm(sequence)

        # Retornamos os estados concatenados (256) para a Atenção 
        # e os individuais (128) para o Predictor usar na inicialização
        return EncoderOutput(
            sequence=output,
            state_h=h,
            state_c=c,
            state_hf=h,
            state_cf=c,
            state_hb=None,
            state_cb=None,
        )

class BallEncoderInput(typing.NamedTuple):
    sequence: typing.Any
    mask: typing.Any

class BallEncoderOutput(typing.NamedTuple):
    sequence: typing.Any
    state_h: typing.Any
    state_c: typing.Any

class BallEncoder(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BallEncoder, self).__init__()
        self.units = units
        self.fwd = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)
        self.bwd = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True, go_backwards=True)
        self.bi_lstm = tf.keras.layers.Bidirectional(self.fwd, backward_layer=self.bwd)
        self.built = True

    def call(self, inputs: BallEncoderInput):
        # A máscara é usada para ignorar frames onde a bola não foi detectada [cite: 58, 59]
        output, state_hf, state_cf, state_hb, state_cb = self.bi_lstm(inputs.sequence, mask=inputs.mask)
        
        return BallEncoderOutput(
            sequence=output,
            state_h=tf.concat([state_hf, state_hb], axis=-1),
            state_c=tf.concat([state_cf, state_cb], axis=-1)
        )