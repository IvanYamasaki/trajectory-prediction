from .decoder import DecoderInitializer, Decoder, DecoderInput, DecoderState, DecoderInitializerInput
from .aggregator import BallAggregator, BallAggregatorInputs
import tensorflow as tf
from .encoder import Encoder, BallEncoder, BallEncoderInput
from .decoder import DecoderInitializer, Decoder, DecoderInput, DecoderState, DecoderInitializerInput
from .shape_checker import ShapeChecker


class Predictor(tf.keras.Model):

    def __init__(self, units, look_back, look_forth, result_dims, use_tf_function=True, forcing=True):
        super(Predictor, self).__init__()
        self.look_forth = look_forth
        self.look_back = look_back
        self.result_dims = result_dims
        self.encoder = Encoder(units)
        self.decoder_init = DecoderInitializer(units)
        self.decoder = Decoder(units, result_dims)
        self.use_tf_function = use_tf_function
        self.shape_checker = ShapeChecker()
        self.forcing = forcing
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.teacher_forcing_prob = tf.Variable(1.0, trainable=False, dtype=tf.float32)

    def _decoder_step(self, decoder_output, target_pos):

        target_pos = tf.reshape(target_pos, (-1, self.result_dims))
        dec_sate = DecoderState(state_h=decoder_output.state_h, state_c=decoder_output.state_c)

        return dec_sate, tf.reshape(decoder_output.sequence, (-1, 1, self.result_dims))

    def _train_step(self, data):
        pass

    @tf.function
    def _tf_train_step(self, data):
        return self._train_step(data)

    def train_step(self, data):
        self.shape_checker = ShapeChecker()
        x, y, _ = tf.keras.utils.unpack_x_y_sample_weight(data)

        if self.use_tf_function:
            return self._tf_train_step((x, y))
        else:
            return self._train_step((x, y))

    def save_model(self, name):
        self.save_weights(name + '.weights.h5')

    def load_model(self, name):
        self.load_weights(name + '.weights.h5')

    def _forecast(self, input_seq):
        [robot_seq, ball_seq, ball_mask] = input_seq
        robot_enc_output = self.encoder(robot_seq)
        ball_enc_output = self.ball_encoder(
            BallEncoderInput(sequence=ball_seq, mask=ball_mask)
        )

        enc_context = self.ball_aggregator(
            BallAggregatorInputs(
                ball_seq=ball_enc_output.sequence,
                robot_seq=robot_enc_output.sequence,
            )
        )

        h_init = tf.concat([robot_enc_output.state_hf, robot_enc_output.state_hb], axis=-1)
        c_init = tf.concat([robot_enc_output.state_cf, robot_enc_output.state_cb], axis=-1)

        new_tokens = robot_seq[:, (self.look_back - 1):self.look_back, 2:4]
        dec_state = self.decoder_init(DecoderInitializerInput(state_h=h_init, state_c=c_init))

        result_tokens = []
        for _ in range(self.look_forth):
            dec_input = DecoderInput(
                new_tokens=tf.concat([new_tokens, enc_context], axis=2),
                enc_output=robot_enc_output.sequence,
            )
            dec_output = self.decoder(dec_input, dec_state)
            dec_state = DecoderState(state_h=dec_output.state_h, state_c=dec_output.state_c)
            new_tokens = dec_output.sequence
            result_tokens.append(new_tokens)

        return tf.concat(result_tokens, axis=1)

    @tf.function
    def _tf_forecast(self, input_seq):
        return self._forecast(input_seq)

    def forecast(self, input_seq):
        if self.use_tf_function:
            return self._tf_forecast(input_seq)
        else:
            return self._forecast(input_seq)

    def call(self, inputs, training=None, mask=None):
        if self.use_tf_function:
            return self._tf_forecast(inputs)

        return self._forecast(inputs)
    
    @property
    def metrics(self):
        return [self.loss_tracker]

class RobotOnlyPredictor(Predictor):
    def __init__(self, units, look_back, look_forth, result_dims, use_tf_function=True, forcing=True):
        super(RobotOnlyPredictor, self).__init__(units, look_back, look_forth, result_dims, use_tf_function, forcing)
    
    def _train_step(self, data):
        x, y = data
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)

        from .losses import SequenceLoss
        loss_fn = SequenceLoss()

        with tf.GradientTape() as tape:
            enc_output = self.encoder(x)
            enc_seq = enc_output.sequence

            dec_state = self.decoder_init(
                DecoderInitializerInput(state_h=enc_output.state_h, state_c=enc_output.state_c)
            )

            tokens = x[:, -1:, 2:4]  # v0 (velocidade do último frame do passado)
            preds = []

            for t in range(self.look_forth):
                out = self.decoder(DecoderInput(new_tokens=tokens, enc_output=enc_seq), dec_state)
                dec_state = DecoderState(state_h=out.state_h, state_c=out.state_c)

                v_pred = out.sequence  # VELOCIDADE ABSOLUTA
                preds.append(v_pred)
                tokens = y[:, t:t+1, :] if self.forcing else v_pred

            seq = tf.concat(preds, axis=1)
            loss = loss_fn(y, seq)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": loss}


    def _loop_step_no_forcing(self, input_pos, target_pos, enc_output, dec_state):
        decoder_input = DecoderInput(
            new_tokens=input_pos,
            enc_output=enc_output
        )

        decoder_output = self.decoder(decoder_input, dec_state)
        return self._decoder_step(decoder_output, target_pos)

    def _loop_step(self, new_seq, enc_output, dec_state):
        input_pos, target_pos = new_seq[:, 0:1, :], new_seq[:, 1:2, :]
        dec_state, seq = self._loop_step_no_forcing(input_pos, target_pos, enc_output, dec_state)

        return dec_state, seq

    def _forecast(self, input_seq):
        input_seq = tf.cast(input_seq, tf.float32)

        enc_output = self.encoder(input_seq)
        enc_seq = enc_output.sequence

        dec_state = self.decoder_init(
            DecoderInitializerInput(state_h=enc_output.state_h, state_c=enc_output.state_c)
        )

        tokens = input_seq[:, -1:, 2:4]  # v0
        preds = []

        for _ in range(self.look_forth):
            out = self.decoder(DecoderInput(new_tokens=tokens, enc_output=enc_seq), dec_state)
            dec_state = DecoderState(state_h=out.state_h, state_c=out.state_c)

            v_pred = out.sequence
            preds.append(v_pred)
            tokens = v_pred

        return tf.concat(preds, axis=1)
    
    def test_step(self, data):
        x, y, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)

        from .losses import SequenceLoss
        loss_fn = SequenceLoss()

        y_pred = self(x, training=False)
        loss = loss_fn(y, y_pred)
        return {"loss": loss}

class BallRobotPredictor(Predictor):
    def __init__(self, units, look_back, look_forth, result_dims, use_tf_function=True, forcing=True):
        super(BallRobotPredictor, self).__init__(units, look_back, look_forth, result_dims, use_tf_function, forcing)
        self.ball_encoder = BallEncoder(units)
        self.ball_aggregator = BallAggregator(units)

    def _train_step(self, data):
        x, y = data
        robot_x, ball_x, ball_mask = x

        with tf.GradientTape() as tape:
            enc_output = self.encoder(robot_x)
            init_h = enc_output.state_h
            init_c = enc_output.state_c
            dec_state = self.decoder_init(DecoderInitializerInput(state_h=init_h, state_c=init_c))

            tokens = robot_x[:, -1:, 2:4]
            preds = []

            for t in range(self.look_forth):
                out = self.decoder(DecoderInput(new_tokens=tokens, enc_output=enc_output.sequence), dec_state)
                dec_state = DecoderState(state_h=out.state_h, state_c=out.state_c)
                preds.append(out.sequence)
                tokens = y[:, t:t+1, :] if self.forcing else out.sequence

            seq = tf.concat(preds, axis=1)
            weights = tf.linspace(1.0, 2.5, self.look_forth)
            weights = tf.reshape(weights, (1, self.look_forth, 1))
            loss = tf.reduce_mean(weights * tf.square(y - seq))

        self.optimizer.apply_gradients(zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return {'batch_loss': loss}

    def _forecast(self, input_seq):
        robot_x, ball_x, ball_mask = input_seq

        enc_output = self.encoder(robot_x)
        init_h = enc_output.state_h
        init_c = enc_output.state_c
        dec_state = self.decoder_init(DecoderInitializerInput(state_h=init_h, state_c=init_c))

        tokens = robot_x[:, -1:, 2:4]
        preds = []
        for _ in range(self.look_forth):
            out = self.decoder(DecoderInput(new_tokens=tokens, enc_output=enc_output.sequence), dec_state)
            dec_state = DecoderState(state_h=out.state_h, state_c=out.state_c)
            tokens = out.sequence
            preds.append(out.sequence)

        return tf.concat(preds, axis=1)