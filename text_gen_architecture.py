# Model Class
import tensorflow as tf

class GenerationModel(tf.keras.Model):

    def __init__(self, paramaters, **kwargs):
        super().__init__(self, **kwargs)

        # save attributes
        self.batch_size = paramaters._batch_size
        self.checkpoint_manager = None
        self.checkpoint = None
        self.paramaters = paramaters
        self.gru_layer_names = ['char_GRU_1', 'char_GRU_2',
                                'word_GRU_1', 'word_GRU_2', 'GRU_Output']

        # imports - Keras
        from tf.keras.layers import Input, Embedding, Concatenate, \
                                    Dense, GRU, Dropout, \
                                    BatchNormalization, Lambda

        # imports Tensorflow Probability
        if self.paramaters._use_probability_layers:
            import tensorflow_probability as tfp
            from tensorflow_probability import layers as tfpl
            from tensorflow_probability import distributions as tfd

        # # Tensorflow HUB for using pretrained embeddings (word models)
        if self.paramaters._use_word_path:
            import tensorflow_hub as hub

        # set model params
        use_word_path = paramaters._use_word_path
        use_probability_layers = paramaters._use_probability_layers
        num_words = paramaters._num_trailing_words
        vocab_size = paramaters._vocab_size
        embedding_dim = paramaters._embedding_dim
        char_rnn_units = paramaters._char_rnn_units
        word_rnn_units = paramaters._word_rnn_units
        merge_dim = paramaters._merge_dim

        # Encoder layers
        if use_word_path:
            self.bert_tokenizer, self.bert_packer, self.bert_encoder = \
                self.get_word_encoder()

        self.char_embedding = Embedding(input_dim=vocab_size,
                                        output_dim=embedding_dim,
                                        mask_zero=True,
                                        name='char_embedding')

        # ## Character Path Layers
        self.char_GRU_1 = GRU(units=char_rnn_units, return_state=True,
                              return_sequences=True, name='char_GRU_1')
        self.char_Batch_Norm_1 = BatchNormalization(name='char_Batch_Norm_1')
        self.char_GRU_2 = GRU(units=char_rnn_units, return_state=True,
                              return_sequences=True, name='char_GRU_2')
        self.char_Batch_Norm_2 = BatchNormalization(name='char_Batch_Norm_2')

        # Word Encoding Path Layers
        if use_word_path:
            self.word_GRU_1 = \
                GRU(units=word_rnn_units, return_state=True,
                    return_sequences=True, name='word_GRU_1', )
            self.word_Batch_Norm_1 = BatchNormalization(name='word_Batch_Norm_1')
            self.word_GRU_2 = \
                GRU(units=word_rnn_units, return_state=True,
                    return_sequences=False, name='word_GRU_2', )
            self.word_Batch_Norm_2 = BatchNormalization(name='word_Batch_Norm_2')

        # Merge Layers
        if use_word_path:
            self.char_Dense_merge = \
                Dense(units=merge_dim, activation=None, name='char_Dense_merge')
            self.word_Dense_merge = \
                Dense(units=merge_dim, activation=None, name='word_Dense_merge')
            self.word_reshape = \
                Lambda(lambda x: tf.expand_dims(x, axis=1), name='word_reshape')
            self.merged_layers = \
                Lambda(lambda x: tf.concat([x[0][:, 1:, :],  # drop first value so shape is preserved after concat
                                            x[1]], axis=1), name='merged_layers')

        else:
            self.rename = Lambda(lambda x: x, name='rename_variable')

        self.Batch_Norm_output = BatchNormalization(name='Batch_Norm_output')

        # Character prediction (logits)
        if use_probability_layers:
            # Dense layer with probabalistic weights
            self.dense_reparam = tfpl.DenseReparameterization(
                units=tfpl.OneHotCategorical.params_size(vocab_size),
                activation=None)
            self.dist_outputs = tfpl.OneHotCategorical(
                vocab_size,
                convert_to_tensor_fn=tfd.OneHotCategorical.logits,
                name='Decoding')
        else:
            self.dense_outputs = \
                Dense(units=vocab_size, activation=None, name='Decoding')

    def call(self, inputs, initial_states=None, stateful=False, **kwargs):

        # set model params
        use_word_path = self.paramaters._use_word_path
        use_probability_layers = self.paramaters._use_probability_layers

        # initialize states as TensorArray
        # (This is similar to dictionary but works with @tf.function)
        if stateful:
            states = tf.TensorArray(tf.float32, size=len(self.gru_layer_names),
                                    clear_after_read=True)
            states.trainable = False

            if initial_states is None:
                indx_state = None

            else:  # unpack values into 'states' array
                states_list = tf.unstack(initial_states)
                for indx in range(len(self.gru_layer_names)):
                    states = states.write(indx, states_list[indx])

        else:
            initial_states = None
            indx_state = None

        # inputs
        input_1 = inputs[0]
        input_2 = inputs[1]

        # ## Character Path Layers
        x1 = self.char_embedding(input_1)

        # Char GRU 1
        if stateful:
            indx = self.gru_layer_names.index('char_GRU_1')
            if initial_states is not None:
                indx_state = states.read(indx)
            x1, new_indx_state = self.char_GRU_1(x1, initial_state=indx_state)
            states = states.write(indx, new_indx_state)
        else:
            x1, _ = self.char_GRU_1(x1, initial_state=None)
        x1 = self.char_Batch_Norm_1(x1)

        # Char GRU 2
        if stateful:
            indx = self.gru_layer_names.index('char_GRU_2')
            if initial_states is not None:
                indx_state = states.read(indx)
            x1, new_indx_state = self.char_GRU_2(x1, initial_state=indx_state)
            states = states.write(indx, new_indx_state)
        else:
            x1, _ = self.char_GRU_2(x1, initial_state=None)
        x1 = self.char_Batch_Norm_2(x1)

        # Word Encoding Path Layers
        if use_word_path:
            # encoding
            x2 = self.bert_tokenizer(input_2)  # tokenize
            x2 = self.bert_packer([x2])  # pack inputs for encoder
            x2 = self.bert_encoder(x2)['sequence_output']  # encoding

            # Word GRU 1
            if stateful:
                indx = self.gru_layer_names.index('word_GRU_1')
                if initial_states is not None:
                    indx_state = states.read(indx)
                x2, new_indx_state = self.word_GRU_1(x2, initial_state=indx_state)
                states = states.write(indx, new_indx_state)
            else:
                x2, _ = self.word_GRU_1(x2, initial_state=None)
            x2 = self.word_Batch_Norm_1(x2)

            # Word GRU 2
            if stateful:
                indx = self.gru_layer_names.index('word_GRU_2')
                if initial_states is not None:
                    indx_state = states.read(indx)
                x2, new_indx_state = self.word_GRU_2(x2, initial_state=indx_state)
                states = states.write(indx, new_indx_state)
            else:
                x2, _ = self.word_GRU_2(x2, initial_state=None)
            x2 = self.word_Batch_Norm_2(x2)

        # Merge Layers
        if use_word_path:
            x1 = self.char_Dense_merge(x1)
            x2 = self.word_Dense_merge(x2)
            x2 = self.word_reshape(x2)
            x = self.merged_layers((x1, x2))

        else:  # update variable id to match next step
            x = self.rename(x1)

        x = self.Batch_Norm_output(x)

        # Character prediction (logits)
        if use_probability_layers:
            # Dense layer with probabalistic weights
            x = self.dense_reparam(x)
            y_pred = self.dist_outputs(x)

        else:
            y_pred = self.dense_outputs(x)

        if stateful:
            new_states = states.stack()
            states.close()
        else:
            new_states = tf.constant(1)  # dummy entry to avoid error in model.save()

        return y_pred, new_states

    def get_word_encoder(self):

        import tensorflow_hub as hub

        # Word Embeddings
        # Selects file locations for BERT or ELECTRA pretrained encoders
        if self.paramaters._use_electra:
            encoder_url = 'https://tfhub.dev/google/electra_small/2'
        else:
            encoder_url = 'https://tfhub.dev/tensorflow/' \
                          + 'small_bert/bert_en_uncased_L-2_H-128_A-2/1'
        preprocessor_url = 'https://tfhub.dev/tensorflow/' \
                           + 'bert_en_uncased_preprocess/3'

        # Get Encoder / Preprocessing layers
        preprocessor = hub.load(preprocessor_url)

        bert_tokenizer = hub.KerasLayer(preprocessor.tokenize, name='bert_tokenizer')

        bert_packer = hub.KerasLayer(
            preprocessor.bert_pack_inputs,
            arguments=dict(seq_length=self.paramaters._num_trailing_words),
            name='bert_input_packer')

        word_encoder = hub.KerasLayer(encoder_url, trainable=False,
                                      name='Word_encoder')

        return bert_tokenizer, bert_packer, word_encoder