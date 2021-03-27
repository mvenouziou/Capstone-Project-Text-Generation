class Paramaters:
    def __init__(self,
                 # integrations
                 use_gdrive, use_anvil,
                 # model architecture
                 use_probability_layers,  # implements TensorFlow Probability
                 use_word_path,  # note: TFP layers not recommended with word-level model
                 use_electra,  # use False for BERT embeddings (fewer params, word model only)
                 # datasets
                 author, data_files,
                 datasets_dir='https://raw.githubusercontent.com/mvenouziou/text_generator/main/',
                 # model params
                 num_trailing_words=5, padded_example_length=300, batch_size=32):

        # save param choices
        # note: additional attributes are added below
        self._use_gdrive = use_gdrive
        self._use_anvil = use_anvil
        self._author = author
        self._num_trailing_words = num_trailing_words
        self._padded_example_length = padded_example_length
        self._batch_size = batch_size
        self._use_probability_layers = use_probability_layers
        self._use_word_path = use_word_path
        self._use_electra = use_electra
        self._data_files = list(data_files)
        self._datasets_dir = datasets_dir
        self._embedding_dim = 32 * 8
        self._char_rnn_units = 512
        self._word_rnn_units = 64
        self._merge_dim = 32 * 8
        self.anvil_code = '53NFXI7IX7IE233XQTVJDXUM-PUGRV2WON2LETWBG'

        # Filepath Structure
        # path name conventions due to model structure
        if self._use_probability_layers:
            self._author += '/probability/'
        if self._use_word_path:
            self._author += '_words_model/'
        if self._use_electra:
            self._author += 'electra/'

        # models / checkpoints directories
        # (Google Drive)
        self._filepath = self._gdrive_dir + 'MyDrive/Colab_Notebooks/models/text_generation/' + self._author
        self._checkpoint_dir = self._filepath + '/checkpoints/'

        ###self._prediction_model_dir = self._filepath + '/prediction_model/'
        self._training_model_dir = self._filepath + '/training_model/'
        self._processed_data_dir = self._filepath + '/proc_data/'
        self._tensorboard_dir = self._checkpoint_dir + '/logs/'

        # Create Tokenizer / Set Vocab Size
        # character tokenizer
        def create_character_tokenizer():
            from string import printable
            from tensorflow.keras.preprocessing import text

            char_tokens = printable  # string.printable
            filters = '#$%&()*+-/<=>@[]^_`{|}~\t'

            # Initialize standard keras tokenizer
            tokenizer = text.Tokenizer(  # tf.keras.preprocessing.text.Tokenizer
                num_words=None,
                filters=filters,
                lower=False,  # conversion to lowercase letters
                char_level=True,
                oov_token=None,  # drop unknown characters
            )
            # fit tokenizer
            tokenizer.fit_on_texts(char_tokens)

            return tokenizer

        self._character_tokenizer = create_character_tokenizer()
        self._vocab_size = len(self._character_tokenizer.word_index) + 1

    def load_imports(self):
        """
        # Additional Imports
        # # Tensorflow HUB for using pretrained embeddings (word models)
        if self._use_word_path:
            import tensorflow_hub as hub

        # # Tensorflow Probability (probability distribution model architecture)
        if self._use_probability_layers:
            import tensorflow_probability as tfp
            tfpl = tfp.layers
            tfd = tfp.distributions
        """

        # # Google Drive:
        if self._use_gdrive:
            self._gdrive_dir = '/content/gdrive/'
            from google.colab import drive
            drive.mount(self._gdrive_dir)
        else:
            self._gdrive_dir = ''

        # # Anvil's web app server
        if self._use_anvil:
            !pip install - q anvil-uplink
            import anvil.server
