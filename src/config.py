from dataclasses import dataclass


@dataclass
class Config:
    """
    Configuration class for setting hyperparameters and model settings.

    :param epochs: Number of training epochs.
    :type epochs: int
    :param lr: Learning rate for the optimizer.
    :type lr: float
    :param sequence_length: Maximum length of the input and target sequences.
    :type sequence_length: int
    :param dims: Dimensionality of the model's hidden layers.
    :type dims: int
    :param n_heads: Number of attention heads in each multi-head attention layer.
    :type n_heads: int
    :param d_ff: Dimensionality of the feed-forward layer.
    :type d_ff: int
    :param n: Number of encoder and decoder blocks.
    :type n: int
    :param dropout: Dropout rate applied in the attention and feed-forward layers.
    :type dropout: float
    :param train_batch_size: Batch size for training.
    :type train_batch_size: int
    :param val_batch_size: Batch size for validation.
    :type val_batch_size: int
    :param test_batch_size: Batch size for testing.
    :type test_batch_size: int
    :param n_val_losses: Number of validation losses to keep track of.
    :type n_val_losses: int
    :param n_inference: Number of inference passes during evaluation.
    :type n_inference: int
    :param source_lang: Language code for the source language.
    :type source_lang: str
    :param target_lang: Language code for the target language.
    :type target_lang: str
    :param dataset: Name of the dataset used for training.
    :type dataset: str
    :param model_path: Path to save the model weights.
    :type model_path: str
    :param model_name: Filename for saving the model weights.
    :type model_name: str
    :param get_latest_weights: Flag indicating whether to retrieve the latest weights.
    :type get_latest_weights: bool
    :param tokenizer_path: Path template for saving the tokenizer data.
    :type tokenizer_path: str
    """

    epochs = 30
    lr = 1e-4
    sequence_length = 350
    dims = 512
    n_heads = 8
    n = 6
    dropout = 0.1
    train_batch_size = 8
    val_batch_size = 8
    test_batch_size = 1
    n_val_losses = 50
    n_inference = 4
    source_lang = 'en'
    target_lang = 'fr'
    dataset = 'opus_books'
    model_path = '../data/weights'
    model_name = 'latest.pt'
    get_latest_weights = False
    tokenizer_path = '../data/tokenizer_data_{}.json'
