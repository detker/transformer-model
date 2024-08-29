import os
import time

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import random_split, Dataset, DataLoader

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from config import Config
from architecture import Transformer
from dataset import BilangDS
from utils import Utils


def yield_data(data,
               lang):
    """
    Yield sentences from the data in the specified language.

    :param data: Dataset containing translations
    :type data: Dataset
    :param lang: Language code to extract sentences from
    :type lang: str
    :yield: Sentences in the specified language
    :rtype: generator
    """
    for sent in data:
        yield sent['translation'][lang]


def get_tokenizer(config,
                  data,
                  lang):
    """
    Get or create a tokenizer for the specified language.

    :param config: Configuration object containing tokenizer path
    :type config: Config
    :param data: Dataset containing translations
    :type data: Dataset
    :param lang: Language code for which to create the tokenizer
    :type lang: str
    :return: Tokenizer for the specified language
    :rtype: Tokenizer
    """
    tokenizer_path = os.path.join(config.tokenizer_path.format(lang))
    if not os.path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'],
                                   min_frequency=2)
        tokenizer.train_from_iterator(yield_data(data, lang), trainer=trainer)
        tokenizer.save(tokenizer_path)
    else:
        tokenizer = Tokenizer.from_file(tokenizer_path)

    return tokenizer


def get_data(config):
    """
    Load and preprocess the dataset, split into train, validation, and test sets.

    :param config: Configuration object containing dataset parameters
    :type config: Config
    :return: Data loaders for training, validation, and test sets, 
                and source and target tokenizers
    :rtype: tuple(DataLoader, DataLoader, DataLoader, Tokenizer, Tokenizer)
    """
    raw_data = load_dataset(config.dataset,
                            '{}-{}'.format(config.source_lang, config.target_lang),
                            split='train')

    source_tokenizer = get_tokenizer(config, raw_data, config.source_lang)
    target_tokenizer = get_tokenizer(config, raw_data, config.target_lang)

    n_train_data = int(0.8 * len(raw_data))
    n_val_data = int(0.1 * len(raw_data))
    n_test_data = len(raw_data) - n_train_data - n_val_data
    train_raw_data, val_raw_data, test_raw_data = random_split(raw_data,
                                                               [n_train_data,
                                                                n_val_data,
                                                                n_test_data])

    train_data = BilangDS(train_raw_data, source_tokenizer,
                          target_tokenizer, config.source_lang,
                          config.target_lang, config.sequence_length)
    val_data = BilangDS(val_raw_data, source_tokenizer,
                        target_tokenizer, config.source_lang,
                        config.target_lang, config.sequence_length)
    test_data = BilangDS(test_raw_data, source_tokenizer,
                         target_tokenizer, config.source_lang,
                         config.target_lang, config.sequence_length)

    max_source_seq_len = max_target_seq_len = 0
    for x in raw_data:
        source_seq_len = source_tokenizer.encode(x['translation'][config.source_lang]).ids
        target_seq_len = target_tokenizer.encode(x['translation'][config.target_lang]).ids
        max_source_seq_len = max(max_source_seq_len, len(source_seq_len))
        max_target_seq_len = max(max_target_seq_len, len(target_seq_len))
    print(f'{max_source_seq_len=}, {max_target_seq_len=}')

    train_data_loader = DataLoader(train_data, batch_size=config.train_batch_size, shuffle=True)
    val_data_loader = DataLoader(val_data, batch_size=config.val_batch_size, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=config.test_batch_size, shuffle=True)

    return (train_data_loader,
            val_data_loader,
            test_data_loader,
            source_tokenizer,
            target_tokenizer)


@torch.no_grad
def evaluate_model(config,
                   model,
                   train_data_loader,
                   val_data_loader,
                   loss_function,
                   device):
    """
    Evaluate the model on training and validation data.

    :param config: Configuration object containing evaluation parameters
    :type config: Config
    :param model: The model to evaluate
    :type model: Transformer
    :param train_data_loader: DataLoader for the training set
    :type train_data_loader: DataLoader
    :param val_data_loader: DataLoader for the validation set
    :type val_data_loader: DataLoader
    :param loss_function: Loss function to use for evaluation
    :type loss_function: nn.Module
    :param device: Device on which to perform evaluation
    :type device: torch.device
    :return: Dictionary containing training and validation losses
    :rtype: dict
    """
    t0 = time.time()
    result = {}
    model.eval()
    train_iterator = iter(train_data_loader)
    val_iterator = iter(val_data_loader)
    train_losses = torch.zeros(config.n_val_losses)
    val_losses = torch.zeros(config.n_val_losses)
    for k in range(config.n_val_losses):
        train_batch = next(train_iterator)
        val_batch = next(val_iterator)
        X_source_train, X_target_train, Y_train = (train_batch['source_tokens'],
                                                   train_batch['target_tokens'],
                                                   train_batch['y'])
        X_source_train, X_target_train, Y_train = (X_source_train.to(device),
                                                   X_target_train.to(device),
                                                   Y_train.to(device))
        X_source_val, X_target_val, Y_val = (val_batch['source_tokens'],
                                             val_batch['target_tokens'],
                                             val_batch['y'])
        X_source_val, X_target_val, Y_val = (X_source_val.to(device),
                                             X_target_val.to(device),
                                             Y_val.to(device))
        src_mask_train, tgt_mask_train, src_mask_val, tgt_mask_val = (train_batch['source_mask'],
                                                                      train_batch['target_mask'],
                                                                      val_batch['source_mask'],
                                                                      val_batch['target_mask'])
        src_mask_train, tgt_mask_train, src_mask_val, tgt_mask_val = (src_mask_train.to(device),
                                                                      tgt_mask_train.to(device),
                                                                      src_mask_val.to(device),
                                                                      tgt_mask_val.to(device))
        logits_train = model(X_source_train, X_target_train, src_mask_train, tgt_mask_train)
        train_losses[k] = loss_function(logits_train.view(-1, config.target_vocab_size),
                                        Y_train.view(-1)).item()
        logits_val = model(X_source_val, X_target_val, src_mask_val, tgt_mask_val)
        val_losses[k] = loss_function(logits_val.view(-1, config.target_vocab_size),
                                      Y_val.view(-1)).item()
    result['train'] = train_losses.mean().item()
    result['val'] = val_losses.mean().item()
    model.train()
    dt = time.time() - t0
    print(f'evaluate_model() took: {dt:.2f} sec.')
    return result


def greedy_search(model,
                  enc_output,
                  source_mask_test,
                  target_tokenizer,
                  max_seq_len,
                  source,
                  device):
    """
    Perform greedy search for sequence generation.

    :param model: The model used for sequence generation
    :type model: Transformer
    :param enc_output: Encoder output
    :type enc_output: torch.Tensor
    :param source_mask_test: Source mask for the test set
    :type source_mask_test: torch.Tensor
    :param target_tokenizer: Tokenizer for the target language
    :type target_tokenizer: Tokenizer
    :param max_seq_len: Maximum sequence length
    :type max_seq_len: int
    :param source: Source tokens
    :type source: torch.Tensor
    :param device: Device on which to perform the search
    :type device: torch.device
    :return: Generated target sequence
    :rtype: torch.Tensor
    """
    sos = target_tokenizer.token_to_id('[SOS]')
    eos = target_tokenizer.token_to_id('[EOS]')

    target_input = torch.empty((1, 1)).fill_(sos).type_as(source).to(device)  # (1, 1)

    while True:
        if target_input.shape[1] == max_seq_len: break

        tgt_mask_test = Utils.get_triu_mask(target_input.shape[1]).to(device)
        dec_out = model.decode(target_input, enc_output, source_mask_test, tgt_mask_test)  # (1, S, D)
        logits = model.project(dec_out[:, -1])  # (1, S, tgt_vocab_size)
        probs = F.softmax(logits, dim=-1)
        idx = torch.multinomial(probs, num_samples=1)
        target_input = torch.cat([target_input,
                                  torch.empty((1, 1)).fill_(idx.item()).type_as(source).to(device)],
                                 dim=1)

        if idx.item() == eos: break
    return target_input.squeeze(0)


@torch.no_grad
def test_model(config,
               model,
               test_data_loader,
               target_tokenizer,
               device):
    """
    Test the model on the test dataset.

    :param config: Configuration object containing test parameters
    :type config: Config
    :param model: The model to test
    :type model: Transformer
    :param test_data_loader: DataLoader for the test set
    :type test_data_loader: DataLoader
    :param target_tokenizer: Tokenizer for the target language
    :type target_tokenizer: Tokenizer
    :param device: Device on which to perform testing
    :type device: torch.device
    """
    model.eval()
    test_iterator = iter(test_data_loader)
    for _ in range(config.n_inference):
        test_batch = next(test_iterator)
        X_source_test = test_batch['source_tokens']
        X_source_test = X_source_test.to(device)
        src_mask_test = test_batch['source_mask']
        src_mask_test = src_mask_test.to(device)
        source_sent, target_sent = test_batch['source_sent'], test_batch['target_sent']

        enc_output = model.encode(X_source_test, src_mask_test)
        output_tokens = greedy_search(model, enc_output, src_mask_test, target_tokenizer, config.sequence_length,
                                      X_source_test, device)
        y_hat = target_tokenizer.decode(output_tokens.detach().cpu().numpy())

        print(f'INPUT TXT: {source_sent[0]}')
        print(f'DESIRED OUTPUT: {target_sent[0]}')
        print(f"MODEL'S OUTPUT: {y_hat}")

    model.train()


def train_model(config):
    """
    Train the model with the given configuration.

    :param config: Configuration object containing training parameters
    :type config: Config
    """
    device = 'cpu'
    cuda = False
    mps = True
    if cuda and torch.cuda_is_available():
        device = 'cuda'
    elif mps and (torch.backends.mps.is_built() or torch.backends.mps.is_available()):
        device = 'mps'
    print(f'Using device: {device}')
    device = torch.device(device)

    torch.set_float32_matmul_precision('high')

    (train_data_loader,
     val_data_loader,
     test_data_loader,
     source_tokenizer,
     target_tokenizer) = get_data(config)
    config.source_vocab_size = source_tokenizer.get_vocab_size()
    config.target_vocab_size = target_tokenizer.get_vocab_size()
    model = Transformer.build(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, eps=1e-9)

    init_epoch = 0
    os.makedirs(config.model_path, exist_ok=True)
    model_path = os.path.join(config.model_path, config.model_name)
    if config.get_latest_weights:
        state = torch.load(model_path)
        model.load_state_dict(state['model_state_dict'])
        init_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
    else:
        print('no model state found. starting from scratch')

    loss_function = nn.CrossEntropyLoss(ignore_index=source_tokenizer.token_to_id('[PAD]'),
                                        label_smoothing=0.1).to(device)

    for epoch in range(init_epoch, config.epochs):
        torch.cuda.empty_cache()
        step = 0
        dt_acum = 0
        for batch in train_data_loader:
            if step % 100 == 0:
                print('-' * 40)
                losses = evaluate_model(config, model, train_data_loader, val_data_loader,
                                        loss_function, device)
                print(f"""epoch: {epoch + 1} | step: {step}/{len(train_data_loader)} | 
                      loss: {losses['train']:.6f} | val_loss: {losses['val']:.6f}""")
                if step == 0:
                    print('train_model() took: x sec. (training has not started yet)')
                else:
                    print(f'train_model() took: {(dt_acum / 2):.2f} sec.')
                print('-' * 40)
                dt_acum = 0

                test_model(config, model, test_data_loader, target_tokenizer, device)

            if step == 0: print("STARTING MODEL'S TRAINING")
            t_0 = time.time()
            X_source, X_target, Y = (batch['source_tokens'],
                                     batch['target_tokens'],
                                     batch['y'])  # (B, S)
            X_source, X_target, Y = (X_source.to(device),
                                     X_target.to(device),
                                     Y.to(device))
            source_mask, target_mask = (batch['source_mask'].to(device),
                                        batch['target_mask'].to(device))

            enc_output = model.encode(X_source, source_mask)  # (B, S, D)
            dec_output = model.decode(X_target, enc_output, source_mask, target_mask)  # (B, S, D)
            logits = model.project(dec_output)  # (B, S, target_vocab_size)
            loss = loss_function(logits.view(-1, config.target_vocab_size), Y.view(-1))

            model.zero_grad(True)
            loss.backward()
            optimizer.step()
            step += 1
            dt_acum += (time.time() - t_0)

        print(f'END OF {epoch+1} EPOCH')
        model_path = os.path.join(config.model_path, config.model_name)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, model_path)


def main():
    """
    Main function to initialize and start the training process.
    """
    config = Config()
    train_model(config)


if '__main__' == __name__:
    main()
