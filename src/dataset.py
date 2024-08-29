import torch
from torch.utils.data import Dataset

from utils import Utils


class BilangDS(Dataset):
    """
    A dataset class for bilingual text data, preparing source 
    and target sequences for training.

    :param dataset: The dataset containing bilingual text samples.
    :type dataset: list of dict
    :param source_tokenizer: Tokenizer for the source language.
    :type source_tokenizer: Tokenizer
    :param target_tokenizer: Tokenizer for the target language.
    :type target_tokenizer: Tokenizer
    :param source_lang: Language code for the source language.
    :type source_lang: str
    :param target_lang: Language code for the target language.
    :type target_lang: str
    :param max_seq_length: Maximum sequence length for source and target sequences.
    :type max_seq_length: int
    """

    def __init__(self, 
                 dataset, 
                 source_tokenizer, 
                 target_tokenizer, 
                 source_lang, 
                 target_lang, 
                 max_seq_length):
        self.dataset = dataset
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_seq_length = max_seq_length
        
        self.pad = torch.tensor([self.source_tokenizer.token_to_id('[PAD]')], 
                                dtype=torch.long)
        self.sos = torch.tensor([self.source_tokenizer.token_to_id('[SOS]')], 
                                dtype=torch.long)
        self.eos = torch.tensor([self.source_tokenizer.token_to_id('[EOS]')], 
                                dtype=torch.long)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset and processes it into tensors.

        :param idx: Index of the sample to retrieve.
        :type idx: int
        :return: A dictionary containing processed source and target tokens, 
                    masks, and original sentences.
        :rtype: dict
        :raises AssertionError: If the sequence lengths do not match 
                                the maximum sequence length.
        """
        sent = self.dataset[idx]
        source_sent = sent['translation'][self.source_lang]
        target_sent = sent['translation'][self.target_lang]

        source_tokens = self.source_tokenizer.encode(source_sent, ).ids
        target_tokens = self.target_tokenizer.encode(target_sent, ).ids

        assert((n_pad_source_tokens := (self.max_seq_length - len(source_tokens) - 2)) >= 0 and 
               (n_pad_target_tokens := (self.max_seq_length - len(target_tokens) - 1)) >= 0)

        fin_source_tokens = torch.cat([
            self.sos,
            torch.tensor(source_tokens, dtype=torch.long),
            self.eos,
            torch.empty(n_pad_source_tokens, dtype=torch.long).fill_(self.pad.item()),
        ], dim=0)

        fin_target_tokens = torch.cat([
            self.sos,
            torch.tensor(target_tokens, dtype=torch.long),
            torch.empty(n_pad_target_tokens, dtype=torch.long).fill_(self.pad.item())
        ], dim=0)

        y_hat = torch.cat([
            torch.tensor(target_tokens, dtype=torch.long),
            self.eos,
            torch.empty(n_pad_target_tokens, dtype=torch.long).fill_(self.pad.item())
        ], dim=0)

        assert all(
            [item == self.max_seq_length for item in [fin_source_tokens.shape[0], 
                                                      fin_target_tokens.shape[0], 
                                                      y_hat.shape[0]]]
        )

        return {
            'source_tokens': fin_source_tokens, # (max_seq_length, )
            'target_tokens': fin_target_tokens, # (max_seq_length, )
            'source_mask': ((fin_source_tokens != self.pad)
                            .unsqueeze(0).unsqueeze(0).int()), # (1, 1, max_seq_length)
            'target_mask': (((fin_source_tokens != self.pad)
                            .unsqueeze(0).unsqueeze(0).int()) & (
                            Utils.get_triu_mask(self.max_seq_length))), # (1, max_seq_length, max_seq_length)
            'y': y_hat, # (max_seq_length, ) 
            'source_sent': source_sent,
            'target_sent': target_sent
        }

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        :return: The number of samples in the dataset.
        :rtype: int
        """
        return len(self.dataset)
