import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data.sampler import RandomSampler
from lib import agcorpus
from lib import vocabulary
from lib import utils


class Dataset(data.Dataset):

    def __init__(self, labels, texts, vocab, args=None, return_word_idx=True):
        self.labels = labels
        self.texts = texts
        self.vocab = vocab
        self.return_word_idx = return_word_idx
        self.args = args

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        text = self.texts[idx]

        if self.return_word_idx:
            words, _ = self.vocab.encode(text)
            words = utils.pad_to_n(words,
                                   self.args.max_sent_len,
                                   self.vocab.w2i['<PAD>'])
            words = torch.from_numpy(np.array(words))
        else:
            _, words = self.vocab.encode()
        item_dict = {'label': label, 'words': words}
        return item_dict


def get_dataloader(args):
    corpus = agcorpus.AGCorpus(args.data_dir)
    train_labels, train_texts, test_labels, test_texts = corpus.load_dataset()

    vocab = vocabulary.Vocabulary(train_texts, args)
    train_dataset = Dataset(train_labels, train_texts, vocab, args)
    test_dataset = Dataset(test_labels, test_texts, vocab, args)

    train_dataloader = data.DataLoader(train_dataset,
                                       args.batch_size,
                                       sampler=RandomSampler(train_dataset))
    test_dataloader = data.DataLoader(test_dataset,
                                      args.batch_size,
                                      sampler=RandomSampler(test_dataset))

    return train_dataloader, test_dataloader
