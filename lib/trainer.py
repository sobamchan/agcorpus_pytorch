import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from lib import dataset
from lib import utils
from lib import models


class Trainer:

    def __init__(self, args):
        self.args = args
        train_dataloader, test_dataloader = dataset.get_dataloader(args)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.vocab = train_dataloader.dataset.vocab

        model = models.Model(self.vocab, args)
        if args.use_cuda:
            self.model = model.cuda()
        else:
            self.model = model

        self.optimizer = optim.SGD(self.model.parameters(), args.lr)
        self.criteria = nn.CrossEntropyLoss(ignore_index=0)

    def train_one_epoch(self, i_epoch):
        args = self.args
        losses = []
        for i, dict_ in enumerate(self.train_dataloader):
            label = dict_['label']
            words = dict_['words']
            words = utils.pad_to_n(words,
                                   self.args.max_sent_len,
                                   self.vocab.w2i['<PAD>'])

            label = torch.from_numpy(label).type(torch.LongTensor)
            words = torch.from_numpy(words).type(torch.LongTensor)
            if args.use_cuda:
                label = label.cuda()
                words = words.cuda()

            preds = self.model(words)
            loss = self.criteria(label, preds)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.data[0])
        return np.mean(losses)