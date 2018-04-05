import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from lib import dataset
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
        total = int(len(self.train_dataloader.dataset) / self.args.batch_size)
        self.model.zero_grad()
        for i, dict_ in tqdm(enumerate(self.train_dataloader), total=total):
            label = dict_['label']
            words = dict_['words']

            label = Variable(label)
            words = Variable(words)
            if args.use_cuda:
                label = label.cuda()
                words = words.cuda()

            preds = self.model(words)
            loss = self.criteria(preds, label)
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.data[0]))
        return np.mean(losses)
