import torch.nn as nn
import torch.nn.functional as F

# from icecream import ic


class Linear(nn.Module):

    def __init__(self, d_in, d_out, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(d_in, d_out, bias=bias)
        nn.init.xavier_normal(self.linear.weight)

    def forward(self, x):
        return self.linear(x)


class Bottle(nn.Module):

    def forward(self, x):
        if len(x.size()) <= 2:
            return super(Bottle, self).forward(x)
        size = x.size()
        out = super(Bottle, self).forward(x.view(size[0], -1))
        return out


class BottleLinear(Bottle, Linear):
    pass


class Model(nn.Module):

    def __init__(self, vocab, args):
        super(Model, self).__init__()
        self.args = args
        self.embed = nn.Embedding(len(vocab),
                                  args.embedding_dim,
                                  padding_idx=vocab.w2i['<PAD>'])
        self.li1 = BottleLinear(args.max_sent_len * args.embedding_dim, 100)
        self.li2 = BottleLinear(100, args.class_n)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embed(x)
        x = F.relu(self.li1(x))
        x = self.li2(x)
        x = self.softmax(x)
        return x


class CNN(nn.Module):

    def __init__(self, vocab, args):
        super(CNN, self).__init__()
        self.args = args
        self.embed = nn.Embedding(len(vocab),
                                  args.embedding_dim,
                                  padding_idx=vocab.w2i['<PAD>'])
        self.cnn = nn.Conv2d(1, 3, (2, self.args.embedding_dim))
        self.pool = nn.MaxPool2d((1, 2))
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.embed(x)
        size = x.size()
        x = x.view(size[0], 1, size[1], size[2])
        x = self.cnn(x)
        x = self.pool(x)
        return self.softmax(x)
