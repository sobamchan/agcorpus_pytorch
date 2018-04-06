import torch
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

        self.cnn_n2 = nn.Conv2d(1, 3, (2, self.args.embedding_dim))
        self.cnn_n3 = nn.Conv2d(1, 3, (3, self.args.embedding_dim))
        self.cnn_n4 = nn.Conv2d(1, 3, (4, self.args.embedding_dim))

        self.linear = BottleLinear(1 * 3 * 3, args.class_n)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embed(x)
        size = x.size()
        batch_size = size[0]
        x = x.view(batch_size, 1, size[1], size[2])

        x_n2 = self.cnn_n2(x)
        x_n2 = F.max_pool2d(x_n2, (x_n2.size(2), 1))
        x_n2 = x_n2.squeeze(3).view(batch_size, -1)

        x_n3 = self.cnn_n3(x)
        x_n3 = F.max_pool2d(x_n3, (x_n3.size(2), 1))
        x_n3 = x_n3.squeeze(3).view(batch_size, -1)

        x_n4 = self.cnn_n4(x)
        x_n4 = F.max_pool2d(x_n4, (x_n4.size(2), 1))
        x_n4 = x_n4.squeeze(3).view(batch_size, -1)

        x_cat = torch.cat((x_n2, x_n3, x_n4), 1)
        x_linear = self.linear(x_cat)

        return self.softmax(x_linear)
