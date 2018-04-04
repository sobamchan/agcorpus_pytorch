import torch.nn as nn


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
        size = x.size()[:2]
        out = super(Bottle, self).forward(x.view(size[0] * size[1], -1))
        return out.view(size[0], size[1], -1)


class BottleLinear(Bottle, Linear):
    pass


class Model(nn.Module):

    def __init__(self, vocab, args):
        super(Model, self).__init__()
        self.embed = nn.Embedding(len(vocab),
                                  args.embedding_dim,
                                  padding_idx=vocab.w2i['<PAD>'])
        self.li1 = BottleLinear(args.embedding_dim * args.max_sent_len, 500)
        self.li2 = BottleLinear(500, args.class_n)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.embed(x)
        x = nn.ReLU(self.li1(x))
        x = self.li2(x)
        return self.softmax(x)
