import os
import argparse
from distutils.util import strtobool
import torch
from lib.trainer import Trainer


class Args:

    def __init__(self):
        self.batch_size = 512
        self.lr = 0.001
        self.use_cuda = True
        self.max_sent_len = 20
        self.data_dir = './'
        self.vocab_size = 50000
        self.embedding_dim = 300
        self.class_n = 4
        self.model = 'CNN'
        self.epoch = 100
        self.gpu_id = 0


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--use-cuda', type=strtobool, default='1')
    parser.add_argument('--max-sent-len', type=int, default=30)
    parser.add_argument('--data-dir', type=str, default='./')
    parser.add_argument('--vocab-size', type=int, default=50000)
    parser.add_argument('--embedding-dim', type=int, default=300)
    parser.add_argument('--class-n', type=int, default=4)
    parser.add_argument('--model', type=str, default='CNN')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--gpu-id', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    # args = Args()
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    print('using GPU id: ', os.environ['CUDA_VISIBLE_DEVICES'])

    if args.use_cuda and torch.cuda.is_available():
        args.use_cuda = True
    else:
        args.use_cuda = False

    trainer = Trainer(args)

    for i_epoch in range(1, args.epoch + 1):
        loss = trainer.train_one_epoch(1)
        print('%d th epoch: training loss -> %f' % (i_epoch, loss))
        loss, acc = trainer.test()
        print('test loss:  %f, test acc: %f' % (loss, acc))
