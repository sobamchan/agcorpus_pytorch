from lib.trainer import Trainer


class Args:

    def __init__(self):
        self.batch_size = 32
        self.lr = 0.001
        self.use_cuda = True
        self.max_sent_len = 20
        self.data_dir = './'
        self.vocab_size = 75000
        self.embedding_dim = 300
        self.class_n = 4


if __name__ == '__main__':
    args = Args()
    trainer = Trainer(args)
    trainer.train_one_epoch(1)
