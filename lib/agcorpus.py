import os
import tarfile
import csv
import requests


class AGCorpus:

    download_link = ('https://drive.google.com/uc'
                     '?export=download&id=0Bz8a_Dbh9QhbUDNpeUdjb0wxRms')

    def __init__(self, save_dir='./'):
        self.save_dir = save_dir
        self.tar_file_path = os.path.join(save_dir, 'ag_news_csv.tar.gz')
        self.data_dir = os.path.join(save_dir, 'ag_news_csv')
        self.train_path = os.path.join(self.data_dir, 'train.csv')
        self.test_path = os.path.join(self.data_dir, 'test.csv')
        self.train_labels = None
        self.train_texts = None
        self.test_labels = None
        self.test_texts = None

    def download(self):
        dest_path = self.tar_file_path
        if os.path.exists(dest_path):
            print('tar file already exists')
            return
        print('downloading dataset')
        r = requests.get(self.download_link, stream=True)
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

    def expand(self):
        if os.path.exists(self.data_dir):
            print('tar file has been already expanded')
            return
        f = tarfile.open(self.tar_file_path)
        f.extractall(self.save_dir)
        f.close()

    def prepare(self):
        self.download()
        self.expand()

    def load_dataset(self):
        self.prepare()
        with open(self.train_path, 'r') as f:
            train_labels, train_texts = list(zip(*(
                  (int(row[0]) - 1, '{} {}'.format(row[1], row[2]).lower())
                  for row in csv.reader(f))))
        with open(self.test_path, 'r') as f:
            test_labels, test_texts = list(zip(*(
                  (int(row[0]) - 1, '{} {}'.format(row[1], row[2]).lower())
                  for row in csv.reader(f))))

        self.train_labels = train_labels
        self.train_texts = train_texts
        self.test_labels = test_labels
        self.test_texts = test_texts

        return train_labels, train_texts, test_labels, test_texts
