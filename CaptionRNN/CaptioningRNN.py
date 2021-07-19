import tensorboard
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import argparse


class CocoCaptionDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class CocoCaptionsLT(pl.LightningDataModule):

    def __init__(self, folder_path, batch_size=64, worker=6, caption_idx=2, transformer=None):
        '''
        Input:
          folder_path: A folder that contains both train and validation images and annotation (.json)
          caption_idx: Original file contain 4 or 5 captions per image. We will only pick one according to index.
          transformers: Pytorch data transforms that will used during preprocess data. 
                        **All data are prepocessed during set up phase. i.e. no transform are used during training**
        '''
        assert caption_idx < 4

        super().__init__()
        self.folder_path = folder_path
        self.batch_size = batch_size
        self.caption_idx = caption_idx
        self.transformer = transformer
        self.worker = worker

        try:
            print(f'Loading coco2017_caption{self.caption_idx}_data_dict')
            self.data_dict = torch.load(os.path.join(
                folder_path, f'coco2017_caption{self.caption_idx}_data_dict'))
        except:
            self.data_dict = None
            train_img_path = os.path.join(folder_path, 'train2017')
            train_caption_path = os.path.join(
                folder_path, 'annotations/captions_train2017.json')
            val_img_path = os.path.join(folder_path, 'val2017')
            val_caption_path = os.path.join(
                folder_path, 'annotations/captions_val2017.json')

            if self.transformer is None:
                self.transformer = transforms.Compose([
                    transforms.Resize((112, 112)),
                    transforms.ToTensor()
                ])

            self.train_dataset = torchvision.datasets.CocoCaptions(
                train_img_path, train_caption_path, transform=self.transformer)
            self.val_dataset = torchvision.datasets.CocoCaptions(
                val_img_path, val_caption_path, transform=self.transformer)

    def prepare_data(self):

        if self.data_dict is None:
            print('Load fail. Build dataset from scratch. This will take a while!!')
            self.vocab_to_idx, self.idx_to_vocab = self._create_vocab_idx_mapping(
                self.train_dataset, self.caption_idx)

            self.train_dataset = self._tokenize_captions(
                self.train_dataset, self.caption_idx, self.vocab_to_idx)

            self.val_dataset = self._tokenize_captions(
                self.val_dataset, self.caption_idx, self.vocab_to_idx)

            self.data_dict = {"vocab_to_idx": self.vocab_to_idx,
                              "idx_to_vocab": self.idx_to_vocab,
                              "train_dataset": self.train_dataset,
                              "val_dataset": self.val_dataset,
                              'vocab_count': self.vocab_count}

            print('saving dataset....')
            torch.save(self.data_dict, os.path.join(self.folder_path,
                       f'coco2017_caption{self.caption_idx}_data_dict'))
        else:
            print('Load successful.')
            self.vocab_to_idx = self.data_dict["vocab_to_idx"]
            self.idx_to_vocab = self.data_dict["idx_to_vocab"]
            self.train_dataset = self.data_dict['train_dataset']
            self.val_dataset = self.data_dict['val_dataset']
            self.vocab_count = self.data_dict['vocab_count']

    def train_dataloader(self):
        trainDataLoader = torch.utils.data.DataLoader(self.train_dataset, self.batch_size,
                                                      shuffle=True, num_workers=self.worker, pin_memory=True)
        return trainDataLoader

    def val_dataloader(self):
        valDataLoader = torch.utils.data.DataLoader(self.val_dataset, self.batch_size,
                                                    shuffle=False, num_workers=self.worker, pin_memory=True)
        return valDataLoader

    def sample(self, num_samples=4):
        idx = torch.randint(0, len(self.train_dataset),
                            (num_samples,), dtype=torch.int)

        for i in idx:
            img, cap = self.train_dataset[i]
            plt.imshow(img.permute(1, 2, 0).numpy())
            plt.title(self._decode_captions(cap))
            plt.axis('off')
            plt.show()

    def _tokenize_captions(self, dataset, caption_idx, vocab_to_idx, fix_length=16):
        '''
        Add <'START'> token at the beginning of the caption,
        Add <'END'> token at the end of the caption,
        Replace unknown words into <'UNK'> token,
        Pad caption to a fix length with <'NULL'> tokens
        and tanslate all into index and return as a Pytorch dataset

        Input:
          dataset: dataset which are gonna be used to count words
          caption_idx: Each image in original Coco dataset has 4-5 captions. 
                       We will only use *ONE* of them according to the index.
                       Index should be less than 4
        Return:
          dataset: Pytorch dataset which can be used in dataloader

        '''
        assert caption_idx < 4
        print("Start tokenize dataset. This will take a while!!")

        master = []
        for image, captions in tqdm(dataset):

            caption = captions[caption_idx]
            caption = caption.lower().split()
            words = [vocab_to_idx['<START>']]
            for i in range(fix_length):

                if i < len(caption):
                    word = caption[i].strip('.')
                    try:
                        words.append(vocab_to_idx[word])
                    except:
                        words.append(vocab_to_idx['<UNK>'])

                elif i == len(caption):
                    words.append(vocab_to_idx['<END>'])

                else:
                    words.append(vocab_to_idx['<NULL>'])

            master.append((image, torch.tensor(words)))
        return CocoCaptionDataset(master)

    def _create_vocab_idx_mapping(self, dataset, caption_idx, num_words=1000):
        '''
        Use the first 5k examples to map vocabulary to idx and
        grab the first 1k most frequent words in the datasets

        Input:
          dataset: dataset which are gonna be used to count words
          caption_idx: Each image in original Coco dataset has 4-5 captions. 
                       We will only use *ONE* of them according to the index.
                       Index should be less than 4

        Output:
          vocab_to_idx: dictionary
          idx_to_vocab: list
        '''
        assert caption_idx < 4
        from collections import defaultdict
        print('Start creating vocab index mapping. This will take several minutes!!')

        smaller_dataset, _ = torch.utils.data.random_split(
            dataset, [5000, len(dataset)-5000])
        vocab_count = defaultdict(int)
        for _, captions in tqdm(smaller_dataset):
            caption = captions[caption_idx]
            caption = caption.lower().split()

            for word in caption:
                word = word.strip('.')
                if len(word) > 0:
                    vocab_count[word] += 1

        special_tokens = [('<NULL>', 0), ('<START>', 0),
                          ('<END>', 0), ('<UNK>', 0)]
        ordered = list(sorted(vocab_count.items(),
                       key=lambda item: item[1], reverse=True))
        first1KTuple = sorted(
            ordered[:num_words-len(special_tokens)], key=lambda item: item[0])

        vocab_to_idx = {x[0]: i for x, i in zip(
            special_tokens + first1KTuple, range(num_words))}
        idx_to_vocab = [k for k, v in vocab_to_idx.items()]

        for i in range(num_words):
            word = idx_to_vocab[i]
            assert vocab_to_idx[word] == i, 'Vocab mapping failed, words are missed match'

        self.vocab_count = vocab_count
        return vocab_to_idx, idx_to_vocab

    def _decode_captions(self, captions):
        """
        Decoding caption indexes into words.
        Inputs:
        - captions: Caption indexes in a tensor of shape (Nx)T.
        - idx_to_word: Mapping from the vocab index to word.
        Outputs:
        - decoded: A sentence (or a list of N sentences).
        """
        singleton = False
        if captions.ndim == 1:
            singleton = True
            captions = captions[None]
        decoded = []
        N, T = captions.shape
        for i in range(N):
            words = []
            for t in range(T):
                word = self.idx_to_vocab[captions[i, t]]
                if word != '<NULL>':
                    words.append(word)
                if word == '<END>':
                    break
            decoded.append(' '.join(words))
        if singleton:
            decoded = decoded[0]
        return decoded


class CaptioningRNN(pl.LightningModule):

    def __init__(self, lr=1e-3, img_size=2048, time_stamps=16, hidden_size=512, vocab_vec_size=128, vocab_size=1000):

        super().__init__()
        self.hparams = {'lr': lr}
        self.T = time_stamps
        self.save_hyperparameters()

        model = torchvision.models.resnet50(pretrained=True)
        # remove the last classifer
        self.backBone = nn.Sequential(*list(model.children())[:-1])

        # unfreeze backbone parameters
        for param in self.backBone.parameters():
            param.requires_grad = True

        self.feat_to_h0 = nn.Linear(img_size, hidden_size)

        self.cell = nn.LSTMCell(vocab_vec_size, hidden_size)

        self.wordEmbed = nn.Embedding(vocab_size, vocab_vec_size)

        self.cellOuput_to_vocab = nn.Linear(hidden_size, vocab_size)

        # information need to know when load data
        self.NULL = 0
        self.START = 1

    def _extract_image_features(self, x):
        feat = self.backBone(x)
        feat = feat.squeeze()
        return feat

    def _loss(self, x, y):
        return F.cross_entropy(x.reshape(-1, x.shape[2]), y.reshape(-1,), ignore_index=self.NULL, reduction='sum') / x.shape[0]

    def forward(self, x):

        B = x.shape[0]

        img_feat = self._extract_image_features(x)  # (B, img_size)

        # Initial hidden state as image features
        h = self.feat_to_h0(img_feat)

        # Initial cell state as zeros
        c = torch.zeros_like(h)

        # Initial start words
        words = self.wordEmbed(torch.full((B,), self.START, device=self.device))

        rawVocabScores = []

        for t in range(self.T):
            h, c = self.cell(words, (h, c))

            # Affine transform hidden state to vocab size
            vocabScore = self.cellOuput_to_vocab(h)

            # Grab the maximum idx of words
            predict = torch.argmax(vocabScore, dim=1)

            # update words for the next iteration
            words = self.wordEmbed(predict)

            # saved for outputs
            rawVocabScores.append(vocabScore)

        out = torch.stack(rawVocabScores, dim=1)

        return out

    def training_step(self, batch, batch_idx):
        x, y = batch

        captions_in = y[:, :-1]
        captions_out = y[:, 1:]

        # Grab embedding of ground truth words
        # (B, sentence_length, word_vec_size)
        GTwords = self.wordEmbed(captions_in)

        # Extract image features
        img_feat = self._extract_image_features(x)  # (B, img_size)

        # Affine transform image features to match word embeding size
        h = self.feat_to_h0(img_feat)

        # initial cell state
        c = torch.zeros_like(h)

        output = []

        # Use image features as initial hidden and cell state
        for t in range(self.T):
            h, c = self.cell(GTwords[:, t, :], (h, c))
            output.append(h)

        output = torch.stack(output, dim=1)
        # Affine transform output to vocabulary size
        # (B, sentence_length, vocab_size)
        vocabScore = self.cellOuput_to_vocab(output)

        loss = self._loss(vocabScore, captions_out)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        captions_out = y[:, 1:]

        vocabScore = self(x)  # calling forward here

        loss = self._loss(vocabScore, captions_out)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams['lr'])

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)


if __name__ == '__main__':

    pl.seed_everything(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=512, help='size of each batch for train and validation dataloader')
    parser.add_argument('--worker', type=int, default=0, help='number of workers for dataloader')
    parser.add_argument('--epoch', type=int, default=None, help='number of max epoch for training')

    args = parser.parse_args()

    folder_path = "/home/fred/datasets/coco/"
    DataModule = CocoCaptionsLT(folder_path, batch_size = args.batch_size, worker=args.worker)


    model = CaptioningRNN()
    trainer = pl.Trainer(gpus = 1, precision=16, max_epochs=args.epoch)

    trainer.fit(model, DataModule)

