# Imports
import os
import pandas as pd
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
spacy_eng = spacy.load("en_core_web_sm")

# 1. Vocabulary mapping each word to a index
# 2. Pytorch custom dataset
# 3. Padding to make sure all examples in batch are of same length

class Vocabulary:
    def __init__(self, freq_thresh):
        self.itos = {0:"<PAD>", 1:"<SOS>", 2:"<EOS>", 3:"<UNK>"}
        self.stoi = {"<PAD>":0, "<SOS>":1, "<EOS>":2, "<UNK>":3}
        self.freq_thresh = freq_thresh

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [token.text.lower() for token in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4  # 0-3 are predefined

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                frequencies[word] = frequencies.get(word,0) + 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text]

class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform

        # Get img,caption pair
        self.imgs = self.df["image"]
        self.captions = self.df["captions"]

        # Initialize and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.covab.build_vocab(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        img_name = self.imgs[idx]
        img = Image.open(os.path.join(self.root_dir,img_name)).convert("RBG")

        if self.transform:
            img = self.transform(img)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption = [self.vocab.stoi["<EOS>"]]

        return img, torch.tensor(numericalized_caption)

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False)
        return imgs, targets

def get_loader(
        root_folder,
        annotation_file,
        transform,
        batch_size = 32,
        num_workers = 4,
        shuffle = True,
        pin_memory = True
):
    dataset = FlickrDataset(root_folder)
    pad_idx = dataset.vocab.stoi["<PAD>"]
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx)
    )
    return loader
path_name = 'xyz'
dataloader = get_loader(path_name,annotation_file=path_name,transform=None)

'''
Note : 
In PyTorch, the `collate_fn` parameter in the `DataLoader` can be either a function or an object of a class. Both approaches are valid, and the choice depends on your preference and the complexity of your collation logic.

1. Function as `collate_fn`:
def my_collate_fn(batch):
    # Your custom collation logic here
    return processed_batch
# Use the function with DataLoader
train_loader = DataLoader(dataset, batch_size=64, collate_fn=my_collate_fn)

2. Class as `collate_fn`:
class MyCollateClass:
    def __call__(self, batch):
        # Your custom collation logic here
        return processed_batch
# Instantiate the class and use it with DataLoader
my_collate_instance = MyCollateClass()
train_loader = DataLoader(dataset, batch_size=64, collate_fn=my_collate_instance)

Using a class allows you to maintain state between batches if needed, as the class instance retains its state between calls. This can be beneficial if your collation logic requires some persistent information.

The key point is that the `collate_fn` parameter should be a callable (a function or an object with a `__call__` method) that takes a list of batch data and returns the processed batch. The processing typically involves padding sequences, converting data types, or any other necessary steps to prepare the batch for the model.
'''




