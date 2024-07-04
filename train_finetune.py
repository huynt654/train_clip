import torch
from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything
from models import CustomCLIPWrapper
from torchvision.models import resnet50
from transformers import AutoTokenizer, AutoModel
import pandas as pd 
from data.dataset import Dataset

def build_loader(dataframe, tokenizer, mode, batch_size=2):
    # transforms = get_transforms(mode=mode)
    dataset = Dataset(
        dataframe["image_paths"].values,
        dataframe["descriptions"].values,
        dataframe["labels"].values,
        tokenizer=tokenizer
        # transforms=transforms,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True if mode == "train" else False,
    )
    return dataloader
    

def build_loaders(dataframe, tokenizer, batch_size):
    # Split dataframe
    train_dataframe = dataframe[dataframe['set'] == 'train'].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe['set'] == 'valid'].reset_index(drop=True)
    test_dataframe = dataframe[dataframe['set'] == 'test'].reset_index(drop=True)
    # Build dataloader
    train_loader = build_loader(train_dataframe, tokenizer, mode="train", batch_size=batch_size)
    valid_loader = build_loader(valid_dataframe, tokenizer, mode="valid", batch_size=batch_size)
    test_loader = build_loader(test_dataframe, tokenizer, mode="test", batch_size=batch_size)
    return train_loader, valid_loader, test_loader


def main(hparams):

    seed_everything(16)


    tokenizer = AutoTokenizer.from_pretrained("johngiorgi/declutr-sci-base")
    df = pd.read_csv('annotation.csv')
        
    train_loader, valid_loader, test_loader = build_loaders(df, tokenizer, batch_size=256)
        
    

    # if hparams.minibatch_size < 1:
    #     hparams.minibatch_size = hparams.batch_size


    img_encoder = resnet50(pretrained=True)
    img_encoder.fc = torch.nn.Linear(2048, 768)
    txt_encoder = AutoModel.from_pretrained("johngiorgi/declutr-sci-base")
    model = CustomCLIPWrapper(img_encoder, txt_encoder, minibatch_size=256, avg_word_embs=True)
    # dm = TextImageDataModule.from_argparse_args(hparams, custom_tokenizer=tokenizer)



    trainer = Trainer.from_argparse_args(hparams, precision=16, max_epochs=32)
    trainer.fit(model, train_loader)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--minibatch_size', type=int, default=0)
    # parser = TextImageDataModule.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
