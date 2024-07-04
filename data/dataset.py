import os
import cv2
import torch

import torchvision.transforms.v2 as transforms
from PIL import Image
from torchvision import transforms as T
import clip 


class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, captions, labels, tokenizer): #, transforms):
        """
        image_paths and cpations must have the same length; so, if there are
        multiple captions for each image, the image_paths must have repetitive
        file names 
        """

        self.image_paths = image_paths
        self.captions = list(captions)
        self.label = labels
        self.custom_tokenizer = False
        self.encoded_captions = tokenizer(
            list(captions), padding=True, truncation=True, max_length=200
        )
        # self.transforms = transforms
        self.transforms = T.Compose([
            T.Lambda(self.fix_img),
            T.RandomResizedCrop(640,
                                scale=(self.resize_ratio, 1.),
                                ratio=(1., 1.)),
            T.ToTensor(),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

    def __getitem__(self, idx):
        # item = {
        #     key: torch.tensor(values[idx])
        #     for key, values in self.encoded_captions.items()
        # }

        # image = Image.open(self.image_paths[idx]).convert("RGB")
        img_path = self.image_paths[idx]
        img_path = img_path.replace(img_path, f"{img_path}")
        # img_path = img_path.replace('', "")

        image = Image.open(img_path).convert('RGB')
        image = self.transforms(image)
        # item['image'] = image
        description = self.captions[idx]
        tokenized_text = description if self.custom_tokenizer else clip.tokenize(description)[0]

        # item['label'] = torch.tensor([1, 0]) if self.label[idx] else torch.tensor([0, 1])

        return image, tokenized_text   # , self.captions[idx]



    def __len__(self):
        return len(self.captions)

def get_transforms(mode="train"):
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),  # Always applied
            # transforms.ToTensor(),
            transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]),                    # Always applied
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Always applied
        ]
    )
