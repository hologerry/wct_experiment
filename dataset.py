import os
from os import listdir

import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def default_loader(path):
    return Image.open(path).convert('RGB')


class TestDataset(data.Dataset):
    def __init__(self, contentPath, stylePath, fineSize):
        super(TestDataset, self).__init__()
        self.contentPath = contentPath
        self.image_list = [x for x in listdir(contentPath) if is_image_file(x)]
        self.stylePath = stylePath
        self.fineSize = fineSize
        # self.normalize = transforms.Normalize(mean=[103.939,116.779,123.68],std=[1, 1, 1])
        # normalize = transforms.Normalize(mean=[123.68,103.939,116.779],std=[1, 1, 1])
        self.prep = transforms.Compose([
                    transforms.Resize(fineSize),
                    transforms.ToTensor(),
                    # transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                    ])

    def __getitem__(self, index):
        contentImgPath = os.path.join(self.contentPath, self.image_list[index])
        styleImgPath = os.path.join(self.stylePath, self.image_list[index])
        contentImg = default_loader(contentImgPath)
        styleImg = default_loader(styleImgPath)

        # resize
        if(self.fineSize != 0):
            w, h = contentImg.size
            if(w > h):
                if(w != self.fineSize):
                    neww = self.fineSize
                    newh = int(h*neww/w)
                    contentImg = contentImg.resize((neww, newh))
                    styleImg = styleImg.resize((neww, newh))
            else:
                if(h != self.fineSize):
                    newh = self.fineSize
                    neww = int(w*newh/h)
                    contentImg = contentImg.resize((neww, newh))
                    styleImg = styleImg.resize((neww, newh))

        # Preprocess Images
        contentImg = transforms.ToTensor()(contentImg)
        styleImg = transforms.ToTensor()(styleImg)
        return contentImg.squeeze(0), styleImg.squeeze(0), self.image_list[index]

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.image_list)


class TrainDataset(data.Dataset):
    def __init__(self, img_dir, img_size):
        super(TrainDataset, self).__init__()
        self.img_dir = img_dir
        self.img_list = [x for x in listdir(self.img_dir) if is_image_file(x)]
        self.trans = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_list[index])
        img = default_loader(img_path)
        img = self.trans(img)
        return img.squeeze(0)

    def __len__(self):
        return len(self.img_list)
