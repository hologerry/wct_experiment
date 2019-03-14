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
    def __init__(self, content_path, style_path, fine_size):
        super(TestDataset, self).__init__()
        self.content_path = content_path
        self.image_list = [x for x in listdir(content_path) if is_image_file(x)]
        self.style_path = style_path
        self.fine_size = fine_size
        # self.normalize = transforms.Normalize(mean=[103.939,116.779,123.68],std=[1, 1, 1])
        # normalize = transforms.Normalize(mean=[123.68,103.939,116.779],std=[1, 1, 1])
        self.prep = transforms.Compose([
                    transforms.Resize(fine_size),
                    transforms.ToTensor(),
                    # transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]),  # turn to BGR
                    ])

    def __getitem__(self, index):
        content_img_path = os.path.join(self.content_path, self.image_list[index])
        style_img_path = os.path.join(self.style_path, self.image_list[index])
        content_img = default_loader(content_img_path)
        style_img = default_loader(style_img_path)

        # resize
        if(self.fine_size != 0):
            w, h = content_img.size
            if(w > h):
                if(w != self.fine_size):
                    neww = self.fine_size
                    newh = int(h*neww/w)
                    content_img = content_img.resize((neww, newh))
                    style_img = style_img.resize((neww, newh))
            else:
                if(h != self.fine_size):
                    newh = self.fine_size
                    neww = int(w*newh/h)
                    content_img = content_img.resize((neww, newh))
                    style_img = style_img.resize((neww, newh))

        # Preprocess Images
        content_img = transforms.ToTensor()(content_img)
        style_img = transforms.ToTensor()(style_img)
        return content_img.squeeze(0), style_img.squeeze(0), self.image_list[index]

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
