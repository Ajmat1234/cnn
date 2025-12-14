import torch
from torch.utils.data import Dataset
import lmdb
from PIL import Image
import six
import torchvision.transforms as transforms

class lmdbDataset(Dataset):
    def __init__(self, root, transform=None):
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False,
                             readahead=False, meminit=False)
        if not self.env:
            raise Exception(f'Cannot open LMDB at {root}')

        with self.env.begin() as txn:
            self.nSamples = int(txn.get(b'num-samples') or 0)

        self.transform = transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index < len(self), 'Index out of range'
        index += 1
        with self.env.begin() as txn:
            img_key = f'image-{index:09d}'.encode()
            imgbuf = txn.get(img_key)

            label_key = f'label-{index:09d}'.encode()
            label = txn.get(label_key).decode('utf-8')

            img = Image.open(six.BytesIO(imgbuf)).convert('L')  # Grayscale

        if self.transform:
            img = self.transform(img)

        return img, label

class resizeNormalize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img

class alignCollate(object):
    def __init__(self, imgH=32, imgW=100):
        self.imgH = imgH
        self.imgW = imgW
        self.transform = resizeNormalize((imgW, imgH))  # Fixed: init mein bana

    def __call__(self, batch):
        images, labels = zip(*batch)
        images = [self.transform(img) for img in images]
        images = torch.cat([img.unsqueeze(0) for img in images], 0)
        return images, labels
