import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from models.crnn import CRNN
import dataset
from utils import strLabelConverter

parser = argparse.ArgumentParser()
parser.add_argument('--trainRoot', required=True, help='path to train LMDB')
parser.add.add_argument('--valRoot', required=True, help='path to val LMDB')
parser.add_argument('--batchSize', type=int, default=64)
parser.add_argument('--imgH', type=int, default=32)
parser.add_argument('--imgW', type=int, default=200, help='wider for handwriting')
parser.add_argument('--nh', type=int, default=256)
parser.add_argument('--nepoch', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--alphabet', type=str, 
                    default="अआइईउऊएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहािीुूेैोौंःँ्०१२३४५६७८९")
parser.add_argument('--cuda', action='store_true', default=True)
parser.add_argument('--expr_dir', default='expr', help='save checkpoints here')
opt = parser.parse_args()

os.makedirs(opt.expr_dir, exist_ok=True)

# Datasets
train_dataset = dataset.lmdbDataset(root=opt.trainRoot)
val_dataset = dataset.lmdbDataset(root=opt.valRoot,
                                  transform=dataset.resizeNormalize((opt.imgW, opt.imgH)))

train_loader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True,
                          collate_fn=dataset.alignCollate(opt.imgH, opt.imgW))
val_loader = DataLoader(val_dataset, batch_size=opt.batchSize, shuffle=False)

# Model
nclass = len(opt.alphabet) + 1
crnn = CRNN(opt.imgH, nc=1, nclass=nclass, nh=opt.nh)

if opt.cuda and torch.cuda.is_available():
    crnn.cuda()

criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = torch.optim.Adam(crnn.parameters(), lr=opt.lr)
converter = strLabelConverter(opt.alphabet)

print(f"Classes: {nclass}, Alphabet length: {len(opt.alphabet)}")
print("Training started...")

for epoch in range(opt.nepoch):
    crnn.train()
    total_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        if opt.cuda and torch.cuda.is_available():
            images = images.cuda()

        preds = crnn(images)
        preds_size = torch.IntTensor([preds.size(0)] * images.size(0))
        text, length = converter.encode(labels)

        if opt.cuda and torch.cuda.is_available():
            text = text.cuda()
            length = length.cuda()

        loss = criterion(preds.log_softmax(2), text, preds_size, length)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{opt.nepoch}] | Batch [{i+1}] | Loss: {loss.item():.4f}')

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} completed | Avg Loss: {avg_loss:.4f}")

    # Save checkpoint
    torch.save(crnn.state_dict(), f"{opt.expr_dir}/crnn_epoch_{epoch+1}.pth")

print("Training finished!")
