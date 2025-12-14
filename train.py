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
parser.add_argument('--valRoot', required=True, help='path to val LMDB')
parser.add_argument('--batchSize', type=int, default=64)
parser.add_argument('--imgH', type=int, default=32)
parser.add_argument('--imgW', type=int, default=200, help='wider for handwriting')
parser.add_argument('--nh', type=int, default=256)
parser.add_argument('--nepoch', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--alphabet', type=str, 
                    default="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZअआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहा ि ी ु ू ृ े ै ो ौं ः ँ ्०१२३४५६७८९")
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
val_loader = DataLoader(val_dataset, batch_size=opt.batchSize, shuffle=False,
                        collate_fn=dataset.alignCollate(opt.imgH, opt.imgW))

# Model
nclass = len(opt.alphabet) + 1
crnn = CRNN(opt.imgH, nc=1, nclass=nclass, nh=opt.nh)

device = torch.device('cuda' if opt.cuda and torch.cuda.is_available() else 'cpu')
crnn.to(device)

criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = torch.optim.Adam(crnn.parameters(), lr=opt.lr)
converter = strLabelConverter(opt.alphabet)

print(f"Classes: {nclass}, Alphabet length: {len(opt.alphabet)}")
print("Training started...")

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds = model(images).log_softmax(2)
            preds_size = torch.IntTensor([preds.size(0)] * images.size(0))
            text, length = converter.encode(labels)
            text = text.to(device)
            length = length.to(device)
            loss = criterion(preds, text, preds_size, length)
            total_loss += loss.item()
    return total_loss / len(loader)

for epoch in range(opt.nepoch):
    crnn.train()
    total_train_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        preds = crnn(images).log_softmax(2)  # Fixed: consistent log_softmax
        preds_size = torch.IntTensor([preds.size(0)] * images.size(0))
        text, length = converter.encode(labels)
        text = text.to(device)
        length = length.to(device)
        loss = criterion(preds, text, preds_size, length)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{opt.nepoch}] | Batch [{i+1}/{len(train_loader)}] | Loss: {loss.item():.4f}')

    avg_train_loss = total_train_loss / len(train_loader)
    val_loss = validate(crnn, val_loader, criterion, device)
    print(f"Epoch {epoch+1} completed | Avg Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Save checkpoint
    torch.save(crnn.state_dict(), f"{opt.expr_dir}/crnn_epoch_{epoch+1}.pth")

print("Training finished!")
