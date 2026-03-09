import torch
import numpy as np
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from cs231n.classifiers.transformer import VisionTransformer

train_data = CIFAR10(root='data', train=True, transform=transforms.ToTensor(), download=True)
batch = next(iter(DataLoader(train_data, batch_size=64, shuffle=False)))
imgs, target = batch

configs = [
    dict(lr=1e-4, wd=1e-4, dropout=0.0, epochs=100),
    dict(lr=3e-4, wd=0.0, dropout=0.0, epochs=150),
    dict(lr=1e-3, wd=0.0, dropout=0.0, epochs=150),
    dict(lr=3e-3, wd=0.0, dropout=0.0, epochs=150),
    dict(lr=1e-3, wd=0.0, dropout=0.0, epochs=300),
]

for cfg in configs:
    torch.manual_seed(231)
    np.random.seed(231)
    model = VisionTransformer(dropout=cfg['dropout'])
    loss_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    model.train()
    best = 0.0
    for epoch in range(cfg['epochs']):
        out = model(imgs)
        loss = loss_criterion(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        top1 = (out.argmax(-1) == target).float().mean().item()
        best = max(best, top1)
    print(cfg, 'final_loss=', float(loss.item()), 'final_top1=', top1, 'best_top1=', best)
