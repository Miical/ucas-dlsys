import time
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# 训练集增强
trans_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),        # 随机裁剪+padding
    transforms.RandomHorizontalFlip(),           # 随机水平翻转
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),   # 颜色抖动（增强）
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 验证集增强
trans_valid = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

trainset = torchvision.datasets.CIFAR10(
    root='/home/featurize/data',
    train=True,
    download=True,
    transform=trans_train
)

trainloader = torch.utils.data.DataLoader(
    dataset=trainset,
    batch_size=256,
    shuffle=True,
    num_workers=2
)

testset = torchvision.datasets.CIFAR10(
    root='/home/featurize/data',
    train=False,
    download=True,
    transform=trans_valid
)

testloader = torch.utils.data.DataLoader(
    dataset=testset,
    batch_size=256,
    shuffle=False,
    num_workers=2
)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

dataiter = iter(trainloader)
images, labels = next(dataiter)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head=64, dropout=0.):
        super(Attention, self).__init__()

        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, dim=512, depth=6, heads=8, dim_head=64, mlp_dim=512, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth,
                 heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super(ViT, self).__init__()

        image_height, image_width = image_size, image_size
        patch_height, patch_width = patch_size, patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool must be either cls or mean'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

net = ViT(
    image_size = 32,
    patch_size = 4,
    num_classes = 10,
    dim = 256,
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    pool = 'cls',
    channels = 3,
    dim_head = 64,
    dropout = 0.1,
    emb_dropout = 0.1
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.4, patience=2)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.4, patience=2, verbose=True)
# scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
criterion = nn.CrossEntropyLoss()
# net.load_state_dict(torch.load('checkpoint/ckpt_best.pth')['net'])

def progress_bar(current, total, msg=None):
    bar_len = 65
    filled_len = int(bar_len * current // total)
    bar = '█' * filled_len + '-' * (bar_len - filled_len)
    percent = f'{100.0 * current / total:.2f}'
    if msg:
        print(f'\r|{bar}| {percent}% {msg}', end='\r')
    else:
        print(f'\r|{bar}| {percent}%', end='\r')
    if current == total:
        print()

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | LR: %.5f'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, optimizer.param_groups[0]['lr']))
    return train_loss / (batch_idx + 1)

best_acc = 0.

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    scheduler.step(test_loss / (batch_idx + 1))
    # scheduler.step()
    acc = 100. * correct / total
    if acc > best_acc:
        best_acc = acc
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, 'checkpoint/ckpt.pth')
        best_acc = acc

    os.makedirs('log', exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch: {epoch} | Loss: {test_loss/(batch_idx+1)} | Acc: {acc}% | LR: {optimizer.param_groups[0]["lr"]}\n'
    print(content)
    with open('log/log.txt', 'a') as f:
        f.write(content)
    return test_loss / (batch_idx + 1)


if __name__ == '__main__':
    for epoch in range(1, 1001):
        train_loss = train(epoch)
        test_loss = test(epoch)
        if epoch % 5 == 0:
            torch.save(net.state_dict(), f'checkpoint/ckpt_{epoch}.pth')
