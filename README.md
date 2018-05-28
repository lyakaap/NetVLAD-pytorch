# NetVLAD-pytorch
Pytorch implementation of NetVLAD &amp; Online Hardest Triplet Loss.
In NetVLAD, broadcasting is used to calculate residuals of clusters and it makes whole calculation time much faster. 

NetVLAD: https://arxiv.org/abs/1511.07247

In Defense of the Triplet Loss for Person Re-Identification: https://arxiv.org/abs/1703.07737 https://omoindrot.github.io/triplet-loss

## Usage
```
import torch
import torch.nn as nn
from torch.autograd import Variable

from netvlad import NetVLAD
from netvlad import EmbedNet
from hard_triplet_loss import HardTripletLoss
from torchvision.models import resnet18


# Discard layers at the end of base network
base_model = nn.Sequential(*list(resnet18(pretrained=True).children())[:6])
dim = list(base_model.parameters())[-1].shape[0]  # last channels (128)

# Define model for embedding
net_vlad = NetVLAD(num_clusters=32, dim=dim, alpha=1.0)
model = EmbedNet(base_model, net_vlad).cuda()

# Define loss
criterion = HardTripletLoss(margin=0.1).cuda()

# It is just toy example. Typically, the number of samples in each classes are 4.
labels = torch.randint(0, 10, (40, )).long()
x = Variable(torch.rand(40, 3, 128, 128)).cuda()
output = model(x)

triplet_loss = criterion(output, labels)
```
