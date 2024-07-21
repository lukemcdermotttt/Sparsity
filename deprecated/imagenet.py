from datasets import load_dataset
from utils import *
from Sparsity.models.convnext import convnextv2_tiny

access_token = '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'
ds = load_dataset("imagenet-1k", token=access_token)
train_ds = ds["train"]
print(train_ds[0]["image"])  # a PIL Image


device = torch.device('cuda:0')
model = convnextv2_tiny(num_classes=1000).to(device)

#batch size 128 but across 8 gpus so gradient accumulation maybe to size 1024
#weight decay .05
#momentum .9
#optimizer adamw
#cosine lr scheduler
#train(model, train_loader, test_loader, './weights.pth', device, num_epochs = num_epochs, lr = 4e-3)