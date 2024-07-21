from Sparsity.models.resnet import *
from Sparsity.models.convnext import * 
from utils import *
import torchvision.transforms as transforms
import torchvision
from Sparsity.models.mlp_mixer import MLPMixer

set_seed()

#Hyperparameters
batch_size = 128
lr = 1e-3
num_epochs = 200


device = torch.device('cuda:0')
model = model = MLPMixer(in_channels=3, image_size=32, patch_size=4, num_classes=10,
                dim=128, depth=8, token_dim=256, channel_dim=512).to(device) #
#model = ResNet18().to(device) #convnextv2_atto_cifar(num_classes=10).to(device) 

train_transform = transforms.Compose([transforms.ToTensor(),  transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])
test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])
train_dataset = torchvision.datasets.CIFAR10(root = './data', train = True, transform = train_transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root = './data', train = False, transform = test_transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size)

#prune.global_unstructured(get_parameters_to_prune(model), pruning_method=prune.L1Unstructured,amount=0)
#model.load_state_dict(torch.load('./weights/mlp_ltr/initialization.pth'))
early_training_rewinding(model, train_loader, test_loader, './weights/mlp_k=5/', device, num_epochs = num_epochs, lr = lr, k = 5)