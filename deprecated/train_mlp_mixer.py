from Sparsity.models.resnet import *
from Sparsity.models.convnext import * 
from utils import *
from Sparsity.models.mlp_mixer import MLPMixer
import torchvision.transforms as transforms
import torchvision
import torch.nn.utils.prune as prune

#Hyperparameters
batch_size = 256
lr = 1e-3
num_epochs = 100



device = torch.device('cuda:0')
#model = MLPMixer(in_channels=3, image_size=224, patch_size=16, num_classes=10,
#                 dim=512, depth=8, token_dim=256, channel_dim=2048).to(device)

#look at github for a cifar-10 implementation. you have this open
model = MLPMixer(in_channels=3, image_size=32, patch_size=4, num_classes=10,
                 dim=128, depth=8, token_dim=256, channel_dim=512).to(device)

prune.global_unstructured(get_parameters_to_prune(model), pruning_method=prune.L1Unstructured,amount=0)
model.load_state_dict(torch.load('/home/archy2/luke/Sparsity/weights/mlp_ltr/trained_iter_1.pth'))


#ResNet18().to(device) #convnextv2_atto_cifar(num_classes=10).to(device) 
#train_transform = transforms.Compose([transforms.ToTensor(),  transforms.Resize((256,256)), transforms.RandomCrop((224,224)), transforms.RandomHorizontalFlip(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])
#test_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224)), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])
train_transform = transforms.Compose([transforms.ToTensor(),  transforms.RandomHorizontalFlip(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])
test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])
train_dataset = torchvision.datasets.CIFAR10(root = './data', train = True, transform = train_transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root = './data', train = False, transform = test_transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size)

#train(model, train_loader, test_loader, './weights.pth', device, num_epochs = num_epochs, lr = lr)
evaluate(model, test_loader, device)