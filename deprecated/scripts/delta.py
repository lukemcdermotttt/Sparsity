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
#model = MLPMixer(in_channels=3, image_size=32, patch_size=4, num_classes=10,
#                 dim=128, depth=8, token_dim=256, channel_dim=512).to(device)
model = ResNet18().to(device) #convnextv2_atto_cifar(num_classes=10).to(device) 

train_transform = transforms.Compose([transforms.ToTensor(),  transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])
test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])
train_dataset = torchvision.datasets.CIFAR10(root = './data', train = True, transform = train_transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root = './data', train = False, transform = test_transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size)

delta_pruning(model, train_loader, test_loader, './weights/rn18_delta/', device, num_prune_iters = 40, num_epochs = num_epochs, lr = lr)








"""
print('OG Model')
print(model.mixer_blocks[0].token_mix[2].net[0].weight.data[:3,:3])
replace_with_delta(model)
model = model.to(device)
print('Store the nn.Module Layer')
print(model.mixer_blocks[0].token_mix[2].net[0].layer.weight.data[:3,:3])
print('Save the initialization as base_weight')
print(model.mixer_blocks[0].token_mix[2].net[0].base_weight[:3,:3])
print('Initialize a delta weight of zeros')
print(model.mixer_blocks[0].token_mix[2].net[0].delta_weight.data[:3,:3])
prune.global_unstructured(get_delta_parameters_to_prune(model), pruning_method=prune.L1Unstructured,amount=.0)
print('Initialize the delta weight sparsity mask by pruning 0%')
print(model.mixer_blocks[0].token_mix[2].net[0].delta_weight_mask[:3,:3])
#prune.global_unstructured(get_delta_parameters_to_prune(model), pruning_method=prune.L1Unstructured,amount=.99)
#print('Check that this sparsity mask is updated')
#print(model.mixer_blocks[0].token_mix[2].net[0].delta_weight_mask[:3,:3])


delta_train(model, train_loader, test_loader, './weights/mlp_delta/test.pth',device,1)
print('Store the nn.Module Layer')
print(model.mixer_blocks[0].token_mix[2].net[0].layer.weight.data[:3,:3])
print('Save the initialization as base_weight')
print(model.mixer_blocks[0].token_mix[2].net[0].base_weight[:3,:3])
print(model.mixer_blocks[0].token_mix[2].net[0].delta_weight.data[:3,:3])
"""
