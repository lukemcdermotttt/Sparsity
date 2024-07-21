from Sparsity.models.resnet import *
from Sparsity.models.convnext import * 
from utils import *
from Sparsity.models.mlp_mixer import MLPMixer
import torchvision.transforms as transforms
import torchvision

#Hyperparameters
batch_size = 128
lr = 1e-3
num_epochs = 200


device = torch.device('cuda:0')
#model = ResNet18().to(device) #convnextv2_atto_cifar(num_classes=10).to(device) 
model = MLPMixer(in_channels=3, image_size=32, patch_size=4, num_classes=10,
                                dim=128, depth=8, token_dim=256, channel_dim=512).to(device) #
#If loading pruned model:
#prune.global_unstructured(get_parameters_to_prune(model), pruning_method=prune.L1Unstructured,amount=0)
#model.load_state_dict(torch.load('/home/archy2/luke/Sparsity/weights/mlp_ltr/trained_iter_10.pth'))
#prune.global_unstructured(get_parameters_to_prune(model), pruning_method=prune.L1Unstructured,amount=0)
#for layer, name in get_parameters_to_prune(model):
#    print(name, 1 - torch.mean(layer.weight_mask), layer.weight_mask.numel())

#If loading delta pruned model:
replace_with_delta(model)
model = model.to(device)
prune.global_unstructured(get_delta_parameters_to_prune(model, skip_last=True), pruning_method=prune.L1Unstructured,amount=0)
model.load_state_dict(torch.load('/home/archy2/luke/Sparsity/weights/mlp_delta/trained_iter_40.pth'))
prune.global_unstructured(get_delta_parameters_to_prune(model, skip_last=True), pruning_method=prune.L1Unstructured,amount=0)
print(get_delta_sparsity(model))
count, total = 0, 0
for layer, name in get_delta_parameters_to_prune(model):
    count += torch.sum(layer.delta_weight_mask).item()
    total += layer.delta_weight_mask.numel()
    print(f'Sparsity: {100 * (1 - torch.mean(layer.delta_weight_mask).item()):.3f}%, only {torch.sum(layer.delta_weight_mask).item():.0f}/{layer.delta_weight_mask.numel()} weights remaining.')
print(f'{count:.0f} weights left out of {total} total.')


train_transform = transforms.Compose([transforms.ToTensor(),  transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])
test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])
train_dataset = torchvision.datasets.CIFAR10(root = './data', train = True, transform = train_transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root = './data', train = False, transform = test_transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size)

#train(model, train_loader, test_loader, './weights.pth', device, num_epochs = num_epochs, lr = lr)
evaluate(model, test_loader, device)