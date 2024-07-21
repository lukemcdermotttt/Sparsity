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
train_transform = transforms.Compose([transforms.ToTensor(),  transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])
test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])
train_dataset = torchvision.datasets.CIFAR10(root = './data', train = True, transform = train_transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root = './data', train = False, transform = test_transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size)


experiment = 'mlp_lrr'
accuracies = []
for i in range(41):
    if 'rn18' in experiment:
        model = ResNet18().to(device)
    else:
        model = MLPMixer(in_channels=3, image_size=32, patch_size=4, num_classes=10,
                        dim=128, depth=8, token_dim=256, channel_dim=512).to(device)
    

    if 'delta' in experiment:
        replace_with_delta(model)
        model = model.to(device)
        prune.global_unstructured(get_delta_parameters_to_prune(model, skip_last=True), pruning_method=prune.L1Unstructured,amount=0)
        model.load_state_dict(torch.load(f'weights/{experiment}/trained_iter_{i}.pth'))
        prune.global_unstructured(get_delta_parameters_to_prune(model, skip_last=True), pruning_method=prune.L1Unstructured,amount=0)
    else:
        prune.global_unstructured(get_parameters_to_prune(model, skip_last=True), pruning_method=prune.L1Unstructured,amount=0)
        model.load_state_dict(torch.load(f'weights/{experiment}/trained_iter_{i}.pth'))
        prune.global_unstructured(get_parameters_to_prune(model, skip_last=True), pruning_method=prune.L1Unstructured,amount=0)

    
    
    print(f'PRUNE ITERATION {i} AT SPARSITY {100*(1-.8**i):.5f}%')
    acc = evaluate(model, test_loader, device)
    accuracies.append(acc)
    torch.save(torch.tensor(accuracies), f'/home/archy2/luke/Sparsity/weights/{experiment}/accuracies.pt')
