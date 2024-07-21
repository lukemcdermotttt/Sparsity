import argparse
from models import build_model
from utils import *
import torchvision.transforms as transforms
import torchvision

set_seed()

def parse_args():
    parser = argparse.ArgumentParser(description="Prune a model on CIFAR-10.")
    #parser.add_argument("--config-file", default="configs/train_config.py")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--model", type=str, default='resnet-18', 
                        choices=['resnet-18', 'mlp_mixer', 'convnext'])
    parser.add_argument("--prune_method", type=str, default='lottery_ticket_rewinding', 
                        choices=['lottery_ticket_rewinding', 
                                'early_training_rewinding',
                                'learning_rate_rewinding',
                                'delta_pruning',
                                'delta_pruning_no_rewinding'])

    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--num_prune_iters", type=int, default=40)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    device = torch.device('cuda:0')

    model = build_model(args.model).to(device)

    train_transform = transforms.Compose([transforms.ToTensor(),  transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])
    
    train_dataset = torchvision.datasets.CIFAR10(root = './data', train = True, transform = train_transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root = './data', train = False, transform = test_transform, download=True)

    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = args.batch_size)

    prune = get_prune_method(args.prune_method)
    prune(model, train_loader, test_loader, args.save_dir, device, num_prune_iters = args.num_prune_iters, num_epochs = args.num_epochs, lr = args.lr)

