import argparse
from models import build_model
from utils import *
import torchvision.transforms as transforms
import torchvision

set_seed()

def parse_args():
    parser = argparse.ArgumentParser(description="Take a low rank decomposition of the delta & measure loss in accuracy.")
    #parser.add_argument("--config-file", default="configs/train_config.py")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--model", type=str, default='resnet-18', 
                        choices=['resnet-18', 'mlp_mixer', 'convnext'])
    parser.add_argument("--save_dir", type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    device = torch.device('cuda:0')


    train_transform = transforms.Compose([transforms.ToTensor(),  transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])
    
    train_dataset = torchvision.datasets.CIFAR10(root = './data', train = True, transform = train_transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root = './data', train = False, transform = test_transform, download=True)

    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = args.batch_size)

    results = torch.zeros((30,10)) #prune_iter X rank_reduction
    for i in range(0,31):
        for rank_idx, rank_reduction in enumerate(.1 * torch.arange(1,11)):
            model = build_model(args.model).to(device)

            if 'delta' in args.save_dir:
                replace_with_delta(model)
                model = model.to(device)
                prune.global_unstructured(get_delta_parameters_to_prune(model, skip_last=True), pruning_method=prune.L1Unstructured,amount=0)
                model.load_state_dict(torch.load(f'{args.save_dir}/trained_iter_{i}.pth'))
                prune.global_unstructured(get_delta_parameters_to_prune(model, skip_last=True), pruning_method=prune.L1Unstructured,amount=0)
            else:
                prune.global_unstructured(get_parameters_to_prune(model, skip_last=True), pruning_method=prune.L1Unstructured,amount=0)
                model.load_state_dict(torch.load(f'{args.save_dir}/trained_iter_{i}.pth'))
                prune.global_unstructured(get_parameters_to_prune(model, skip_last=True), pruning_method=prune.L1Unstructured,amount=0)

            init_dict = torch.load(f'{args.save_dir}/init_iter_{i}.pth')
            delta_dict = model.state_dict()
            for name, param in model.named_parameters():
                if 'weight' in name and len(param.size()) == 2:
                    init_weight = init_dict[name]
                    delta_weight = param - init_weight
                    U, S, V = torch.svd(delta_weight)
                    S[int(rank_reduction*len(S)):] *= 0
                    delta_weight = U @ torch.diag(S) @ V.T
                    delta_dict[name] = delta_weight + init_weight

            print(f'Prune Iteration: {i}, Rank Reduction {rank_reduction}')
            model.load_state_dict(delta_dict)
            acc = evaluate(model, test_loader, device)
            print()

            results[i, rank_idx] = acc
            torch.save(results, f'{args.save_dir}/low_rank_delta.pth')
            
            

        
