import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import copy

def evaluate(model, val_loader, device):
    num_correct, total, val_loss = 0, 0, 0
    criterion = nn.CrossEntropyLoss()

    for step, (images, labels) in enumerate(val_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        val_loss += criterion(outputs, labels).item() / len(images)

        num_correct += torch.sum((torch.argmax(outputs, dim=1) == labels))
        total += labels.size(0)

    acc = num_correct / total

    print('-------- Evaluation Statistics --------')
    print(f'Test Accuracy: {acc:.2f}')
    print('Test Loss:   ', val_loss / len(val_loader))

    return acc

def train(model, train_loader, val_loader, save_path, device, num_epochs = 200, lr = 1e-3):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=.9, weight_decay=1e-3)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1,num_epochs // 2, 3 * num_epochs // 4, 7 * num_epochs // 8], gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * num_epochs)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader))
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc, best_train_acc = 0, 0 #record the best (val in case of split) accuracy,
    for epoch in range(num_epochs):     
        num_correct, total, train_loss = 0, 0, 0

        #Train one epoch
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            #Record accuracy
            num_correct += torch.sum((torch.argmax(outputs, dim=1) == labels))
            total += labels.size(0)

            loss = criterion(outputs, labels)
            train_loss += loss.item() / len(images)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

        acc = num_correct / total
        print('Epoch: ',epoch)
        print('-------- Training Statistics --------')
        print(f'Train Accuracy: {acc:.2f}')
        print('Train Loss: ', train_loss / len(train_loader))
        print()

        #Save best model according to val accuracy, if tie pick best train accuracy.
        if val_loader is not None:
            val_acc = evaluate(model, val_loader, device)
            print()

            if val_acc > best_val_acc or (val_acc == best_val_acc and acc > best_train_acc):
                torch.save(model.state_dict(), save_path)
                best_val_acc = val_acc
                best_train_acc = acc
        
        elif acc > best_train_acc:
            torch.save(model.state_dict(), save_path)
            best_train_acc = acc

    return best_val_acc

def get_prune_method(prune_method):
    if prune_method == 'lottery_ticket_rewinding':
        return lottery_ticket_rewinding
    elif prune_method == 'early_training_rewinding':
        return early_training_rewinding
    elif prune_method == 'delta_pruning':
        return delta_pruning

"""
Lottery Ticket Rewinding: train, prune, rewind weights back to initialization
"""
def lottery_ticket_rewinding(model, train_loader, val_loader, save_dir, device, num_prune_iters = 30, num_epochs = 200, lr = 1e-2):
    prune.global_unstructured(get_parameters_to_prune(model), pruning_method=prune.L1Unstructured,amount=0)
    torch.save(model.state_dict(), save_dir + 'initialization.pth')
    
    for prune_iter in range(0, num_prune_iters + 1):
        rewind(model, torch.load(save_dir + 'initialization.pth', map_location=device))
        torch.save(model.state_dict(), save_dir + f'init_iter_{prune_iter}.pth')
        print(f'TRAINING MODEL AT PRUNE_ITER {prune_iter}/{num_prune_iters} AT SPARSITY {get_sparsity(model): .2f}')
        train(model, train_loader, val_loader, save_dir + f'trained_iter_{prune_iter}.pth', device, num_epochs = num_epochs, lr = lr)
        prune.global_unstructured(get_parameters_to_prune(model), pruning_method=prune.L1Unstructured,amount=.2)

"""
Early Training Rewinding: train, prune, rewind weights back to the dense model at k epochs
"""
def early_training_rewinding(model, train_loader, val_loader, save_dir, device, num_prune_iters = 30, num_epochs = 200, lr = 1e-2, k=5):
    prune.global_unstructured(get_parameters_to_prune(model), pruning_method=prune.L1Unstructured,amount=0)
    train(model, train_loader, val_loader, save_dir + 'initialization.pth', device, num_epochs = k, lr = lr)
    
    for prune_iter in range(0, num_prune_iters + 1):
        rewind(model, torch.load(save_dir + 'initialization.pth', map_location=device))
        torch.save(model.state_dict(), save_dir + f'init_iter_{prune_iter}.pth')
        print(f'TRAINING MODEL AT PRUNE_ITER {prune_iter}/{num_prune_iters} AT SPARSITY {get_sparsity(model): .2f}')
        train(model, train_loader, val_loader, save_dir + f'trained_iter_{prune_iter}.pth', device, num_epochs = num_epochs, lr = lr)
        prune.global_unstructured(get_parameters_to_prune(model), pruning_method=prune.L1Unstructured,amount=.2)

"""
Learning Rate Rewinding: train, prune
"""
def learning_rate_rewinding(model, train_loader, val_loader, save_dir, device, num_prune_iters = 30, num_epochs = 200, lr = 1e-2):
    prune.global_unstructured(get_parameters_to_prune(model), pruning_method=prune.L1Unstructured,amount=0)
    torch.save(model.state_dict(), save_dir + 'initialization.pth')
    
    for prune_iter in range(0, num_prune_iters + 1):
        print(f'TRAINING MODEL AT PRUNE_ITER {prune_iter}/{num_prune_iters} AT SPARSITY {get_sparsity(model): .2f}')
        train(model, train_loader, val_loader, save_dir + f'trained_iter_{prune_iter}.pth', device, num_epochs = num_epochs, lr = lr)
        prune.global_unstructured(get_parameters_to_prune(model), pruning_method=prune.L1Unstructured,amount=.2)


"""
Delta Pruning w/o Rewinding: train, prune delta / gradients, retrain, etc.
"""
def delta_pruning(model, train_loader, val_loader, save_dir, device, num_prune_iters = 30, num_epochs = 35, lr = 1e-1):
    replace_with_delta(model)
    model = model.to(device)
    prune.global_unstructured(get_delta_parameters_to_prune(model), pruning_method=prune.L1Unstructured,amount=0)
    torch.save(model.state_dict(), save_dir + 'initialization.pth')

    for prune_iter in range(0, num_prune_iters + 1):
        torch.save(model.state_dict(), save_dir + f'init_iter_{prune_iter}.pth')
        
        #Keep a running total of all the gradient movements over training.
        test_accuracy = train(model, train_loader, val_loader, save_dir + f'trained_iter_{prune_iter}.pth', device, num_epochs = num_epochs, lr = lr)
        
        #Save results in a .txt
        with open(save_dir + 'accuracies.txt', "a") as file:
            file.write(f'PRUNE ITERATION {prune_iter} AT SPARSITY {100*(1-.8**prune_iter):.5f}%\n')
            file.write(f'Test Accuracy: {test_accuracy}\n\n')

        prune.global_unstructured(get_delta_parameters_to_prune(model), pruning_method=prune.L1Unstructured,amount=.2)
        
def get_parameters_to_prune(model, bias = False, skip_last = True):
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, 'weight'))
            if bias and module.bias != None:
                parameters_to_prune.append((module, 'bias'))

    if skip_last:
        return tuple(parameters_to_prune)[:-1]
    return tuple(parameters_to_prune)

def get_delta_parameters_to_prune(model, bias = False, skip_last = True):
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, DeltaLayer):
            parameters_to_prune.append((module.delta, 'weight'))
            if bias and module.delta_bias != None:
                parameters_to_prune.append((module.delta, 'bias'))

    if skip_last:
        return tuple(parameters_to_prune)[:-1]
    return tuple(parameters_to_prune)

def get_sparsity(model):
    num_zero_elements, total_elements = 0, 0
    for name, module in model.named_modules():
        if (isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear)) and hasattr(module, 'weight_mask'):
            num_zero_elements += torch.sum(module.weight_mask == 0).float()
            total_elements += module.weight_mask.numel()

            if hasattr(module, 'bias_mask'):
                num_zero_elements += torch.sum(module.bias_mask == 0).float()
                total_elements += module.bias_mask.numel()

    return num_zero_elements / total_elements

def get_delta_sparsity(model):
    num_zero_elements, total_elements = 0, 0
    for name, module in model.named_modules():
        if isinstance(module, DeltaLayer) and hasattr(module.delta, 'weight_mask'):
            num_zero_elements += torch.sum(module.delta.weight_mask == 0).float()
            total_elements += module.delta.weight_mask.numel()
            if hasattr(module.delta, 'bias_mask'):
                num_zero_elements += torch.sum(module.delta.bias_mask == 0).float()
                total_elements += module.delta.bias_mask.numel()

    return num_zero_elements / total_elements

def rewind(model, state_dict, bias = False):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            if hasattr(module, 'weight_orig'):
                with torch.no_grad():
                    module.weight_orig.copy_(state_dict[name + '.weight_orig'])
                    if bias:
                        module.bias_orig.copy_(state_dict[name + '.bias_orig'])
            else:
                with torch.no_grad():
                    module.weight.copy_(state_dict[name + '.weight'])
                    if bias:
                            module.bias.copy_(state_dict[name + '.bias'])

def replace_with_delta(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            setattr(model, name, DeltaLayer(module))
        else:
            replace_with_delta(module) # Recursively replace in sub-modules

class DeltaLayer(nn.Module):
    def __init__(self, layer):
        super(DeltaLayer, self).__init__()
        
        #Store Layer as our initialization
        self.use_bias = layer.bias is not None
        self.initialization = layer
        self.initialization.weight.requires_grad = False
        if self.use_bias: self.initialization.bias.requires_grad = False

        #Create a zero-filled delta
        self.delta = copy.deepcopy(layer)
        self.delta.weight.data *= 0
        self.delta.weight.requires_grad = True
        if self.use_bias: 
            self.delta.bias.data *= 0
            self.delta.bias.requires_grad = True

    def forward(self, x):
        return self.initialization(x) + self.delta(x) 
    
def set_seed(seed = 0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
#UNDER CONSTRUCTION:
def structured_lottery_ticket_rewinding(model, train_loader, val_loader, save_dir, device, num_prune_iters = 30, num_epochs = 200, lr = 1e-2):
    prune.global_unstructured(get_parameters_to_prune(model), pruning_method=prune.L1Unstructured,amount=0)
    prune.global_structured(get_parameters_to_prune(model), amount=0.2, n=2, dim=0)
    torch.save(model.state_dict(), save_dir + 'initialization.pth')
    
    for prune_iter in range(0, num_prune_iters + 1):
        rewind(model, torch.load(save_dir + 'initialization.pth', map_location=device))
        torch.save(model.state_dict(), save_dir + f'init_iter_{prune_iter}.pth')
        print(f'TRAINING MODEL AT PRUNE_ITER {prune_iter}/{num_prune_iters} AT SPARSITY {get_sparsity(model): .2f}')
        train(model, train_loader, val_loader, save_dir + f'trained_iter_{prune_iter}.pth', device, num_epochs = num_epochs, lr = lr)
        prune.global_unstructured(get_parameters_to_prune(model), pruning_method=prune.L1Unstructured,amount=.2)


"""
NOT TESTED ANYMORE. THIS CODE IS NOT SUPPORTED.
Delta Pruning w/ Rewinding: train, prune delta / gradients, rewind weights back to initialization.
"""
def delta_pruning_w_rewinding(model, train_loader, val_loader, save_dir, device, num_prune_iters = 30, num_epochs = 35, lr = 4.5e-4):
    replace_with_delta(model)
    model = model.to(device)
    prune.global_unstructured(get_delta_parameters_to_prune(model), pruning_method=prune.L1Unstructured,amount=0)
    torch.save(model.state_dict(), save_dir + 'initialization.pth')

    for prune_iter in range(0, num_prune_iters + 1):
        for (layer, _) in get_delta_parameters_to_prune(model):
            layer.reset_delta()

        torch.save(model.state_dict(), save_dir + f'init_iter_{prune_iter}.pth')
        print(f'TRAINING MODEL AT PRUNE_ITER {prune_iter}/{num_prune_iters} AT SPARSITY {get_delta_sparsity(model): .2f}')

        #Keep a running total of all the gradient movements over training.
        delta_train(model, train_loader, val_loader, save_dir + f'trained_iter_{prune_iter}.pth', device, num_epochs = num_epochs, lr = lr)

        #Place the running total of delta movements onto layer.delta_weight, prune the delta's that move the least.
        for (layer, _) in get_delta_parameters_to_prune(model):
            layer.prepare_for_pruning()
        prune.global_unstructured(get_delta_parameters_to_prune(model), pruning_method=prune.L1Unstructured,amount=.2)
