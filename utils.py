import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

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

def train(model, train_loader, val_loader, save_path, device, num_epochs = 200, lr = 1e-2):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=.9, weight_decay=5e-4)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2 * num_epochs // 3, 5 * num_epochs // 6], gamma=0.2)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * num_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader))
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
Delta Pruning: train, prune delta / gradients, rewind weights back to initialization.
"""
def delta_pruning(model, train_loader, val_loader, save_dir, device, num_prune_iters = 30, num_epochs = 35, lr = 4.5e-4):
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

"""
Delta Pruning w/o Rewinding: train, prune delta / gradients, retrain, etc.
"""
def delta_pruning_no_rewinding(model, train_loader, val_loader, save_dir, device, num_prune_iters = 30, num_epochs = 35, lr = 4.5e-4):
    replace_with_delta(model)
    model = model.to(device)
    prune.global_unstructured(get_delta_parameters_to_prune(model), pruning_method=prune.L1Unstructured,amount=0)
    torch.save(model.state_dict(), save_dir + 'initialization.pth')

    for prune_iter in range(0, num_prune_iters + 1):
        for (layer, _) in get_delta_parameters_to_prune(model):
            layer.reload_weights()
        print(f'TESTING delta_pruning_no_rewinding, ITERATION {prune_iter}, Test acc: {evaluate(model, val_loader, device)}')
        torch.save(model.state_dict(), save_dir + f'init_iter_{prune_iter}.pth')
        print(f'TRAINING MODEL AT PRUNE_ITER {prune_iter}/{num_prune_iters} AT SPARSITY {get_delta_sparsity(model): .2f}')

        #Keep a running total of all the gradient movements over training.
        delta_train(model, train_loader, val_loader, save_dir + f'trained_iter_{prune_iter}.pth', device, num_epochs = num_epochs, lr = lr)

        #Place the running total of delta movements onto layer.delta_weight, prune the delta's that move the least.
        for (layer, _) in get_delta_parameters_to_prune(model):
            layer.prepare_for_pruning()
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
            parameters_to_prune.append((module, 'delta_weight'))
            if bias and module.delta_bias != None:
                parameters_to_prune.append((module, 'delta_bias'))

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
        if isinstance(module, DeltaLayer) and hasattr(module, 'delta_weight_mask'):
            num_zero_elements += torch.sum(module.delta_weight_mask == 0).float()
            total_elements += module.delta_weight_mask.numel()
            if hasattr(module, 'delta_bias_mask'):
                num_zero_elements += torch.sum(module.delta_bias_mask == 0).float()
                total_elements += module.delta_bias_mask.numel()

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
        self.base_weight = layer.weight.data.clone() #Save the weight initialization
        self.delta_weight = nn.Parameter(torch.zeros(layer.weight.size())) #This our zero-initialized layer.
        if layer.bias is not None:
            self.base_bias = layer.bias.data
            self.delta_bias = nn.Parameter(torch.zeros(layer.bias.size()))
        self.layer = layer #Store the actual nn.Module as well.
        #self.set_delta()
        self.prepruned_weight_data = None
        self.prepruned_weight_orig_data = None
        self.sum_of_deltas = torch.zeros(layer.weight.size()).to(self.base_weight.device) #Stores the gradient movements during training

    #This might be useless. If it breaks when we remove it, try sending delta_weight to device.
    def set_delta(self):
        self.delta_weight.data = self.layer.weight.data - self.base_weight

    #Zeros the entire delta.
    def reset_delta(self):
        self.delta_weight_orig.data *= 0
        self.delta_weight.data *= 0
        self.set_layer()

    def update_delta(self):
        self.sum_of_deltas += torch.abs((self.layer.weight.data - self.base_weight) - self.delta_weight.data)
        self.delta_weight.data = self.layer.weight.data - self.base_weight
        if self.layer.bias is not None:
            self.delta_bias.data = self.layer.bias.data - self.base_bias

    def set_layer(self):
        self.layer.weight.data = self.base_weight + self.delta_weight.data
        if self.layer.bias is not None:
            self.layer.bias.data = self.base_bias + self.delta_bias.data

    #Sets the sum of all delta movements to delta_weight, so that the pruning function uses magnitude(sum_of_deltas) as our criteria.
    def prepare_for_pruning(self):
        self.prepruned_weight_data = self.delta_weight.data
        self.prepruned_weight_orig_data = self.delta_weight_orig.data
        self.delta_weight_orig.data = self.sum_of_deltas.clone()
        self.delta_weight.data = self.sum_of_deltas.clone()
        self.sum_of_deltas *= 0

    def reload_weights(self):
        if self.prepruned_weight_data is None:
            self.delta_weight_orig.data *= 0
            self.delta_weight.data *= 0
        else:
            #We have ran prepare_for_pruning before
            self.delta_weight_orig.data = self.prepruned_weight_data * self.delta_weight_mask.data
            self.delta_weight.data = self.prepruned_weight_orig_data * self.delta_weight_mask.data


    def forward(self,x):
        return self.layer(x)
    
    #Set gradients to 0 where we have a pruned delta.
    def mask_gradient(self, optimizer):
        self.layer.weight.grad *= self.delta_weight_mask
        if 'momentum_buffer' in optimizer.state[self.layer.weight]:
            optimizer.state[self.layer.weight]['momentum_buffer'] *= self.delta_weight_mask
                
def delta_train(model, train_loader, val_loader, save_path, device, num_epochs = 200, lr = 1e-2):

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=.9, weight_decay=5e-4)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2 * num_epochs // 3, 5 * num_epochs // 6], gamma=0.2)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * num_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
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

            #Mask off Gradients for Pruned Delta
            for (layer, _) in get_delta_parameters_to_prune(model):
                layer.mask_gradient(optimizer)

                #Keep track of how delta moves over time.
                if step % 3 == 0:
                    layer.update_delta()

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
