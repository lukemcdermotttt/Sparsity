from datasets import load_dataset
from torch.utils.data import Dataset
from utils import *
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import copy

device = torch.device('cuda:0')

class ImageNet(Dataset):
    def __init__(self, split='validation'):
        access_token = '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' #NOTE: Careful pushing this.
        self.ds = load_dataset("imagenet-1k", token=access_token)[split]
        self.transform = transforms.Compose([lambda x: F.pil_to_tensor(x) / 255.0, transforms.Resize((224,224)), lambda x: x.repeat(3,1,1) if x.size(0) < 3 else x, transforms.Normalize([.5,.5,.5], [.5,.5,.5])])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.transform(self.ds[idx]['image']), self.ds[idx]['label']

def evaluate(model, val_loader, device):
    num_correct, total, val_loss = 0, 0, 0
    #criterion = nn.CrossEntropyLoss()

    for step, (images, labels) in enumerate(val_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images).logits
        
        #val_loss += criterion(outputs, labels).item() / len(images)
        num_correct += torch.sum((torch.argmax(outputs, dim=1) == labels))
        total += labels.size(0)

        if step % 100 == 0: print(step, len(val_loader))

    acc = num_correct / total

    print('-------- Evaluation Statistics --------')
    print(f'Test Accuracy: {acc:.2f}')
    print('Test Loss:   ', val_loss / len(val_loader))

    return acc

def get_SVD(W,rank):

  Uw, Sw, Vw = torch.svd(W)
  Sw[rank:] = 0
  W_svd = Uw @ torch.diag(Sw) @ Vw.T

  return W_svd

def get_DRONE(W,rank,X):

  Uw, Sw, Vw = torch.svd(W)
  Ux, Sx, Vx = torch.svd(X)
  Sw, Sx = torch.diag(Sw), torch.diag(Sx)

  Z = Sw @ Vw.T @ Ux @ Sx
  Uz, Sz, Vz = torch.svd(Z)
  Sz[rank:] = 0
  Sz = torch.diag(Sz)

  Ustar = W @ Vw @ torch.inverse(Sw) @ Uz @ Sz
  VstarT = Vz.T @ torch.inverse(Sx) @ Ux.T
  W_drone = Ustar @ VstarT

  return W_drone

#NOTE THIS IS CHEATING USING SCALE > 0.
def get_Monarch(W):
  n = W.size(0)
  m = int(n**(1/2))
  W_monarch = copy.deepcopy(W)
  for i in range(m):
    for j in range(m):
      Um,Sm,Vm = torch.svd(W_monarch[i*m:(i+1)*m, j*m:(j+1)*m])
      Sm[1:] = 0
      W_monarch[i*m:(i+1)*m, j*m:(j+1)*m] = Um @ torch.diag(Sm) @ Vm.T

  return W_monarch 
 
def get_DRONE_Monarch(W, X):
  n = W.size(0)
  m = int(n**(1/2))
  print(m)

  W = copy.deepcopy(W)
  for i in range(m):
    x = X[i*m:(i+1)*m,:] #little x block
    for j in range(m):
      w = W[i*m:(i+1)*m, j*m:(j+1)*m] #little w block
      
      Uw, Sw, Vw = torch.svd(w)
      Ux, Sx, Vx = torch.svd(x)
      Sw, Sx = torch.diag(Sw), torch.diag(Sx)

      Z = Sw @ Vw.T @ Ux @ Sx
      Uz, Sz, Vz = torch.svd(Z)
      Sz[1:] = 0
      Sz = torch.diag(Sz)

      Ustar = w @ Vw @ torch.inverse(Sw) @ Uz @ Sz
      VstarT = Vz.T @ torch.inverse(Sx) @ Ux.T
      W[i*m:(i+1)*m, j*m:(j+1)*m] = Ustar @ VstarT

  return W

def get_Monarch_SVD(W, rank):
   
   W_svd = get_SVD(W, rank)
   W_diff = W - W_svd
   W_monarch = get_Monarch(W_diff)

   return W_svd + W_monarch

train_dataset = ImageNet(split='validation')
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = 64)
val_dataset = ImageNet(split='validation')
val_loader = torch.utils.data.DataLoader(dataset = val_dataset, batch_size = 128)

model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224").to(device)

#print(model)
for n,p in model.named_parameters():
   p.requires_grad = False

rank = 50
for l in range(len(model.vit.encoder.layer)):
#model.vit.encoder.layer[l].attention.query, attention.output.dense, layer[l].intermediate.dense, layer[l].output.dense
    #Projection
    model.vit.encoder.layer[l].attention.output.dense.weight.data = get_Monarch_SVD(model.vit.encoder.layer[l].attention.output.dense.weight.data, rank=rank)
    model.vit.encoder.layer[l].attention.output.dense.weight.requires_grad = True
    
    #QKV
    """
    model.vit.encoder.layer[l].attention.attention.query.weight.data = get_Monarch(model.vit.encoder.layer[l].attention.attention.query.weight.data)
    model.vit.encoder.layer[l].attention.attention.query.weight.requires_grad = True
    model.vit.encoder.layer[l].attention.attention.key.weight.data = get_Monarch(model.vit.encoder.layer[l].attention.attention.key.weight.data)
    model.vit.encoder.layer[l].attention.attention.key.weight.requires_grad = True
    model.vit.encoder.layer[l].attention.attention.value.weight.data = get_Monarch(model.vit.encoder.layer[l].attention.attention.value.weight.data)
    model.vit.encoder.layer[l].attention.attention.value.weight.requires_grad = True
    """
    #FC1
    #fc1 = model.vit.encoder.layer[l].intermediate.dense.weight.data #3072,768
    #fc1[:768], fc1[768:2*768], fc1[2*768:3*768], fc1[3*768:] = get_Monarch(fc1[:768]), get_Monarch(fc1[768:2*768]), get_Monarch(fc1[2*768:3*768]), get_Monarch(fc1[3*768:])
    #model.vit.encoder.layer[l].intermediate.dense.weight.data = fc1

    #FC2
    #fc2 = model.vit.encoder.layer[l].output.dense.weight.data #768,3072
    #fc2[:,:768], fc2[:,768:2*768], fc2[:,2*768:3*768], fc2[:,3*768:] = get_Monarch(fc2[:,:768]), get_Monarch(fc2[:,768:2*768]), get_Monarch(fc2[:,2*768:3*768]), get_Monarch(fc2[:,3*768:])
    #model.vit.encoder.layer[l].output.dense.weight.data = fc2

"""

optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader))
criterion = nn.CrossEntropyLoss()
num_epochs = 1

best_val_acc, best_train_acc = 0, 0 #record the best (val in case of split) accuracy,
for epoch in range(num_epochs):     
    num_correct, total, train_loss = 0, 0, 0

    #Train one epoch
    for step, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images).logits

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
"""

#evaluate(model, val_loader, device)
#Dense Val Accuracy is 80%
#Monarch Value Accuracy on every layer (q,k,v,p,fc1,fc2) is 0%
#Monarch Value Accuracy on every layer (q,k,v,p) is 0%

#Monarch p accuracy is 3%
#Monarch p + .5 diff is 77%
#Monarch p + .25 diff is 56%

#Monarch+SVDr16 p accuracy is 13%
#Monarch+SVDr16 + 1 epoch p accuracy is 19%