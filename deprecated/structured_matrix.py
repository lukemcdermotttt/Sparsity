from utils import *
import copy
from transformers import AutoModelForImageClassification

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

device = torch.device('cuda:0')
model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224").to(device)
rank = 16

#Store results here
ranks = [16,24,27,32,64,128]
modules = ['W_proj', 'W_q', 'W_k', 'W_v', 'W_fc1', 'W_fc2']
method_names = ['svd_'+str(rank) for rank in ranks] + ['monarch'] + ['monarch_'+str(rank) for rank in ranks]
methods = [lambda x: get_SVD(x, rank) for rank in ranks] + [lambda x: get_Monarch(x)] + [lambda x: get_Monarch_SVD(x, rank) for rank in ranks]

Norm = {}
for name in method_names:
    Norm[name] = {}
    for module in modules:
        Norm[name][module] = []

for l in range(len(model.vit.encoder.layer)):

    W_proj = model.vit.encoder.layer[l].attention.output.dense.weight.data

    W_q = model.vit.encoder.layer[l].attention.attention.query.weight.data
    W_k = model.vit.encoder.layer[l].attention.attention.key.weight.data
    W_v = model.vit.encoder.layer[l].attention.attention.value.weight.data

    #W_fc1 = model.vit.encoder.layer[l].intermediate.dense.weight.data #3072,768
    #W_fc1_blocks = W_fc1[:768], W_fc1[768:2*768], W_fc1[2*768:3*768], W_fc1[3*768:]

    #W_fc2 = model.vit.encoder.layer[l].output.dense.weight.data #768,3072
    #W_fc2_blocks = W_fc2[:,:768], W_fc2[:,768:2*768], W_fc2[:,2*768:3*768], W_fc2[:,3*768:]

    for i in range(len(methods)):
        Norm[method_names[i]]['W_proj'].append(torch.norm(W_proj - methods[i](W_proj))) 
        Norm[method_names[i]]['W_q'].append(torch.norm(W_q - methods[i](W_q))) 
        Norm[method_names[i]]['W_k'].append(torch.norm(W_k - methods[i](W_k))) 
        Norm[method_names[i]]['W_v'].append(torch.norm(W_v - methods[i](W_v))) 

    


