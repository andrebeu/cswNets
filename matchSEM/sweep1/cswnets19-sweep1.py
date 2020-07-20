import sys
import torch as tr
import numpy as np
from CSW import *

rootdir = 'matchSEM/sweep1/'

## sweep vars
block_len0 = int(sys.argv[1])
stsize = int(sys.argv[2])
learn_rate = float(sys.argv[3])
bpmode = sys.argv[4]

print(block_len0,stsize,learn_rate,bpmode) 

# fix vars
nnets = 20
neps = 200

## task init
taskL = [CSWTask(1.0),CSWTask(0.0)]

# eval paths
eval_pathL = [
  [0,10,1,4,5],       
  [0,11,1,3,5],
  [0,11,2,4,6],
  ]

# format for nn input
xeval = taskL[0].format_Xeval(eval_pathL)
xeval = tr.tensor(xeval)

# eval array
tdim,sm_dim=5,12
ysm = -np.ones([nnets,3,neps,tdim,sm_dim])
acc = -np.ones([nnets,neps])

## train setup
softmax = lambda ulog: tr.softmax(ulog,-1)
lossop = tr.nn.CrossEntropyLoss()

# loop over seeds
for seed in range(nnets):
  # init net
  net = CSWNet(stsize,seed)
  optiop = tr.optim.Adam(net.parameters(), lr=learn_rate)
  # init loop vars
  task_int = 0
  block_len = block_len0 
  # train loop
  for ep in range(neps):
    # curriculum select graph
    if ep == 160:
      block_len = 1
    if ep%block_len==0:
      task_int = (task_int+1)%2
      task = taskL[task_int]
      filler_id = 10+task_int 
    # forward prop  
    path = task.sample_path()
    xtrain,ytrain = task.dataset_onestory_with_marker(path=path,filler_id=filler_id,depth=1)
    yh = net(xtrain) # (time,smunits)
    # accuracy on train data
    yh_sm = softmax(yh).detach().numpy()
    yt = ytrain.detach().numpy().squeeze()
    acc[seed,ep] = np.mean(np.equal(np.argmax(yh_sm[2:4],1),yt[2:4]))
    ## eval dataset
    for idx,xev in enumerate(xeval):
      ysm_t = softmax(net(xev)).detach().numpy()
      ysm[seed,idx,ep] = ysm_t
    if bpmode=='scene':
      loss = 0
      for tstep in range(len(xtrain)):
        loss = lossop(yh[tstep].unsqueeze(0),ytrain[tstep])
        optiop.zero_grad()
        loss.backward(retain_graph=True)
        optiop.step()
    elif bpmode=='story':
      loss = 0
      for tstep in range(len(xtrain)):
        loss += lossop(yh[tstep].unsqueeze(0),ytrain[tstep])
      optiop.zero_grad()
      loss.backward(retain_graph=True)
      optiop.step()
    else: 
      assert False


np.save(rootdir+'model_data/ysm-blocklen%i-bpmode_%s-stsize%i-lr%.3f.npy'%(block_len0,bpmode,stsize,learn_rate),ysm)
np.save(rootdir+'model_data/acc-blocklen%i-bpmode_%s-stsize%i-lr%.3f.npy'%(block_len0,bpmode,stsize,learn_rate),acc)

