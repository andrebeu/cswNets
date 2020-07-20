import sys
import torch as tr
import numpy as np
from CSW import *
from scipy.spatial import distance
import itertools

from matplotlib import pyplot as plt


## params
nnets = 20
#task
block_len0 = int(sys.argv[1])
stsize = int(sys.argv[2])
learn_rate = float(sys.argv[3])
#net
neps = 200
envsize = 12
envM = 0
envS = 1

fpath = 'matchSEM/regression_sweep2/model_data/'
fname = 'block-%i_lr-%.3f_stsize-%i'% (block_len0,learn_rate,stsize)


# task
pr = 1.0
taskL = [CSWTask(pr),CSWTask(1-pr)]
lossop = tr.nn.MSELoss()

# eval params
eval_pathL = [
  [0,11,1,4,5],       
    ]
eval_ypathL = [
  [11,1,4,5,7],       
    ]

xeval = tr.tensor(taskL[0].format_Xeval(eval_pathL))
yeval = tr.tensor(taskL[0].format_Xeval(eval_ypathL))


## data arrays
eval_tsteps = 2 # layer2 and layer3
num_eval_paths = len(eval_pathL) 
acc = -np.ones([nnets,neps,eval_tsteps]) # 2 tsteps (layer2 and layer3)
yeval = -np.ones([nnets,num_eval_paths,neps,eval_tsteps,envsize])
lossA = -np.ones([nnets,neps])


## loop over seeds
for seed in range(nnets):
  # init net
  net = CSWNetReg(envsize,stsize,seed)
  Emat = tr.Tensor(np.random.normal(0,1,[12,envsize]))
  optiop = tr.optim.Adam(net.parameters(), lr=learn_rate)
  
  ## train loop
  task_int = 0
  block_len = block_len0 
  for ep in range(neps):
    if ep >= 160:
      block_len = 1
    # select graph
    if ep%block_len==0:
      task_int = (task_int+1)%2
      task = taskL[task_int]
      filler_id = 10+task_int 
      
    ## forward prop  
    path = task.sample_path()
    xtrain,ytrain = task.dataset_onestory_with_marker(path=path,filler_id=filler_id,depth=1)
    xtrain,ytrain = Emat[xtrain],Emat[ytrain]
    yh = net(xtrain) # (time,smunits)
    yt = ytrain.detach().numpy().squeeze()
    
    ## compute accuracy
    acc[seed,ep,0] = distance.cosine(yh[2:3].detach().numpy(),yt[2:3])
    acc[seed,ep,1] = distance.cosine(yh[3:4].detach().numpy(),yt[3:4])

    ## eval
    for path_idx,xev in enumerate(xeval):
      xev = Emat[xev]
      yhev = net(xev).detach().numpy()
      for tstep_idx,yhev_t in enumerate(yhev[2:4]):
        yeval[seed,path_idx,ep,tstep_idx] = distance.cdist([yhev_t],Emat,'euclidean')      
        
    ## train
    loss = 0
    for tstep in range(len(xtrain)):
      loss = lossop(yh[tstep].unsqueeze(0),ytrain[tstep])
      optiop.zero_grad()
      loss.backward(retain_graph=True)
      optiop.step()
    lossA[seed,ep] = loss


np.save(fpath+'acc-'+fname,acc)
np.save(fpath+'yeval-'+fname,yeval)
