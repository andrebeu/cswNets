import sys
import torch as tr
import numpy as np
from CSW import *
from scipy.spatial import distance
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

fpath = 'matchSEM/sweep_regression/model_data/'
fname = 'block-%i_lr-%.3f_stsize-%i'% (block_len0,learn_rate,stsize)

# task
pr = 1.0
taskL = [CSWTask(pr),CSWTask(1-pr)]

# setup eval data
eval_pathL = [
  [0,10,1,4,5],       
    ]
eval_ypathL = [
  [10,1,4,5,7],       
    ]

xeval = tr.tensor(taskL[0].format_Xeval(eval_pathL))
yeval = tr.tensor(taskL[0].format_Xeval(eval_ypathL))


## train setup
lossop = tr.nn.MSELoss()
Emat = tr.Tensor(np.random.normal(envM,envS,[envsize,envsize]))


acc = -np.ones([nnets,neps,2]) # 2 tsteps
# eval array
tdim = 5
num_eval_paths = len(eval_pathL)
yeval = -np.ones([nnets,num_eval_paths,neps,tdim,envsize])
lossA = -np.ones([nnets,neps])


## loop over seeds
for seed in range(nnets):
  # init net
  net = CSWNetReg(envsize,stsize,seed)
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
    # forward prop  
    path = task.sample_path()
    xtrain,ytrain = task.dataset_onestory_with_marker(path=path,filler_id=filler_id,depth=1)
    xtrain,ytrain = Emat[xtrain],Emat[ytrain]
    yh = net(xtrain) # (time,smunits)
    yt = ytrain.detach().numpy().squeeze()
    ## compute accuracy
    acc[seed,ep,0] = distance.cosine(yh[2:3].detach().numpy(),yt[2:3])
    acc[seed,ep,1] = distance.cosine(yh[3:4].detach().numpy(),yt[3:4])

    # eval
    for pidx,xev in enumerate(xeval):
      xev = Emat[xev]
      yh_ev = net(xev).detach().numpy()
      yeval[seed,pidx,ep] = yh_ev

    loss = 0
    for tstep in range(len(xtrain)):
      loss = lossop(yh[tstep].unsqueeze(0),ytrain[tstep])
      optiop.zero_grad()
      loss.backward(retain_graph=True)
      optiop.step()
    lossA[seed,ep] = loss


np.save(fpath+'acc-'+fname,acc)

