import sys
import numpy as np
from cswsims import *

## PARAMS

NUM_RUNS = 50
TOTAL_EP = 50000
EPB = int(sys.argv[1])

NUM_BLOCKS = int(TOTAL_EP/EPB)
curr = [[NUM_BLOCKS,EPB],[10000,1]]

## EVAL ON FOLLOWING PATHS

eval_paths = [[10,0,1,3,5],
              [10,0,2,4,6],
              [11,0,1,4,5],
              [11,0,2,3,6]]


## MAIN LOOPS - reinitializing every run: same seed

netdata = []
for s in range(NUM_RUNS):
  print('\nrun%.2i\n'%s)
  # initialize
  net = NetGraph(50)
  trainer = Trainer(net,0.8)
  # train
  pred_data = trainer.main_loop(curr,eval_paths)
  netdata.append(pred_data)


## SAVING 

def curr_int2str(curr):
  """
  takes an int curriculum: 
    e.g. [[5000,1],[1,1000]]
    [nblocks,epb]
  returns a str curriculum:
    e.g. "5000(1)1(1000)"
    nblocks(epb)
  """
  curstr = ""
  for nb,epb in curr: 
    curstr += "ne(%i)epb(%i)"%(nb*epb,epb) 
  return curstr

fpath = "evaldata/sub01_LSTM50-curr_%s"%(curr_int2str(curr))
np.save(fpath,netdata)

