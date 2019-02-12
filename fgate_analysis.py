import sys
import numpy as np
from glob import glob as glob

from cswsims import *
import tensorflow as tf


shiftpr = int(sys.argv[1])/100

# model params
lstm_size = 50 
nstories = 3
# task/training params
nepochs = 10000
graphpr = 1

seqL = ['AAA','ABA','BAA','BBA']

ML = MetaLearner(lstm_size,nstories)


trainer = Trainer(ML,shift_pr=shiftpr,graph_pr=graphpr)
train_data = trainer.train_loop(nepochs)

netn = len(glob('eval_data/*'))
# eval
for seq in seqL:
  Xeval,Yeval = CSWMLTask(1).get_Xeval2(seq)
  evalstep_data = trainer.eval_step(Xeval,Yeval)
  fpath = 'eval_data/S%i-shift_%i-seq_%s'%(netn,100*shiftpr,seq)
  np.save(fpath,evalstep_data)

