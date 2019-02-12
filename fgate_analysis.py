import numpy as np
from glob import glob as glob

from cswsims import *
import tensorflow as tf

# model params
lstm_size = 50 
nstories = 3
# task/training params
nepochs = 10000
graphpr = 1
shiftpr = 0.5
seqL = ['AAA','ABA','BAA','BBA']

nnets = 10
ML = MetaLearner(lstm_size,nstories)
for netn in range(nnets):
  ML.reinitialize()
  trainer = Trainer(ML,shift_pr=shiftpr,graph_pr=graphpr)
  train_data = trainer.train_loop(nepochs)
  # eval
  for seq in seqL:
    Xeval,Yeval = CSWMLTask(1).get_Xeval2(seq)
    evalstep_data = trainer.eval_step(Xeval,Yeval)
    fpath = 'eval_data/S%i-shift_%i-seq_%s'%(netn,100*shiftpr,seq)
    np.save(fpath,evalstep_data)

