import sys,os
import numpy as np
from glob import glob as glob
import itertools

from cswsims import *
import tensorflow as tf

# model params
stsize = 50 
nstories = 3
# task/training params
graphpr = 0.9


context_strL=["".join(i) for i in itertools.product(
                                    ('A1','A2','B1','B2'),
                                    ('A1','A2','B1','B2'),
                                    ('A1','A2','B1','B2')
                                   )]


for idx in range(150):
  for shiftpr in [10,50,90]:
    print('shift',shiftpr,'idx',idx)
    midx = "%.3i"%idx
    model_dir = 'models/sweep_shiftpr/state_50-nstories_3-shiftpr_%i/%s' %(shiftpr,midx)
    ## restore
    ML = MetaLearner(stsize,nstories)
    ML.saver_op.restore(ML.sess,model_dir+'/trained')
    trainer = Trainer(ML,shiftpr,graphpr)
    ## eval
    eval_data = trainer.eval_loop(context_strL)
    ## save
    model_name = 'state_%i-nstories_%i-shiftpr_%i'%(stsize,nstories,shiftpr*100)
    np.save(model_dir+'/eval_data_full',eval_data)
