import sys,os
from glob import glob as glob

import numpy as np
import itertools

from cswsims import *
import tensorflow as tf


# model params
arch = str(sys.argv[1])
stsize = int(sys.argv[2])
# task/training params
graphpr = int(sys.argv[3])
shiftpr = int(sys.argv[4])
nepochs = 40000

ML = MetaLearner(stsize,feedPE=arch)

# train
trainer = Trainer(ML,shift_pr=shiftpr/100,graph_pr=graphpr/100)
train_data = trainer.train_loop(nepochs)

# eval trained model
eval_context_strL=["".join(i) for i in itertools.product(
                                    ('A1','A2','B1','B2'),
                                    ('A1','A2','B1','B2'),
                                    ('A1','A2','B1','B2'),
                                    ('A1','A2','B1','B2'),
                                    ('A1','A2','B1','B2'),
                                    ('A1','A2','B1','B2')
                                   )]

eval_data = trainer.eval_loop(eval_context_strL)

## save
model_name = 'LSTM_%s-state_%i-csw_%i-shiftpr_%i-nstories_fp6bp3'%(arch,stsize,graphpr,shiftpr)
num_models = len(glob('models/csw_pef/%s/*'%model_name)) 
model_dir = 'models/csw_pef/%s/%.3i'%(model_name,num_models) 
os.makedirs(model_dir)

# model
ML.saver_op.save(ML.sess,model_dir+'/trained')
# data
np.save(model_dir+'/train_data',train_data)
np.save(model_dir+'/eval_data',eval_data)
