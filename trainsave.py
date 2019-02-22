import sys,os
from glob import glob as glob

import numpy as np
import itertools

from cswsims import *
import tensorflow as tf


# model params
stsize = int(sys.argv[1])
nstories = 3
# task/training params
nepochs = 30000
graphpr = int(sys.argv[2])
shiftpr = int(sys.argv[3])


ML = MetaLearner(stsize,nstories)

# train
trainer = Trainer(ML,shift_pr=shiftpr/100,graph_pr=graphpr/100)
train_data = trainer.train_loop(nepochs)

# eval trained model
eval_context_strL=["".join(i) for i in itertools.product(
                                    ('A1','A2','B1','B2'),
                                    ('A1','A2','B1','B2'),
                                    ('A1','A2','B1','B2')
                                   )]

eval_data = trainer.eval_loop(eval_context_strL)

## save
model_name = 'state_%i-csw_%i-nstories_%i-shiftpr_%i'%(stsize,nstories,shiftpr,graphpr)
num_models = len(glob('models/csw_tpgen/%s/*'%model_name)) 
model_dir = 'models/csw_tpgen/%s/%.3i'%(model_name,num_models) 
os.makedirs(model_dir)

# model
ML.saver_op.save(ML.sess,model_dir+'/trained')
# data
np.save(model_dir+'/train_data',train_data)
np.save(model_dir+'/eval_data',eval_data)
