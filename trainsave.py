import sys,os
import numpy as np
from glob import glob as glob

from cswsims import *
import tensorflow as tf


shiftpr = int(sys.argv[1])/100

# model params
stsize = 50 
nstories = 3
# task/training params
nepochs = 5000
graphpr = .9



ML = MetaLearner(stsize,nstories)

# train
trainer = Trainer(ML,shift_pr=shiftpr,graph_pr=graphpr)
train_data = trainer.train_loop(nepochs)

# eval trained model
eval_seqL = ['AAA','AAB','ABA','ABB',
						 'BBB','BBA','BAB','BAA']
eval_data = trainer.eval_loop1(eval_seqL)


## save
model_name = 'state_%i-nstories_%i-shiftpr_%i'%(stsize,nstories,shiftpr*100)
num_models = len(glob('models/sweep_shiftpr/%s/*'%model_name)) 
model_dir = 'models/sweep_shiftpr/%s/%.3i'%(model_name,num_models) 
os.makedirs(model_dir)

# model
ML.saver_op.save(ML.sess,model_dir+'/trained')
# data
np.save(model_dir+'/train_data',train_data)
np.save(model_dir+'/eval_data',eval_data)
