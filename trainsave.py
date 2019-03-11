import numpy as np
import sys
import itertools
from glob import glob as glob

from cswsims import *
import tensorflow as tf


train_epochs = 30000
stsize = 30
shiftpr = int(sys.argv[1])/1e5
cswpr = 1.0

net = CSWNet(stsize)
print(shiftpr)
trainer = Trainer(net,shiftpr,cswpr)
train_data = trainer.train_loop(train_epochs)


fpath = 'models/csw_em/lstm_%icswpr_%i-shiftpr_%ie-5-evalAAABBB-train_data'%(stsize,cswpr*100,shiftpr*100000)
nnets = len(glob(fpath+"*"))
np.save(fpath+'-n%.2i'%(nnets+1),train_data)
