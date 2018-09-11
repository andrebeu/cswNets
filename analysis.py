from BaseGraph import *
from Cells import *
from RNNs import *
from cswNetEngine import *

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import colors




# misc


def mov_avg(data,window=15):
  """ moving average over first dimension """
  mean_arr = []
  for t1 in range(len(data)-window):
    t2 = t1 + window
    mean_arr.append(np.mean(data[t1:t2],0))
  return np.array(mean_arr)


# eval plots

def plt_eval(eval_arr,num_blocks,num_evals_per_block):
  """
  """
  total_num_evals = num_blocks*num_evals_per_block
  fig,axarr = plt.subplots(2,1,figsize=(20,10),sharex=True);

  axarr[1].set_xticks(np.arange(0,total_num_evals+1,1000))

  

  # accuracy
  axarr[0].set_title('accuracy')
  acc_plt = mov_avg(eval_arr['acc'],1)
  xplt = np.linspace(0,total_num_evals,len(acc_plt))
  axarr[0].plot(xplt,acc_plt)
  axarr[0].axhline(0.5,c='r',ls='--',lw=0.8,alpha=0.6)
  axarr[0].axhline(0.8,c='r',ls='--',lw=0.8,alpha=0.6)

  # loss
  axarr[1].set_title('loss')
  axarr[1].plot(eval_arr['loss'])
  
  for ax in axarr: 
    ax.grid(True)
    plt_block_divides(ax,len(eval_arr),num_blocks)

  axarr[0].tick_params(labelsize=30)
  axarr[-1].tick_params(labelsize=30)
  axarr[-1].set_xlabel('epochs',size=30)

  # title = 'eval-%iblocks%iepb'%(num_blocks,num_epochs_per_block)
  # plt.suptitle(title,size=20)


# formating predictions data

def make_pred_df(pred_arr,num_blocks,num_evals_per_block):
  ## get respective data from epoch_num
  # because datapoints are randomized in an epoch, 
  # i cannot assume the data will be aligned between epochs. 
  xb_st = lambda epoch_num: pred_arr[epoch_num]['xbatch'][:,0]
  xb_fl = lambda epoch_num: pred_arr[epoch_num]['xbatch'][:,1]
  yh_st = lambda epoch_num: pred_arr[epoch_num]['yhat'][:,0]
  yh_fl = lambda epoch_num: pred_arr[epoch_num]['yhat'][:,1]
  ## multi index allows tuple indices: .loc[fillerid,stateid]
  # this aligns 
  multi_idx = lambda epoch_num: pd.MultiIndex.from_tuples(
    [(xb_fl(epoch_num)[i],xb_st(epoch_num)[i]) for i in range(len(xb_st(epoch_num)))])
  ## assemble a dataframe with predictions from a given epoch
  trial_df = lambda epoch_num: pd.DataFrame(
                                  data=yh_st(epoch_num),
                                  columns=['yh_st'],
                                  index=multi_idx(epoch_num)
                                ).sort_index()
  ## construct full dataframe
  # initialize dataframe with first epoch
  full_df = trial_df(0) 
  # include all other epochs
  num_epochs = num_blocks * num_evals_per_block
  for epoch_num in range(1,num_epochs):
    full_df['yh%i'%epoch_num] = trial_df(epoch_num)
  return full_df

def get_pred_df(pred_arr_L,blocking):
  pred_df_L = []
  for pred_arr,(num_blocks,num_epochs) in zip(pred_arr_L,blocking):
    pred_df_L.append(make_pred_df(pred_arr,num_blocks,num_epochs))
  return pd.concat(pred_df_L,1)

def code_responses(prediction_df,trainer,eval_fillerid):
  """ 
  given a dataframe containing the predictions of the network
  returns a new dataframe coded as (optimal=2,alt=1,neither=0)
  """
  df_plt = pd.DataFrame(columns=prediction_df.columns,index=prediction_df.loc[eval_fillerid,:].index[1:])
  for frnode in df_plt.index:
    if frnode == 0 :continue
    opt_tonode,alt_tonode = trainer.csw.get_opt_and_alt_tonodes(frnode,eval_fillerid)
    opt_idx = prediction_df.loc[eval_fillerid,frnode] == opt_tonode
    alt_idx = prediction_df.loc[eval_fillerid,frnode] == alt_tonode
    df_plt.loc[frnode][opt_idx] = 2
    df_plt.loc[frnode][alt_idx] = 1
    df_plt.loc[frnode][np.logical_not(opt_idx|alt_idx)] = 0
  return df_plt


# prediction plots

def plt_avg_predictions(coded_df,blocking):
  """ aggregates data from all transitions in a single plot 
  """
  num_epochs_plt = coded_df.shape[1]
  z = 2
  plt.figure(figsize=(10*z,3*z))
  # count responses between node
  for resp,color in zip([2,1,0],['g','b','r']):
    count = np.sum(coded_df == resp,0)
    plt.scatter(np.arange(num_epochs_plt),np.repeat(resp,num_epochs_plt),s=count*count*count,c=color)
  # ticks
  plt.yticks([0,1,2],['neither','contextB','contextA'],size=24)
  plt.ylim(-.5,2.5)
  plt.xlabel('EPOCHS',size=25)
  plt.xticks(size=23)
  plt.title('average between nodes',size=30)
  # include block divides
  current_axes = plt.gca()
  offset=0
  for sess_num,sess in enumerate(blocking):
    num_blocks,block_len=sess
    start,end = get_block_idx(num_blocks,block_len,offset)
    plt_block_divides2(current_axes,start,end)
    offset = end[-1]

  return None

def plt_predictions(df_plt,blocking,num_states_=None):
  """ plots data from transitions individually 
  """
  num_states,num_epochs_plt = df_plt.shape
  num_states_plt = num_states_ or (num_states-3)

  fig,axarr = plt.subplots(num_states_plt,1,figsize=(2*20,num_states_plt*5),sharex=True)

  for frnode,ax in enumerate(axarr):
    # skip 0th node
    frnode = frnode+1

    # separate data for node
    yhat = df_plt.loc[frnode].values
    opt_resp = np.ma.masked_array(data=yhat,mask=yhat!=2)
    alt_resp = np.ma.masked_array(data=yhat,mask=yhat!=1)
    neither = np.ma.masked_array(data=yhat,mask=yhat!=0)

    # plot
    pltsize = 50
    ax.scatter(range(num_epochs_plt),opt_resp,s=pltsize,c='green')
    ax.scatter(range(num_epochs_plt),alt_resp,s=pltsize,c='blue')
    ax.scatter(range(num_epochs_plt),neither,s=pltsize,c='red')

    # labeling
    ax.set_yticks([0,1,2])
    ax.set_ylim(-.5,2.5)
    ax.set_yticklabels(["neither","contextB","contextA"],size=30)
    ax.set_title("Predictions on State%i-ContextA"%(frnode),size=25)

  # include block divides
  start,end = get_block_idx(blocking)
  for ax in axarr: 
    plt_block_divides2(ax,start,end)

  # xticks
  axarr[-1].tick_params(labelsize=30)
  axarr[-1].set_xlabel('epochs',size=30)

  title = 'pred-blocking'
  plt.suptitle(title+'\n'+str(blocking),size=50)


# color in blocks where training on df of filler being viewed

def get_block_idx(num_blocks,block_len,offset=0):
  """ returns the begin and end indices of blocks
  """
  start_idx_L,end_idx_L = [],[]
  for block in range(num_blocks):
    start_idx = block*block_len + offset
    end_idx = start_idx + block_len 
    start_idx_L.append(start_idx)
    end_idx_L.append(end_idx)
  return start_idx_L,end_idx_L

def plt_block_divides(ax,evals_per_block,num_blocks,colors=['grey','white']):
  y0,y1 = ax.get_ylim()
  for idx,dblock_ep in enumerate(range(0,evals_per_block*num_blocks,evals_per_block)):
    ax.fill_between(np.arange(dblock_ep,dblock_ep+evals_per_block),
                    y0,y1,facecolor=colors[idx%2],alpha=0.1)

def plt_block_divides2(ax,start_L,end_L):
  y0,y1 = ax.get_ylim()
  colors = ['green','blue']
  for block in range(len(start_L)):
    ax.fill_between([start_L[block],end_L[block]],y0,y1,
                      facecolor=colors[block%2],alpha=0.1)
    if len(start_L)<100:
      ax.axvline(start_L[block],c='r',alpha=0.1)

