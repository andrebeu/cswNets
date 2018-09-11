import numpy as np
from numpy.random import choice

def gen_path(graph_dict,pr):
  """ graph is a dict{A:[B,C]}
  A is current state B is transitioned to with pr
  each path makes a single training example"""
  path = []
  st = 0
  while st < 7:
    path.append(st)
    coin = np.random.binomial(1,pr)
    if np.random.binomial(1,pr):
      st1 = graph_dict[st][coin]
    else:
      st1 = graph_dict[st][coin]
    st = st1  
  return path


def gen_paths(graph_dict,pr,num_paths):
  """ graph is a dict{A:[B,C]}
  A is current state B is transitioned to with pr
  each path makes a single training example"""
  path = []
  path_num = -1
  st = 0
  while path_num < num_paths:
    if st == 0 or st == 1: path_num += 1
    path.append(st)
    coin = np.random.binomial(1,pr)
    if coin:
      st1 = graph_dict[st][coin]
    else:
      st1 = graph_dict[st][coin]
    st = st1 
  return path



def gen_xy(path,filler_id):
  """ given a path through the graph e.g. [A,C] 
  make set of training examples ([A,x],[C,x])
  where A, is state and x is filler_id """
  X = []
  Y = []
  for i in range(len(path)-1):
    x,y = path[i:i+2]  
    X.extend([x,filler_id])
    Y.extend([y,filler_id])
  return np.array(X),np.array(Y)

def gen_xy2(path):
  """ given a path through the graph e.g. [A,C] 
  make set of training examples ([A,x],[C,x])
  where A, is state and x is filler_id """
  X = []
  Y = []
  for i in range(len(path)-1):
    x,y = path[i:i+2]  
    X.extend([x])
    Y.extend([y])
  return np.array(X),np.array(Y)

def gen_m_samples(m,graph_dict,transition_pr,filler_id):
  X = []
  Y = []
  for itr in range(m):
    path = gen_path(graph_dict,transition_pr)
    x,y = gen_xy(path,filler_id)
    X.append(x)
    Y.append(y)
  X = np.array(X)
  Y = np.array(Y)
  return X,Y

def gen_m_samples2(m,graph_dict,transition_pr,num_paths):
  X = []
  Y = []
  for itr in range(m):
    path = gen_paths(graph_dict,transition_pr,num_paths)
    x,y = gen_xy2(path)
    X.append(x)
    Y.append(y)
  X = np.array(X)
  Y = np.array(Y)
  return X,Y



graph_dict1 = {
    0:[1,2],
    1:[3,4],
    2:[3,4],
    3:[5,6],
    4:[5,6],
    5:[7,7],
    6:[7,7]
  }

graph_dict2 = {
    0:[2,1],
    1:[4,3],
    2:[4,3],
    3:[6,5],
    4:[6,5],
    5:[7,7],
    6:[7,7]
  }

graph_dict3 = {
  0:[2,3],
  1:[2,3],
  2:[4,5],
  3:[4,5],
  4:[0,1],
  5:[0,1]
}
graph_dict4 = {
  0:[3,2],
  1:[3,2],
  2:[5,4],
  3:[5,4],
  4:[1,0],
  5:[1,0]
}


graph_dict_L = [graph_dict1,graph_dict2]


def gen_data_dict(story_size=100,RAND=False,COND=False,graph_dict=None):
  if not graph_dict:
    graph_dict = choice(graph_dict_L)

  if RAND:
    filler_id = 15
  else:
    filler_id = 8

  if COND:
    X1,Y1 = gen_m_samples(m=int(story_size),graph_dict=graph_dict,transition_pr=.8,filler_id=filler_id)
    X2,Y2 = gen_m_samples(m=int(story_size),graph_dict=graph_dict,transition_pr=.2,filler_id=filler_id+1)
    X = np.vstack([X1,X2])
    Y = np.vstack([Y1,Y2])
  else:
    X,Y = gen_m_samples(m=story_size,graph_dict=graph_dict,transition_pr=.8,filler_id=filler_id)

  train_data_dict = {'X_data':X,'Y_data':Y}
  return train_data_dict


graph_dict_L2 = [graph_dict3,graph_dict4]
def gen_data_dict2(num_stories=100,num_paths=5,graph_dict=None):
  if not graph_dict:
    graph_dict = choice(graph_dict_L2)

  X,Y = gen_m_samples2(num_stories,graph_dict,.8,num_paths=num_paths)

  train_data_dict = {'X_data':X,'Y_data':Y}
  return train_data_dict


