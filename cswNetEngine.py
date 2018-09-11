import json
from numpy import random
import numpy as np

# see experiment for more options
SCH_FPATH = "csw_condloc.sch"


def read_json(path):
  """ load schema files"""
  with open(path) as f:
    schema_info_L = json.load(f)
  return schema_info_L


class CSW():
  def __init__(self,conditioned=True):
    self.sch_dict = sch_dict = read_json(SCH_FPATH) 
    # params
    self.conditioned = conditioned
    # conditions
    self.cond_dict = self.make_cond_dict(sch_dict)
    self.fixed_cond = 'location.latent.false'
    # graph building
    self.graph = {} 
    self.assign_edges(sch_dict) 
    self.assign_nodeids(sch_dict)

  # graph building 

  def assign_edges(self,sch_dict):
    """ graph building 
    assigns an edges {cond:edge} to each node in dict"""
    for nodename,edges in sch_dict.items():
      if nodename == 'END': continue
      if 'sent' in edges: edges.pop('sent') 
      self.graph[nodename] = {'edges': edges['edge']}

  def assign_nodeids(self,sch_dict):
    """ graph building 
    assigns an int id to each node """
    for nodeid,node in enumerate(sch_dict):
      if node == 'END':continue
      self.graph[node]['nodeid'] = nodeid

  def make_cond_dict(self,sch_dict):
    """ assigns a unique id to each condition
    """
    # {cond1,cond2...}
    condS = set() # set with every condition
    for n in self.sch_dict.values():
      condS.update(list(n['edge'].keys())) 
    cond_dict = {}
    # {cond:cond_id}
    for cond_id,cond in enumerate(condS):
      cond_dict[cond] = len(sch_dict) + cond_id
    return cond_dict

  # generating path

  def sample_tonode(self,frnode_name,cond):
    """ given name of frnode and condition, 
    returns name of next tonode """
    edge = self.graph[frnode_name]['edges'][cond] 
    next_tonode_L = list(edge.keys())
    next_tonode_pr = list(edge.values())
    next_tonode_name = random.choice(next_tonode_L,p=next_tonode_pr)
    return next_tonode_name

  def gen_path(self,cond_=None):
    """ 
    graph = {node:{edges(dict),id(int)}, edges={cond:edge}, edge={tonode:pr}
    cond = role.property.value (e.g. location.latent.false)
    returns path [nodeid]
    """
    # establish condition
    # for unconditioned transitions pass None
    cond = cond_ or self.fixed_cond
    # start at Begin node
    frnode_name = 'BEGIN'
    frnode_id = self.graph[frnode_name]['nodeid']
    path = [frnode_id]
    while True:
      # draw next tonode
      next_tonode_name = self.sample_tonode(frnode_name,cond) 
      if next_tonode_name == 'END': break
      # collect nodeid
      next_tonode_id = self.graph[next_tonode_name]['nodeid']
      path.append(next_tonode_id)
      # walk
      frnode_name = next_tonode_name
    return path


  # general purpose methods for analysis
  
  def get_node(self,by,value):
    """ general purpose getter for nodes"""
    if by == 'id':
      id_node_dict = {}
      for name,node in self.graph.items():
        id_node_dict[node['nodeid']] = node
      return id_node_dict[value]
    return None

  def id2cond(self,id_):
    """ given an id (e.g. 15), returns condition"""
    for cond,condid in self.cond_dict.items():
      if condid == id_:
        return cond
    return None

  def get_opt_and_alt_tonodes(self,nodeid,fillerid=None):
    """ 
    given a nodeid,fillerid determines 
    which is the optimal and alternative transitions
    returns the ids of the tonodes which are (optimal,alternative)
    """
    # get edge that is dictating transitions
    if self.conditioned: 
      # gets the condition to which fillerid corresponds
      cond = self.id2cond(fillerid) 
    else: 
      cond = self.fixed_cond
    edge = self.get_node('id',nodeid)['edges'][cond]
    # loop through nodes in edge
    for tonode,pr in edge.items():
      if tonode == 'END': return (1,1)
      if pr>0.5:
        optimal_tonode = self.graph[tonode]['nodeid']
      elif pr<0.5:
        alternative_tonode = self.graph[tonode]['nodeid']
      else:
        optimal_tonode, alternative_tonode = 1,2
    return optimal_tonode,alternative_tonode