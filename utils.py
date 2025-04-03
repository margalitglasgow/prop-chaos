import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

def her_to_pol(k):
  if k == 1:
    p_dic = {1: 1}
  if k == 2:
    p_dic = {0: -1, 2: 1}
  if k == 3:
    p_dic = {1: -3, 3: 1}
  if k == 4:
    p_dic = {0: 3, 2: -6, 4: 1}
  if k == 5:
    p_dic = {1: 15, 3: -10, 5: 1}
  if k == 6:
    p_dic = {0: -15, 2: 45, 4: -15, 6: 1}
  for key in p_dic.keys():
    p_dic[key] = p_dic[key]/np.sqrt(math.factorial(k))
  return p_dic

def her_dic_to_pol(dic):
  p_dic = {}
  for k in dic.keys():
    k_pol = her_to_pol(k)
    for k2 in k_pol.keys():
      if k2 not in p_dic.keys():
        p_dic[k2] = dic[k]*k_pol[k2]
      else:
        p_dic[k2] += dic[k]*k_pol[k2]
  return p_dic

def poly_act(x, poly_dict):
  out = torch.zeros(size=x.size())
  for k in poly_dict.keys():
    out += poly_dict[k]*torch.pow(x, k)
  return out
def poly_act_fun(poly_dict):
  return lambda x: poly_act(x, poly_dict)
def poly_label_fun(poly_dict):
  return lambda x: poly_act(x[:, 0], poly_dict)

# Hermite Polynomials up to k = 6, note normalization by 1/k!
def hermite(x, k):
      if k == 1:
          x2 = x
      if k == 2:
          x2 = torch.square(x) - 1
      if k == 3:
          x2 = torch.pow(x, 3) - 3*x
      if k == 4:
          x2 = torch.pow(x, 4) - 6*torch.pow(x, 2) + 3
      if k == 5:
          x2 = torch.pow(x, 5) - 10*torch.pow(x, 3) + 15*x
      if k == 6:
          x2 = torch.pow(x, 6) - 15*torch.pow(x, 4) + 45*torch.pow(x, 2) - 15
      x2 = x2/(np.sqrt(math.factorial(k)))
      return x2

def her(k):
  return lambda x: hermite(x, k)

def ie4np(x):
  xsq = torch.square(x)
  return torch.exp(-xsq/2)*(2*xsq - 1)
def ie4np_fun():
  return  lambda x: ie4np(x)

def ie4np_other(x):
  xsq = torch.square(x)
  return torch.exp(-xsq/2)*(2*xsq - 1) +  torch.exp(-xsq)*xsq
def ie4np_other_fun():
  return  lambda x: ie4np_other(x)

def ge4(x):
    xsq = torch.square(x)
    return 5*(torch.exp(-xsq)*xsq - np.sqrt(3)/9)
def ge4_fun():
  return lambda x: ge4(x)

# Complex Activations

# Hermite k and Hermite 1
def complex_act_low_IE(k):
  return lambda x: hermite(x, k) + hermite(x, 1)

# Hermite k and Hermite k + 2
def complex_act_high_IE(k):
  return lambda x: hermite(x, k) + hermite(x, k+2)

# General activation/label. {k: c_k} ---> returns sum_k c_k He_k(x)
def dict_act(x, dic):
  out = torch.zeros(size=x.size())
  for k in dic.keys():
    out += hermite(x, k)*dic[k]
  return out
def dict_act_fun(dic):
  return lambda x: dict_act(x, dic)
def dict_label_fun(dic):
  return lambda x: dict_act(x[:, 0], dic)

#ReLU
def relu(x):
  return F.relu(x)

# For generating labels (takes X also)
def orth(x, k):
    return 0.5*hermite(x[:, 0], k) + 0.5*hermite(x[:, 1], k)
def orth_fun(k):
  return lambda x: orth(x, k)

def nonorth(x, k):
    return 0.5*hermite(x[:, 0], k) + 0.5*hermite((x[:, 1] + x[:, 0])/np.sqrt(2), k)
def nonorth_fun(k):
  return lambda x: nonorth(x, k)

def single(x, k):
    return hermite(x[:, 0], k)
def single_fun(k):
  return lambda x: single(x, k)

def compl(x, k):
    return 0.5*(hermite(x[:, 0], k) + x[:, 0]) + 0.5*(hermite(x[:, 1], k) + x[:, 1])
def compl_fun(k):
  return lambda x: compl(x, k)

theta = np.pi/3
def complex_nonorth(x, k):
    return 0.5*(hermite(x[:, 0], k) + x[:, 0]) + 0.5*(hermite((np.sin(theta)*x[:, 1] + np.cos(theta)*x[:, 0]), k) + (np.sin(theta)*x[:, 1] + np.cos(theta)*x[:, 0]))
def complex_nonorth_fun(k):
  return lambda x: complex_nonorth(x, k)

#Teacher-Student labels
def teacher(x, wstars, activation):
    K = len(wstars)
    # print(wstars.type())
    projection = torch.matmul(x, wstars.T) # (n x d) @ (d x K)
    link = activation(projection) # should be n x K
    return torch.mean(link, dim=1) # should be size n
def teacher_fun(wstars, activation):
  return lambda x: teacher(x[:, :len(wstars.T)], wstars, activation)

# Manifold unform on circle (Hermite 4)
def man(x, k_ind):
    k_GT = 4
    norms_sq = torch.square(torch.norm(x[:, :k_ind], dim=1))
    return (3*torch.pow(norms_sq, 2)/8 - 3*norms_sq + 3)/(np.sqrt(math.factorial(k_GT)))
def man_fun(k_ind):
    return lambda x: man(x, k_ind)
def man_big_fun(k_ind):
    return lambda x: 4*man(x, k_ind)

#Manifold SIM
# Outputs the equivalent of (alpha) He_k(x) + (1 - alpha) \mathbb{E}_{G \sim N(0, 1)} He_k(beta* x + sqrt(1 - beta^2)*G)
# This can techncially be fit by NN with activation He_k (though unclear if GD will accomplish this.)
def manifold_SIM(x, alpha, beta, k):
    z = (alpha + (1 - alpha)*(beta**k))*hermite(x[:, 0], k)
    return z/(np.sqrt(math.factorial(k)))
def manifold_SIM_fun(alpha, beta, k):
   return lambda x: manifold_SIM(x, alpha, beta, k)


# k-XOR function (product version fors guassian)
def XOR(x, k):
    X_low = x[:, :k]
    prod = torch.ones(len(x))
    for i in range(k):
        prod *= X_low[:, i]
    return prod #/(k*(np.sqrt(math.factorial(k))))
def XOR_fun(k):
  return lambda x: XOR(x, k)

def staircase(x,k):
  res = XOR(x, 1)
  for i in range(k - 1):
      res += XOR(x, i + 2)
  return res
def staircase_fun(k):
  return lambda x: staircase(x, k)/10

def hopcase(x,k):
  res = XOR(x, k) + XOR(x, 2*k)
  return res
def hopcase_fun(k):
  return lambda x: hopcase(x, k)/2

def hop(x,k1, k2):
  res = 0.25*XOR(x, k1) + 0.75*XOR(x, k1 + k2)
  return res
def hop_fun(k1, k2):
  return lambda x: hop(x, k1, k2)