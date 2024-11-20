# Self-organising map pytroch model.
# Author: Mathieu d'Aquin

import torch
from torch.nn import CosineSimilarity
import math
import numpy as np

def euclidean_distance(x,y):
    """returns a distance matrix between the elements of
    the tensor x and the ones of the tensor y"""
    return torch.cdist(x,y,2)

# BUG : does not work in batch !!! 
def cosine_distance(x,y):
    """returns a distance matrix between the elements of 
    the tensor x and the ones of the tensor y"""
    out = CosineSimilarity(dim=1, eps=1e-6)(x,y.repeat(len(x), 1))
    return 1-out.resize(len(out), 1)

# TODO: #13 this does not seem to work...
def nb_ricker(node, dims, coord, nb):
    """
    Ricker wavelet (mexican hat) neighborhood weights between between node (x,y) 
    and all the coordinates in the tensor coord ([(x,y)]) assuming
    it follow the dimensions in dims (height, width).
    nb is the neighborhood radius (i.e. the distance after which 
    the function returns 0).
    """
    # if nb < 1.0: nb = 1.0 # seem to make no difference...
    nodes = node.repeat(dims[0]*dims[1], 1)
    dist = torch.nn.functional.pairwise_distance(nodes, coord)
    dist[int(node[0]*dims[0])+node[1]%dims[0]] = 0.0
    FM = (math.sqrt(6)/(math.pi*(2*nb)))/math.sqrt(3)
    fbit = (1-2*math.pi**2*FM**2*dist**2)
    sbit = -math.pi**2*FM**2*dist**2
    return fbit*torch.exp(sbit)

def nb_gaussian(node, dims, coord, nb):
    """
    Gaussian neighborhood weights between between node (x,y) 
    and all the coordinates in the tensor coord ([(x,y)]) assuming
    it follow the dimensions in dims (height, width).
    nb is the neighborhood radius (i.e. the distance after which 
    the function returns 0).    
    """
    # exp(-(x/(nb/2))**2)
    nodes = node.repeat(dims[0]*dims[1], 1)
    dist = torch.nn.functional.pairwise_distance(nodes, coord)
    dist[int(node[0]*dims[0])+node[1]%dims[0]] = 0.0
    ret = torch.exp(-(dist/(nb/2))**2)
    ret[ret < 0] = 0.0
    return ret    

def nb_linear_o(node1, node2, nb):
    """deprecated non-batch versoin of the linear
    neighborhood function"""
    if node1[0] == node2[0] and node1[1] == node2[1]: return 1.0
    dist = euclidean_distance(node1.view(-1, 2).to(torch.float32),
                              node2.view(-1, 2).to(torch.float32))[0][0]
    return max(0,1-(dist/nb))

def nb_linear(node, dims, coord, nb):
    """linear neighborhood distances between node (x,y) 
    and all the coordinates in the tensor coord ([(x,y)]) assuming
    it follow the dimensions in dims (height, width).
    nb is the neighborhood radius (i.e. the distance after which 
    the function returns 0)."""
    nodes = node.repeat(dims[0]*dims[1], 1)
    dist = torch.nn.functional.pairwise_distance(nodes, coord)
    dist[int(node[0]*dims[0])+node[1]%dims[0]] = 0.0
    ret = 1-(dist/nb)
    ret[ret < 0] = 0.0
    return ret

class SOM(torch.nn.Module):
    """
    Model for a self-organising map (Kohonen map), which is trained by adding data points that will make the map adapt. The general principle for the training is that each point is matched to the closest unit (using the euclidean distance). The weights of this best matching unit (BMU) are then updated to move closer to the input data point (to a rate dependent on the learning rate, alpha). Neighboring units are also updated, to a lesser extent, depending on the neighborhood function and radius. The learning rate and the neighborhood radius decay over time (provided input points). 

Parameters:
-----------

xs: vertical size of the map

ys: horizontal size of the map

dist (default: ksom.euclidean_distance): the fonction to identify the best matching unit (BMU). This takes two 2D vectors and returns a 2D matrix of the distances between them. 

zero_init (default: False): Whether the map is initialised randomly, or by zero vectors. 

alpha_init (default: 1e-3): initial value of learning rate.

alpha_drate (default: 1e-6): decay rate of the learning rate. The learning rate (alpha) decreases linearly based on this rate for each new data point, until it reaches the valyue of alpha_drate. 

neightborgood_init (default: half of the smallest dimension of the map): the initial radius of the neighborhood.

neighborhood_fct (default: ksom.nb_gaussian, other values: nb_linear, nb_ricker): the function to get the neighborhood rate depending on the neighborhood rate and the euclidean distance between units. 

neighborhood_drate (default: 1e-6): rate of decay of the neigtborhood radius. This will decrease linearly from neighborhood_init to neighborhood_drate depending on neighborhood_drate.

Example:  
--------

See https://github.com/mdaquin/KSOM/blob/main/test_img.py for an example of the use of KSOM to build a map of the pixel colors of an image. 
    """
    
    def __init__(self, xs, ys, dim,
                 dist=euclidean_distance, zero_init=False, 
                 alpha_init=1e-2, alpha_drate=1e-6,
                 neighborhood_init=None, neighborhood_fct=nb_gaussian, neighborhood_drate=1e-6, 
                 minval=None, maxval=None, device="cpu"):
        if type(xs) != int or type(ys) != int or type(dim) != int: raise TypeError("size and dimension of SOM should be int")
        if alpha_init <= alpha_drate: raise ValueError("Decay rate of learning rate (alpha_drate) should be smaller than initial value (alpha_init)")
        if neighborhood_init is None: self.neighborhood_init = min(xs,ys)/2 # start with half the map
        else: self.neighborhood_init = neighborhood_init
        if self.neighborhood_init <= neighborhood_drate: raise ValueError("Neighborhood radius decay rate should (neighborhood_drate) should be smaller than initial value (neighborhood_init)")
        super(SOM, self).__init__()
        self.somap = torch.randn(xs*ys, dim).to(device)
        if minval is not None and maxval is not None:
            self.somap = (self.somap + minval) * (maxval - minval)
        self.minval = minval
        self.maxval = maxval
        if zero_init: self.somap = torch.zeros((xs*ys, dim), dtype=torch.float).to(device)
        self.xs = xs
        self.ys = ys
        self.dim = dim
        self.dist = dist
        self.step = 0
        self.neighborhood_drate = neighborhood_drate
        self.neighborhood_fct = neighborhood_fct
        self.alpha_init  = alpha_init
        self.alpha_drate = alpha_drate
        lx = torch.arange(xs).repeat(ys).view(-1, ys).T.reshape(xs*ys)
        ly = torch.arange(ys).repeat(xs)
        self.coord = torch.stack((lx,ly), -1).to(device)

    def to(self, device):
        super(SOM, self).to(device)
        self.somap = self.somap.to(device)
        self.coord = self.coord.to(device)
        
    def forward(self, x):
        """
        Identifies the best matching unit for the data points in x in the current map.

        Parameter:
        ----------

        x: 2D tensor (N, dim) corresponding to N data points of dimension dim. 

        Returns:
        --------

        Returns a tuple including the coordinates of the bmu in the current map and the distance matrix used to find it.
        """
        if type(x) != torch.Tensor: raise TypeError("x should be a tensor of shape (N,dim)")
        if len(x.size()) != 2: raise ValueError("x should be a tensor of shape (N,dim)")
        if x.size()[1] != self.dim: raise ValueError("x should be a tensor of shape (N,dim)")
        dists = self.dist(self.somap, x)
        bmu_ind = dists.min(dim=0).indices
        bmu_ind_x = (bmu_ind/self.xs).to(torch.int32)
        bmu_ind_y = bmu_ind%self.xs
        return torch.stack((bmu_ind_x, bmu_ind_y), -1), dists

    def __1DIndexTo2DIndex(self, ind):
        return torch.Tensor([int(ind/self.xs), ind%self.xs])
    
    def add(self, x):
        """
        Add the data points in x to the current map and update the map to those points (training). 

        Parameters:
        -----------

        x: 2D tensor (N, dim) corresponding to N data points of dimension dim. 

        Returns:
        --------

        Returns the euclidean distance of the SOM's matrix before and after training. Gives an indication of the impact of the added training points on the map.
        """
        if type(x) != torch.Tensor: raise TypeError("x should be a tensor of shape (N,dim)")
        if len(x.size()) != 2: raise ValueError("x should be a tensor of shape (N,dim)")
        if x.size()[1] != self.dim: raise ValueError("x should be a tensor of shape (N,dim)")
        prev_som = self.somap.clone().detach()
        count = 0
        for x_k in x:
            if x_k.isnan().any() or x_k.isinf().any(): continue # do not try to add vector containing nans ! 
            count+=1
            # decreases linearly...
            nb = max(self.neighborhood_drate, self.neighborhood_init - (self.step*self.neighborhood_drate))
            alpha = max(self.alpha_drate, self.alpha_init - (self.step*self.alpha_drate))
            self.step += 1
            bmu = self(x_k.view(-1, x_k.size()[0]))[0][0]
            theta = self.neighborhood_fct(bmu, (self.xs, self.ys), self.coord, nb)
            ntheta = theta.view(-1, theta.size(0)).T
            # TODO: print(ntheta) prob is that neg ones get to quickly neg and then go to infinity...
            self.somap = self.somap + ntheta*(alpha*(x_k-self.somap))
            if torch.isnan(self.somap).any() or torch.isinf(self.somap).any(): 
                print("*** Nan! ***")
                print("bmu", bmu)
                print("theta", theta.min(), theta.max(), theta.mean(), theta.isnan().any())
                print("ntheta", ntheta.min(), ntheta.max(), ntheta.mean(), ntheta.isnan().any())
                print("alpha", alpha)
                print("x_k", x_k.min(), x_k.max(), x_k.mean(), x_k.isnan().any())
            # old non batch (slow) version
            # batch here means calculating the whole map at once,
            # not having a batch of values treated at once. 
            # for i, w_i in enumerate(self.somap):
            #      i2 = self.__1DIndexTo2DIndex(i)
            #      theta = self.neighborhood_fct(bmu, i2, nb)                
            #      self.somap[i] = w_i + theta*alpha*(x_k-w_i)
            # print("o nsomap", self.somap)
                #  wij' = wij + ( n_fct(bmu,ni,nb(s)) * alpha(s) * (x_k - wij) )
        return float(torch.nn.functional.pairwise_distance(prev_som, self.somap).mean()), count
