### Self-organising map pytroch model
import torch
import math
import numpy as np

def euclidean_distance(x,y):
    """returns a distance matrix between the elements of
    the tensor x and the ones of the tensor y"""
    return torch.cdist(x,y,2)

def nb_ricker(node, dims, coord, nb):
    """
    Ricker wavelet (mexican hat) neighborhood weights between between node (x,y) 
    and all the coordinates in the tensor coord ([(x,y)]) assuming
    it follow the dimensions in dims (height, width).
    nb is the neighborhood radius (i.e. the distance after which 
    the function returns 0.
    """
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

    def __init__(self, xs, ys, dim,
                 dist=euclidean_distance,
                 alpha_init=1e-3, alpha_drate=1e-6,
                 neighborhood_init=None, neighborhood_fct=nb_linear, neighborhood_drate=1e-6):
        super(SOM, self).__init__()
        self.somap = torch.randn(xs*ys, dim)
        self.xs = xs
        self.ys = ys
        self.dist = dist
        self.step = 0
        self.neighborhood_drate = neighborhood_drate
        self.neighborhood_fct = neighborhood_fct
        self.alpha_init  = alpha_init
        self.alpha_drate = alpha_drate
        if neighborhood_init is None: self.neighborhood_init = min(xs,ys)/2 # start with half the map
        else: self.neighborhood_init = neighborhood_init
        lx = torch.arange(xs).repeat(ys).view(-1, ys).T.reshape(xs*ys)
        ly = torch.arange(ys).repeat(xs)
        self.coord = torch.stack((lx,ly), -1)
        
    def forward(self, x):
        dists = self.dist(self.somap, x)
        bmu_ind = dists.min(dim=0).indices
        bmu_ind_x = (bmu_ind/self.xs).to(torch.int32)
        bmu_ind_y = bmu_ind%self.xs
        return torch.stack((bmu_ind_x, bmu_ind_y), -1), dists

    def __1DIndexTo2DIndex(self, ind):
        return torch.Tensor([int(ind/self.xs), ind%self.xs])
    
    def add(self, x):
        prev_som = self.somap.clone().detach()
        for x_k in x:
            # decreases linearly...
            nb = max(self.neighborhood_drate, self.neighborhood_init - (self.step*self.neighborhood_drate))
            alpha = max(self.alpha_drate, self.alpha_init - (self.step*self.alpha_drate))
            self.step += 1
            bmu = self(x_k.view(-1, x_k.size()[0]))[0][0]
            theta = self.neighborhood_fct(bmu, (self.xs, self.ys), self.coord, nb)
            ntheta = theta.view(-1, theta.size(0)).T
            self.somap = self.somap + ntheta*(alpha*(x_k-self.somap))
            # old non batch (slow) version
            # batch here means calculating the whole map at once,
            # not having a batch of values treated at once. 
            # for i, w_i in enumerate(self.somap):
            #      i2 = self.__1DIndexTo2DIndex(i)
            #      theta = self.neighborhood_fct(bmu, i2, nb)                
            #      self.somap[i] = w_i + theta*alpha*(x_k-w_i)
            # print("o nsomap", self.somap)
                #  wij' = wij + ( n_fct(bmu,ni,nb(s)) * alpha(s) * (x_k - wij) )
        return float(torch.nn.functional.pairwise_distance(prev_som, self.somap).mean())
