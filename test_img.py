import sys

if len(sys.argv) < 3:
    print("Provide image to analyse, and size (N) of colour map (NxN)")
    sys.exit(-1)
if not sys.argv[2].isnumeric():
    print("Second argument should be a number.")
    sys.exit(-1)

from PIL import Image
from torchvision import transforms
from ksom import SOM, nb_linear, nb_gaussian, nb_ricker
import pygame
import torch
import time

# init display screen and function to display map
screen_size=600 # size of screen 
pygame.init()
surface = pygame.display.set_mode((screen_size,screen_size))

def display(smodel):
    unit = int(screen_size/som_size)
    for i,cs in enumerate(smodel.somap):
        x = int(i/som_size)
        y = i%som_size
        x = x*unit
        y = y*unit
        color = (max(min(255, int(cs[0]*255)), 0),
                 max(min(255, int(cs[1]*255)), 0),
                 max(min(255, int(cs[2]*255)), 0))
        pygame.draw.rect(surface,
                         color,
                         pygame.Rect(x, y, unit, unit))
        pygame.display.flip()


# open image, transform into tensor, and create shuffle index
im= Image.open(sys.argv[1])
x= transforms.ToTensor()(im)
x = x[:-1] if x.size(0) == 4 else x # remove alpha layer if there is one
x = x.view(-1, x.size()[1]*x.size()[2]).transpose(0,1)
perm = torch.randperm(x.size(0))

# init SOM model
som_size = int(sys.argv[2]) # size of som (square, so som_size x som_size)
smodel = SOM(som_size, som_size, 3, zero_init=False,
             alpha_init=0.01, alpha_drate=1e-7,
             neighborhood_fct=nb_gaussian, neighborhood_init=som_size, neighborhood_drate=0.0001)

# train (1 pass through all the pixels) by batches of 100 pixels
for i in range(int(x.size()[0]/100)):
    idx = perm[i*100:(i+1)*100]
    time1 = time.time()
    dist = smodel.add(x[idx])
    print((i+1)*100,"-", dist, "-", round(((time.time()-time1)*1000), 2), "ms")
    display(smodel)

# continue to keep the display alive
while True: time.sleep(10)  

