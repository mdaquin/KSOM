from PIL import Image
from torchvision import transforms
from ksom import SOM, nb_linear
import pygame
import torch
import sys
import time

# BUG :: not working well when xs different from ys

screen_size=600
som_size = int(sys.argv[2])

pygame.init()
surface = pygame.display.set_mode((screen_size,screen_size))

im= Image.open(sys.argv[1])
x= transforms.ToTensor()(im)
x = x[:-1] if x.size(0) == 4 else x # remove alpha layer
x = x.view(-1, x.size()[1]*x.size()[2]).transpose(0,1)

smodel = SOM(som_size, som_size, 3, alpha_init=0.01, alpha_drate=1e-7, neighborhood_fct=nb_linear, neighborhood_init=som_size, neighborhood_drate=0.0001)

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


# TODO: stop when the diff is lower than a treshold for several iterations
perm = torch.randperm(x.size(0))

#idx = perm[0:100]
#time1 = time.time()
#dist = smodel.add(x[idx])
#print(dist, "-", round(((time.time()-time1)*1000), 2), "ms")
#sys.exit(0)

for i in range(int(x.size()[0]/100)):
    idx = perm[i*100:(i+1)*100]
    time1 = time.time()
    dist = smodel.add(x[idx])
    print((i+1)*100,"-", dist, "-", round(((time.time()-time1)*1000), 2), "ms")
    display(smodel)


while True:
    time.sleep(10)
        
#print(smodel(x[:3]))

