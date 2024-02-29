import sys
import torch
import time

if len(sys.argv) < 4:
    print("Provide the name of the neighborhood function to test and the size (N) of map (NxN) and radius of neighborhood function.")
    sys.exit(-1)
if not sys.argv[2].isnumeric():
    print("Second argument should be a number.")
    sys.exit(-1)
if not sys.argv[3].isnumeric():
    print("Third argument should be a number.")
    sys.exit(-1)

from ksom import SOM, nb_linear, nb_gaussian, nb_ricker

if sys.argv[1] == "linear":
    fct = nb_linear
elif sys.argv[1] == "gaussian":
    fct = nb_gaussian
elif sys.argv[1] == "ricker":
    fct = nb_ricker      
else:
    print("Neighboorhood function should be linear or gaussian")
    sys.exit(-1)

import pygame

# init display screen and function to display map
screen_size=600 # size of screen 
pygame.init()
surface = pygame.display.set_mode((screen_size,screen_size))

def display(distmatrix):
    unit = int(screen_size/som_size)
    for i,cs in enumerate(distmatrix):
        x = int(i/som_size)
        y = i%som_size
        x = x*unit
        y = y*unit
        color = (max(min(255, int(-cs*255)), 0),
                 max(min(255, int(cs*255)), 0),
                 max(min(255, int(cs*255)), 0))
        pygame.draw.rect(surface,
                         color,
                         pygame.Rect(x, y, unit, unit))
        pygame.display.flip()

# init SOM model
som_size = int(sys.argv[2]) # size of som (square, so som_size x som_size)
smodel = SOM(som_size, som_size, 3)

node = (int(som_size/2), int(som_size/2))

distmtx = fct(torch.tensor(node), (som_size, som_size), smodel.coord, int(sys.argv[3]))
display(distmtx)

# continue to keep the display alive
while True: time.sleep(10)  

