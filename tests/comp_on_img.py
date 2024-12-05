import sys
import json
import minisom

if len(sys.argv) != 2:
     print("provide a test configuration.")
     sys.exit(-1)

config = json.load(open(sys.argv[1]))

som_size = config["somsize"]
distance = config["distance"]
nfct = config["nfct"]
nsize = config["nsize"]
stop = config["stop"]
disp = True
    
from PIL import Image
from torchvision import transforms
from ksom import SOM, cosine_distance, euclidean_distance, nb_linear, nb_gaussian, nb_ricker
if disp: import pygame
import torch
import time

# init display screen and function to display map
if disp:
  screen_size=600 # size of screen 
  pygame.init()
  surface = pygame.display.set_mode((screen_size,screen_size))

def display(map, save=None):
    for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
    unit = int(screen_size/som_size)
    for i,cs in enumerate(map):
        x = int(i/som_size)
        y = i%som_size
        x = x*unit
        y = y*unit
        try : 
            color = (max(min(255, int(cs[0]*255)), 0),
                     max(min(255, int(cs[1]*255)), 0),
                     max(min(255, int(cs[2]*255)), 0))
        except: 
            print(cs*255)
            sys.exit(-1)
        pygame.draw.rect(surface,
                         color,
                         pygame.Rect(x, y, unit, unit))
    pygame.display.flip()
    pygame.display.update()
    if save is not None:
        pygame.image.save(surface, save)

# open image, transform into tensor, and create shuffle index
im= Image.open("chica.jpg")
x= transforms.ToTensor()(im)
x = x[:-1] if x.size(0) == 4 else x # remove alpha layer if there is one
x = x.view(-1, x.size()[1]*x.size()[2]).transpose(0,1)
perm = torch.randperm(x.size(0))

# init SOM model
samples = x[perm[-(som_size*som_size):]]
smodel = SOM(som_size, som_size, 3, # sample_init=samples, # zero_init=False,
             dist=cosine_distance if distance=="cosine" else euclidean_distance,
             alpha_init=0.01, alpha_drate=1e-7,
             neighborhood_fct=nb_gaussian if nfct == "gaussian" else nb_linear, 
             neighborhood_init=nsize, 
             neighborhood_drate=0.0001
             )

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
    x = x.to(device)
    smodel.to(device)
    print("Running on CUDA")

idx = perm[:stop*1000]
time1 = time.time()
dist,count = smodel.add(x[idx])
timedone = (time.time()-time1)
print(f"ksom trained in {timedone:05.2f}s")
display(smodel.somap, save="ksom_on_img.png")

msom = minisom.MiniSom(som_size, som_size, 3, 
                       activation_distance=distance, 
                       sigma=nsize/som_size,
                       neighborhood_function=nfct)
idx = perm[:stop*1000]
time1 = time.time()
msom.train(x[idx].cpu().numpy(), 1, use_epochs=True)
timedone = (time.time()-time1)
print(f"minisom trained in {timedone:05.2f}s")
if disp: display(msom._weights.flatten().reshape(-1,3), save="minisom_on_img.png")

# continue to keep the display alive
if disp: 
  while True:
    for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()    
    time.sleep(0.1)
    pygame.display.flip()    
    pygame.display.update()