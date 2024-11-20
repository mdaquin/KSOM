import sys

if len(sys.argv) < 3:
     print("No image or map size provided, default parameters (chica.jpg, 6) will be used.")
     img = "chica.jpg"
     som_size = 6
else: 
    img = sys.argv[1]
    if not sys.argv[2].isnumeric():
        print("Second argument should be a number.")
        sys.exit(-1)
    som_size = int(sys.argv[2])
disp = True
if len(sys.argv) >=4: disp = sys.argv[3] != "nodisplay"
    
from PIL import Image
from torchvision import transforms
from ksom import SOM, cosine_distance, nb_linear, nb_gaussian, nb_ricker
if disp: import pygame
import torch
import time

# init display screen and function to display map
if disp:
  screen_size=600 # size of screen 
  pygame.init()
  surface = pygame.display.set_mode((screen_size,screen_size))

def display(smodel):
    for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
    unit = int(screen_size/som_size)
    for i,cs in enumerate(smodel.somap):
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

# open image, transform into tensor, and create shuffle index
im= Image.open(img)
x= transforms.ToTensor()(im)
x = x[:-1] if x.size(0) == 4 else x # remove alpha layer if there is one
x = x.view(-1, x.size()[1]*x.size()[2]).transpose(0,1)
perm = torch.randperm(x.size(0))

# init SOM model
smodel = SOM(som_size, som_size, 3, zero_init=True,
             # dist=cosine_distance,
             alpha_init=0.01, alpha_drate=1e-8,
             neighborhood_fct=nb_gaussian, neighborhood_init=som_size/2, neighborhood_drate=0.00001)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
    x = x.to(device)
    smodel.to(device)
    print("Running on CUDA")

# train (1 pass through all the pixels) by batches of 1000 pixels
for i in range(int(x.size()[0]/1000)):
    idx = perm[i*1000:(i+1)*1000]
    time1 = time.time()
    dist,count = smodel.add(x[idx])
    print(f"{(i+1):06d}K - {dist:.4f} - {(time.time()-time1)*1000:05.2f}ms")
    if disp: display(smodel)
    
# continue to keep the display alive
if display: 
  while True:
    for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()    
    time.sleep(0.1)
    pygame.display.flip()    
    pygame.display.update()
