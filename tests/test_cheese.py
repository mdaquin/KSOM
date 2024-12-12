import math
import time
import pandas as pd
import pygame
from sklearn.decomposition import PCA
import ksom
import torch
import sys

def remspace(x): return x.replace(", ", ",") if type(x) == str else x

def onehotencode(df, col):
    return pd.concat([df, df[col].apply(remspace).str.get_dummies(sep=",")], axis=1).drop(col, axis=1)

def findLabel(i, map, labels):
     idx = abs(map.mean(dim=0)-map[i]).argmax()
     lab = labels[idx]
     if map.mean(dim=0)[idx] > map[i][idx]: lab = "not "+lab
     return lab

def display(map, xoffset=0, labels=None, label_offset=0):
    if map.shape[1] > 3:
        pca = PCA(n_components=3)
        somap = pca.fit_transform(map)
    else: somap = map
    for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
    unit = int(screen_size/smodel.xs)
    for i,cs in enumerate(somap):
        x = int(i/smodel.xs)
        y = i%smodel.xs
        x = (x*unit)+xoffset
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
        if labels is not None:
             lab = findLabel(i, map, labels)
             cp = surface.get_at((int(x+label_offset+unit/20),y+int(unit/5)))
             cl = (200, 200, 200)
             if cp[0] > 100 : cl = (0, 0, 0)
             texts = font.render(lab, False, cl)
             surface.blit(texts, (x+label_offset+unit/20,y+int(unit/5)))

    pygame.display.flip()
    pygame.display.update()

df = pd.read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/refs/heads/main/data/2024/2024-06-04/cheeses.csv", index_col=0)
df = df.drop(["url", "producers", "alt_spellings", "synonyms", "fat_content", "calcium_content", "vegetarian", "vegan"], axis=1)
print(df.columns)
df = onehotencode(df, "milk")
df = onehotencode(df, "country")
df = onehotencode(df, "region")
df = onehotencode(df, "family")
df = onehotencode(df, "type")
df = onehotencode(df, "texture")
df = onehotencode(df, "rind")
df = onehotencode(df, "color")
df = onehotencode(df, "flavor")
df = onehotencode(df, "aroma")

#print(df.aroma.apply(remspace).str.get_dummies(sep=","))
#df["vegetarian"] = df.vegetarian.apply(lambda x: int(x) if type(x)==bool else x)
#df["vegan"] = df.vegan.apply(lambda x: int(x) if type(x)==bool else x)
df = df.dropna()

screen_size=600 # size of screen 
pygame.init()
surface = pygame.display.set_mode((screen_size*2,screen_size))

NBEPOCH = 7
BATCHSIZE = 100
SOMSIZE = 6
DIST = ksom.cosine_distance

pygame.font.init()
font = pygame.font.SysFont('Courrier',int((screen_size/6)/5))

smodel = ksom.SOM(SOMSIZE, SOMSIZE, 
                  len(df.T), 
                  zero_init=True, 
                  # sample_init=torch.Tensor(df.sample(SOMSIZE*SOMSIZE).to_numpy()),
                  dist=DIST)

for epoch in range(NBEPOCH):
    for b in range(math.ceil(len(df)/BATCHSIZE)):
        dist,count = smodel.add(torch.Tensor(df.iloc[b*BATCHSIZE:(b+1)*BATCHSIZE].to_numpy()))
        print(f"{epoch+1:02d}.{b:02d}: distance {dist:.4f} out of {count} objects")
        freqmap = torch.zeros(SOMSIZE*SOMSIZE)
        bmu,dists = smodel(torch.Tensor(df.to_numpy()))
        for i in bmu: freqmap[i[0]*SOMSIZE+i[1]] += 1
        freqmap = (freqmap - freqmap.min())/(freqmap.max()-freqmap.min())
        freqmap = freqmap.view(len(freqmap), 1).repeat((1,3))
        display(freqmap, xoffset=screen_size)
        display(smodel.somap, labels=list(df.columns), label_offset=screen_size)

# continue to keep the display alive
while True:
    for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()    
    time.sleep(0.1)
    pygame.display.flip()    
    pygame.display.update()
