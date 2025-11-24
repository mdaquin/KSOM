import math
import time
import pandas as pd
import pygame
from sklearn.decomposition import PCA
import sys 
sys.path.insert(0, "src/")
import ksom.ksom as ksom
import torch
import sys
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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
        somap = pca.fit_transform(map.detach())
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

df = pd.read_csv("https://mdaquin.github.io/d/cars/all.csv")
df = df.drop("model", axis=1)
df = df.drop("brand", axis=1)
print(df)
# df = onehotencode(df, "brand")
df = onehotencode(df, "transmission")
df = onehotencode(df, "fuelType")

df = df.dropna()
# df = df.sample(10000)

scaler = MinMaxScaler()
df[df.columns] = scaler.fit_transform(df[df.columns])

screen_size=600 # size of screen 
pygame.init()
surface = pygame.display.set_mode((screen_size*2,screen_size))

NBEPOCH = 5
BATCHSIZE = 256
SOMSIZE = 5
DIST = ksom.cosine_distance
LR = 1e-2
alpha = 1e-2
alpha_drate = 5e-8

pygame.font.init()
font = pygame.font.SysFont('Courrier',int((screen_size/6)/5))

smodel = ksom.WSOM(SOMSIZE, SOMSIZE, 
                  len(df.T), 
                  # zero_init=True, 
                  sample_init=torch.Tensor(df.sample(SOMSIZE*SOMSIZE).to_numpy()),
                  dist=DIST, alpha_drate=alpha_drate, alpha_init=alpha)

iweights = {c: float(smodel.weights[i]) for i, c in enumerate(df.columns)}
# sort weights by value
iweights = dict(sorted(iweights.items(), key=lambda item: item[1], reverse=True))
for k, v in iweights.items(): print(f"{k}: {v:.4f}")

optimizer = torch.optim.Adam(smodel.parameters(), lr=LR)
smodel.train()
for epoch in range(NBEPOCH):
    sdist = 0
    sloss = 0
    freqmap = torch.zeros(SOMSIZE*SOMSIZE)
    for b in range(math.ceil(len(df)/BATCHSIZE)):
        dist,count,loss = smodel.add(torch.Tensor(df.iloc[b*BATCHSIZE:(b+1)*BATCHSIZE].to_numpy()), optimizer)
        print(f"{epoch+1:02d}.{b:02d}: distance {dist:.6f} out of {count} objects, loss={loss:.6f}")
        sdist += dist
        sloss += loss 
        bmu,dists = smodel(torch.Tensor(df.iloc[b*BATCHSIZE:(b+1)*BATCHSIZE].to_numpy()))
        for i in bmu: freqmap[i[0]*SOMSIZE+i[1]] += 1
        ffreqmap = (freqmap - freqmap.min())/(freqmap.max()-freqmap.min())
        # freqmap = freqmap.view(len(freqmap), 1).repeat((1,3))
        display(ffreqmap.view(len(freqmap), 1).repeat((1,3)), xoffset=screen_size)
        display(smodel.somap, labels=list(df.columns), label_offset=screen_size)
    print(f"Final {epoch+1:02d}: distance {sdist/(math.ceil(len(df)/BATCHSIZE)):.6f}, loss={sloss/(math.ceil(len(df)/BATCHSIZE)):.6f}")


weights = {c: float(smodel.weights[i]) for i, c in enumerate(df.columns)}
# sort weights by value
weights = dict(sorted(weights.items(), key=lambda item: item[1], reverse=True))
for k, v in weights.items(): print(f"{k}: {v:.4f}")

# continue to keep the display alive
while True:
    for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()    
    time.sleep(0.1)
    pygame.display.flip()    
    pygame.display.update()
