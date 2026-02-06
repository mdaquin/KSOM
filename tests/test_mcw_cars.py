import math
import time
import pandas as pd
import pygame
from sklearn.decomposition import PCA
import sys
sys.path.insert(0, "../src/")
import ksom.ksom as ksom
import torch
import sys
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def remspace(x): return x.replace(", ", ",") if type(x) == str else x

def onehotencode(df, col):
    return pd.concat([df, df[col].apply(remspace).apply(lambda x: str(col[0])+"_"+str(x)).str.get_dummies(sep=",")], axis=1).drop(col, axis=1)

def findLabel(i, map, labels):
     idx = abs(map.mean(dim=0)-map[i]).argmax()
     lab = labels[idx]
     if map.mean(dim=0)[idx] > map[i][idx]: lab = "not "+lab
     return lab

def display(map, som_size, xoffset=0, yoffset=0, labels=None, label_offset=0):
    if map.shape[1] > 3:
        pca = PCA(n_components=3)
        somap = pca.fit_transform(map.detach())
    else: somap = map
    for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
    unit = int(channel_size/som_size)
    for i,cs in enumerate(somap):
        x = int(i/som_size)
        y = i%som_size
        x = (x*unit)+xoffset
        y = (y*unit)+yoffset
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

NBEPOCH = 3
BATCHSIZE = 256
SOMSIZE = 3
DIST = ksom.cosine_distance
LR = 1e-2
alpha = 1e-2
alpha_drate = 5e-8
sparcity_coeff = 1e-2
nchannels = 4
channel_sim_loss_coef = 1e-2

# layout: channels in a grid, each channel shows SOM + frequency map side by side
channel_size = 300
grid_cols = min(nchannels, 2)
grid_rows = math.ceil(nchannels / grid_cols)
screen_width = grid_cols * channel_size * 2
screen_height = grid_rows * channel_size

pygame.init()
surface = pygame.display.set_mode((screen_width, screen_height))
pygame.font.init()
font = pygame.font.SysFont('Courrier', int((channel_size/SOMSIZE)/5))

smodel = ksom.MCWSOM(SOMSIZE, SOMSIZE,
                  len(df.T),
                  # zero_init=True,
                  sample_init=torch.Tensor(df.sample(SOMSIZE*SOMSIZE, random_state=42).to_numpy()),
                  dist=DIST, alpha_drate=alpha_drate, alpha_init=alpha,
                  sparcity_coeff=sparcity_coeff,
                  n_channels=nchannels,
                  channel_sim_loss_coef=channel_sim_loss_coef)

## Initial weights
iweights = []
for i, wsom in enumerate(smodel.channel_wsoms):
    iweights.append({c: float(wsom.weights[j]) for j, c in enumerate(df.columns)})
    iweights[i] = dict(sorted(iweights[i].items(), key=lambda item: item[1], reverse=True))
    for k, v in iweights[i].items(): print(f"{k}: {v:.4f}")
    print("="*50)

## Optimizer and training loop
optimizer = torch.optim.Adam(smodel.parameters(), lr=LR)
smodel.train()
for epoch in range(NBEPOCH):
    sdist = 0
    sloss = 0
    freqmaps = [torch.zeros(SOMSIZE*SOMSIZE) for _ in range(nchannels)]
    for b in range(math.ceil(len(df)/BATCHSIZE)):
        dist,count,loss,wloss = smodel.add(torch.Tensor(df.iloc[b*BATCHSIZE:(b+1)*BATCHSIZE].to_numpy()), optimizer)
        print(f"{epoch+1:02d}.{b:02d}: distance {dist:.6f} out of {count} objects, loss={loss:.6f}, wloss={wloss:.6f}")
        sdist += dist
        sloss += loss
        bmus_dists = smodel(torch.Tensor(df.iloc[b*BATCHSIZE:(b+1)*BATCHSIZE].to_numpy()))
        for c in range(nchannels):
            bmu = bmus_dists[c][0]
            for i in bmu: freqmaps[c][i[0]*SOMSIZE+i[1]] += 1
            ffreqmap = (freqmaps[c] - freqmaps[c].min())/(freqmaps[c].max()-freqmaps[c].min())
            col = c % grid_cols
            row = c // grid_cols
            xoff = col * channel_size * 2
            yoff = row * channel_size
            display(smodel.channel_wsoms[c].somap, SOMSIZE, xoffset=xoff, yoffset=yoff, labels=list(df.columns), label_offset=channel_size)
            display(ffreqmap.view(len(freqmaps[c]), 1).repeat((1,3)), SOMSIZE, xoffset=xoff + channel_size, yoffset=yoff)
    print(f"Final {epoch+1:02d}: distance {sdist/(math.ceil(len(df)/BATCHSIZE)):.6f}, loss={sloss/(math.ceil(len(df)/BATCHSIZE)):.6f}")


## Final weights
iweights = []
for i, wsom in enumerate(smodel.channel_wsoms):
    iweights.append({c: float(wsom.weights[j]) for j, c in enumerate(df.columns)})
    iweights[i] = dict(sorted(iweights[i].items(), key=lambda item: item[1], reverse=True))
    for k, v in iweights[i].items(): print(f"{k}: {v:.4f} torch.sigmoid({v:.4f})={torch.sigmoid(torch.Tensor([v]))[0]:.4f}")
    print("="*50)

## Final display
for c in range(nchannels):
    col = c % grid_cols
    row = c // grid_cols
    xoff = col * channel_size * 2
    yoff = row * channel_size
    display(smodel.channel_wsoms[c].somap, SOMSIZE, xoffset=xoff, yoffset=yoff, labels=list(df.columns), label_offset=channel_size)
    ffreqmap = (freqmaps[c] - freqmaps[c].min())/(freqmaps[c].max()-freqmaps[c].min())
    display(ffreqmap.view(len(freqmaps[c]), 1).repeat((1,3)), SOMSIZE, xoffset=xoff + channel_size, yoffset=yoff)

# continue to keep the display alive
while True:
    for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
    time.sleep(0.1)
