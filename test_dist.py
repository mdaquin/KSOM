import torch
from ksom import euclidean_distance, cosine_distance
from scipy.spatial import distance

som_size = 10
dim = 128
batch_size = 256

som = torch.randn(som_size*som_size, 128)
batch = torch.randn(batch_size, dim)

euc1 = euclidean_distance(batch, som)
euc2 = []

cos1 = cosine_distance(batch, som)
cos2 = []

for s in batch:
    seuc = []
    scos = []
    for c in som:
        seuc.append(distance.euclidean(s, c))
        scos.append(distance.cosine(s, c))
    euc2.append(torch.Tensor(seuc))
    cos2.append(torch.Tensor(scos))
euc2 = torch.cat(euc2).view(batch_size, som_size*som_size)
cos2 = torch.cat(cos2).view(batch_size, som_size*som_size)

print("euclidean distance:", float(abs(euc1-euc2).mean()))
print("cosine distance:", float(abs(cos1-cos2).mean()))