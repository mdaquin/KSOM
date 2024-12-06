import sys
import json
import minisom
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
     print("provide a test configuration.")
     sys.exit(-1)

config = json.load(open(sys.argv[1]))

som_size = config["somsize"]
distance = config["distance"]
nfct = config["nfct"]
nsamples = config["nsamples"]
    
from ksom import SOM, cosine_distance, euclidean_distance, nb_linear, nb_gaussian
import torch
import time

res = {"dim": [], "ksom cpu":[], "ksom gpu": [], "minisom":[]}
for dim in range(100, 10100, 100):
    x = torch.randn((nsamples,dim))
 
    # init SOM model
    smodel = SOM(som_size, som_size, dim, # sample_init=samples, # zero_init=False,
             dist=cosine_distance if distance=="cosine" else euclidean_distance,
             alpha_init=0.01, alpha_drate=1e-7,
             neighborhood_fct=nb_gaussian if nfct == "gaussian" else nb_linear, 
             neighborhood_init=som_size/2, 
             neighborhood_drate=0.0001
             )
    
    x = x.to("cpu")
    smodel.to("cpu")
    
    time1 = time.time()
    dist,count = smodel.add(x)
    tkc = (time.time()-time1)

    # init SOM model
    smodel = SOM(som_size, som_size, dim, # sample_init=samples, # zero_init=False,
             dist=cosine_distance if distance=="cosine" else euclidean_distance,
             alpha_init=0.01, alpha_drate=1e-7,
             neighborhood_fct=nb_gaussian if nfct == "gaussian" else nb_linear, 
             neighborhood_init=som_size/2, 
             neighborhood_drate=0.0001
             )

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        x = x.to(device)
        smodel.to(device)
        print("Running on CUDA")

    time1 = time.time()
    dist,count = smodel.add(x)
    tkg = (time.time()-time1)

    msom = minisom.MiniSom(som_size, som_size, dim, 
                       activation_distance=distance, 
                       sigma=0.5,
                       neighborhood_function=nfct)
    x = x.cpu().numpy()
    time1 = time.time()
    msom.train(x, 1, use_epochs=True)
    tms = (time.time()-time1)
    
    print(f"{dim},{tkc},{tkg},{tms}")
    res["dim"].append(dim)
    res["ksom cpu"].append(tkc)
    res["ksom gpu"].append(tkg)
    res["minisom"].append(tms)

pd.DataFrame(res).set_index("dim").plot()
plt.show()
