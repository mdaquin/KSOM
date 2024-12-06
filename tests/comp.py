import sys
import json
import minisom
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) != 3:
     print("provide a test configuration and a library to test (ksom_cpu, ksom_gpu or minisom).")
     sys.exit(-1)

config = json.load(open(sys.argv[1]))

som_size = config["somsize"]
distance = config["distance"]
nfct = config["nfct"]
nsamples = config["nsamples"]
    
from ksom import SOM, cosine_distance, euclidean_distance, nb_linear, nb_gaussian
import torch
import time

res = {"dim": [], "time":[]}
for dim in range(100, 10100, 100):
    x = torch.randn((nsamples,dim))

    if sys.argv[2] == "ksom_gpu" or sys.argv[2] == "ksom_cpu":
       # init SOM model
        smodel = SOM(som_size, som_size, dim, # sample_init=samples, # zero_init=False,
             dist=cosine_distance if distance=="cosine" else euclidean_distance,
             alpha_init=0.01, alpha_drate=1e-7,
             neighborhood_fct=nb_gaussian if nfct == "gaussian" else nb_linear, 
             neighborhood_init=som_size/2, 
             neighborhood_drate=0.0001
             )
        
        device = "cpu"
        if sys.argv[2] == "ksom_gpu" and torch.cuda.is_available():
            device = "cuda:0"
            x = x.to(device)
            smodel.to(device)
            # print("Running on CUDA")

        time1 = time.time()
        dist,count = smodel.add(x)
        otime = (time.time()-time1)

    elif sys.argv[2] == "minisom":
        msom = minisom.MiniSom(som_size, som_size, dim, 
                       activation_distance=distance, 
                       sigma=0.5,
                       neighborhood_function=nfct)
        x = x.cpu().numpy()
        time1 = time.time()
        msom.train(x, 1, use_epochs=True)
        otime = (time.time()-time1)
    
    print(f"{dim},{otime}")
    res["dim"].append(dim)
    res["time"].append(otime)

df = pd.DataFrame(res).set_index("dim")
df.to_csv("comp_results_"+sys.argv[2]+".csv")
df.plot()
plt.show()
