# KSOM - Simple, but relatively efficient, pytorch-based self organising map

This is a simple implementation of self-organising map training in python, using pytorch for efficiency. This enables to create, train and apply square maps of potentially high dimensions on CPU or GPU.


To install, use 
```
pip install ksom
```

An example is available in ``test/test-img.py`` for a simple use case creating a square color map of an image. Having loaded the data in a tensor x, the code to initialise and train the SOM looks like this:

```python
from ksom import SOM, cosine_distance, nb_gaussian
...
smodel = SOM(6, 6, 3, # size of the map and dimension of units
             sample_init=samples, # initialised with samples
             dist=cosine_distance, # using cosine distance for BMU
             alpha_init=0.01, # learning rate
             alpha_drate=1e-7, # decay of learning rate
             neighborhood_fct=nb_gaussian, # neighboorhood function
             neighborhood_init=som_size, # initial neighbourhood radius
             neighborhood_drate=0.0001) # decay of neighbourhood radius

perm = torch.randperm(x.size(0)) # to shuffle the data
for i in range(int(x.size()[0]/1000)):
    idx = perm[i*1000:(i+1)*1000] 
    time1 = time.time()
    dist,count = smodel.add(x[idx]) # feed the SOM a batch of 1000 pixels
    print(f"{(i+1):06d}K - {dist:.4f} - {(time.time()-time1)*1000:05.2f}ms")
```

The results on the image on the left looks like the map on the right, where each unit is represented by the colour corresponding to its weights.

![map from image](./imgs/chica_map.png)


Another example is included in the ``test/test_cheese.py`` creating a map of cheeses based on various binary attributes. The results is presented below with on the right the map represented by colours for each unit created through PCA with 3 components for the RGB components of the colour. On the left, a frequency map is given that show how many cheese have each unit for BMU (brighter == more cheese) as well as the name of the attribute most different in this unit compared to the average of the whole dataset. 

![map of chesses](./imgs/cheese.gif)
