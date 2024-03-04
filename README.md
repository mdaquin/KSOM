# KSOM - Simple, but relatively efficient, pytorch-based self organising map

This is a simple implementation of self-organising map training in python, using pytorch for efficiency. Check ``test-img.py`` for a simple usecase creating a square color map of an image. To use, try
```
pip install -r requirements.txt
```
and then
```
python test_img.py chica.jpg 6
```
This shoud show an updating 6x6 map of the colours in the ``chica.jpg`` image. Change the image or the size of the map by changing the corresponding parameters.