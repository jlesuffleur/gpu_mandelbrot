# Fast Mandelbrot set explorer

## Features
- **Accelerated on GPU** and CPU using numba CUDA JIT
- **Interactive exploration** using Matplotlib
  - Use mousewheel or left/right click to zoom in/out
  - Use button and slider to change the color palette and the number of iterations
- Save still and animated images
- Smooth iteration coloring, anti-aliasing by oversampling
- **Shading:** stripe average and Lambert normal
- Customizable color palette
- **Precision:** zoom down to float64 precision (approx. 16 significants digits)

All that in a 100% Python code 🐍

## Quick start

```python
from mandelbrot import Mandelbrot
mand = Mandelbrot(gpu = True) # set gpu to False if not available
```

### Explore the set

```python
# Explore the set using interactive Matplotlib GUI
mand.explore()
```
### Draw an image

```python
# Draw an image and save it to file
mand.draw('mandelbrot.jpg')
```
![](img/mandelbrot.png)

### Make a zoom animation

```python
# We decrease the image size to avoid overloading git and the readme page
mand = Mandelbrot(maxiter = 2500, xpixels = 426, gpu = True)
# Point to zoom at
x_real = -1.7492404293151038
x_imag = -0.0002615793835705974
mand.animate(x_real, x_imag, 'mandelbrot.gif')
```
![](img/mandelbrot.gif)

### Run from command line

Use the following command from terminal to launch a GUI with default parameters:

```shell
python ./mandelbrot.py
```

## Runtime 🚀

Computing a sequence of `100` frames of pictures of size `1800*1000` pixels, with `2000` iterations takes approximately **1 second** on a Tesla K80 GPU.

## Requirements
- NumPy
- Matplotlib
- Numba
- (optional, for much faster rendering) A CUDA compatible GPU & CUDA Toolkit
