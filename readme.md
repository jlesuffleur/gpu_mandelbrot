# Fast Mandelbrot set explorer

## Features
- **Accelerated on GPU** and CPU using numba CUDA JIT
- **Interactive exploration** using Matplotlib
  - Use mousewheel or left/right click to zoom in/out
  - Use button and slider to change the color palette and the number of iterations
- Save still and animated images
- Smooth coloring
- Customizable color palette

## Quick start

```python
from mandelbrot import Mandelbrot
mand = Mandelbrot(gpu = True)

# Explore the set using interactive Matplotlib window
mand.explore()
```

```
# Draw an image and save it to file
mand.draw('mandelbrot.png')
```

```
# Zoom to a given point and save the animation as a GIF
x_real = -1.7576871663606164
y_imag = 0.017457512970355783
mand.zoom_gif(x_real, y_imag, 'mandelbrot.gif')
```


## Requirements
- NumPy
- Matplotlib
- Numba
- (optional) A CUDA compatible GPU for much faster rendering
