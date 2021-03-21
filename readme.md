# Fast Mandelbrot set explorer

## Features
- **Accelerated on GPU** and CPU using numba CUDA JIT
- **Interactive exploration** using Matplotlib
  - Use mousewheel or left/right click to zoom in/out
  - Use button and slider to change the color palette and the number of iterations
- Smooth coloring
- Customizable color palette

## Quick start

```python
from mandelbrot import Mandelbrot

mand = Mandelbrot()

# Draw image and save to file
mand.draw(mandelbrot.png)

# Explore the set using interactive Matplotlib window
mand.explore()
```
## Requirements
- NumPy
- Matplotlib
- Numba
