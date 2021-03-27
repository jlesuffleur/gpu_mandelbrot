#!/usr/bin/env python3

"""Compute and draw/explore/animate the Mandelbrot set.

Fast computation of the Mandelbrot set using Numba on CPU or GPU. The set is
smoothly colored with custom colortables. 

  mand = Mandelbrot()
  mand.explore()
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from matplotlib.widgets import Slider, Button
from numba import cuda
from PIL import Image
import imageio


def sin_colortable(rgb_thetas=[.85, .0, .15], ncol=2**12):
    """ Sinusoidal color table
    
    Cyclic color table made with a sinus function for each color channel
    
    Args:
        rgb_thetas: [float, float, float]
            phase for each color channel
        ncol: int 
            number of color in the output table

    Returns:
        ndarray(dtype=uint8, ndim=2): color table
    """
    def colormap(x, rgb_thetas):
        # x in [0,1]
        # Compute the frequency and phase of each channel
        y = x*2*math.pi + math.pi
        y = np.column_stack((y + rgb_thetas[0] * 2 * math.pi,
                             y + rgb_thetas[1] * 2 * math.pi,
                             y + rgb_thetas[2] * 2 * math.pi))
        # Set amplitude to [0,255]
        val = np.around(255*(0.5 + 0.5*np.cos(y))).astype(np.uint8)
        return val

    return colormap(np.linspace(0, 1, ncol), rgb_thetas)

@jit
def smooth_iter(c, maxiter):
    """ Smooth number of iteration in the Mandelbrot set for given c
    
    Args:
        c: complex
            point of the complex plane
        maxiter: int 
            maximal number of iterations

    Returns:
        float: smooth iteration count at escape, 0 if maxiter is reached
    """
    # Escape radius squared: 2**2 is enough, but using a higher radius yields
    # better estimate of the smooth iteration count
    esc_radius_2 = 8**2
    z = complex(0, 0)
    
    # Mandelbrot iteration
    for n in range(maxiter):
        z = z*z + c
        # If escape: save (smooth) iteration count
        # Equivalent to abs(z) > esc_radius
        if z.real*z.real + z.imag*z.imag > esc_radius_2:
            # Smooth iteration count
            return n + 1 - math.log(math.log(abs(z)))/math.log(2)
    # Otherwise: leave iteration count to 0
    return 0

@jit
def compute_set(creal, cim, maxiter, colortable, ncycle):
    """ Compute and color the Mandelbrot set (CPU version)
    
    Args:
        creal: ndarray(dtype=float, ndim=1)
            vector of real coordinates
        cim: ndarray(dtype=float, ndim=1)
            vector of imaginary coordinates
        maxiter: int 
            maximal number of iterations
        colortable: ndarray(dtype=uint8, ndim=2)
            cyclic RGB colortable 
        ncycle: float
            number of iteration before cycling the colortable

    Returns:
        ndarray(dtype=uint8, ndim=3): image of the Mandelbrot set
    """
    xpixels = len(creal)
    ypixels = len(cim)
    ncol = colortable.shape[0] - 1
    
    # Output initialization
    mat = np.zeros((ypixels, xpixels, 3), np.uint8)

    # Looping through pixels
    for x in range(xpixels):
        for y in range(ypixels):
            
            # Initialization of c
            c = complex(creal[x], cim[y])
            # Get smooth iteration count
            niter = smooth_iter(c, maxiter)
            # Power post-transform
            niter = math.pow(niter, .5)
            # If escaped: color the set
            if niter != 0:
                col_i = round(niter % ncycle / ncycle * ncol)
                mat[y,x,0] = colortable[col_i,0]
                mat[y,x,1] = colortable[col_i,1]
                mat[y,x,2] = colortable[col_i,2]
    return mat

@cuda.jit
def compute_set_gpu(mat, xmin, xmax, ymin, ymax, maxiter, colortable, ncycle):
    """ Compute and color the Mandelbrot set (GPU version)
    
    Uses a 1D-grid with xpixels blocks of ypixels threads. For larger images, 
    one may need to change the grid because of the limitations of the GPU.
    
    Args:
        mat: ndarray(dtype=uint8, ndim=3)
            shared data to write the output image of the set
        xmin, xmax, ymin, ymax: float
            coordinates of the set
        maxiter: int 
            maximal number of iterations
        colortable: ndarray(dtype=uint8, ndim=2)
            cyclic RGB colortable 
        ncycle: float
            number of iteration before cycling the colortable

    Returns:
        mat: ndarray(dtype=uint8, ndim=3)
            shared data to write the output image of the set
    """
    # Retrieve x and y from CUDA grid coordinates
    x = cuda.blockIdx.x
    y = cuda.threadIdx.x
    ncol = colortable.shape[0] - 1
    
    # Mapping pixel to C
    creal = xmin + x / mat.shape[1] * (xmax - xmin)
    cim = ymin + y / mat.shape[0] * (ymax - ymin)
    
    # Initialization of c
    c = complex(creal, cim)

    # Get smooth iteration count
    niter = smooth_iter(c, maxiter)
    # Power post-transform
    # We use sqrt since pow can yield unexpected values with numba
    niter = math.sqrt(niter)
    
    # If escaped: color the set
    if niter != 0:
        col_i = round(niter % ncycle / ncycle * ncol)
        mat[y,x,0] = colortable[col_i,0]
        mat[y,x,1] = colortable[col_i,1]
        mat[y,x,2] = colortable[col_i,2]


class Mandelbrot():
    """Mandelbrot set object"""
    def __init__(self, xpixels=1000, maxiter=500,
                 coord=(-2.6, 1.85, -1.25, 1.25), gpu=False, ncycle=32,
                 colortable=None, oversampling_size=1):
        """Mandelbrot set object
    
        Args:
            xpixels: int
                image width (in pixels)
            maxiter: int
                maximal number of iterations
            coord: (float, float, float, float)
                coordinates of the frame in the complex space
            gpu: boolean
                use CUDA on GPU to compute the set
            ncycle: float
                number of iteration before cycling the colortable
            colortable: ndarray(dtype=uint8, ndim=2)
                color table used to color the set (preferably cyclic)
            oversampling_size: int
                for each pixel, a [n, n] grid is computed where n is the
                oversampling_size. Then, the average color of the n*n pixels
                is taken. Set to 1 for no oversampling.
        """
        self.xpixels = xpixels
        self.maxiter = maxiter
        self.coord = coord
        self.gpu = gpu
        self.ncycle = ncycle
        self.os = oversampling_size
        # Compute ypixels so the image is not stretched (1:1 ratio)
        self.ypixels = round(self.xpixels / (self.coord[1]-self.coord[0]) *
                             (self.coord[3]-self.coord[2]))
        # GPU: Number of threads per block is set to ypixels, but usually the 
        # maximum number of threads per block on a GPU is 1024. This require
        # to change the gridsize used.
        if (self.ypixels >= 1024) and self.gpu:
            raise AttributeError('image size is too high for GPU grid size')
            
        # Initialization of colortable
        if colortable is None:
            colortable = sin_colortable()
        self.colortable = colortable
        # Compute the set
        self.update_set()

    def update_set(self):
        """Updates the set
    
        Compute and color the Mandelbrot set, using CPU or GPU
        """
        # Apply ower post-transform to ncycle
        ncycle = math.pow(self.ncycle, .5)
        
        # Oversampling: rescaling by os
        xp = self.xpixels*self.os
        yp = self.ypixels*self.os
        
        if(self.gpu):
            # Pixel mapping is done in compute_self_gpu
            self.set = np.zeros((yp, xp, 3), np.uint8)
            # Compute set with GPU
            compute_set_gpu[xp,
                            yp](self.set, *self.coord, self.maxiter,
                                self.colortable, ncycle)
        else:
            # Mapping pixels to C
            creal = np.linspace(self.coord[0], self.coord[1], xp)
            cim = np.linspace(self.coord[2], self.coord[3], yp)
            # Compute set with CPU
            self.set = compute_set(creal, cim, self.maxiter,
                                   self.colortable, ncycle)
        # Oversampling: reshaping to (ypixels, xpixels, 3)
        if self.os > 1:
            self.set = (self.set
                        .reshape((self.ypixels, self.os,
                                  self.xpixels, self.os, 3))
                        .mean(3).mean(1).astype(np.uint8))

    def draw_pil(self, filename=None):
        """Draw or save, using PIL"""
        img = Image.fromarray(self.set, 'RGB')
        if filename is not None:
            img.save(filename) # fast (save in jpg) (compare reading as well)
        else:
            img.show() # slow
        
    def draw(self, filename=None, dpi=72):
        """Draw or save, using Matplotlib"""
        plt.subplots(figsize=(self.xpixels/dpi, self.ypixels/dpi))
        plt.imshow(self.set, extent=self.coord, origin='lower')
        # Remove axis and margins
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.axis('off')
        plt.show()
        # Write figure to file
        if filename is not None:
            plt.savefig(filename, dpi=dpi)
        
    def zoom_at(self, x, y, s):
        xrange = (self.coord[1] - self.coord[0])/2
        yrange = (self.coord[3] - self.coord[2])/2
        self.coord = [x - xrange * s,
                      x + xrange * s,
                      y - yrange * s,
                      y + yrange * s]
        
    def szoom_at(self, x, y, s):
        """Soft zoom (continuous)"""
        xrange = (self.coord[1] - self.coord[0])/2
        yrange = (self.coord[3] - self.coord[2])/2
        x = x * (1-s**2) + (self.coord[1] + self.coord[0])/2 * s**2
        y = y * (1-s**2) + (self.coord[3] + self.coord[2])/2 * s**2
        self.coord = [x - xrange * s,
                      x + xrange * s,
                      y - yrange * s,
                      y + yrange * s]      
        
    def animate(self, x, y, file_out, n_frames=100, loop=True):
        """Note that the Mandelbrot object is modified by this function"""
        # Zoom scale: gaussian shape, from 0% (s=1) to 40% (s=0.6)
        # => zoom scale (i.e. speed) is increasing, then decreasing
        def gaussian(n, sig = 1):
            x = np.linspace(-1, 1, n)
            return np.exp(-np.power(x, 2.) / (2 * np.power(sig, 2.)))
        s = 1 - gaussian(n_frames, 1/2)*.4
        
        # Update in case it was not up to date (e.g. parameters changed)
        self.update_set()
        images = [self.set]
        # Making list of images
        for i in range(1, n_frames):
            # Zoom at (x,y)
            self.szoom_at(x,y,s[i])
            # Update the set
            self.update_set()
            images.append(self.set)
            
        # Go backward, one image in two (i.e. 2x speed)
        if(loop):
            images += images[::-2]
        # Make GIF
        imageio.mimsave(file_out, images)   
    
    def explore(self, dpi=72):
        # It is important to keep track of the object in a variable, so the
        # slider and button are responsive
        self.explorer = Mandelbrot_explorer(self, dpi)


class Mandelbrot_explorer():
    def __init__(self, mand, dpi=72):
        self.mand = mand
        # Update in case it was not up to date (e.g. parameters changed)
        self.mand.update_set()
        # Plot the set
        self.fig, self.ax = plt.subplots(figsize=(mand.xpixels/dpi,
                                                  mand.ypixels/dpi))
        self.graph = plt.imshow(mand.set,
                                extent=mand.coord, origin='lower')
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.axis('off')
        # Add a slider of number of iterations
        self.ax_sld = plt.axes([0.3, 0.005, 0.4, 0.02])
        self.sld_maxiter = Slider(self.ax_sld, 'Iterations', 0,
                                 max(5000, self.mand.maxiter),
                             valinit=mand.maxiter, valstep=50)
        # Add a button to randomly change the color table
        self.ax_button = plt.axes([0.45, 0.03, 0.1, 0.035])
        self.button = Button(self.ax_button, 'Random colors')
        plt.sca(self.ax)
        
        # Note that it is mandatory to keep track of those objects so they are
        # not deleted by Matplotlib, and callbacks can be used
        # We call the same function for all event: self.onclick
        self.button.on_clicked(self.onclick)
        self.sld_maxiter.on_changed(self.onclick)
        # Responsiveness for any click or scroll
        self.cid1 = self.fig.canvas.mpl_connect('scroll_event', self.onclick)
        self.cid2 = self.fig.canvas.mpl_connect('button_press_event',
                                                self.onclick)
        plt.show()
        
    def onclick(self, event):
        # This function is called by any click/scroll, button click or slider
        # value change
        
        update = False
        
        # If event is an integer: it comes from the Slider
        if(isinstance(event, int)):
            # Slider event: update maxiter
            self.mand.maxiter = event
            update = True
        # Otherwise: check which axe was clicked
        else:    
            if event.inaxes == self.ax:
                # Click or scroll in the main axe: zoom event
                # Default: zoom in
                zoom = 1/2
                if event.button in ('down', 3):
                    # If right click or scroll down: zoom out
                    zoom = 1/zoom
                # Zoom and update figure coordinates
                self.mand.zoom_at(event.xdata, event.ydata, zoom)
                self.graph.set_extent(self.mand.coord)
                update = True
            elif ((event.inaxes == self.ax_button) and
                  (event.name == 'button_press_event')):
                # If the button is pressed: randomly change colortable
                rgb_thetas = np.random.uniform(size=3)
                self.mand.colortable = sin_colortable(rgb_thetas)
                update = True
        if update:
            # Updating the figure
            self.mand.update_set()
            self.graph.set_data(self.mand.set)
            plt.draw()       
            plt.show()


if __name__ == "__main__":
    Mandelbrot().explore()