#!/usr/bin/env python3

import math
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from matplotlib.widgets import Slider, Button
from numba import cuda
from PIL import Image
import imageio

@jit
def smooth_iter(c, maxiter):
    # Escape radius squared: 2**2 is enough, but using a higher radius yields
    # better estimate of the smooth iteration count
    esc_radius_2 = 8**2
    z = complex(0, 0)
    
    # Mandelbrot iteration
    for n in range(maxiter):
        z = z*z + c
        # If unbounded: save (smooth) iteration count
        # Equivalent to abs(z) > esc_radius
        if z.real*z.real + z.imag*z.imag > esc_radius_2:
            # Smooth iteration count
            return(n + 1 - math.log(math.log(abs(z)))/math.log(2))
    # Otherwise: leave iteration count to 0
    return(0)

@jit
def compute_set(creal, cim, maxiter, colortable, ncycle):
    
    xpixels = len(creal)
    ypixels = len(cim)
    mat = np.zeros((ypixels, xpixels, 3), np.uint8)
    ncol = colortable.shape[0] - 1
    
    # Looping through pixels
    for x in range(xpixels):
        for y in range(ypixels):
            
            # Initialisation of C
            c = complex(creal[x], cim[y])
            niter = smooth_iter(c, maxiter)
            if niter != 0:
                col_i = round(niter % ncycle / ncycle * ncol)
                mat[y,x,0] = colortable[col_i,0]
                mat[y,x,1] = colortable[col_i,1]
                mat[y,x,2] = colortable[col_i,2]
    return(mat)

@cuda.jit
def compute_set_gpu(mat, xmin, xmax, ymin, ymax, maxiter, colortable, ncycle):
    
    x = cuda.blockIdx.x
    y = cuda.threadIdx.x
    ncol = colortable.shape[0] - 1
    
    # Mapping pixel to C
    creal = xmin + x / mat.shape[1] * (xmax - xmin)
    cim = ymin + y / mat.shape[0] * (ymax - ymin)
    
    # Initialisation of C and Z
    c = complex(creal, cim)

    niter = smooth_iter(c, maxiter)
    if niter != 0:
        col_i = round(niter % ncycle / ncycle * ncol)
        mat[y,x,0] = colortable[col_i,0]
        mat[y,x,1] = colortable[col_i,1]
        mat[y,x,2] = colortable[col_i,2]

class Mandelbrot():
    """Compute the Mandelbrot set
    
        Args:
            xpixels (int): lenght of x-axis
            maxiter (int): maximal number of iterations
            xmin, xmax (float): min and max coordinates for x-axis (real part)
            ymin, ymax (float): min and max coordinates for y-axix (imaginary part)
    
        Returns:
            array (numpy.ndarray): the Mandelbrot set as a 2D array of shape (xpixels, ypixels)
    """
    def __init__(self, xpixels=1000, maxiter=100, coord=(-2.6, 1.85, -1.25, 1.25),
                 gpu = False, ncycle=40, rgb_thetas=[3.3, 4, 4.4]):
        self.xpixels = xpixels
        self.maxiter = maxiter
        self.coord = coord
        self.gpu = gpu
        self.ncycle = ncycle
        self.rgb_thetas = rgb_thetas
        # Compute ypixels so the image is not stretched (1:1 ratio)
        self.ypixels = round(self.xpixels / (self.coord[1]-self.coord[0]) *
                             (self.coord[3]-self.coord[2]))
        # GPU: Number of threads per block is set to ypixels, but usually the 
        # maximum number of threads per block on a GPU is 1024. This require
        # to change the gridsize used.
        if (self.ypixels >= 1024) & self.gpu:
            raise AttributeError('ypixels is too high for chosen GPU grid size')
            
        # Initialisation of colortable and set
        self.update_colortable()
        self.update_set()

        
    def update_set(self, color = False):
        if(self.gpu):
            # Pixel mapping is done in compute_self_gpu
            self.set = np.zeros((self.ypixels, self.xpixels, 3), np.uint8)
            # Compute set with GPU
            compute_set_gpu[self.xpixels,
                            self.ypixels](self.set, *self.coord, self.maxiter,
                                          self.colortable, self.ncycle)
        else:
            # Mapping pixels to C
            creal = np.linspace(self.coord[0], self.coord[1], self.xpixels)
            cim = np.linspace(self.coord[2], self.coord[3], self.ypixels)
            # Compute set with CPU
            self.set = compute_set(creal, cim, self.maxiter,
                                   self.colortable, self.ncycle)
            
    def update_colortable(self, ncol = 2**12):
        def colormap(x, theta = [0, 0, 0]):
            y = x*2*math.pi + math.pi
            y = np.column_stack((y + theta[0],
                                 y + theta[1],
                                 y + theta[2]))
            val = np.around(255*(0.5 + 0.5*np.cos(y))).astype(np.uint8)
            return(val)

        lin = np.linspace(0, 1, ncol)
        self.colortable = colormap(lin, self.rgb_thetas)

    def draw_pil(self, filename = None):
        img = Image.fromarray(self.set, 'RGB')
        if filename is not None:
            img.save(filename) # fast (save in jpg) (compare reading as well)
        else:
            img.show() # slow
        
    def draw(self, filename = None, dpi = 72):
        plt.subplots(figsize=(self.xpixels/dpi, self.ypixels/dpi))
        plt.imshow(self.set, extent = self.coord, origin = 'lower')
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.axis('off')
        plt.show()
        # Write figure to file
        if filename is not None:
            plt.savefig(filename, dpi=dpi)
            
    def zoom_gif(self, x, y, out, n_frames = 100, loop = True):
        """Note that the Mandelbrot object is modified by the zoom_gif"""
        # Zoom scale: gaussian shape, from 0% (s=1) to 40% (s=0.6)
        def gaussian(n, sig = 1):
            x = np.linspace(-1, 1, n)
            return np.exp(-np.power(x, 2.) / (2 * np.power(sig, 2.)))
        s = 1 - gaussian(n_frames, 1/2)*.4
        
        images = [self.set]
        # Making list images
        for i in range(1, n_frames):
    
            self.szoom_at(x,y,s[i])
            # add some iterations while zooming
            self.update_set()
            images.append(self.set)
            
        # Go backward, one image in two
        if(loop):
            images += images[::-2]
        # Make GIF
        imageio.mimsave(out, images)   
    
    def explore(self, dpi = 72):
        self.explorer = Mandelbrot_explorer(self, dpi)
        
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

class Mandelbrot_explorer():
    
    def __init__(self, mand, dpi = 72):
        self.mand = mand
        self.fig, self.ax = plt.subplots(figsize=(mand.xpixels/dpi,
                                                  mand.ypixels/dpi))
        self.graph = plt.imshow(mand.set,
                                extent = mand.coord, origin = 'lower')
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.axis('off')
        self.ax_sld = plt.axes([0.3, 0.005, 0.4, 0.02])
        self.sld_maxiter = Slider(self.ax_sld, 'Iterations', 0, 2000,
                             valinit=mand.maxiter, valstep=50)
        self.ax_button = plt.axes([0.45, 0.03, 0.1, 0.035])
        self.button = Button(self.ax_button, 'Random colors')
        plt.sca(self.ax)
        plt.show()
        
        self.button.on_clicked(self.onclick)
        self.sld_maxiter.on_changed(self.onclick)
        self.cid1 = self.fig.canvas.mpl_connect('scroll_event', self.onclick)
        self.cid2 = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        
    def onclick(self, event):
        
        update = False
        
        # If event is an integer: it comes from the Slider
        if(isinstance(event, int)):
            self.mand.maxiter = event
            update = True
        # Otherwise: check which axe was clicked
        else:    
            if event.inaxes == self.ax:
                zoom = 1/2
                if event.button in ('down', 3):
                    zoom = 1/zoom
                self.mand.zoom_at(event.xdata, event.ydata, zoom)
                self.graph.set_extent(self.mand.coord)
                update = True
            elif (event.inaxes == self.ax_button) & (event.name == 'button_press_event'):
                self.mand.rgb_thetas = np.random.uniform(size=3)
                self.mand.update_colortable()
                update = True
        if update:
            # Updating the figure
            self.mand.update_set()
            self.graph.set_data(self.mand.set)
            plt.draw()       
            plt.show()
