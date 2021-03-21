#!/usr/bin/env python3

import math
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from matplotlib.widgets import Slider, Button

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
def compute_set(creal, cim, maxiter):
    
    xpixels = len(creal)
    ypixels = len(cim)
    mat = np.zeros((xpixels, ypixels))
    
    # Looping through pixels
    for x in range(xpixels):
        for y in range(ypixels):
            
            # Initialisation of C
            c = complex(creal[x], cim[y])
            mat[x,y] = smooth_iter(c, maxiter)
    return(mat)



@jit
def colorize(mat, colortable, ncycle):

    mat_col = np.zeros((*mat.shape, 3)).astype(np.uint8)
    ncol = colortable.shape[0] - 1
    
    # Looping through pixels
    for x in range(mat.shape[0]):
        for y in range(mat.shape[1]):
            val = mat[x,y]
            if val != 0:
                col_i = round(val % ncycle / ncycle * ncol)
                mat_col[x,y,0] = colortable[col_i,0]
                mat_col[x,y,1] = colortable[col_i,1]
                mat_col[x,y,2] = colortable[col_i,2]
    return(mat_col)


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
    def __init__(self, xpixels=1000, maxiter=100, coord=(-2.6, 1.85, -1.25, 1.25)):
        self.xpixels = xpixels
        self.maxiter = maxiter
        self.coord = coord
        self.rgb_thetas = [3.3, 4, 4.4]
        # Compute ypixels so the image is not stretched (1:1 ratio)
        self.ypixels = round(self.xpixels / (self.coord[1]-self.coord[0]) *
                             (self.coord[3]-self.coord[2]))
        # Initialisation of output matrix to 0
        self.set = np.zeros((self.xpixels, self.ypixels))   
        self.update_set()
        
    def update_set(self):
        # Mapping pixels to C
        creal = np.linspace(self.coord[0], self.coord[1], self.xpixels)
        cim = np.linspace(self.coord[2], self.coord[3], self.ypixels)
    
        self.set = compute_set(creal, cim, self.maxiter)

    def to_image(self, ncol = 2**12, ncycle = 40):
        
        def colormap(x, theta = [0, 0, 0]):
            y = x*2*math.pi + math.pi
            y = np.column_stack((y + theta[0],
                                 y + theta[1],
                                 y + theta[2]))
            val = np.around(255*(0.5 + 0.5*np.cos(y))).astype(np.uint8)
            return(val)

        lin = np.linspace(0, 1, ncol)
        colortable = colormap(lin, self.rgb_thetas)
        mat_col = colorize(self.set.T, colortable, ncycle)
        return(mat_col)
        
    def draw(self, filename = None, dpi = 72):
        plt.subplots(figsize=(self.xpixels/dpi, self.ypixels/dpi))
        plt.imshow(self.to_image(), extent = self.coord, origin = 'lower')
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.axis('off')
        plt.show()
        # Write figure to file
        if filename is not None:
            plt.savefig(filename, dpi=dpi)
            
    def explore(self, dpi = 72):
        Mandelbrot_explorer(self, dpi)

class Mandelbrot_explorer():
    
    def __init__(self, mand, dpi = 72):
        self.mand = mand
        self.fig, self.ax = plt.subplots(figsize=(mand.xpixels/dpi,
                                                  mand.ypixels/dpi))
        self.graph = plt.imshow(mand.to_image(),
                                extent = mand.coord, origin = 'lower')
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.axis('off')
        self.ax_sld = plt.axes([0.3, 0.005, 0.4, 0.02])
        self.sld_maxiter = Slider(self.ax_sld, 'Iterations', 0, 1000,
                             valinit=mand.maxiter, valstep=50)
        self.ax_button = plt.axes([0.45, 0.03, 0.1, 0.035])
        self.button = Button(self.ax_button, 'Random colors')
        plt.sca(self.ax)
        plt.show()
        
        self.button.on_clicked(self.onclick)
        self.sld_maxiter.on_changed(self.onclick)
        self.fig.canvas.mpl_connect('scroll_event', self.onclick)
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        
    def onclick(self, event):
        # If event is an integer: it comes from the Slider
        if(isinstance(event, int)):
            self.mand.maxiter = event
            self.mand.update_set()
        # Otherwise: check which axe was clicked
        else:    
            if event.inaxes == self.ax:
                zoom = 1/2
                if event.button in ('down', 3):
                    zoom = 1/zoom
                # Compute new coordinates
                xrange = (plt.axis()[1] - plt.axis()[0])/2
                yrange = (plt.axis()[3] - plt.axis()[2])/2
                self.mand.coord = [event.xdata - xrange * zoom,
                                   event.xdata + xrange * zoom,
                                   event.ydata - yrange * zoom,
                                   event.ydata + yrange * zoom]
                self.graph.set_extent(self.mand.coord)
                self.mand.update_set()
            elif (event.inaxes == self.ax_button) & (event.name == 'button_press_event'):
                self.mand.rgb_thetas = np.random.uniform(size=3)
    
        # Updating the figure
        self.graph.set_data(self.mand.to_image())
        plt.draw()       
        plt.show()
        
if __name__ == '__main__':
    mand = Mandelbrot(1200, 500)
    mand.draw()
