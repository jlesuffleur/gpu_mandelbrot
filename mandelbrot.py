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
from numba import jit, cuda
from matplotlib.widgets import Slider
from PIL import Image
import imageio

def sin_colortable(rgb_thetas=(.85, .0, .15), ncol=2**12):
    """ Sinusoidal color table
   
    Cyclic and smooth color table made with a sinus function for each color
    channel   
    Args:
        rgb_thetas: (float, float, float)
            phase for each color channel
        ncol: int
            number of color in the output table

    Returns:
        ndarray(dtype=float, ndim=2): color table
    """
    def colormap(x, rgb_thetas):
        # x in [0,1]
        # Compute the frequency and phase of each channel
        y = np.column_stack(((x + rgb_thetas[0]) * 2 * math.pi,
                             (x + rgb_thetas[1]) * 2 * math.pi,
                             (x + rgb_thetas[2]) * 2 * math.pi))
        # Set amplitude to [0,1]
        val = 0.5 + 0.5*np.sin(y)
        return val
    return colormap(np.linspace(0, 1, ncol), rgb_thetas)

@jit
def blinn_phong(normal, light):
    """ Blinn-Phong shading algorithm
   
    Brightess computed by Blinn-Phong shading algorithm, for one pixel,
    given the normal and the light vectors

    Returns:
        float: Blinn-Phong brightness
    """
    ## Lambert normal shading (diffuse light)
    normal = normal / abs(normal)    
    
    # theta: light azimuth; phi: light elevation
    # light vector: [cos(theta)cos(phi), sin(theta)cos(phi), sin(phi)]
    # normal vector: [normal.real, normal.imag, 1]
    # Diffuse light = dot product(light, normal)
    ldiff = (normal.real*math.cos(light[0])*math.cos(light[1]) +
             normal.imag*math.sin(light[0])*math.cos(light[1]) +
             1*math.sin(light[1]))
    # Normalization
    ldiff = ldiff/(1+1*math.sin(light[1]))
    
    ## Specular light: Blinn Phong shading
    # Phi half: average between pi/2 and phi (viewer elevation)
    # Specular light = dot product(phi_half, normal)
    phi_half = (math.pi/2 + light[1])/2
    lspec = (normal.real*math.cos(light[0])*math.sin(phi_half) +
             normal.imag*math.sin(light[0])*math.sin(phi_half) +
             1*math.cos(phi_half))
    # Normalization
    lspec = lspec/(1+1*math.cos(phi_half))
    #spec_angle = max(0, spec_angle)
    lspec = lspec ** light[6] # shininess
    
    ## Brightness = ambiant + diffuse + specular
    bright = light[3] + light[4]*ldiff + light[5]*lspec
    ## Add intensity
    bright = bright * light[2] + (1-light[2])/2 
    return bright
    
@jit
def smooth_iter(c, maxiter, stripe_s, stripe_sig):
    """ Smooth number of iteration in the Mandelbrot set for given c
   
    Args:
        c: complex
            point of the complex plane
        maxiter: int
            maximal number of iterations
        stripe_s:
            frequency parameter of stripe average coloring
        stripe_sig:
            memory parameter of stripe average coloring

    Returns: (float, float, float, complex)
        - smooth iteration count at escape, 0 if maxiter is reached
        - stripe average coloring value, in [0,1]
        - dem: estimate of distance to the nearest point of the set
        - normal, used for shading
    """
    # Escape radius squared: 2**2 is enough, but using a higher radius yields
    # better estimate of the smooth iteration count and the stripes
    esc_radius_2 = 10**10
    z = complex(0, 0)
   
    # Stripe average coloring if parameters are given
    stripe = (stripe_s > 0) and (stripe_sig > 0)
    stripe_a =  0
    # z derivative
    dz = 1+0j
   
    # Mandelbrot iteration
    for n in range(maxiter):
        # derivative update
        dz = dz*2*z + 1
        # z update
        z = z*z + c
        if stripe:
            # Stripe Average Coloring
            # See: Jussi Harkonen On Smooth Fractal Coloring Techniques
            # cos instead of sin for symmetry
            # np.angle inavailable in CUDA
            # np.angle(z) = math.atan2(z.imag, z.real)
            stripe_t = (math.sin(stripe_s*math.atan2(z.imag, z.real)) + 1) / 2
       
        # If escape: save (smooth) iteration count
        # Equivalent to abs(z) > esc_radius
        if z.real*z.real + z.imag*z.imag > esc_radius_2:
           
            modz = abs(z)
            # Smooth iteration count: equals n when abs(z) = esc_radius
            log_ratio = 2*math.log(modz)/math.log(esc_radius_2)
            smooth_i = 1 - math.log(log_ratio)/math.log(2)

            if stripe:
                # Stripe average coloring
                # Smoothing + linear interpolation
                # spline interpolation does not improve
                stripe_a = (stripe_a * (1 + smooth_i * (stripe_sig-1)) +
                            stripe_t * smooth_i * (1 - stripe_sig))
                # Same as 2 following lines:
                #a2 = a * stripe_sig + stripe_t * (1-stripe_sig)
                #a = a * (1 - smooth_i) + a2 * smooth_i            
                # Init correction, init weight is now:
                # stripe_sig**n * (1 + smooth_i * (stripe_sig-1))
                # thus, a's weight is 1 - init_weight. We rescale
                stripe_a = stripe_a / (1 - stripe_sig**n *
                                       (1 + smooth_i * (stripe_sig-1)))

            # Normal vector for lighting
            u = z/dz
            normal = u # 3D vector (u.real, u.imag. 1)

            # Milton's distance estimator
            dem = modz * math.log(modz) / abs(dz) / 2

            # real smoothiter: n+smooth_i (1 > smooth_i > 0)
            # so smoothiter <= niter, in particular: smoothiter <= maxiter
            return (n+smooth_i, stripe_a, dem, normal)
       
        if stripe:
            stripe_a = stripe_a * stripe_sig + stripe_t * (1-stripe_sig)
           
    # Otherwise: set parameters to 0
    return (0,0,0,0)
           
@jit
def color_pixel(matxy, niter, stripe_a, step_s, dem, normal, colortable,
                ncycle, light):
    """ Colors given pixel, in-place
   
    Coloring is based on the smooth iteration count niter which cycles through
    the colortable (every ncycle). Then, shading is added using the stripe
    average coloring, distance estimate and normal for lambert shading.
   
    Args:
        matxy: ndarray(dtype=float, ndim=1)
            pixel to color, 3 values in [0,1]
        niter: float
            smooth iteration count
        stripe_a: float
            stripe average coloring value
        dem: float
            boundary distance estimate
        normal: complex
            normal
        colortable: ndarray(dtype=uint8, ndim=2)
            cyclic RGB colortable
        ncycle: float
            number of iteration before cycling the colortable
           

    Returns: (float, float, float, complex)
        - smooth iteration count at escape, 0 if maxiter is reached
        - stripe average coloring value, in [0,1]
        - dem: estimate of distance to the nearest point of the set
        - normal, used for shading
    """

    ncol = colortable.shape[0] - 1
    # Power post-transform and mapping to [0,1]
    niter = math.sqrt(niter) % ncycle / ncycle
    # Cycle through colortable
    col_i = round(niter * ncol)

    def overlay(x, y, gamma):
        """x, y  and gamma floats in [0,1]. Returns float in [0,1]"""
        if (2*y) < 1:
            out = 2*x*y
        else:
            out = 1 - 2 * (1 - x) * (1 - y)
        return out * gamma + x * (1-gamma)
    
    # brightness with Blinn Phong shading
    bright = blinn_phong(normal, light)
    
    # dem: log transform and sigmoid on [0,1] => [0,1]
    dem = -math.log(dem)/12
    dem = 1/(1+math.exp(-10*((2*dem-1)/2)))

    # Shaders: steps and/or stripes
    nshader = 0
    shader = 0
    # Stripe shading
    if stripe_a > 0:
        nshader += 1
        shader = shader + stripe_a
    # Step shading
    if step_s > 0:
        # Color update: constant color on each major step
        step_s = 1/step_s
        col_i = round((niter - niter % step_s)* ncol)
        # Major step: step_s frequency
        x = niter % step_s / step_s
        light_step = 6*(1-x**5-(1-x)**100)/10
        # Minor step: n for each major step
        step_s = step_s/8
        x = niter % step_s / step_s
        light_step2 = 6*(1-x**5-(1-x)**30)/10
        # Overlay merge between major and minor steps
        light_step = overlay(light_step2, light_step, 1)
        nshader += 1
        shader = shader + light_step
    # Applying shaders to brightness
    if nshader > 0:
        bright = overlay(bright, shader/nshader, 1) * (1-dem) + dem * bright
    # Set pixel color with brightness
    for i in range(3):
        # Pixel color
        matxy[i] = colortable[col_i,i]
        # Brightness with overlay mode
        matxy[i] = overlay(matxy[i], bright, 1)
        # Clipping to [0,1]
        matxy[i] = max(0,min(1, matxy[i]))
        
@jit
def compute_set(creal, cim, maxiter, colortable, ncycle, stripe_s, stripe_sig,
                step_s, diag, light):
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
        stripe_s:
            frequency parameter of stripe average coloring
        stripe_sig:
            memory parameter of stripe average coloring

    Returns:
        ndarray(dtype=uint8, ndim=3): image of the Mandelbrot set
    """
    xpixels = len(creal)
    ypixels = len(cim)

    # Output initialization
    mat = np.zeros((ypixels, xpixels, 3))

    # Looping through pixels
    for x in range(xpixels):
        for y in range(ypixels):
            # Initialization of c
            c = complex(creal[x], cim[y])
            # Get smooth iteration count
            niter, stripe_a, dem, normal = smooth_iter(c, maxiter, stripe_s,
                                                      stripe_sig)
            # If escaped: color the set
            if niter > 0:
                # dem normalization by diag
                color_pixel(mat[y,x,], niter, stripe_a, step_s, dem/diag,
                            normal, colortable,
                            ncycle, light)
    return mat

@cuda.jit
def compute_set_gpu(mat, xmin, xmax, ymin, ymax, maxiter, colortable, ncycle,
                    stripe_s, stripe_sig, step_s, diag, light):
    """ Compute and color the Mandelbrot set (GPU version)
   
    Uses a 1D-grid with blocks of 32 threads.
   
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
        stripe_s:
            frequency parameter of stripe average coloring
        stripe_sig:
            memory parameter of stripe average coloring

    Returns:
        mat: ndarray(dtype=uint8, ndim=3)
            shared data to write the output image of the set
    """
    # Retrieve x and y from CUDA grid coordinates
    index = cuda.grid(1)
    x, y = index % mat.shape[1], index // mat.shape[1]
    #ncol = colortable.shape[0] - 1
   
    # Check if x and y are not out of mat bounds
    if (y < mat.shape[0]) and (x < mat.shape[1]):
        # Mapping pixel to C
        creal = xmin + x / (mat.shape[1] - 1) * (xmax - xmin)
        cim = ymin + y / (mat.shape[0] - 1) * (ymax - ymin)
        # Initialization of c
        c = complex(creal, cim)
        # Get smooth iteration count
        niter, stripe_a, dem, normal = smooth_iter(c, maxiter, stripe_s,
                                                   stripe_sig)
        # If escaped: color the set
        if niter > 0:
            color_pixel(mat[y,x,], niter, stripe_a, step_s, dem/diag, normal,
                        colortable, ncycle, light)

class Mandelbrot():
    """Mandelbrot set object"""
    def __init__(self, xpixels=1280, maxiter=500,
                 coord=(-2.6, 1.845, -1.25, 1.25), gpu=True, ncycle=32,
                 rgb_thetas=(.0, .15, .25), oversampling=3, stripe_s=0,
                 stripe_sig=.9, step_s=0,
                 light = (45., 45., .75, .2, .5, .5, 20)):
        """Mandelbrot set object
   
        Args:
            xpixels: int
                image width (in pixels)
            maxiter: int
                maximal number of iterations
            coord: (float, float, float, float)
                coordinates of the frame in the complex space. Default to the
                main view of the Set, with a 16:9 ratio.
            gpu: boolean
                use CUDA on GPU to compute the set
            ncycle: float
                number of iteration before cycling the colortable
            rgb_thetas: (float, float, float)
                phase for each color channel
            oversampling: int
                for each pixel, a [n, n] grid is computed where n is the
                oversampling_size. Then, the average color of the n*n pixels
                is taken. Set to 1 for no oversampling.
            stripe_s:
                stripe density: frequency parameter of stripe average coloring.
                Set to 0 for no stripes.
            stripe_sig:
                memory parameter of stripe average coloring
            step_s:
                step density: frequency parameter of step coloring. Set to 0
                for no steps.
            light: (float, float, float)
                light vector: angle azimuth [0-360], angle elevation [0-90],
                opacity [0,1], k_ambiant, k_diffuse, k_spectral, shininess
           
        """
        self.explorer = None
        self.xpixels = xpixels
        self.maxiter = maxiter
        self.coord = coord
        self.gpu = gpu
        self.ncycle = ncycle
        self.os = oversampling
        self.rgb_thetas = rgb_thetas
        self.stripe_s = stripe_s
        self.stripe_sig = stripe_sig
        self.step_s = step_s
        # Light angles mapping
        self.light = np.array(light)
        self.light[0] = 2*math.pi*self.light[0]/360
        self.light[1] = math.pi/2*self.light[1]/90
        # Compute ypixels so the image is not stretched (1:1 ratio)
        self.ypixels = round(self.xpixels / (self.coord[1]-self.coord[0]) *
                             (self.coord[3]-self.coord[2]))
        # Initialization of colortable
        self.colortable = sin_colortable(self.rgb_thetas)
        # Compute the set
        self.update_set()

    def update_set(self):
        """Updates the set
   
        Compute and color the Mandelbrot set, using CPU or GPU
        """
        # Apply ower post-transform to ncycle
        ncycle = math.sqrt(self.ncycle)
        diag = math.sqrt((self.coord[1]-self.coord[0])**2 +
                  (self.coord[3]-self.coord[2])**2)
        # Oversampling: rescaling by os
        xp = self.xpixels*self.os
        yp = self.ypixels*self.os
       
        if self.gpu:
            # Pixel mapping is done in compute_self_gpu
            self.set = np.zeros((yp, xp, 3))
            # Compute set with GPU:
            # 1D grid, with n blocks of 32 threads
            npixels = xp * yp
            nthread = 32
            nblock = math.ceil(npixels / nthread)
            compute_set_gpu[nblock,
                            nthread](self.set, *self.coord, self.maxiter,
                                    self.colortable, ncycle, self.stripe_s,
                                    self.stripe_sig, self.step_s, diag,
                                    self.light)
        else:
            # Mapping pixels to C
            creal = np.linspace(self.coord[0], self.coord[1], xp)
            cim = np.linspace(self.coord[2], self.coord[3], yp)
            # Compute set with CPU
            self.set = compute_set(creal, cim, self.maxiter,
                                   self.colortable, ncycle, self.stripe_s,
                                   self.stripe_sig, self.step_s, diag,
                                   self.light)
        self.set = (255*self.set).astype(np.uint8)
        # Oversampling: reshaping to (ypixels, xpixels, 3)
        if self.os > 1:
            self.set = (self.set
                        .reshape((self.ypixels, self.os,
                                  self.xpixels, self.os, 3))
                        .mean(3).mean(1).astype(np.uint8))
   
    def draw(self, filename = None):
        """Draw or save, using PIL"""
        # Reverse x-axis (equivalent to matplotlib's origin='lower')
        img = Image.fromarray(self.set[::-1,:,:], 'RGB')
        if filename is not None:
            img.save(filename) # fast (save in jpg) (compare reading as well)
        else:
            img.show() # slow
           
    def draw_mpl(self, filename=None, dpi=72):
        """Draw or save, using Matplotlib"""
        plt.subplots(figsize=(self.xpixels/dpi, self.ypixels/dpi))
        plt.imshow(self.set, extent=self.coord, origin='lower')
        # Remove axis and margins
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.axis('off')
        # Write figure to file
        if filename is not None:
            plt.savefig(filename, dpi=dpi)
        else:
            plt.show()
       
    def zoom_at(self, x, y, s):
        """Zoom at (x,y): center at (x,y) and scale by s"""
        xrange = (self.coord[1] - self.coord[0])/2
        yrange = (self.coord[3] - self.coord[2])/2
        self.coord = [x - xrange * s,
                      x + xrange * s,
                      y - yrange * s,
                      y + yrange * s]
       
    def szoom_at(self, x, y, s):
        """Soft zoom (continuous) at (x,y): partial centering"""
        xrange = (self.coord[1] - self.coord[0])/2
        yrange = (self.coord[3] - self.coord[2])/2
        x = x * (1-s**2) + (self.coord[1] + self.coord[0])/2 * s**2
        y = y * (1-s**2) + (self.coord[3] + self.coord[2])/2 * s**2
        self.coord = [x - xrange * s,
                      x + xrange * s,
                      y - yrange * s,
                      y + yrange * s]      
       
    def animate(self, x, y, file_out, n_frames=150, loop=True):
        """Animated zoom to GIF file
   
        Note that the Mandelbrot object is modified by this function
       
        Args:
            x: float
                real part of point to zoom at
            y: float
                imaginary part of point to zoom at
            file_out: str
                filename to save the GIF output
            n_frames: int
                number of frames in the output file
            loop: boolean
                loop back to original coordinates
        """        
        # Zoom scale: gaussian shape, from 0% (s=1) to 30% (s=0.7)
        # => zoom scale (i.e. speed) is increasing, then decreasing
        def gaussian(n, sig = 1):
            x = np.linspace(-1, 1, n)
            return np.exp(-np.power(x, 2.) / (2 * np.power(sig, 2.)))
        s = 1 - gaussian(n_frames, 1/2)*.3
       
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
        if loop:
            images += images[::-2]
        # Make GIF
        imageio.mimsave(file_out, images)  
   
    def explore(self, dpi=72):
        """Run the Mandelbrot explorer: a Matplotlib GUI"""
        # It is important to keep track of the object in a variable, so the
        # slider and button are responsive
        self.explorer = MandelbrotExplorer(self, dpi)


class MandelbrotExplorer():
    """A Matplotlib GUI to explore the Mandelbrot set"""
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
        
        ## Sliders class matplotlib.widgets.Slider(ax, label, v, valmax, *, valinit=0.5, valfmt=None, closedmin=True, closedmax=True, slidermin=None, slidermax=None, dragging=True, valstep=None, orientation='horizontal', initcolor='r', track_color='lightgrey', handle_style=None, **kwargs)
        self.sld_maxit = Slider(ax=plt.axes([0.1, 0.005, 0.2, 0.02]), label='Iterations',
                                valmin=0, valmax=5000, valinit=mand.maxiter, valstep=5)
        self.sld_maxit.on_changed(self.update_val)
        self.sld_r = Slider(ax=plt.axes([0.1, 0.04, 0.2, 0.02]), label='R',
                           valmin=0, valmax=1, valinit=mand.rgb_thetas[0], valstep=.001)
        self.sld_r.on_changed(self.update_val)
        self.sld_g = Slider(ax=plt.axes([0.1, 0.06, 0.2, 0.02]), label='G',
                            valmin=0,valmax= 1, valinit=mand.rgb_thetas[1], valstep=.001)
        self.sld_g.on_changed(self.update_val)
        self.sld_b = Slider(ax=plt.axes([0.1, 0.08, 0.2, 0.02]), label='B',
                            valmin=0,valmax= 1, valinit=mand.rgb_thetas[2], valstep=.001)
        self.sld_b.on_changed(self.update_val)
        self.sld_n = Slider(ax=plt.axes([0.1, 0.10, 0.2, 0.02]), label='ncycle',
                            valmin=0,valmax= 200, valinit=mand.ncycle, valstep=1)
        self.sld_n.on_changed(self.update_val)
        self.sld_p = Slider(ax=plt.axes([0.1, 0.12, 0.2, 0.02]), label='phase',
                            valmin=0,valmax= 1, valinit=0, valstep=0.001)
        self.sld_p.on_changed(self.update_val)
        self.sld_st = Slider(ax=plt.axes([0.7, 0.19, 0.2, 0.02]), label='step_s',
                             valmin=0,valmax= 100, valinit=mand.step_s, valstep=1)
        self.sld_st.on_changed(self.update_val)
        self.sld_s = Slider(ax=plt.axes([0.7, 0.17, 0.2, 0.02]), label='stripe_s',
                            valmin=0,valmax= 32, valinit=mand.stripe_s, valstep=1)
        self.sld_s.on_changed(self.update_val)
        self.sld_li1 = Slider(ax=plt.axes([0.7, 0.14, 0.2, 0.02]), label='light_azimuth',
                              valmin=0,valmax= 360, valinit=360*mand.light[0]/(2*math.pi), valstep=1)
        self.sld_li1.on_changed(self.update_val)
        self.sld_li2 = Slider(ax=plt.axes([0.7, 0.12, 0.2, 0.02]), label='light_elevation',
                              valmin=0,valmax= 90,valinit=90*mand.light[1]/(math.pi/2), valstep=1)
        self.sld_li2.on_changed(self.update_val)
        self.sld_li3 = Slider(ax=plt.axes([0.7, 0.10, 0.2, 0.02]), label='light_i',
                              valmin=0,valmax= 1, valinit=mand.light[2], valstep=.01)
        self.sld_li3.on_changed(self.update_val)
        self.sld_li4 = Slider(ax=plt.axes([0.7, 0.08, 0.2, 0.02]), label='k_ambiant',
                              valmin=0,valmax= 1, valinit=mand.light[3], valstep=.01)
        self.sld_li4.on_changed(self.update_val)
        self.sld_li5 = Slider(ax=plt.axes([0.7, 0.06, 0.2, 0.02]), label='k_diffuse',
                              valmin=0,valmax= 1, valinit=mand.light[4], valstep=.01)
        self.sld_li5.on_changed(self.update_val)
        self.sld_li6 = Slider(ax=plt.axes([0.7, 0.04, 0.2, 0.02]), label='k_specular',
                              valmin=0,valmax= 1, valinit=mand.light[5], valstep=.01)
        self.sld_li6.on_changed(self.update_val)
        self.sld_li7 = Slider(ax=plt.axes([0.7, 0.02, 0.2, 0.02]), label='shininess',
                              valmin=1,valmax= 100, valinit=mand.light[6], valstep=1)
        self.sld_li7.on_changed(self.update_val)
        
        ## Zoom events
        plt.sca(self.ax)
        # Note that it is mandatory to keep track of those objects so they are
        # not deleted by Matplotlib, and callbacks can be used
        # Responsiveness for any click or scroll
        self.cid1 = self.fig.canvas.mpl_connect('scroll_event', self.onclick)
        self.cid2 = self.fig.canvas.mpl_connect('button_press_event',
                                                self.onclick)
        plt.show()
       
    def update_val(self, _):
        """Slider interactivity: update object values"""
        rgb = [x + self.sld_p.val for x in [self.sld_r.val, self.sld_g.val,
                                            self.sld_b.val]]
        self.mand.rgb_thetas = tuple(rgb)
        self.mand.colortable = sin_colortable(rgb)
        self.mand.maxiter = self.sld_maxit.val
        self.mand.ncycle = self.sld_n.val
        self.mand.stripe_s = self.sld_s.val
        self.mand.step_s = self.sld_st.val
        self.mand.light = (2*math.pi*self.sld_li1.val/360,
                           math.pi/2*self.sld_li2.val/90,
                           self.sld_li3.val,
                           self.sld_li4.val, self.sld_li5.val,
                           self.sld_li6.val, self.sld_li7.val)
        self.mand.update_set()
        self.graph.set_data(self.mand.set)
        plt.draw()      
        plt.show()
       
    def onclick(self, event):
        """Click & scroll interactivity: zoom in/out"""
        # This function is called by any click/scroll
        if event.inaxes == self.ax:
            # Click or scroll in the main axe: zoom event
            # Default: zoom in
            zoom = 1/4
            if event.button in ('down', 3):
                # If right click or scroll down: zoom out
                zoom = 1/zoom
            # Zoom and update
            self.mand.zoom_at(event.xdata, event.ydata, zoom)
            self.mand.update_set()
            # Updating the graph
            self.graph.set_data(self.mand.set)
            self.graph.set_extent(self.mand.coord)
            plt.draw()      
            plt.show()


if __name__ == "__main__":
    Mandelbrot().explore()
    