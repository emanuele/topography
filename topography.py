import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab

def topography(value, x, y, cmap=plt.cm.jet, nx=512, ny=512, plotsensors=True, vmin=None, vmax=None):
    """Simple plot of a topography given one value per channel and its
    position through a layout.

    Most of the code is taken from:
      http://stackoverflow.com/questions/3864899/resampling-irregularly-spaced-data-to-a-regular-grid-in-python
    with some minor additions.

    nx, ny = size of regular grid.
    """
    z = value

    xmin = x.min()
    ymin = y.min()
    xmax = x.max()
    ymax = y.max()

    # Generate a regular grid to interpolate the data.
    xi = np.linspace(xmin, xmax, nx)
    yi = np.linspace(ymin, ymax, ny)
    xi, yi = np.meshgrid(xi, yi)
    
    # Normalize coordinate system

    def normalize_x(data):
        data = data.astype(np.float)
        return (data - xmin) / (xmax - xmin)

    def normalize_y(data):
        data = data.astype(np.float)
        return (data - ymin) / (ymax - ymin)
    
    x_new, xi_new = normalize_x(x), normalize_x(xi)
    y_new, yi_new = normalize_y(y), normalize_y(yi)

    # Interpolate using delaunay triangulation
    zi = mlab.griddata(x_new, y_new, z, xi_new, yi_new)
    
    # Plot the results
    plt.pcolormesh(xi,yi,zi, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(shrink=0.75)
    if plotsensors:
        plt.scatter(x,y,c='w')

    # plt.axis([xmin*0.95, xmax*1.05, ymin*0.95, ymax*1.05])
    plt.axis('equal')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    from mne.layouts import read_layout
    layout = read_layout('Vectorview-mag.lout')
    x = layout.pos[:,0]
    y = layout.pos[:,1]
    # generate some values:
    # value = np.sin((layout.pos[:,:2]**2).sum(1)*10)
    value = np.random.rand(x.size)
    plt.figure()
    topography(value, x, y)
    # plt.show()
