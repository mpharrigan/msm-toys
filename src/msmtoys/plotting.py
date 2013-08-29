"""Functions for visualizing things."""

from __future__ import division
from matplotlib import pyplot as pp
import numpy as np


def get_grid(bounds, resolution):
    """Grid up a space

    bounds - the limits of the grid
    resolution - how fine the grid should be
    """

    assert len(bounds) == 4, 'Bounds must be len 4 array'
    minx = bounds[0]
    maxx = bounds[1]
    miny = bounds[2]
    maxy = bounds[3]
    grid_width = max(maxx - minx, maxy - miny) / resolution

    grid = np.mgrid[minx : maxx : grid_width, miny : maxy : grid_width]
    return grid

def plot2d(potential, ax=None, bounds=None, resolution=200, ** kwargs):
    """Plot a countour plot of a 2d potential.

    potential - the class with a .potential() method
    ax - matplotlib plotting scheme
    bounds - a list of the form [xmin, xmax, ymin, ymax] for the limits
             of plotting
    **kwargs - arguments to pass to the plotting command.
    """

    if bounds is None:
        bounds = potential.bounds
    grid = get_grid(bounds, resolution)

    ax = kwargs.pop('ax', None)
    potential = potential.potential(grid[0], grid[1])

    # clip off any values greater than 200, since they mess up
    # the color scheme
    if ax is None:
        ax = pp

    ax.contourf(grid[0], grid[1], potential.clip(max=200), 40, **kwargs)

def plot2d_vector(vec, translate_func, scale=1.0):
    """Plot the centroids of a particular clustering scheme.

    If marker_sizes is given, it will use this array as the sizes for the
    various centroids. This is useful for visualizing eigenvectors.

    translate_func(i) should take state i to coordinate (x,y)
    """

    for i in xrange(len(vec)):
        marker_size = np.abs(vec[i])
        if vec[i] < 0:
            color_string = 'ro'
        else:
            color_string = 'wo'

        x, y = translate_func(i)
        pp.plot(x, y, color_string, markersize=12 * scale * marker_size,
                zorder=10)


