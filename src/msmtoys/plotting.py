from __future__ import division
import numpy as np
from matplotlib import pyplot as pp



def _get_grid(potential, bounds, resolution):

    if bounds is None:
        bounds = potential.bounds

    minx = bounds[0]
    maxx = bounds[1]
    miny = bounds[2]
    maxy = bounds[3]
    grid_width = max(maxx - minx, maxy - miny) / resolution

    grid = np.mgrid[minx : maxx : grid_width, miny : maxy : grid_width]
    return grid, grid_width

def _get_xys(potential, bounds, resolution):
    # FIXME
    grid = _get_grid(potential, bounds, resolution)
    gshape = grid.shape
    grid = np.reshape(grid, (gshape[1] * gshape[2], 2))
    return grid

def plot2d(potential, ax=None, bounds=None, resolution=200, ** kwargs):
    """Plot a countour plot of a 2d potential.

    potential - the class with a .potential() method
    ax - matplotlib plotting scheme
    bounds - a list of the form [xmin, xmax, ymin, ymax] for the limits
             of plotting
    **kwargs - arguments to pass to the plotting command.
    """

    grid, _ = _get_grid(potential, bounds, resolution)

    ax = kwargs.pop('ax', None)
    potential = potential.potential(grid[0], grid[1])

    # clip off any values greater than 200, since they mess up
    # the color scheme
    if ax is None:
        ax = pp

    ax.contourf(grid[0], grid[1], potential.clip(max=200), 40, **kwargs)


