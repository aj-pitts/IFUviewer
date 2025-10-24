import numpy as np
from ifuviewer.interactiveifu import InteractiveIFUViewer
from util import defaults
import matplotlib.pyplot as plt
import os

# Example usage
if __name__ == "__main__":
    dirpath = defaults.get_default_path()
    plt.style.use(os.path.join(dirpath, 'configs/figures.mplstyle'))

    # Create sample data
    ny, nx = 50, 50
    nwave = 1000
    
    # Create sample maps
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)
    
    maps = {
        'Velocity': 100 * np.sin(2 * np.pi * X / nx) * np.cos(2 * np.pi * Y / ny),
        'Dispersion': 50 + 20 * np.sqrt((X - nx/2)**2 + (Y - ny/2)**2) / (nx/4),
        'Flux': 1000 * np.exp(-((X - nx/2)**2 + (Y - ny/2)**2) / (2 * (nx/6)**2))
    }
    
    # Create sample datacube with varying spectra
    datacube = np.zeros((nwave, ny, nx))
    wavelength = np.linspace(5000, 7000, nwave)
    
    # Create a sample binmap with Voronoi-like bins
    from scipy.spatial import Voronoi
    
    # Generate random bin centers
    n_bins = 100
    np.random.seed(42)
    bin_centers = np.random.rand(n_bins, 2) * np.array([nx, ny])
    
    # Assign each spaxel to nearest bin
    binmap = np.zeros((ny, nx), dtype=int)
    for i in range(ny):
        for j in range(nx):
            distances = np.sqrt((bin_centers[:, 0] - j)**2 + (bin_centers[:, 1] - i)**2)
            binmap[i, j] = np.argmin(distances)
    
    for i in range(ny):
        for j in range(nx):
            # Create a simple emission line spectrum
            center = 6000 + maps['Velocity'][i, j] / 3e5 * 6000  # Doppler shift
            width = maps['Dispersion'][i, j]
            amplitude = maps['Flux'][i, j]
            datacube[:, i, j] = amplitude * np.exp(-(wavelength - center)**2 / (2 * width**2))
            # Add continuum
            datacube[:, i, j] += 100
    
    # Create viewer
    viewer = InteractiveIFUViewer(maps, datacube, wavelength=wavelength, binmap=binmap)
    viewer.show()