import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, RadioButtons
import numpy as np
from astropy.io import fits
import argparse
from typing import Dict, List, Optional, Tuple, Union
from modules import util, file_handler
from astropy.io import fits
import os

class InteractiveIFUViewer:
    def __init__(self, fluxcube: np.ndarray, modelcube: np.ndarray, wavelength:np.ndarray, datadict: Dict[str, np.ndarray], binmap:np.ndarray, 
                 kwargs_dict: dict, redshiftmap: np.ndarray, datamask: Optional[np.ndarray] = None, MCMC_cube: Optional[np.ndarray] = None):
        
        self._validate_inputs(fluxcube, wavelength, datadict, binmap, redshiftmap, datamask, MCMC_cube)

        # unpack and store data
        HDUkeyword, datamap = next(iter(datadict.items()))

        self.fluxcube = fluxcube
        self.modelcube = modelcube
        self.wavelength = wavelength
        self.binmap = binmap
        self.datamap = datamap
        self.MCMC_cube = MCMC_cube

        self.redshiftmap = redshiftmap
        self.datamask = datamask

        # data properties
        self.nwave, self.ny, self.nx = self.fluxcube.shape
        self.HDUkeyword = HDUkeyword
        self.kwargs_dict = kwargs_dict

        # figure and axes
        self.fig = plt.figure()
        self._setup_layout()

        # plots
        self._setup_map_plot()
        self._setup_spectrum_plot()
        self._setup_controls()

        # connect events
        self._connect_events()

        # track click state
        self.last_clicked = None
        self.current_spectrum_artis = None

    def _validate_inputs(self, fluxcube, wavelength, datadict, binmap, redshiftmap, datamask, MCMC_cube):
        if fluxcube.ndim != 3:
            raise ValueError("Fluxcube must be 3D array with shape (nwave, ny, nx)")
        
        if wavelength.ndim != 1:
            raise ValueError("Wavelength must be 1D array")
        
        if len(wavelength) != fluxcube.shape[0]:
            raise ValueError("Wavelength length must match datacube spectral dimension")
        
        if not isinstance(datadict, dict):
            raise ValueError(f"Datadict must be a single key-value pair dictionary")
        
        HDUkeyword, datamap = next(iter(datadict.items()))

        if (fluxcube.shape[1], fluxcube.shape[2]) != datamap.shape:
            raise ValueError("Measurement map must match the spatial dimensions of the flux datacube")
        
        if (fluxcube.shape[1], fluxcube.shape[2]) != binmap.shape:
            raise ValueError("Bin ID map must match the spatial dimensions of the flux datacube")
        
        if (fluxcube.shape[1], fluxcube.shape[2]) != redshiftmap.shape:
            raise ValueError("Redshift map must match the spatial dimensions of the flux datacube")
            
        if datamask is not None:
            if datamask.shape != datamap.shape:
                raise ValueError(f"Data mask has shape {datamask.shape} while Map data has shape {datamap.shape}")
            
        if MCMC_cube is not None:
            if MCMC_cube.ndim != 3 or MCMC_cube.shape[0] != 4:
                raise ValueError("MCMC_cube bust be a 3D array with shape (4, ny, nx)")

            
    def _setup_layout(self):
        # create main axes
        self.ax_map = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
        self.ax_spectrum = plt.subplot2grid((2, 3), (0, 2), rowspan=2)

        # adjust spacing
        plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, 
                           wspace=0.3, hspace=0.3)
    
    def _setup_map_plot(self):
        # isolate map data
        plotdata = self.datamap
        
        # mask data if necessary
        if self.datamask is not None:
            plotdata[self.datamask.astype(bool)] = np.nan
        
        kwargs = self.kwargs_dict
        # setup the imshow
        self.im = self.ax_map.imshow(plotdata, origin='lower', aspect='equal',
                                     cmap = kwargs['cmap'], vmin=kwargs['vmin'], vmax=kwargs['vmax'])
        self.ax_map.set_facecolor('lightgray')

        self.cbar = plt.colorbar(self.im, ax=self.ax_map, shrink=0.95, aspect=20, pad=0.01)
        self.cbar.set_label(rf"{kwargs['v_str']}")

        # self.ax_map.set_xlabel(r'$\Delta \alpha$ (arcsec)')
        # self.ax_map.set_ylabel(r'$\Delta \delta$ (arcsec)')
        self.ax_map.set_xlabel('X Spaxel')
        self.ax_map.set_ylabel('Y Spaxel')

        self.ax_map.grid(True, alpha=0.3)

        # initialized click_marker
        self.click_marker = None

    def _setup_spectrum_plot(self):
        """Initialize the spectrum display."""
        self.ax_spectrum.set_xlabel(r'Wavelength $\mathrm{\AA}$')
        self.ax_spectrum.set_ylabel('Normalized Flux')
        self.ax_spectrum.set_title('Click on map to show spectrum')
        self.ax_spectrum.grid(True, alpha=0.3)
        
        # Add instruction text
        self.ax_spectrum.text(0.5, 0.5, 'Click on a pixel\nin the map to\nview spectrum',
                             ha='center', va='center', transform=self.ax_spectrum.transAxes,
                             fontsize=12, alpha=0.6)
        
    def _setup_controls(self):
        return
        """Set up interactive controls.""" 
        # Only create radio buttons if we have multiple maps
        if len(self.map_names) > 1:
            # Create radio buttons for map selection
            radio_ax = plt.axes([0.02, 0.7, 0.15, 0.2])
            self.radio = RadioButtons(radio_ax, self.map_names)
            self.radio.on_clicked(self._change_map)
            
            # Style radio buttons (handle different matplotlib versions)
            try:
                # Newer matplotlib versions
                for circle in self.radio.circles:
                    circle.set_radius(0.05)
            except AttributeError:
                # Older matplotlib versions or alternative approach
                try:
                    # Try alternative attribute names
                    circles = getattr(self.radio, '_circles', None)
                    if circles:
                        for circle in circles:
                            circle.set_radius(0.05)
                except (AttributeError, TypeError):
                    # If styling fails, just continue without custom styling
                    pass

    def _connect_events(self):
        """Connect mouse events."""
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_hover)
    
    def _on_click(self, event):
        """Handle mouse click events."""
        if event.inaxes != self.ax_map:
            return
            
        # Get pixel coordinates
        x, y = int(np.round(event.xdata)), int(np.round(event.ydata))
        
        # Check bounds
        if not (0 <= x < self.nx and 0 <= y < self.ny):
            return
            
        # Update click marker
        if self.click_marker:
            self.click_marker.remove()
            
        self.click_marker = patches.Rectangle((x-0.5, y-0.5), 1, 1,
                                            linewidth=2, edgecolor='red',
                                            facecolor='none')
        self.ax_map.add_patch(self.click_marker)
        
        # Plot spectrum
        self._plot_spectrum(x, y)
        
        # Store last clicked position
        self.last_clicked = (x, y)
        
        # Refresh display
        self.fig.canvas.draw()

    def _on_hover(self, event):
        if event.inaxes != self.ax_map:
            return
            
        x, y = int(np.round(event.xdata)), int(np.round(event.ydata))
        
        if not (0 <= x < self.nx and 0 <= y < self.ny):
            return
            
        bin_id = self.binmap[y, x]
        map_value = self.datamap[y, x]
        
        title = f'{self.HDUkeyword} | Bin ID: {bin_id:.0f} | Value: {map_value:.3f}'
        self.ax_map.set_title(title)
        self.fig.canvas.draw_idle()

    def _plot_spectrum(self, x: int, y: int):
        self.ax_spectrum.clear()
        bin_id = self.binmap[y, x]

        if bin_id == -1:
            self.ax_spectrum.text(0.5, 0.5, 'Bins with ID = -1\nare not included\nin the model',
                                ha='center', va='center', transform=self.ax_spectrum.transAxes,
                                fontsize=14, color='red', weight='bold')
            self.ax_spectrum.set_title(f"Bin {bin_id} - Not Included")
            self.ax_spectrum.set_xlabel(r'Wavelength $(\mathrm{\AA})$')
            self.ax_spectrum.set_ylabel('Normalized Flux')
            return  # Exit early, don't plot spectrum
        
        spectrum = self.fluxcube[:, y, x]
        model = self.modelcube[:, y, x]

        normflux = spectrum/model
        redshiftmap = self.redshiftmap
        wavelength = self.wavelength

        z = redshiftmap[y, x]
        restwave = wavelength / (1 + z)

        nad_lims = (5870, 5920)
        select = (restwave>=nad_lims[0]) & (restwave<=nad_lims[1])

        flux_nad = normflux[select]
        wave_nad = restwave[select]

        if self.MCMC_cube is not None:
            from modules import model_nai
            theta = self.MCMC_cube[:,y,x]
            mod_data = model_nai.model_NaI(theta, z, restwave)
            self.ax_spectrum.plot(mod_data['modwv'], mod_data['modflx'], 'b', linewidth=1.5)

        self.ax_spectrum.plot(wave_nad, flux_nad, 'k', drawstyle='steps-mid', linewidth=2)

        self.ax_spectrum.set_xlabel(r'Wavelength $(\mathrm{\AA})$')
        self.ax_spectrum.set_ylabel('Normalized Flux')

        self.ax_spectrum.set_title(f"Na D profile in Bin {bin_id}")

        self.ax_spectrum.set_xlim(nad_lims[0]+10, nad_lims[1]-10)
        self.ax_spectrum.set_ylim(0.75, 1.2)

        self.ax_spectrum.grid(True, alpha = 0.3)

        # Force square aspect ratio based on data ranges
        x_range = nad_lims[1] - nad_lims[0]  # 50 Å
        y_range = 1.3 - 0.75                 # 0.6 flux units
        aspect_ratio = x_range / y_range     # ~83
        self.ax_spectrum.set_aspect(aspect_ratio)

    def show(self):
        """Display the interactive plot."""
        plt.show()

def get_imshow_kwargs(hdu_keyword):
    main_dict = {
        'EW_NOEM':dict(cmap = 'rainbow', vmin=-0.2, vmax=2.5, v_str = r'$\mathrm{EW_{Na\ D}}\ \left( \mathrm{\AA} \right)$'),
        'EW_NAI':dict(cmap = 'rainbow', vmin=-0.2, vmax=2, v_str = r'$\mathrm{EW_{Na\ D}}\ \left( \mathrm{\AA} \right)$'),
        'V_NAI':dict(cmap = 'seismic', vmin = -250, vmax = 250, v_str = r'$v_{\mathrm{cen}}\ \left( \mathrm{km\ s^{-1}} \right)$'),
        'SFRSD':dict(cmap = 'rainbow', vmin=-2.5, vmax=0, v_str = r'$\mathrm{log\ \Sigma_{SFR}}\ \left( \mathrm{M_{\odot}\ kpc^{-2}\ yr^{-1}\ spaxel^{-1}} \right)$'),
    }
    return main_dict[hdu_keyword]

def main_viewer(galname, bin_method, HDU_key):
    print(f"Launching interactive viewer for {galname} {bin_method} {HDU_key}")
    print("\nInstructions:")
    print("- Click on any pixel in the map to view its spectrum")
    print("- Use radio buttons to switch between different measurement maps")
    print("- Hover over pixels to see bin IDs and values")
    print("- Common emission lines are marked in red on spectra")
    datapaths = file_handler.init_datapaths(galname, bin_method)
    local = fits.open(datapaths['LOCAL'])
    cube = fits.open(datapaths['LOGCUBE'])

    flux = cube['FLUX'].data
    model = cube['MODEL'].data 
    binmap = cube['BINID'].data[0]
    wavelength = cube['WAVE'].data 

    redshifts = local['REDSHIFT'].data 
    datamap = local[HDU_key].data 
    datamask = local[f"{HDU_key}_mask"].data

    datadict = {HDU_key:datamap}
    kwargs_dict = get_imshow_kwargs(HDU_key)

    viewer = InteractiveIFUViewer(flux, model, wavelength, datadict,
                                  binmap, kwargs_dict, redshifts, datamask)
    
    viewer.show()
    return viewer

def arguments():
    parser = argparse.ArgumentParser(
    description="An interactive plotter for easy inspection of IFU data/measurements"
    )

    parser.add_argument("galname", type=str, nargs="?", help="Input galaxy name (default: NGC4030)", default = "NGC4030")
    parser.add_argument("bin_method", type=str, nargs="?", help="Input DAP spatial binning method (default: SQUARE0.6)", default = "SQUARE0.6")
    parser.add_argument(
        "HDU_keyword",
        type=str,
        nargs="?",
        help="Input HDU keyword of the MAP to be viewed from local_maps.fits (default: EW_NOEM)",
        default="EW_NOEM",
    )

    return parser.parse_args()



if __name__ == "__main__":
    args = arguments()
    print(f"Setting up viewer for {args.galname} {args.bin_method} {args.HDU_keyword}")
    plt.style.use(util.defaults.matplotlib_rc())


    viewer = main_viewer(args.galname, args.bin_method, args.HDU_keyword)