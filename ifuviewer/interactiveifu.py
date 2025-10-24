import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, RadioButtons
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

class InteractiveIFUViewer:
    def __init__(self, maps: Dict[str, np.ndarray], datacube: np.ndarray, binmap: Optional[np.ndarray] = None, wavelength: Optional[np.ndarray] = None,
                 map_kwargs: Dict[str, dict] = None, plot_kwargs: dict = None):
        """
        This class manages the initialization of the data that will be displayed in the interactive plotter. The maps are the primary input
        which are input as a dictionary where each value is a 2D numpy array of measurements.

        Parameters:
        ----------
        maps : Dict[str, np.ndarray]
            A dictionary containing the 2D numpy arrays to be plotted as values.
        datacube : np.ndarray
            The flux datacube corresponding to `maps` with the spectral axis as the 0th. The spectral
            axis will be plotted upon interacting with a given spaxel.
        binmap : np.ndarray, Optional
            An optional 2D map of unique spatial bin identifiers.
        wavelength : np.ndarray, Optional
            An optional spectral wavelength to be plotted with `datacube`. If wavelength is a 3D array,
            the wavelength of the corresponding spaxel will be plotted.
        map_kwargs : Dict[str, dict], Optional
            An optional dictionary containing dictionaries as values to be passed into `matplotlib.pyplot.imshow`
            as `**kwargs`. The every key of `map_kwargs` must be identical to a key of `maps`. If `None`,
            default values will be used.
        plot_kwargs : dict, Optional
            An optional dictionary to be passed directly into `matplotlib.pyplot.plot` as `**kwargs` for the
            spectrum plot. If `None`, default values will be used.
        """
        # Main data
        self.maps = maps
        self.datacube = datacube
        self.binmap = binmap
        self.wavelength = wavelength
        self.map_kwargs = map_kwargs
        self.plot_kwargs = plot_kwargs

        # validate inputs
        self._validate()

        # data properties
        self.nwave, self.ny, self.nx = self.datacube.shape

        # current map tracking (starts with first)
        self.current_map_name = list(self.maps.keys())[0]
        self.current_map = self.maps[self.current_map_name]

        base_w, base_h = plt.rcParams['figure.figsize']
        # figure and axes
        self.fig = plt.figure(figsize=(base_w * 2, base_h))
        self._setup_layout()

        # plot artists
        self.map_image = None
        self.colorbar = None
        self.click_marker = None
        self.clicked_bin_contour = None
        self.hovered_bin_contour = None
        
        # setup plots
        self._setup_map_plot()
        self._setup_spectrum_plot()
        self._setup_controls()

        # connect events
        self._connect_events()

        # track click state
        self.last_clicked = None
        self.current_spectrum_artist = None

    def _validate(self):
        if self.datacube.ndim !=3:
            raise ValueError(f"datacube should be a 3D array")
        
        zdim, ydim, xdim = self.datacube.shape

        for key, value in self.maps.items():
            if value.shape != (ydim, xdim):
                raise ValueError(f"Array of maps['{key}'] does not match spatial dimension of datacube."
                                 f"Datacube has dimensions {self.datacube.shape} while maps['{key}'] has dimensions {value.shape}")
            
        if self.wavelength is not None:
            if self.wavelength.ndim !=3 and self.wavelength.ndim != 1:
                raise ValueError(f"wavelength must be either a 1D array of the spectral axis, or a 3D array containing the spectral axes of each spaxel")
            if self.wavelength.ndim == 1:
                if len(self.wavelength) != zdim:
                    raise ValueError(f"wavelength does not match the spectral dimensions of input datacube.")
            if self.wavelength.ndim == 3:
                if self.wavelength.shape != self.datacube.shape:
                    raise ValueError(f"wavelength datacube does not match the dimensions of flux datacube.")
                
        if self.binmap is not None:
            if self.binmap.shape != (ydim, xdim):
                raise ValueError(f"bin identification map does not match the spatial dimensions of the datacube")
            
        if self.map_kwargs is not None:
            for key in list(self.map_kwargs.keys()):
                if key not in list(self.maps.keys()):
                    raise ValueError(f"map_kwargs {key} not in maps dictionary.")

            
    def _setup_layout(self):
        """Initialize the main axes and adjust spacing"""
        self.ax_map = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
        self.ax_spectrum = plt.subplot2grid((2, 3), (0, 2), rowspan=2)

        plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, wspace=0.3, hspace=0.3)

    def _setup_map_plot(self):
        """Initialize the current map plot display"""
        self.map_image = self.ax_map.imshow(self.current_map, origin='lower', cmap='rainbow')

        self.ax_map.set_xlabel('Spaxel')
        self.ax_map.set_ylabel('Spaxel')

        self.colorbar = plt.colorbar(self.map_image, ax=self.ax_map)

    def _setup_spectrum_plot(self):
        """Initialize spectrum display"""
        self.ax_spectrum.set_xlabel('Wavelength' if self.wavelength is not None else 'Channel')
        self.ax_spectrum.set_ylabel('Flux')

    def _setup_controls(self):
        """Setup interactive controls"""
        # only set up radio buttons if more than one map
        if len(list(self.maps.values())) > 1:
            radio_ax = plt.axes([0.01, 0.75, 0.1, 0.2])
            self.radio = RadioButtons(radio_ax, list(self.maps.keys()))
            self.radio.on_clicked(self._change_map)

            # style radio buttons, attempt to handle different matplotlib version
            try:
                for circle in self.radio.circles:
                    circle.set_radius(0.1)
            except AttributeError:
                try:
                    circles = getattr(self.radio, '_circles', None)
                    if circles:
                        for circle in circles:
                            circle.set_radius(0.1)
                except (AttributeError, TypeError):
                    # If styling fails, continue without custom styling
                    pass

    def _change_map(self, label):
        """Change the displayed map when radio button is clicked"""
        self.current_map_name = label
        self.current_map = self.maps[label]

        self.map_image.set_data(self.current_map)
        self.map_image.set_clim(vmin=np.nanmin(self.current_map),
                                vmax=np.nanmax(self.current_map))
        self.ax_map.set_title(self.current_map_name)
        self.fig.canvas.draw_idle()
        
    def _connect_events(self):
        """Connect mouse events"""
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_hover)

    def _on_click(self, event):
        """Handles mouse click events"""
        if event.inaxes != self.ax_map:
            return
        
        # pixel coordinates
        x, y = int(np.round(event.xdata)), int(np.round(event.ydata))

        # check bounds
        if not (0 <= x < self.nx and 0 <= y < self.ny):
            return
        
        # update click marker
        if self.click_marker:
            self.click_marker.remove()
        
        self.click_marker = patches.Rectangle((x-0.5, y-0.5), 1, 1,
                                              linewidth = 1, edgecolor='green',
                                              facecolor = 'none')
        self.ax_map.add_patch(self.click_marker)

        # draw the contour around the clicked bin
        if self.binmap is not None:
            self._draw_bin_contours(x, y, 'clicked')

        # plot the spectrum
        self._plot_spectrum(x,y)

        # update click state
        self.last_clicked = (x, y)

        self.fig.canvas.draw()

    def _on_hover(self, event):
        """Handles mouse hovering events"""
        if event.inaxes != self.ax_map:
            return
        
        # pixel coords
        x, y = int(np.round(event.xdata)), int(np.round(event.ydata))

        # check bounds
        if not (0 <= x < self.nx and 0 <= y < self.ny):
            return
        
        map_value = self.current_map[y, x]
        title = f'{self.current_map_name}: {map_value:.3f} at ({x}, {y})'

        if self.binmap is not None:
            binid = self.binmap[y, x]
            title+= f' | Bin {binid}'

            self._draw_bin_contours(x, y, 'hovered')

        self.ax_map.set_title(title)
        self.fig.canvas.draw_idle()

    def _draw_bin_contours(self, x, y, contour_type):
        """Draw contour lines around current bin"""

        current_bin = self.binmap[y, x]

        # binary mask
        binmask = (self.binmap == current_bin).astype(float)

        # clear the current contours
        if contour_type == 'hovered':
            if self.hovered_bin_contour is not None:
                self.hovered_bin_contour.remove()
                self.hovered_bin_contour = None

            color = 'dimgray'
            linewidth = 1.5

        elif contour_type == 'clicked':
            if self.clicked_bin_contour is not None:
                self.clicked_bin_contour.remove()
                self.clicked_bin_contour = None
            color = 'green'
            linewidth = 2

        # draw the new contour at the edge of the bin
        contour = self.ax_map.contour(
            binmask,
            levels = [0.5],
            colors = color,
            linewidths = linewidth,
            alpha = 1
        )

        # store the contour
        if contour_type == 'clicked':
            self.clicked_bin_contour = contour
        else:
            self.hovered_bin_contour = contour

    def _plot_spectrum(self, x: int, y: int):
        """Plot the spectrum of spaxel y, x"""
        # clear the previous spectrum
        self.ax_spectrum.clear()

        # extract the 1D spectrum
        spectrum = self.datacube[:, y, x]

        if self.wavelength is not None:
            if self.wavelength.ndim == 1:
                wave = self.wavelength
            else:
                wave = self.wavelength[:, y, x]
            xlabel = 'Wavelength'
        else:
            wave = np.arange(self.nwave)
            xlabel = 'Channel'

        self.current_spectrum_artist = self.ax_spectrum.plot(wave, spectrum, 'k', drawstyle = 'steps-mid')
        self.ax_spectrum.set_xlabel(xlabel)
        self.ax_spectrum.set_ylabel('Flux')

    def show(self):
        """Display the interactive plot"""
        plt.show()

    