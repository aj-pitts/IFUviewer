import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, RadioButtons
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import inspect
import os

class InteractiveIFUViewer:
    def __init__(self, maps: Dict[str, np.ndarray], datacube: np.ndarray, 
                 maps_masks: Optional[Dict[str, Union[float, np.ndarray]]] = None, modelcube: Optional[np.ndarray] = None, 
                 continuumcube: Optional[np.ndarray] = None, binmap: Optional[np.ndarray] = None, wavelength: Optional[np.ndarray] = None,
                 map_kwargs: Optional[Dict[str, dict]] = None, map_titles: Optional[dict[str, str]] = None,
                 spec_kwargs: Optional[dict] = None, specmod_kwargs: Optional[dict] = None):
        """
        This class manages the interactive plotting for IFU data.

        Parameters:
        ----------
        maps : dict of {str: np.ndarray}
            A dictionary containing the 2D numpy arrays of measurements to be plotted as values using
            `matplotlib.pyplot.imshow`.
        datacube : np.ndarray
            The flux datacube of the IFU data, spatially corresponding to `maps`, with the spectral axis 
            as the 0th. The 1D spectrum of (y, x) will be plotted upon interacting with the spaxel at (y, x).
        maps_masks : dict of {str: float or np.ndarray}, Optional
            A dictionary containing either 2D numpy arrays of boolean values or integers, or single values 
            to be used to mask out values of `maps` when plotting. If the values of `map_masks` are floats,
            values of the corresponding `map` equal to the float will be masked. If `None`, no masking will
            occur. Every key of `map_masks` must correspond to one key of `maps`. A mask value with the key
            'all' may be included in `maps_masks` to be applied to every map in addition to map-specific masks.
            The 'all' mask must be a 2D array.
        modelcube : np.ndarray, Optional
            An additional datacube containing flux data corresponding to each 1D spectrum of `datacube`. The
            spectra of `modelcube` will be plotted with those of `datacube`, or `datacube / continuumcube` if 
            `continuumcube` is not `None`. *Note:* the values of units of this data should be normalized flux 
            if `continuumcube` is not `None`; the input of `modelcube` is not normalized by `continuumcube`.
        continuumcube : np.ndarray, Optional
            An additional flux datacube to be used to normalize each 1D spectrum of `datacube` when plotting,
            e.g., the flux continuum. The shape of `continuumcube` must be identical to `datacube`. If `None`,
            no normalization will occur.
        binmap : np.ndarray, Optional
            An optional 2D map of unique spatial bin identifiers.
        wavelength : np.ndarray, Optional
            An optional spectral wavelength to be plotted with `datacube`. If wavelength is a 3D array,
            the wavelength of the corresponding spaxel will be plotted.
        map_kwargs : dict of {str: dict}, Optional
            An optional dictionary containing dictionaries as values to be passed into `matplotlib.pyplot.imshow`
            as `**kwargs`. The every key of `map_kwargs` must be identical to a key of `maps`. If `None`,
            default values will be used.
        map_titles : dict of {str: str}, Optional
            An optional dictionary containing strings as values to be used as titles/labels for each map. Every key
            of `map_titles` must correspond to a key of `maps`. If `None`, or for any `maps` value without a 
            corresponding `map_titles` entry will be labeled by the key(s) of `maps`.
        spec_kwargs : dict, Optional
            An optional dictionary to be passed directly into `matplotlib.pyplot.plot` as `**kwargs` for each
            1D spectrum plot of `datacube` or `datacube / continuumcube`. If `None`, default values will be used.
        specmod_kwargs : dict, Optional
            An optional dictionary to be passed directly into `matplotlib.pyplot.plot` as `**kwargs` for each
            1D model spectrum plot of `modelcube`. If `None`, default values will be used.
        """
        rootpath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        configpath = os.path.join(rootpath, 'configs')
        plt.style.use(os.path.join(configpath, 'figures.mplstyle'))
        # validate inputs 
        self._validate(maps, datacube, maps_masks, modelcube, continuumcube, binmap, wavelength, map_kwargs, map_titles, spec_kwargs, specmod_kwargs)
        
        # Main data
        self.maps = maps
        self.maps_masks = maps_masks if maps_masks is not None else {}
        self.datacube = datacube
        self.modelcube = modelcube
        self.continuumcube = continuumcube
        self.binmap = binmap
        self.wavelength = wavelength

        # plotting kwargs and additionals
        self.map_titles = map_titles if map_titles is not None else {}
        self.map_kwargs = map_kwargs if map_kwargs is not None else {}
        self.spec_kwargs = spec_kwargs if spec_kwargs is not None else {}
        self.specmod_kwargs = specmod_kwargs if specmod_kwargs is not None else {}

        # data properties
        self.nwave, self.ny, self.nx = self.datacube.shape

        # current map tracking (starts with first)
        self.current_map_name = list(self.maps.keys())[0]
        self.current_map = self.maps[self.current_map_name]
        self.current_mask = self._get_mask(self.current_map_name)
        self.current_map_title = self.map_titles.get(self.current_map_name, self.current_map_name)

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

        # track click state and current spectra
        self.last_clicked = None
        self.current_spectrum_artists = []

    def _validate(self, maps, datacube, maps_masks, modelcube, continuumcube, binmap, wavelength, map_kwargs, map_titles, spec_kwargs, specmod_kwargs):
        # validate datacube
        if not isinstance(datacube, np.ndarray) or datacube.ndim !=3:
            raise ValueError(f"datacube should be a 3D array")
        
        # validate maps
        zdim, ydim, xdim = datacube.shape
        for key, value in maps.items():
            if not isinstance(value, np.ndarray):
                raise ValueError(f"Value of `maps['{key}']` is not of type np.ndarray.")
            if value.shape != (ydim, xdim):
                raise ValueError(f"Array of `maps['{key}']` does not match spatial dimension of input datacube. "
                                 f"datacube has dimensions {datacube.shape} while `maps['{key}']` has dimensions {value.shape}")
        
        # validate maps_masks
        if maps_masks is not None:
            for key, value in maps_masks.items():
                if key not in list(maps.keys()):
                    raise ValueError(f"Key: {key} of `maps_masks` does not correspond to one of `maps`.")
                if isinstance(value, np.ndarray):
                    if value.shape != (ydim, xdim):
                        raise ValueError(f"Array of `maps_masks['{key}']` does not match spatial dimensions of input datacube. Datacube has dimensions {datacube.shape} while `maps_masks['{key}']` has dimensions {value.shape}")
                elif not isinstance(value, float):
                    raise ValueError(f"Value of `maps_masks['{key}']` is not of type float or np.ndarray.")
                
        # validate modelcube
        if modelcube is not None:
            if not isinstance(modelcube, np.ndarray) or modelcube.ndim != 3:
                raise ValueError(f"datacube should be a 3D array")
            if modelcube.shape != datacube.shape:
                raise ValueError(f"Dimensions of modelcube and datacube are not identical")
            
        # validate continnumcube
        if continuumcube is not None:
            if not isinstance(continuumcube, np.ndarray) or continuumcube.ndim != 3:
                raise ValueError(f"datacube should be a 3D array")
            if continuumcube.shape != datacube.shape:
                raise ValueError(f"Dimensions of continuumcube and datacube are not identical")

        # validate binmap
        if binmap is not None:
            if not isinstance(binmap, np.ndarray) or binmap.ndim != 2:
                raise ValueError("binmap must be a 2D array")
            if binmap.shape != (ydim, xdim):
                raise ValueError(f"binmap map does not match the spatial dimensions of the datacube")
            
        # validate wavelength
        if wavelength is not None:
            if wavelength.ndim !=3 and wavelength.ndim != 1:
                raise ValueError(f"wavelength must be either a 1D array of the spectral axis, or a 3D array containing the spectral axes of each spaxel")
            if wavelength.ndim == 1:
                if len(wavelength) != zdim:
                    raise ValueError(f"wavelength does not match the spectral dimensions of input datacube.")
            if wavelength.ndim == 3:
                if wavelength.shape != datacube.shape:
                    raise ValueError(f"wavelength datacube does not match the dimensions of flux datacube.")

        # validate map_kwargs 
        if map_kwargs is not None:
            for key, value in map_kwargs.items():
                if key not in list(maps.keys()):
                    raise ValueError(f"map_kwargs {key} not in maps dictionary.")
                if not isinstance(value, dict):
                    raise ValueError(f"maps_kwargs['{key}'] is not of type dict")
                valid_imshow_params = inspect.signature(plt.imshow).parameters
                invalid_imshow_keys = [k for k in value if k not in valid_imshow_params]
                if invalid_imshow_keys:
                    raise ValueError(f"Invalid imshow kwargs in maps_kwargs['{key}']: {invalid_imshow_keys}")
                
        # validate map_titles 
        if map_titles is not None:
            for key, value in map_titles.items():
                if key not in list(maps.keys()):
                    raise ValueError(f"map_titles {key} not in maps dictionary.")
                if not isinstance(value, str):
                    raise ValueError(f"map_titles['{key}'] is not of type str")
                
        # validate spec_kwargs
        if spec_kwargs is not None:
            if not isinstance(spec_kwargs, dict):
                raise ValueError(f"spec_kwargs is not of type dict")
            valid_plot_params = inspect.signature(plt.plot).parameters
            invalid_plot_keys = [k for k in spec_kwargs if k not in valid_plot_params]
            if invalid_plot_keys:
                raise ValueError(f"Invalid matplotlib.pyplot.plot kwargs in spec_kwargs: {invalid_plot_keys}")
        
        # validate specmod_kwargs
        if specmod_kwargs is not None:
            if not isinstance(specmod_kwargs, dict):
                raise ValueError(f"spec_kwargs is not of type dict")
            valid_plot_params = inspect.signature(plt.plot).parameters
            invalid_plot_keys = [k for k in specmod_kwargs if k not in valid_plot_params]
            if invalid_plot_keys:
                raise ValueError(f"Invalid matplotlib.pyplot.plot kwargs in specmod_kwargs: {invalid_plot_keys}")
            
    def _setup_layout(self):
        """Initialize the main axes and adjust spacing"""
        self.ax_map = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
        self.ax_spectrum = plt.subplot2grid((2, 3), (0, 2), rowspan=2)

        plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, wspace=0.3, hspace=0.3)

    def _setup_map_plot(self):
        """Initialize the current map plot display"""

        mapdata = np.copy(self.current_map)
        mapdata[self.current_mask] = np.nan

        kwargs = self.map_kwargs.get(self.current_map_name, {})

        if 'origin' not in kwargs:
            kwargs['origin'] = 'lower'
        if 'interpolation' not in kwargs:
            kwargs['interpolation'] = 'nearest'
        if 'cmap' not in kwargs:
            kwargs['cmap'] = 'viridis'
        if 'vmin' not in kwargs or kwargs['vmin'] == None:
            kwargs['vmin'] = np.nanmin(mapdata)
        if 'vmax' not in kwargs or kwargs['vmax'] == None:
            kwargs['vmax'] = np.nanmax(mapdata)

        self.map_image = self.ax_map.imshow(mapdata, **kwargs)

        self.ax_map.set_xlabel('Spaxel')
        self.ax_map.set_ylabel('Spaxel')
        self.ax_map.set_facecolor('lightgray')
        # else:
        #     self.ax_map.set_xlabel(r'$\Delta \alpha$ (arcsec)')
        #     self.ax_map.set_ylabel(r'$\Delta \delta$ (arcsec)')

        # colorbar
        divider = make_axes_locatable(self.ax_map)
        cax = divider.append_axes("top", size="5%", pad=0.01)
        self.colorbar = plt.colorbar(self.map_image, ax=self.ax_map, cax=cax, orientation = 'horizontal')
        cax.xaxis.set_ticks_position('top')
        cax.xaxis.set_label_position('top')

    def _setup_spectrum_plot(self):
        """Initialize spectrum display"""
        self.ax_spectrum.set_xlabel('Wavelength' if self.wavelength is not None else 'Channel')
        self.ax_spectrum.set_ylabel('Flux')
        self.ax_spectrum.set_box_aspect(1)

    def _setup_controls(self):
        """Setup interactive controls"""
        # only set up radio buttons if more than one map
        if len(list(self.maps.values())) > 1:
            radio_ax = plt.axes([0.01, 0.15, 0.1, 0.8])
            #keylist = [self.map_titles.get(key, key) for key in self.maps.keys()]
            self.radio = RadioButtons(radio_ax, list(self.maps.keys()))
            self.radio.on_clicked(self._change_map)

            # style radio buttons, attempt to handle different matplotlib version
            try:
                for circle in self.radio.circles:
                    circle.set_radius(1)
            except AttributeError:
                try:
                    circles = getattr(self.radio, '_circles', None)
                    if circles:
                        for circle in circles:
                            circle.set_radius(1)
                except (AttributeError, TypeError):
                    # If styling fails, continue without custom styling
                    pass

    def _change_map(self, label):
        """Change the displayed map when radio button is clicked"""
        self.current_map_name = label
        self.current_map_title = self.map_titles.get(label, label)
        self.current_map = self.maps[label]
        self.current_mask = self._get_mask(label)

        # get the kwargs for this map
        kwargs = self.map_kwargs.get(label, {})

        # mask the map data
        mapdata = np.copy(self.current_map)
        mapdata[self.current_mask] = np.nan
        
        # update the data
        self.map_image.set_data(mapdata)

        # handle default kwargs
        if 'cmap' not in kwargs:
            kwargs['cmap'] = 'viridis'
        if 'vmin' not in kwargs or kwargs['vmin'] == None:
            kwargs['vmin'] = np.nanmin(mapdata)
        if 'vmax' not in kwargs or kwargs['vmax'] == None:
            kwargs['vmax'] = np.nanmax(mapdata)

        # update cmap and vmin vmax
        self.map_image.set_clim(vmin = kwargs['vmin'], vmax = kwargs['vmax'])
        self.map_image.set_cmap(kwargs['cmap'])
        
        # remove colorbar and reassign
        self.colorbar.remove()
        divider = make_axes_locatable(self.ax_map)
        cax = divider.append_axes("top", size="5%", pad=0.01)
        self.colorbar = plt.colorbar(self.map_image, ax=self.ax_map, cax=cax, orientation = 'horizontal')
        self.colorbar.mappable.set_clim(kwargs['vmin'], kwargs['vmax'])
        self.colorbar.update_ticks()
        self.colorbar._draw_all()
        cax.xaxis.set_ticks_position('top')
        cax.xaxis.set_label_position('top')

        # title
        self.ax_map.set_title(self.current_map_title, pad=50)

        # draw new artists
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
        masked_flag = '' if not self.current_mask[y, x] else '[MASKED]'
        title = f'{self.current_map_title} = {map_value:.3g} {masked_flag} | ({x}, {y})'

        if self.binmap is not None:
            binid = self.binmap[y, x]
            title+= f' | Bin {binid}'

            self._draw_bin_contours(x, y, 'hovered')

        self.ax_map.set_title(title, pad=50)
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

    def _plot_spectrum(self, x: int, y: int, clear_previous = True):
        """Plot the spectrum of spaxel y, x; optionally, replace the current plot with clear_previous"""

        # clear the previous spectrum
        if clear_previous:
            self.ax_spectrum.clear()
            self.current_spectrum_artists = []

        # extract the 1D spectrum
        spectrum = self.datacube[:, y, x]

        # set the ylabel here
        ylabel = 'Flux'

        # extract the 1D continuum if not None and normalize spectrum
        if self.continuumcube is not None:
            continuum = self.continuumcube[:, y, x]
            spectrum /= continuum
            ylabel = 'Normalized Flux'

        if self.wavelength is not None:
            if self.wavelength.ndim == 1:
                wave = self.wavelength
            else:
                wave = self.wavelength[:, y, x]
            xlabel = r'Wavelength $\left( \mathrm{\AA} \right)$'
        else:
            wave = np.arange(self.nwave)
            xlabel = 'Channel'

        # plot the spectrum
        specline = self.ax_spectrum.plot(wave, spectrum, 'k', drawstyle = 'steps-mid')
        self.current_spectrum_artists.append(specline)

        # extract the 1D model if model is not None and plot
        if self.modelcube is not None:
            specmod = self.modelcube[:, y, x]
            modline = self.ax_spectrum.plot(wave, specmod, '#0063ff', drawstyle = 'steps-mid')
            self.current_spectrum_artists.append(modline)

        
        if clear_previous:
            self.ax_spectrum.set_xlabel(xlabel)
            self.ax_spectrum.set_ylabel(ylabel)
            self.ax_spectrum.set_box_aspect(1)

    def _get_mask(self, label):
        all_mask = self.maps_masks.get('all', None)
        value = self.maps_masks.get(label, np.zeros((self.ny, self.nx))) 
        mask = value.astype(bool) if not isinstance(value, (float, int)) else (self.current_map == value)
        if all_mask is not None:
            mask += all_mask
        return mask
        
        

    def show(self):
        """Display the interactive plot"""
        plt.show()

    