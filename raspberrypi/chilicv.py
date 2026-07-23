# Ref code: https://github.com/danforthcenter/plantcv
from skimage.morphology import remove_small_objects
import math
import matplotlib
import numpy as np
from math import sqrt
import sys, cv2, os
from matplotlib import pyplot as plt
# from altair.vegalite.v5.api import FacetChart, LayerChart, Chart
# from xarray import DataArray

"""ChiliCV: A simple computer vision program for processing chili pepper images."""

"""READ IMAGE"""
def readimage(filename):
    """Read an image from file and convert to RGB color space.

    Inputs:
    filename = the path to the image file

    Returns:
    img = the image in RGB color space
    """
    img = cv2.imread(filename, -1)
    if img is None:
        print("Error: could not read image from " + filename)
        sys.exit()
    else:
        return img

# Convert RBG to Grey using Blue-Yellow Channel 

class Params:
    """ChiliCV parameters class."""

    def __init__(self, device=0, debug=None, debug_outdir=".", line_thickness=5,
                 line_color=(255, 0, 255), dpi=100, text_size=0.55,
                 text_thickness=2, marker_size=60, color_scale="gist_rainbow", color_sequence="sequential",
                 sample_label="default", saved_color_scale=None, verbose=True, unit="pixels", px_height=1, px_width=1):
        """Initialize parameters.

        Keyword arguments/parameters:
        device            = Device number. Used to count steps in the pipeline. (default: 0)
        debug             = None, print, or plot. Print = save to file, Plot = print to screen. (default: None)
        debug_outdir      = Debug images output directory. (default: .)
        line_thickness    = Width of line drawings. (default: 5)
        line_color        = Color of line annotations (default = (255, 0, 255))
        dpi               = Figure plotting resolution, dots per inch. (default: 100)
        text_size         = Size of plotting text. (default: 0.55)
        text_thickness    = Thickness of plotting text. (default: 2)
        marker_size       = Size of plotting markers (default: 60)
        color_scale       = Name of plotting color scale (matplotlib colormap). (default: gist_rainbow)
        color_sequence    = Build color scales in "sequential" or "random" order. (default: sequential)
        sample_label      = Sample name prefix. Used in analyze functions. (default: "default")
        saved_color_scale = Saved color scale that will be applied next time color_palette is called. (default: None)
        verbose           = Whether or not in verbose mode. (default: True)
        unit              = Units of size trait outputs. (default: "pixels")
        px_height         = Size scaling information about pixel height (default: 1)
        px_width          = Size scaling information about pixel width (default: 1)


        :param device: int
        :param debug: str
        :param debug_outdir: str
        :param line_thickness: numeric
        :param dpi: int
        :param text_size: float
        :param text_thickness: int
        :param marker_size: int
        :param color_scale: str
        :param color_sequence: str
        :param sample_label: str
        :param saved_color_scale: list
        :param verbose: bool
        :param unit: str
        :param px_height: float
        :param px_width: float

        """
        self.device = device
        self.debug = debug
        self.debug_outdir = debug_outdir
        self.line_thickness = line_thickness
        self.line_color = line_color
        self.dpi = dpi
        self.text_size = text_size
        self.text_thickness = text_thickness
        self.marker_size = marker_size
        self.color_scale = color_scale
        self.color_sequence = color_sequence
        self.sample_label = sample_label
        self.saved_color_scale = saved_color_scale
        self.verbose = verbose
        self.unit = unit
        self.px_height = px_height
        self.px_width = px_width

params = Params()

def fatal_error(error):
    """Print out the error message that gets passed, then quit the program.

    Inputs:
    error = error message text

    :param error: str
    :return:
    """
    raise RuntimeError(error)
def _rgb2lab(rgb_img, channel):
    """Convert image from RGB colorspace to LAB colorspace. Returns the specified subchannel as a gray image.

    Parameters
    ----------
    rgb_img : numpy.ndarray
        RGB image data
    channel : str
        color subchannel (l = lightness, a = green-magenta, b = blue-yellow)

    Returns
    -------
    numpy.ndarray
        grayscale image from one LAB color channel
    """
    # The allowable channel inputs are l, a or b
    channel = channel.lower()
    if channel not in ["l", "a", "b"]:
        fatal_error("Channel " + str(channel) + " is not l, a or b!")

    # Convert the input BGR image to LAB colorspace
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2LAB)
    # Split LAB channels
    l, a, b = cv2.split(lab)
    # Create a channel dictionaries for lookups by a channel name index
    channels = {"l": l, "a": a, "b": b}
    return channels[channel]
def _show_dataarray(img, **kwargs):
    """
    Plot facetted images from xarray dataarray

    Inputs:
    img     - dataarray to display
    kwargs  - key-value arguments to xarray.plot method

    :param img: xr.core.dataarray.DataArray
    :param kwargs: dict of arguments recognized by xarray.plot.plot
    """
    # check for kwargs col and row. col and row are removed from kwargs!
    # the default for col and row are None
    col = kwargs.pop('col', None)
    row = kwargs.pop('row', None)

    col_or_row_given = col is not None or row is not None
    if not col_or_row_given:
        fatal_error('You need to specify `col` or `row` with which to facet the xarray images. '
                    'We only support xarray facetted image plots using pcolormesh() in pcv functions. '
                    'For other types of plots please use xarray plotting methods and matplotlib directly.')

    contains_xy = len([dim for dim in ['x', 'y'] if dim in img.dims]) == 2
    if not contains_xy:
        fatal_error('You are missing x and y dimensions. '
                    'We only support xarray facetted image plots using pcolormesh() in pcv functions. '
                    'For other types of plots please use xarray plotting methods and matplotlib directly.')

    try:
        # need to force pcolormesh() for case when dim in col or row has length 1
        # https://github.com/pydata/xarray/issues/620
        fig_handle = img.plot.pcolormesh(col=col, row=row, **kwargs)
    except ValueError as err:
        raise ValueError(f'You are trying to plot shape {img.shape} but you should have exactly 2 dimensions in '
                         'addition those specified by `col` and `row`.') from err

    return fig_handle

class PSII_data:
    """PSII data class"""

    def __init__(self):
        self.ojip_dark = None
        self.ojip_light = None
        self.pam_dark = None
        self.pam_light = None
        self.spectral = None
        self.chlorophyll = None
        self.datapath = None
        self.filename = None

    def __repr__(self):
        mvars = []
        for k, v in self.__dict__.items():
            if v is not None:
                mvars.append(k)
        return "PSII variables defined:\n" + '\n'.join(mvars)

    def add_data(self, protocol):
        """Input:
        protocol: xr.DataArray with name equivalent to initialized attributes
        """
        self.__dict__[protocol.name] = protocol

PSII_data = PSII_data()

def print_image(img, filename, **kwargs):
    """
    Save image to file.

    Inputs:
    img      = image object
    filename = name of file to save image to
    kwargs   = key-value arguments to xarray.plot method

    :param img: numpy.ndarray, matplotlib.figure.Figure, ggplot, xarray.core.dataarray.DataArray
    :param filename: string
    :param kwargs: dict
    :return:
    """
    # Print numpy array type images
    if isinstance(img, np.ndarray):
        cv2.imwrite(filename, img)

    # Print matplotlib type images
    elif isinstance(img, matplotlib.figure.Figure):
        img.savefig(filename, dpi=params.dpi)

    # # Print altair type images
    # elif isinstance(img, (FacetChart, LayerChart, Chart)):
    #     img.save(filename, ppi=params.dpi)

    # elif isinstance(img, DataArray):
    #     fig_handle = _show_dataarray(img, **kwargs)
    #     # fig_handle comes back as a tuple if xarray makes a histogram
    #     # fig_handle comes back as a list len 1 containing matplotlib.lines.Line2D if xarray makes a line plot
    #     # will this ever happen? I think _show_dataarray and xarray will fail first
    #     fig_handle.fig.savefig(filename, dpi=params.dpi)

    elif isinstance(img, PSII_data):
        fatal_error("You need to provide an underlying DataArray.")

    else:
        fatal_error(f"Error writing file {filename}: input img is {str(type(img))}, not a numpy.ndarray, "
                    "matplotlib.figure, plotnine.ggplot, or xarray.core.dataarray.DataArray and cannot get "
                    "saved out with print_image.")

def plot_image(img, cmap=None, **kwargs):
    """
    Plot an image to the screen.

    :param img: numpy.ndarray, ggplot, xarray.core.dataarray.DataArray
    :param cmap: str
    :param kwargs: key-value arguments to xarray.plot method
    :return:
    """
    dimensions = np.shape(img)

    if isinstance(img, np.ndarray):
        matplotlib.rcParams['figure.dpi'] = params.dpi
        # If the image is color then OpenCV stores it as BGR, we plot it as RGB
        if len(dimensions) == 3:
            plt.figure()
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.show()

        elif cmap is None and len(dimensions) == 2:
            plt.figure()
            plt.imshow(img, cmap="gray")
            plt.show()

        elif cmap is not None and len(dimensions) == 2:
            plt.figure()
            plt.imshow(img, cmap=cmap)
            plt.show()

    elif isinstance(img, matplotlib.figure.Figure):
        fatal_error(
            "Error, matplotlib Figure not supported. Instead try running without plot_image.")

    # elif isinstance(img, DataArray):
    #     _show_dataarray(img, **kwargs)

    # Altair FacetChart
    # elif isinstance(img, (FacetChart, LayerChart, Chart)):
    #     img.display()

    elif isinstance(img, PSII_data):
        fatal_error("You need to plot an underlying DataArray.")

    else:
        fatal_error(f"Plotting {type(img)} is not supported.")

def _debug(visual, filename=None, **kwargs):
    """
    Save or display a visual for debugging.

    Inputs:
    visual   - An image or plot to display for debugging
    filename - An optional filename to save the visual to (default: None)
    kwargs - key-value arguments to xarray.plot method

    :param visual: numpy.ndarray
    :param filename: str
    :param kwargs: dict
    """
    # Auto-increment the device counter
    params.device += 1

    if params.debug == "print":
        # If debug is print, save the image to a file
        print_image(img=visual, filename=filename, **kwargs)
    elif params.debug == "plot":
        # If debug is plot, print to the plotting device
        plot_image(img=visual, **kwargs)

def _rgb2hsv(rgb_img, channel):
    """Convert image from RGB colorspace to HSV colorspace. Returns the specified subchannel as a gray image.

    Parameters
    ----------
    rgb_img : numpy.ndarray
        RGB image data
    channel : str
        color subchannel (h = hue, s = saturation, v = value/intensity/brightness)

    Returns
    -------
    numpy.ndarray
        grayscale image from one HSV color channel
    """
    # The allowable channel inputs are h, s or v
    channel = channel.lower()
    if channel not in ["h", "s", "v"]:
        fatal_error("Channel " + str(channel) + " is not h, s or v!")

    # Convert the input BGR image to HSV colorspace
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
    # Split HSV channels
    h, s, v = cv2.split(hsv)
    # Create a channel dictionaries for lookups by a channel name index
    channels = {"h": h, "s": s, "v": v}

    return channels[channel]

def _rgb2cmyk(rgb_img, channel):
    """Convert image from RGB colorspace to CMYK colorspace. Returns the specified subchannel as a gray image.

    Parameters
    ----------
    rgb_img : numpy.ndarray
        RGB image data
    channel : str
        color subchannel (c = cyan, m = magenta, y = yellow, k=black)

    Returns
    -------
    numpy.ndarray
        grayscale image from one CMYK color channel
    """
    # Set NumPy to ignore divide by zero errors
    _ = np.seterr(divide='ignore', invalid='ignore')
    # The allowable channel inputs are c, m , y or k
    channel = channel.lower()
    if channel not in ["c", "m", "y", "k"]:
        fatal_error("Channel " + str(channel) + " is not c, m, y or k!")

    # Create float
    bgr = rgb_img.astype(float)/255.

    # K channel
    k = 1 - np.max(bgr, axis=2)

    # C Channel
    c = (1 - bgr[..., 2] - k) / (1 - k)

    # M Channel
    m = (1 - bgr[..., 1] - k) / (1 - k)

    # Y Channel
    y = (1 - bgr[..., 0] - k) / (1 - k)

    # Convert the input BGR image to LAB colorspace
    cmyk = (np.dstack((c, m, y, k)) * 255).astype(np.uint8)
    # Split CMYK channels
    y, m, c, k = cv2.split(cmyk)
    # Create a channel dictionaries for lookups by a channel name index
    channels = {"c": c, "m": m, "y": y, "k": k}

    return channels[channel]


def _rgb2gray(rgb_img):
    """Convert image from RGB colorspace to Gray.

    Parameters
    ----------
    rgb_img : numpy.ndarray
        RGB image data

    Returns
    -------
    numpy.ndarray
        grayscale image
    """
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)

    return gray

def rgb2gray_hsv(rgb_img, channel):
    """Convert image from RGB colorspace to HSV colorspace. Returns the specified subchannel as a gray image.

    Parameters
    ----------
    rgb_img : numpy.ndarray
        RGB image data
    channel : str
        color subchannel (h = hue, s = saturation, v = value/intensity/brightness)

    Returns
    -------
    numpy.ndarray
        grayscale image from one HSV color channel
    """
    # Convert RGB to HSV and return the specified subchannel as a gray image
    gray_img = _rgb2hsv(rgb_img=rgb_img, channel=channel)

    # The allowable channel inputs are h, s or v
    names = {"h": "hue", "s": "saturation", "v": "value"}

    _debug(visual=gray_img,
           filename=os.path.join(params.debug_outdir, f"{params.device}_hsv_{names[channel.lower()]}.png"), cmap='gray')

    return gray_img

def rgb2gray_lab(rgb_img, channel):
    """Convert image from RGB colorspace to LAB colorspace. Returns the specified subchannel as a gray image.

    Parameters
    ----------
    rgb_img : numpy.ndarray
        RGB image data
    channel : str
        color subchannel (l = lightness, a = green-magenta, b = blue-yellow)

    Returns
    -------
    numpy.ndarray
        grayscale image from one LAB color channel
    """
    # Convert RGB to LAB and return the specified subchannel as a gray image
    gray_img = _rgb2lab(rgb_img=rgb_img, channel=channel)

    # The allowable channel inputs are l, a or b
    names = {"l": "lightness", "a": "green-magenta", "b": "blue-yellow"}

    # Display debug image
    _debug(visual=gray_img,
           filename=os.path.join(params.debug_outdir, f"{params.device}_lab_{names[channel.lower()]}.png"), cmap="gray")

    return gray_img

def rgb2gray_cmyk(rgb_img, channel):
    """Convert image from RGB colorspace to CMYK colorspace. Returns the specified subchannel as a gray image.

    Parameters
    ----------
    rgb_img : numpy.ndarray
        RGB image data
    channel : str
        color subchannel (c = cyan, m = magenta, y = yellow, k=black)

    Returns
    -------
    numpy.ndarray
        grayscale image from one CMYK color channel
    """
    # Convert RGB to CMYK and return the specified subchannel as a gray image
    gray_img = _rgb2cmyk(rgb_img=rgb_img, channel=channel)

    # The allowable channel inputs are c, m , y or k
    names = {"c": "cyan", "m": "magenta", "y": "yellow", "k": "black"}

    # Save or display the grayscale image
    _debug(visual=gray_img,
           filename=os.path.join(params.debug_outdir, f"{params.device}_cmyk_{names[channel.lower()]}.png"), cmap="gray")

    return gray_img

def rgb2gray(rgb_img):
    """Convert image from RGB colorspace to Gray.

    Parameters
    ----------
    rgb_img : numpy.ndarray
        RGB image data

    Returns
    -------
    numpy.ndarray
        grayscale image
    """
    gray = _rgb2gray(rgb_img=rgb_img)

    _debug(visual=gray, filename=os.path.join(params.debug_outdir, f"{params.device}_gray.png"))

    return gray

# Triangle autothreshold

def _call_threshold(gray_img, threshold, threshold_method, method_name):
    """Calls the OpenCV threshold function to reduce code duplication

    Parameters
    ----------
    gray_img : numpy.ndarray
        Grayscale image data
    threshold : int
        Threshold value
    threshold_method : int
        OpenCV thresholding method
    method_name : str
        Name of the method used for debugging purposes

    Returns
    -------
    numpy.ndarray
        Thresholded, binary image
    """
    # Threshold the image
    _, bin_img = cv2.threshold(gray_img, threshold, 255, threshold_method)

    if bin_img.dtype != 'uint16':
        bin_img = np.uint8(bin_img)

    # Print or plot the binary image if debug is on
    _debug(visual=bin_img, filename=os.path.join(params.debug_outdir,
                                                 str(params.device) + method_name + str(threshold) + '.png'))

    return bin_img

# Internal plotting function for the triangle autothreshold method
def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 4))

    ax.plot(x, 'b', lw=1)
    if ind.size:
        label = 'valley' if valley else 'peak'
        label = label + 's' if ind.size > 1 else label
        ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                label=f'{ind.size} {label}')
        ax.legend(loc='best', framealpha=.5, numpoints=1)
    ax.set_xlim(-.02 * x.size, x.size * 1.02 - 1)
    ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
    yrange = ymax - ymin if ymax > ymin else 1
    ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
    ax.set_xlabel('Data #', fontsize=14)
    ax.set_ylabel('Amplitude', fontsize=14)
    mode = 'Valley detection' if valley else 'Peak detection'
    ax.set_title(f"{mode} ({mph=}, {mpd=}, {threshold=}, {edge=})")
    plt.show()

def _detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising', valley=False, show=False, ax=None):
    """Marcos Duarte, https://github.com/demotu/BMC; version 1.0.4; license MIT

    Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indices of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    Examples
    --------
    from detect_peaks import detect_peaks
    x = np.random.randn(100)
    x[60:81] = np.nan
    # detect all peaks and plot data
    ind = detect_peaks(x, show=True)
    print(ind)

    x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    # set minimum peak height = 0 and minimum peak distance = 20
    detect_peaks(x, mph=0, mpd=20, show=True)

    x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    # set minimum peak distance = 2
    detect_peaks(x, mpd=2, show=True)

    x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    # detection of valleys instead of peaks
    detect_peaks(x, mph=0, mpd=20, valley=True, show=True)

    x = [0, 1, 1, 0, 1, 1, 0]
    # detect both edges
    detect_peaks(x, edge='both', show=True)

    x = [-2, 1, -2, 2, 1, 1, 3, 0]
    # set threshold = 2
    detect_peaks(x, threshold = 2, show=True)
    """
    x = np.atleast_1d(x).astype('float64')

    # It is always the case that x.size=256 since 256 hardcoded in line 186 ->
    # cv2.calcHist([gray_img], [0], None, [256], [0, 255])
    # if x.size < 3:
    #     return np.array([], dtype=int)

    # # Where this function is used it is hardcoded to use the default valley=False so this will never be used
    # if valley:
    #     x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    # indnan = np.where(np.isnan(x))[0]

    # x will never contain NaN since calcHist will never return NaN
    # if indnan.size:
    #     x[indnan] = np.inf
    #     dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)

    if edge.lower() in ['rising', 'both']:
        ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # x will never contain NaN since calcHist will never return NaN
    # if ind.size and indnan.size:
    #     # NaN's and values close to NaN's cannot be peaks
    #     ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]
    # first and last values of x cannot be peaks
    # if ind.size and ind[0] == 0:
    #     ind = ind[1:]
    # if ind.size and ind[-1] == x.size - 1:
    #     ind = ind[:-1]
    # We think the above code will never be reached given some of the hardcoded properties used

    # # Where this function is used has hardcoded mph=None so this will never be used
    # # remove peaks < minimum peak height
    # if ind.size and mph is not None:
    #     ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold

    if show:
        # x will never contain NaN since calcHist will never return NaN
        # if indnan.size:
        #     x[indnan] = np.nan
        # # Where this function is used it is hardcoded to use the default valley=False so this will never be used
        # if valley:
        #     x = -x
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind

def triangle(gray_img, object_type="light", xstep=1):
    """Creates a binary image from a grayscale image using Zack et al.'s (1977) thresholding.

    Inputs:
    gray_img     = Grayscale image data
    object_type  = "light" or "dark" (default: "light")
                   - If object is lighter than the background then standard thresholding is done
                   - If object is darker than the background then inverse thresholding is done
    xstep        = value to move along x-axis to determine the points from which to calculate distance recommended to
                   start at 1 and change if needed)

    Returns:
    bin_img      = Thresholded, binary image

    :param gray_img: numpy.ndarray
    :param object_type: str
    :param xstep: int
    :return bin_img: numpy.ndarray
    """
    # Calculate automatic threshold value based on triangle algorithm
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 255])

    # Make histogram one array
    newhist = []
    for item in hist:
        newhist.extend(item)

    # Detect peaks
    show = False
    if params.debug == "plot":
        show = True
    ind = _detect_peaks(newhist, mph=None, mpd=1, show=show)

    # Find point corresponding to highest peak
    # Find intensity value (y) of highest peak
    max_peak_int = max(newhist[i] for i in ind)
    # Find value (x) of highest peak
    max_peak = [i for i, x in enumerate(newhist) if x == max(newhist)]
    # Combine x,y
    max_peak_xy = [max_peak[0], max_peak_int]

    # Find final point at end of long tail
    end_x = len(newhist) - 1
    end_y = newhist[end_x]
    end_xy = [end_x, end_y]

    # Define the known points
    points = [max_peak_xy, end_xy]
    x_coords, y_coords = zip(*points)

    # Get threshold value
    peaks = []
    dists = []

    for i in range(x_coords[0], x_coords[1], xstep):
        distance = (((x_coords[1] - x_coords[0]) * (y_coords[0] - hist[i])) -
                    ((x_coords[0] - i) * (y_coords[1] - y_coords[0]))) / math.sqrt(
            (float(x_coords[1]) - float(x_coords[0])) *
            (float(x_coords[1]) - float(x_coords[0])) +
            ((float(y_coords[1]) - float(y_coords[0])) *
             (float(y_coords[1]) - float(y_coords[0]))))
        peaks.append(i)
        dists.append(distance)
    autothresh = [peaks[x] for x in [i for i, x in enumerate(list(dists)) if x == max(list(dists))]]
    autothreshval = autothresh[0]

    # Set the threshold method
    threshold_method = ""
    if object_type.upper() == "LIGHT":
        threshold_method = cv2.THRESH_BINARY + cv2.THRESH_OTSU
    elif object_type.upper() == "DARK":
        threshold_method = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    else:
        fatal_error('Object type ' + str(object_type) + ' is not "light" or "dark"!')

    params.device += 1

    # Threshold the image
    bin_img = _call_threshold(gray_img, autothreshval, threshold_method, "_triangle_threshold_")

    # Additional figures created by this method, if debug is on
    if params.debug is not None:
        if params.debug == 'print':
            _, ax = plt.subplots()
            ax.plot(hist)
            ax.set(title=f"Threshold value = {autothreshval}")
            ax.axis([0, 256, 0, max(hist)])
            ax.grid(True)
            fig_name_hist = os.path.join(params.debug_outdir,
                                         str(params.device) + '_triangle_thresh_hist_' + str(autothreshval) + ".png")
            # write the figure to current directory
            plt.savefig(fig_name_hist, dpi=params.dpi)
            # close pyplot plotting window
            plt.clf()
        elif params.debug == 'plot':
            print(f"Threshold value = {autothreshval}")
            _, ax = plt.subplots()
            ax.plot(hist)
            ax.axis([0, 256, 0, max(hist)])
            ax.grid(True)
            plt.show()

    return bin_img

def _identity(x, **kwargs):
    """Identity function for use in _rect_filter
    This may be useful if there are several outputs from a function passed to _rect_filter
    which would otherwise be difficult to manage
    Parameters
    ----------
    x : any
      An object
    **kwargs
      Other keyword arguments, ignored.
    """
    return x

def _rect_filter(img, roi=None, function=None, **kwargs):
    """Subset a rectangular section of image to apply function to
    Parameters
    ----------
    img : numpy.ndarray
        An image
    roi : plantcv Objects class
        A rectangular ROI as returned by plantcv.roi.rectangle
    function : function
        analysis function to apply to each submask
    **kwargs
        Other keyword arguments to pass to the analysis function.
    Returns
    -------
    any
        Return value depends on the function that is called. If no function is called then this is a numpy.ndarray.
    """
    if roi is None:
        xstart = 0
        ystart = 0
        xend = np.shape(img)[1]
        yend = np.shape(img)[0]
    else:
        xstart = roi.contours[0][0][0][0][0].astype("int32")
        ystart = roi.contours[0][0][0][0][1].astype("int32")
        xend = roi.contours[0][0][2][0][0].astype("int32")
        yend = roi.contours[0][0][2][0][1].astype("int32")
    # slice image to subset rectangle
    sub_img = img[ystart:yend, xstart:xend]
    # apply function
    if function is None:
        function = _identity

    return function(sub_img, **kwargs)

def _rect_replace(img, sub_img, roi):
    """
    Parameters
    ----------
    img : numpy.ndarray
        Full sized image
    sub_img : numpy.ndarray
        output from _rect_filter
    roi : plantcv Objects class
        A rectangular ROI as returned by plantcv.roi.rectangle
    Returns
    -------
    numpy.ndarray
    """
    if roi is None:
        # if no ROI then no subsetting was done, just return sub_img
        return sub_img

    # if subsetting was done then get coordinates, slice into main image, and return
    xstart = roi.contours[0][0][0][0][0].astype("int32")
    ystart = roi.contours[0][0][0][0][1].astype("int32")
    xend = roi.contours[0][0][2][0][0].astype("int32")
    yend = roi.contours[0][0][2][0][1].astype("int32")
    full_img = np.copy(img)
    full_img[ystart:yend, xstart:xend] = sub_img
    return full_img

def fill(bin_img, size, roi=None):
    """
    Identifies objects and fills objects that are less than size.

    Inputs:
    bin_img      = Binary image data
    size         = minimum object area size in pixels (integer)
    roi          = optional Objects class rectangular ROI

    Returns:
    filtered_img = image with objects filled

    :param bin_img: numpy.ndarray
    :param size: int
    :param roi: plantcv.plantcv.Objects
    :return filtered_img: numpy.ndarray
    """
    # Make sure the image is binary
    if len(np.shape(bin_img)) != 2 or len(np.unique(bin_img)) > 2:
        fatal_error("Image is not binary")

    # Cast binary image to boolean
    bool_img = bin_img.astype(bool)

    # Find and fill contours, possibly within bounding rectangle
    bool_img = _rect_filter(bool_img,
                            roi=roi,
                            function=remove_small_objects,
                            **{"min_size" : size})
    # Cast boolean image to binary and make a copy of the binary image for returning
    filtered_img = np.copy(bool_img.astype(np.uint8) * 255)
    # slice the subset image back into full size binary image
    replaced_img = _rect_replace(bin_img.astype(bool) * 255, filtered_img, roi)

    _debug(visual=replaced_img,
           filename=os.path.join(params.debug_outdir, str(params.device) + "_fill" + str(size) + '.png'))

    return replaced_img

def rescale(gray_img, min_value=0, max_value=255):
    """Rescale image.

    Inputs:
    gray_img  = Grayscale image data
    min_value = (optional) new minimum value for range of interest. default = 0
    max_value = (optional) new maximum value for range of interest. default = 255

    Returns:
    rescaled_img = rescaled image

    :param gray_img: numpy.ndarray
    :param min_value: int
    :param max_value: int
    :return c: numpy.ndarray
    """
    if len(np.shape(gray_img)) != 2:
        fatal_error("Image is not grayscale")

    rescaled_img = np.interp(gray_img, (np.nanmin(gray_img), np.nanmax(gray_img)), (min_value, max_value))
    rescaled_img = (rescaled_img).astype('uint8')

    _debug(visual=rescaled_img, filename=os.path.join(params.debug_outdir, str(params.device) + '_rescaled.png'))

    return rescaled_img

def apply_mask(img, mask, mask_color):
    """Apply white image mask to image, with bitwise AND operator bitwise NOT operator and ADD operator.

    Inputs:
    img        = RGB image data
    mask       = Binary mask image data
    mask_color = 'white' or 'black'

    Returns:
    masked_img = masked image data

    :param img: numpy.ndarray
    :param mask: numpy.ndarray
    :param mask_color: str
    :return masked_img: numpy.ndarray
    """
    if mask_color.upper() == "WHITE":
        color_val = 255
    elif mask_color.upper() == "BLACK":
        color_val = 0
    else:
        fatal_error('Mask Color ' + str(mask_color) + ' is not "white" or "black"!')

    array_data = img.copy()

    # Mask the array
    array_data[np.where(mask == 0)] = color_val

    # Check the array data format
    if len(np.shape(array_data)) > 2 and np.shape(array_data)[-1] > 3:
        # Replace this part with _make_pseudo_rgb
        num_bands = np.shape(array_data)[2]
        med_band = int(num_bands / 2)
        debug = params.debug
        params.debug = None
        pseudo_rgb = cv2.merge((rescale(array_data[:, :, 0]),
                                rescale(array_data[:, :, med_band]),
                                rescale(array_data[:, :, num_bands - 1])))
        params.debug = debug

        _debug(visual=pseudo_rgb,
               filename=os.path.join(params.debug_outdir, str(params.device) + '_masked.png'))
    else:
        _debug(visual=array_data,
               filename=os.path.join(params.debug_outdir, str(params.device) + '_masked.png'))

    return array_data

"""WHITE BALANCE"""
def _hist(img, hmax, x, y, h, w, data_type):
    """Corrects the exposure of an image based on its histogram.

    Parameters
    ----------
    img : numpy.ndarray
        An RGB image on which to perform the correction
    hmax : int
        The maximum pixel intensity value
    x : int
        The x-coordinate of the top left corner of the ROI
    y : int
        The y-coordinate of the top left corner of the ROI
    h : int
        The height of the ROI
    w : int
        The width of the ROI
    data_type : type
        The data type of the image

    Returns
    -------
    numpy.ndarray
        Image after exposure correction
    """
    _, bins = np.histogram(img[y:y + h, x:x + w], bins='auto')
    max1 = np.amax(bins)
    alpha = hmax / float(max1)
    corrected = np.asarray(np.where(img <= max1, np.multiply(alpha, img), hmax), data_type)

    return corrected


def _max(img, hmax, mask, x, y, h, w, data_type):
    """Corrects the exposure of an image based on the maximum pixel intensity value.

    Parameters
    ----------
    img : numpy.ndarray
        An RGB image on which to perform the correction
    hmax : int
        The maximum pixel intensity value
    mask : numpy.ndarray
        An image mask
    x : int
        The x-coordinate of the top left corner of the ROI
    y : int
        The y-coordinate of the top left corner of the ROI
    h : int
        The height of the ROI
    w : int
        The width of the ROI
    data_type : type
        The data type of the image

    Returns
    -------
    numpy.ndarray
        Image after exposure correction
    """
    imgcp = np.copy(img)
    cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)
    mask_binary = mask[:, :, 0]
    _, mask_binary = cv2.threshold(mask_binary, 254, 255, cv2.THRESH_BINARY)
    masked = apply_mask(imgcp, mask_binary, 'black')
    max1 = np.amax(masked)
    alpha = hmax / float(max1)
    corrected = np.asarray(np.where(img <= max1, np.multiply(alpha, img), hmax), data_type)

    return corrected


def white_balance(img, mode='hist', roi=None):
    """
    Corrects the exposure of an image based on its histogram.

    Inputs:
    img     = An RGB image on which to perform the correction, correction is done on each channel and then reassembled,
              alternatively a single channel can be input but is not recommended.
    mode    = 'hist or 'max'
    roi     = A list of 4 points (x, y, width, height) that form the rectangular ROI of the white color standard.
              If a list of 4 points is not given, whole image will be used.

    Returns:
    img     = Image after exposure correction

    :param img: numpy.ndarray
    :param mode: str
    :param roi: list
    :return finalcorrected: numpy.ndarray
    """
    ori_img = np.copy(img)

    if mode not in ('hist', 'max'):
        fatal_error('Mode must be either "hist" or "max" but ' + mode + ' was input.')

    if roi is not None:
        roiint = all(isinstance(item, (list, int)) for item in roi)

        if len(roi) != 4:
            fatal_error('If ROI is used ROI must have 4 elements as a list and all must be integers')
        elif roiint is False:
            fatal_error('If ROI is used ROI must have 4 elements as a list and all must be integers')
    else:
        pass

    if len(np.shape(img)) == 3:
        iy, ix, _ = np.shape(img)
        hmax = 255
        data_type = np.uint8
    else:
        iy, ix = np.shape(img)
        if img.dtype == 'uint8':
            hmax = 255
            data_type = np.uint8
        elif img.dtype == 'uint16':
            hmax = 65536
            data_type = np.uint16

    mask = np.zeros((iy, ix, 3), dtype=np.uint8)

    if roi is None:
        x = 0
        y = 0
        w = ix
        h = iy
    else:
        x = roi[0]
        y = roi[1]
        w = roi[2]
        h = roi[3]

    if len(np.shape(img)) == 3:
        cv2.rectangle(ori_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        c1 = img[:, :, 0]
        c2 = img[:, :, 1]
        c3 = img[:, :, 2]
        if mode.upper() == 'HIST':
            channel1 = _hist(c1, hmax, x, y, h, w, data_type)
            channel2 = _hist(c2, hmax, x, y, h, w, data_type)
            channel3 = _hist(c3, hmax, x, y, h, w, data_type)
        elif mode.upper() == 'MAX':
            channel1 = _max(c1, hmax, mask, x, y, h, w, data_type)
            channel2 = _max(c2, hmax, mask, x, y, h, w, data_type)
            channel3 = _max(c3, hmax, mask, x, y, h, w, data_type)

        finalcorrected = np.dstack((channel1, channel2, channel3))

    else:
        cv2.rectangle(ori_img, (x, y), (x + w, y + h), (255, 255, 255), 3)
        if mode.upper() == 'HIST':
            finalcorrected = _hist(img, hmax, x, y, h, w, data_type)
        elif mode.upper() == 'MAX':
            finalcorrected = _max(img, hmax, mask, x, y, h, w, data_type)

    _debug(visual=ori_img,
           filename=os.path.join(params.debug_outdir, str(params.device) + '_whitebalance_roi.png'),
           cmap='gray')
    _debug(visual=finalcorrected,
           filename=os.path.join(params.debug_outdir, str(params.device) + '_whitebalance.png'),
           cmap='gray')

    return finalcorrected


"""Convert mean RGB to LAB"""
def rgb_to_lab(r, g, b):
    # 1. Normalize RGB variables to 0-1 range
    var_R = r / 255.0
    var_G = g / 255.0
    var_B = b / 255.0

    # 2. Linearize sRGB (apply gamma correction)
    var_R = ((var_R + 0.055) / 1.055) ** 2.4 if var_R > 0.04045 else var_R / 12.92
    var_G = ((var_G + 0.055) / 1.055) ** 2.4 if var_G > 0.04045 else var_G / 12.92
    var_B = ((var_B + 0.055) / 1.055) ** 2.4 if var_B > 0.04045 else var_B / 12.92

    # 3. Convert to XYZ space using D65 matrix profile
    X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805
    Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722
    Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505

    # 4. Normalize for D65 white point reference
    var_X = X / 0.95047
    var_Y = Y / 1.00000
    var_Z = Z / 1.08883

    # 5. Non-linear transformation for LAB calculation
    var_X = var_X ** (1/3) if var_X > 0.008856 else (7.787 * var_X) + (16 / 116)
    var_Y = var_Y ** (1/3) if var_Y > 0.008856 else (7.787 * var_Y) + (16 / 116)
    var_Z = var_Z ** (1/3) if var_Z > 0.008856 else (7.787 * var_Z) + (16 / 116)

    # 6. Final LAB calculations
    L = (116 * var_Y) - 16
    a = 500 * (var_X - var_Y)
    b = 200 * (var_Y - var_Z)

    return L, a, b
