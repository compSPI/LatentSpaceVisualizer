import time

import numpy as np

from bokeh import events
import bokeh.io
from bokeh.io import output_notebook
from bokeh.io import show
from bokeh.layouts import row
from bokeh.models import CustomJS, Div, LinearColorMapper, ColorBar, Range1d, HoverTool

from bokeh.plotting import ColumnDataSource
import bokeh.palettes

from kora.bokeh import figure

from sklearn import datasets
from sklearn import preprocessing
from sklearn.decomposition import PCA

from PIL import Image
import base64
from io import BytesIO

from six import string_types

import h5py as h5

"""
Deeban Ramalingam (deebanr@slac.stanford.edu)
"""


"""
Functions for converting from quaternion to (azimuth, elevation, rotation angle)
"""

def angle_axis_representation(quaternion):
    q_r = quaternion[0]
    q_ijk = quaternion[1:]
    # https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    q_norm = np.linalg.norm(q_ijk)
    axis = q_ijk / q_norm
    theta = 2 * np.arctan2(q_norm, q_r)
    return axis, theta

def azimuth_elevation_representation(unit_vector):
    x = unit_vector[0]
    y = unit_vector[1]
    z = unit_vector[2]
    # https://en.wikipedia.org/wiki/Spherical_coordinate_system
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))
    return azimuth, elevation

def get_azimuth_elevation_rotation_angles_from_orientations(orientations):
    n_orientations = orientations.shape[0]
    azimuth_angles = np.zeros((n_orientations,))
    elevation_angles = np.zeros((n_orientations,))
    rotation_angles = np.zeros((n_orientations,))
    
    for orientation_idx, orientation in enumerate(orientations):
        axis, theta = angle_axis_representation(orientation)
        azimuth, elevation = azimuth_elevation_representation(axis)

        azimuth_angles[orientation_idx] = azimuth
        elevation_angles[orientation_idx] = elevation
        rotation_angles[orientation_idx] = theta
    
    return azimuth_angles, elevation_angles, rotation_angles

# Adapted from: https://sscc.nimh.nih.gov/pub/dist/bin/linux_gcc32/meica.libs/nibabel/quaternions.py
def quat2mat(q):
    ''' Calculate rotation matrix corresponding to quaternion

    Parameters
    ----------
    q : 4 element array-like

    Returns
    -------
    M : (3,3) array
      Rotation matrix corresponding to input quaternion *q*

    Notes
    -----
    Rotation matrix applies to column vectors, and is applied to the
    left of coordinate vectors.  The algorithm here allows non-unit
    quaternions.

    References
    ----------
    Algorithm from
    http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion

    Examples
    --------
    >>> import numpy as np
    >>> M = quat2mat([1, 0, 0, 0]) # Identity quaternion
    >>> np.allclose(M, np.eye(3))
    True
    >>> M = quat2mat([0, 1, 0, 0]) # 180 degree rotn around axis 0
    >>> np.allclose(M, np.diag([1, -1, -1]))
    True
    '''
    w, x, y, z = q
    Nq = w*w + x*x + y*y + z*z
    FLOAT_EPS = np.finfo(np.float).eps
    if Nq < FLOAT_EPS:
        return np.eye(3)
    
    s = 2.0 / Nq
    X = x * s
    Y = y * s
    Z = z * s
    wX = w * X
    wY = w * Y
    wZ = w * Z
    xX = x * X
    xY = x * Y
    xZ = x * Z
    yY = y * Y
    yZ = y * Z
    zZ = z * Z
    
    return np.array(
           [[ 1.0 - (yY + zZ), xY - wZ, xZ + wY ],
            [ xY + wZ, 1.0 - (xX + zZ), yZ - wX ],
            [ xZ - wY, yZ + wX, 1.0 - (xX + yY) ]])

def get_azimuth_elevation_from_orientations_applied_to_reference_vector(orientations, orientation_ref_vector = np.array([[1, 0, 0]]).T):
    n_orientations = orientations.shape[0]
    azimuth_angles = np.zeros((n_orientations,))
    elevation_angles = np.zeros((n_orientations,))
    
    for orientation_idx, orientation in enumerate(orientations):
        rotation_matrix_3d = quat2mat(orientation)
        rotated_orientation_ref_vector = np.matmul(rotation_matrix_3d, orientation_ref_vector)
        azimuth, elevation = azimuth_elevation_representation(rotated_orientation_ref_vector)
        
        azimuth_angles[orientation_idx] = azimuth
        elevation_angles[orientation_idx] = elevation
    
    return azimuth_angles, elevation_angles

"""
Functions for coloring points in the scatter plot according to rotation angle
"""

def get_color_from_continuous_value(x, color_bar_palette, vmin, vmax):
    n = len(color_bar_palette)
    return color_bar_palette[int((x - vmin) / (vmax - vmin) * n)]

def get_colors_from_continuous_values(data, color_bar_palette, color_bar_vmin, color_bar_vmax): 
    colors = []
    for data_point in data:
        color = get_color_from_continuous_value(data_point, color_bar_palette, color_bar_vmin, color_bar_vmax)
        colors.append(color)
    
    return colors

def get_colors_from_discrete_values(data, color_bar_palette):
    colors = []
    for data_point in data:
        color = color_bar_palette[data_point]
        colors.append(color)
        
    return colors

"""
Function for displaying real-space XY projection plot using quaternions and atomic coordinates
"""

def display_real2d_plot_using_orientations(orientations, real2d_plot_data_source, ref_vector_azimuth, ref_vector_elevation, atomic_coordinates):
    return CustomJS(args=dict(orientations=orientations, real2d_plot_data_source=real2d_plot_data_source, ref_vector_azimuth=ref_vector_azimuth, ref_vector_elevation=ref_vector_elevation, atomic_coordinates=atomic_coordinates), code="""
        // Adapted from:
        // 1. https://stackoverflow.com/questions/27205018/multiply-2-matrices-in-javascript
        // 2. https://en.wikipedia.org/wiki/Transpose
        function transposeSecondArgThenMultiplyMatrices(m1, m2) {
            var result = [];
            for (var i = 0; i < m1.length; i++) {
                result[i] = [];
                for (var j = 0; j < m2.length; j++) {
                    var sum = 0;
                    for (var k = 0; k < m1[0].length; k++) {                        
                        sum += m1[i][k] * m2[j][k];
                    }
                    result[i][j] = sum;
                }
            }
            return result;
        }
        
        // Adapted from: https://sscc.nimh.nih.gov/pub/dist/bin/linux_gcc32/meica.libs/nibabel/quaternions.py
        function quat2mat(quaternion) {
            var w = quaternion[0];
            var x = quaternion[1];
            var y = quaternion[2];
            var z = quaternion[3];

            var Nq = w*w + x*x + y*y + z*z;

            var FLOAT_EPS = 2.220446049250313e-16;

            if (Nq < FLOAT_EPS) {
                 return [[1, 0, 0], 
                         [0, 1, 0], 
                         [0, 0, 1]];
            }

            var s = 2.0 / Nq;

            var X = x * s;
            var Y = y * s;
            var Z = z * s;

            var wX = w * X; 
            var wY = w * Y; 
            var wZ = w * Z;

            var xX = x * X; 
            var xY = x * Y; 
            var xZ = x * Z;

            var yY = y * Y; 
            var yZ = y * Z; 

            var zZ = z * Z;

            return [[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
                    [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
                    [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]];
        }   
    
        // Get the mouse position on the scatter plot
        var mouse_x = Number(cb_obj['x']);
        var mouse_y = Number(cb_obj['y']);
    
        // Find the index of the data point that is closest to the mouse position on the scatter plot
        var minDiffIndex = -1;
        var minDiff = 99999;
        var n = ref_vector_azimuth.length;
        var squareDiff;
        for (var i = 0; i < n; i++) {
            squareDiff = (mouse_x - ref_vector_azimuth[i]) ** 2 + (mouse_y - ref_vector_elevation[i]) ** 2;
            if (squareDiff < minDiff) {
                minDiff = squareDiff;
                minDiffIndex = i;
            }
        }
                
        if (minDiffIndex > -1) {
            // identify the orientation that corresponds to the nearest (azimuth, elevation) point
            var orientation = orientations[minDiffIndex];
            
            // compute the rotation matrix that corresponds to the orientation
            var rotation_matrix = quat2mat(orientation);

            // rotate atomic_coordinates using rotation_matrix
            // Adapted from: /reg/neh/home/dujardin/pysingfel/examples/scripts/gui.py
            var rotated_atomic_coordinates = transposeSecondArgThenMultiplyMatrices(rotation_matrix, atomic_coordinates);

            // scatter plot the rotated_atomic_coordinates
            // Adapted from: /reg/neh/home/dujardin/pysingfel/examples/scripts/gui.py
            real2d_plot_data_source.data['x'] = rotated_atomic_coordinates[1]; // x-axis represents the y-coordinate
            real2d_plot_data_source.data['y'] = rotated_atomic_coordinates[0]; // y-axis represents the x-coordinate
            real2d_plot_data_source.change.emit();
        }
    """)

"""
Function for generating PNG image byte strings
"""

def gnp2im(image_np):
    """
    Converts an image stored as a 2-D grayscale Numpy array into a PIL image.
    """
    rescaled = (255.0 / image_np.max() * (image_np - image_np.min())).astype(np.uint8)
    im = Image.fromarray(rescaled, mode='L')
    return im

def to_base64(png):
    return "data:image/png;base64," + base64.b64encode(png).decode("utf-8")

def get_images(data):
    images = []
    for gnp in data:
        im = gnp2im(gnp)
        memout = BytesIO()
        im.save(memout, format='png')
        images.append(to_base64(memout.getvalue()))
    return images

"""
Functions for displaying image plots
"""

def display_image_plot(image_plot_div, latent_variable_1, latent_variable_2, static_images, image_plot_image_brightness, image_plot_index_label_text_style = 'font-size:20px;text-align:center'):
    return CustomJS(args=dict(image_plot_div=image_plot_div, latent_variable_1=latent_variable_1, latent_variable_2=latent_variable_2, static_images=static_images, image_plot_image_brightness=image_plot_image_brightness), code="""
        // Get the mouse position on the scatter plot
        var mouse_x = Number(cb_obj['x']);
        var mouse_y = Number(cb_obj['y']);
    
        // Find the index of the data point that is closest to the mouse position on the scatter plot
        var minDiffIndex = -1;
        var minDiff = 99999;
        var n = latent_variable_1.length;
        var squareDiff;
        for (var i = 0; i < n; i++) {
            squareDiff = (mouse_x - latent_variable_1[i]) ** 2 + (mouse_y - latent_variable_2[i]) ** 2;
            if (squareDiff < minDiff) {
                minDiff = squareDiff;
                minDiffIndex = i;
            }
        }
        
        if (minDiffIndex > -1) {    
            // Identify the image source
            var img_src = static_images[minDiffIndex];
            
            // Style the image
            var img_tag_attrs = "style='filter: brightness(" + image_plot_image_brightness + "); transform: scaleY(-1);'";

            // Create the image
            var img_tag = "<div><img src='" + img_src + "' " + img_tag_attrs + "></img></div>";

            // Create the image index label text
            var img_idx_label_text = "<p style=%r>" + (minDiffIndex + 1) + "</p>";

            // Create the image plot contents
            var img_plot_contents = img_tag + img_idx_label_text;

            // Display the image plot contents in the image plot div
            image_plot_div.text = img_plot_contents;
        }
    """ % (image_plot_index_label_text_style))

def display_image_plot_on_slac_pswww(image_plot_div, latent_variable_1, latent_variable_2, image_plot_slac_username, image_plot_slac_dataset_name, image_plot_image_brightness, image_plot_index_label_text_style='font-size:20px;text-align:center'):
    return CustomJS(args=dict(image_plot_div=image_plot_div, latent_variable_1=latent_variable_1, latent_variable_2=latent_variable_2, image_plot_slac_username=image_plot_slac_username, image_plot_slac_dataset_name=image_plot_slac_dataset_name, image_plot_image_brightness=image_plot_image_brightness), code="""        
        // Get the mouse position on the scatter plot
        var mouse_x = Number(cb_obj['x']);
        var mouse_y = Number(cb_obj['y']);
    
        // Find the index of the data point that is closest to the mouse position on the scatter plot
        var minDiffIndex = -1;
        var minDiff = 99999;
        var n = latent_variable_1.length;
        var squareDiff;
        for (var i = 0; i < n; i++) {
            squareDiff = (mouse_x - latent_variable_1[i]) ** 2 + (mouse_y - latent_variable_2[i]) ** 2;
            if (squareDiff < minDiff) {
                minDiff = squareDiff;
                minDiffIndex = i;
            }
        }
        
        if (minDiffIndex > -1) {       
            // Identify the image source
            var img_src = "https://pswww.slac.stanford.edu/jupyterhub/user/" + image_plot_slac_username + "/files/" + image_plot_slac_dataset_name + "/images/diffraction-pattern-" + minDiffIndex + ".png";
            
            // Style the image
            var img_tag_attrs = "style='filter: brightness(" + image_plot_image_brightness + "); transform: scaleY(-1);'";

            // Create the image
            var img_tag = "<div><img src='" + img_src + "' " + img_tag_attrs + "></img></div>";

            // Create the image index label text
            var img_idx_label_text = "<p style=%r>" + (minDiffIndex + 1) + "</p>";

            // Create the image plot contents
            var img_plot_contents = img_tag + img_idx_label_text;

            // Display the image plot contents in the image plot div
            image_plot_div.text = img_plot_contents;
        }
    """ % (image_plot_index_label_text_style))

def display_image_plot_using_image_source_location(image_plot_image_source_location, image_plot_div, latent_variable_1, latent_variable_2, image_plot_image_brightness, image_plot_index_label_text_font_size, image_plot_slac_username=None, image_plot_slac_dataset_name=None, dataset_file=None, image_type=None):
    # Determine where to load images from
    if image_plot_image_source_location == "slac-pswww":
        if image_plot_slac_username is None:
            raise Exception("Please provide: slac_username")

        if image_plot_slac_dataset_name is None:
            raise Exception("Please provide: slac_dataset_name")

        # Directly display images from SLAC PSWWW
        display_image_plot_fn = display_image_plot_on_slac_pswww(image_plot_div, 
                                                        latent_variable_1, 
                                                        latent_variable_2, 
                                                        image_plot_slac_username, 
                                                        image_plot_slac_dataset_name, 
                                                        image_plot_image_brightness, 
                                                        image_plot_index_label_text_style='font-size:{};text-align:center'.format(image_plot_index_label_text_font_size))    
    elif image_plot_image_source_location is None:
        # Load images from HDF5 file
        with h5.File(dataset_file, "r") as dataset_file_handle:
            images = dataset_file_handle[image_type][:]

        # Convert loaded images to PNG image byte strings
        tic = time.time()
        static_images = get_images(images)
        toc = time.time()
        print("It takes {:.2f} seconds to generate static images in memory.".format(toc-tic))

        # Display PNG image byte strings generated from image arrays in HDF5 file
        display_image_plot_fn = display_image_plot(image_plot_div, 
                                                    latent_variable_1, 
                                                    latent_variable_2,
                                                    static_images,
                                                    image_plot_image_brightness, 
                                                    image_plot_index_label_text_style='font-size:{};text-align:center'.format(image_plot_index_label_text_font_size))
    else:
        raise Exception("Unknown image source location.")
    
    return display_image_plot_fn

"""
Function for displaying Bokeh plots in Jupyter notebook
"""
    
def output_notebook():
    """
    Display Bokeh plots in Jupyter notebook
    """
    bokeh.io.output_notebook()

"""
Function for visualizing data from an HDF5 file
"""

def visualize_orientations(
    dataset_file, 
    image_type, 
    figure_height = 450, 
    figure_width = 450, 
    scatter_plot_x_axis_label_text_font_size='15pt',
    scatter_plot_y_axis_label_text_font_size='15pt', 
    scatter_plot_color_bar_height = 400,
    scatter_plot_color_bar_width = 120,
    scatter_plot_ref_vector_as_a_list = [1, 0, 0],
    image_plot_image_size_scale_factor = 0.9,
    image_plot_image_brightness=1.0,
    image_plot_image_source_location=None, 
    image_plot_slac_username=None,
    image_plot_slac_dataset_name=None,
    image_plot_index_label_text_font_size='20px',
    real2d_plot_particle_property=None,
    real2d_plot_particle_atoms_random_sample_size=1000,
    real2d_plot_x_lower=None,
    real2d_plot_x_upper=None,
    real2d_plot_y_lower=None,
    real2d_plot_y_upper=None,
    real2d_plot_x_axis_label_text_font_size='15pt',
    real2d_plot_y_axis_label_text_font_size='15pt'
    ):
    
    """
    Visualize the orientations from an HDF5 file
    
    :param dataset_file: The path to the HDF5 file, as a str
    :param image_type: A key in the HDF5 file that represents the type of image, as a str
    :param latent_method: A key in the HDF5 file that represents the latent method used to build the latent space, as a str
    :param figure_height: height of the figure containing the plots, as an int
    :param figure_width: width of the figure containing the plots, as an int
    :param scatter_plot_x_axis_label_text_font_size: Font size of the x axis label for the scatter plot, as a str
    :param scatter_plot_y_axis_label_text_font_size: Font size of the y axis label for the scatter plot, as a str
    :param scatter_plot_color_bar_height: height of the color bar, as an int
    :param scatter_plot_color_bar_width: width of the color bar, as an int
    :param scatter_plot_ref_vector_as_a_list: reference vector for the azimuth-elevation plot, as a list of 3 floats
    :param image_plot_image_size_scale_factor: scale the image size according to a fraction of the figure size, as a float
    :param image_plot_image_brightness: Brightness of the image displayed in the image plot, as a float
    :param image_plot_image_source_location: An alias for the determining where to display images from, as a str
    :param image_plot_slac_username: A valid SLAC username, as a str
    :param image_plot_slac_dataset_name: The name of the dataset in SLAC Compute Cluster, as a str
    :param image_plot_index_label_text_font_size: Font size of the label for the index below the image plot, as a float
    """
    
    # Load data from the HDF5 file
    tic = time.time()
    with h5.File(dataset_file, "r") as dataset_file_handle:
        orientations = dataset_file_handle["orientations"][:]
        atomic_coordinates = dataset_file_handle[real2d_plot_particle_property][:]

    toc = time.time()
    print("It takes {:.2f} seconds to load the latent vectors and metadata from the HDF5 file.".format(toc-tic))
  
    # Real-space XY projection plots
    real2d_plot = figure(width=figure_width, height=figure_height, tools="pan,wheel_zoom,box_zoom,reset")
    
    # Add axis labels
    real2d_plot.xaxis.axis_label_text_font_size = real2d_plot_x_axis_label_text_font_size
    real2d_plot.yaxis.axis_label_text_font_size = real2d_plot_y_axis_label_text_font_size
    
    # Data source
    real2d_plot_data_source = ColumnDataSource({'x': [], 'y': []})

    # Scatter plot
    real2d_plot.scatter('x', 'y', source=real2d_plot_data_source)
    
    # Assuming the beam travels in the +Z direction, the real-space XY projection plot faces the beam.
    real2d_plot.xaxis.axis_label = "Y"
    real2d_plot.yaxis.axis_label = "X"
    
    # Axes limits
    if real2d_plot_x_lower is not None and real2d_plot_x_upper is not None and real2d_plot_y_lower is not None and real2d_plot_y_upper is not None:
        real2d_plot.x_range = Range1d(real2d_plot_x_lower, real2d_plot_x_upper)
        real2d_plot.y_range = Range1d(real2d_plot_y_lower, real2d_plot_y_upper)

    # Scatter plot for the orientations
    scatter_plot = figure(width=figure_width, height=figure_height, tools="pan,wheel_zoom,box_zoom,reset")

    # Quaternion -> 3D rotation matrix -> apply to reference vector -> (azimuth, elevation)
    ref_vector_azimuth, ref_vector_elevation = get_azimuth_elevation_from_orientations_applied_to_reference_vector(orientations, orientation_ref_vector = np.array([scatter_plot_ref_vector_as_a_list]).T)

    # Quaternion -> angle-axis -> (azimuth, elevation), rotation angle about axis
    azimuth, elevation, rotation_angles = get_azimuth_elevation_rotation_angles_from_orientations(orientations)    

    # Use rotation_angles to color the points on the scatter plots
    color_palette = bokeh.palettes.plasma(256)
    color_vmin = 0.0
    color_vmax = 2 * np.pi
    scatter_plot_colors = get_colors_from_continuous_values(rotation_angles, color_palette, color_vmin, color_vmax)
    color_bar_color_mapper = LinearColorMapper(palette=color_palette, low=color_vmin, high=color_vmax)
    
    # Data source for the scatter plot
    scatter_plot_data_source = ColumnDataSource(data=dict(
        ref_vector_azimuth=ref_vector_azimuth,
        ref_vector_elevation=ref_vector_elevation,
        azimuth=azimuth,
        elevation=elevation,
        rotation_angles=rotation_angles,
        colors=scatter_plot_colors
    ))

    # Populate the scatter plot
    scatter_plot.circle('ref_vector_azimuth', 'ref_vector_elevation', fill_color='colors', source=scatter_plot_data_source, fill_alpha=0.6, line_color=None)

    # Add axis labels
    scatter_plot.xaxis.axis_label = "Azimuth"
    scatter_plot.yaxis.axis_label = "Elevation"
    
    # Change the axis label font size
    scatter_plot.xaxis.axis_label_text_font_size = scatter_plot_x_axis_label_text_font_size
    scatter_plot.yaxis.axis_label_text_font_size = scatter_plot_y_axis_label_text_font_size

    # Limit the azimuth and elevation
    scatter_plot_x_lower, scatter_plot_x_upper, scatter_plot_y_lower, scatter_plot_y_upper = -np.pi, np.pi, -np.pi / 2, np.pi / 2
    scatter_plot.x_range = Range1d(scatter_plot_x_lower, scatter_plot_x_upper)
    scatter_plot.y_range = Range1d(scatter_plot_y_lower, scatter_plot_y_upper)

    # Color bar
    color_bar_plot = figure(title="Rotation", 
                            title_location="right", 
                            height=scatter_plot_color_bar_height, 
                            width=scatter_plot_color_bar_width, 
                            min_border=0, 
                            outline_line_color=None,
                            toolbar_location=None)
    
    color_bar_plot.title.align = "center"
    color_bar_plot.title.text_font_size = "12pt"
    color_bar_plot.scatter([], []) # removes Bokeh warning 1000 (MISSING_RENDERERS)
    
    color_bar = ColorBar(color_mapper=color_bar_color_mapper, label_standoff=12, border_line_color=None, location=(0,0))
    color_bar_plot.add_layout(color_bar, "right")

    # Container to display the images
    div_width = int(figure_width * image_plot_image_size_scale_factor)
    div_height = int(figure_height * image_plot_image_size_scale_factor)
    image_plot_div = Div(width=div_width, height=div_height)

    # Build layout for plots
    layout = row(real2d_plot, scatter_plot, color_bar_plot, image_plot_div)

    # Display the corresponding image when mouse is near a point in scatter plot
    display_image_plot_fn = display_image_plot_using_image_source_location(image_plot_image_source_location, 
                                                        image_plot_div, 
                                                        ref_vector_azimuth, 
                                                        ref_vector_elevation, 
                                                        image_plot_image_brightness, 
                                                        image_plot_index_label_text_font_size, 
                                                        image_plot_slac_username=image_plot_slac_username, 
                                                        image_plot_slac_dataset_name=image_plot_slac_dataset_name, 
                                                        dataset_file=dataset_file, 
                                                        image_type=image_type)
    scatter_plot.js_on_event(events.MouseMove, display_image_plot_fn)
    
    # To prevent a lag when displaying particles with several atoms, randomly select particle_random_sample_size atoms
    n_atomic_coordinates = len(atomic_coordinates)
    particle_atoms_random_sample_size = min(real2d_plot_particle_atoms_random_sample_size, n_atomic_coordinates)
    particle_atoms_random_sample_idx = np.random.choice(n_atomic_coordinates, particle_atoms_random_sample_size, replace=False)
    atomic_coordinates_random_sample = atomic_coordinates[particle_atoms_random_sample_idx]
    
    # Display real-space XY projection of particle when mouse is near a point in scatter plot
    display_real2d_plot_fn = display_real2d_plot_using_orientations(orientations, 
                                                real2d_plot_data_source, 
                                                ref_vector_azimuth, 
                                                ref_vector_elevation, 
                                                atomic_coordinates_random_sample)
    scatter_plot.js_on_event(events.MouseMove, display_real2d_plot_fn)

    # Display the plots
    tic = time.time()
    show(layout)
    toc = time.time()
    print("It takes {:.2f} seconds to display the plots.".format(toc-tic))

def visualize_latent_space(
    dataset_file, 
    image_type, 
    latent_method, 
    latent_idx_1=0, 
    latent_idx_2=1,
    figure_height = 450, 
    figure_width = 450, 
    scatter_plot_x_axis_label_text_font_size='15pt',
    scatter_plot_y_axis_label_text_font_size='15pt', 
    scatter_plot_color_bar_height = 400,
    scatter_plot_color_bar_width = 120,
    scatter_plot_type = "hexbin",
    image_plot_image_size_scale_factor = 0.9,
    image_plot_image_brightness=1.0,
    image_plot_image_source_location=None, 
    image_plot_slac_username=None,
    image_plot_slac_dataset_name=None,
    image_plot_index_label_text_font_size='20px'
    ):
    
    """
    Visualize the latent space from an HDF5 file
    
    :param dataset_file: The path to the HDF5 file, as a str
    :param image_type: A key in the HDF5 file that represents the type of image, as a str
    :param latent_method: A key in the HDF5 file that represents the latent method used to build the latent space, as a str
    :param figure_height: height of the figure containing the plots, as an int
    :param figure_width: width of the figure containing the plots, as an int
    :param scatter_plot_x_axis_label_text_font_size: Font size of the x axis label for the scatter plot, as a str
    :param scatter_plot_y_axis_label_text_font_size: Font size of the y axis label for the scatter plot, as a str
    :param scatter_plot_color_bar_height: height of the color bar, as an int
    :param scatter_plot_color_bar_width: width of the color bar, as an int
    :param image_plot_image_size_scale_factor: scale the image size according to a fraction of the figure size, as a float
    :param image_plot_image_brightness: Brightness of the image displayed in the image plot, as a float
    :param image_plot_image_source_location: An alias for the determining where to display images from, as a str
    :param image_plot_slac_username: A valid SLAC username, as a str
    :param image_plot_slac_dataset_name: The name of the dataset in SLAC Compute Cluster, as a str
    :param image_plot_index_label_text_font_size: Font size of the label for the index below the image plot, as a float
    """
    
    # Load data from the HDF5 file
    tic = time.time()
    with h5.File(dataset_file, "r") as dataset_file_handle:        
        latent_variable_1 = dataset_file_handle[latent_method][:, latent_idx_1 - 1]
        latent_variable_2 = dataset_file_handle[latent_method][:, latent_idx_2 - 1] 
        
        if scatter_plot_type == "hexbin_colored_by_single_hit_flag":
            # Load the single-hit mask from the HDF5 file
            single_hit_mask_key = "single_hit_mask"
            single_hit_mask = dataset_file_handle[single_hit_mask_key][:].astype(np.bool)
        
        elif scatter_plot_type == "colored_by_training_mask":
            # Load the training set mask from the HDF5 file
            training_set_mask_key = "training_set_mask"
            training_set_mask = dataset_file_handle[training_set_mask_key][:].astype(np.bool)

        # unclear on how to plot targets
        # labels = np.zeros(len(images))

    toc = time.time()
    print("It takes {:.2f} seconds to load the {} latent vectors and metadata from the HDF5 file.".format(toc-tic, len(latent_variable_1)))
    
    # unclear on how to plot targets
    # n_labels = len(np.unique(labels))
    
    # Scatter plot for the latent vectors
    scatter_plot = figure(match_aspect=True, width=figure_width, height=figure_height, tools="pan,wheel_zoom,box_zoom,reset", background_fill_color="#440154")

    if scatter_plot_type == "hexbin":
        # Make scatter plot grid invisible
        scatter_plot.grid.visible = False
        
        # Data source for the scatter plot
        scatter_plot_data_source = ColumnDataSource(data=dict(latent_variable_1=latent_variable_1, latent_variable_2=latent_variable_2))

        # Hexbin the scatter plot
        scatter_plot_hexbin_renderer, scatter_plot_hexbin_bins = scatter_plot.hexbin(latent_variable_1, latent_variable_2, size=0.5, hover_color="pink", hover_alpha=0.8)

        # Populate the scatter plot
        #scatter_plot.circle('latent_variable_1', 'latent_variable_2', source=scatter_plot_data_source, fill_alpha=0.6, line_color=None)
        scatter_plot.circle('latent_variable_1', 'latent_variable_2', source=scatter_plot_data_source, color='white', size=0.1)
        
        # Display data point count for each hex tile in the hexbin scatter plot
        scatter_plot.add_tools(HoverTool(
            tooltips=[("count", "@c")],
            mode="mouse", point_policy="follow_mouse", renderers=[scatter_plot_hexbin_renderer]
        ))
    
    elif scatter_plot_type == "hexbin_colored_by_single_hit_flag":
        # Color the points in the scatter plot according to the single-hit mask
        scatter_plot_colors = get_colors_from_discrete_values(single_hit_mask, bokeh.palettes.Set1[3][:2])

        # Define the legend 
        data_point_type = np.empty((len(latent_variable_1),), dtype=np.object)
        data_point_type[single_hit_mask] = "Single-hit"
        data_point_type[np.invert(single_hit_mask)] = "Outlier"

#         # Data source for the scatter plot
#         scatter_plot_data_source = ColumnDataSource(data=dict(latent_variable_1=latent_variable_1, latent_variable_2=latent_variable_2, data_point_type=data_point_type, scatter_plot_colors=scatter_plot_colors))
        
        # Make scatter plot grid invisible
        scatter_plot.grid.visible = False
        
        # Data source for the scatter plot
        scatter_plot_data_source = ColumnDataSource(data=dict(latent_variable_1=latent_variable_1, latent_variable_2=latent_variable_2, data_point_type=data_point_type, scatter_plot_colors=scatter_plot_colors))

        # Hexbin the scatter plot
        scatter_plot_hexbin_renderer, scatter_plot_hexbin_bins = scatter_plot.hexbin(latent_variable_1, latent_variable_2, size=0.5, hover_color="pink", hover_alpha=0.8)

        # Populate the scatter plot
        #scatter_plot.circle('latent_variable_1', 'latent_variable_2', source=scatter_plot_data_source, fill_alpha=0.6, line_color=None)
        scatter_plot.circle('latent_variable_1', 'latent_variable_2', fill_color='scatter_plot_colors', source=scatter_plot_data_source, fill_alpha=0.6, legend_field="data_point_type", line_color=None,  size=0.5)
        
        # Display data point count for each hex tile in the hexbin scatter plot
        scatter_plot.add_tools(HoverTool(
            tooltips=[("count", "@c")],
            mode="mouse", point_policy="follow_mouse", renderers=[scatter_plot_hexbin_renderer]
        ))
    
    elif scatter_plot_type == "colored_by_training_mask":
        
        # Color the points in the scatter plot according to the training set mask    
        scatter_plot_colors = get_colors_from_discrete_values(training_set_mask, bokeh.palettes.Set1[3][:2])

        # Define the legend 
        data_point_type = np.empty((len(latent_variable_1),), dtype=np.object)
        data_point_type[training_set_mask] = "Train"
        data_point_type[np.invert(training_set_mask)] = "Test"

        # Data source for the scatter plot
        scatter_plot_data_source = ColumnDataSource(data=dict(latent_variable_1=latent_variable_1, latent_variable_2=latent_variable_2, data_point_type=data_point_type, scatter_plot_colors=scatter_plot_colors))

        # Populate the scatter plot
        # Adapted from: https://stackoverflow.com/questions/50083062/how-to-add-legend-inside-pythons-bokeh-circle-plot
        scatter_plot.circle('latent_variable_1', 'latent_variable_2', fill_color='scatter_plot_colors', source=scatter_plot_data_source, fill_alpha=0.6, legend_field="data_point_type", line_color=None)
    
    else:
        # Data source for the scatter plot
        scatter_plot_data_source = ColumnDataSource(data=dict(latent_variable_1=latent_variable_1, latent_variable_2=latent_variable_2))
        
        # Populate the scatter plot
        scatter_plot.circle('latent_variable_1', 'latent_variable_2', source=scatter_plot_data_source, fill_alpha=0.6, line_color=None)        

    # Add axis labels
    if latent_method == "principal_component_analysis":
        scatter_plot.xaxis.axis_label = "PC {}".format(latent_idx_1)
        scatter_plot.yaxis.axis_label = "PC {}".format(latent_idx_2)
    elif latent_method == "diffusion_map":          
        scatter_plot.xaxis.axis_label = "DC {}".format(latent_idx_1)
        scatter_plot.yaxis.axis_label = "DC {}".format(latent_idx_2)
    elif latent_method == "incremental_principal_component_analysis":
        scatter_plot.xaxis.axis_label = "PC {}".format(latent_idx_1)
        scatter_plot.yaxis.axis_label = "PC {}".format(latent_idx_2)
    elif latent_method == "ensemble_pca":
        scatter_plot.xaxis.axis_label = "PC {}".format(latent_idx_1)
        scatter_plot.yaxis.axis_label = "PC {}".format(latent_idx_2)
    elif latent_method == "ensemble_pca_mpi":
        scatter_plot.xaxis.axis_label = "PC {}".format(latent_idx_1)
        scatter_plot.yaxis.axis_label = "PC {}".format(latent_idx_2)
    else:
        raise Exception("Unrecognized latent method. Please choose from: principal_component_analysis, diffusion_map")
    
    # Change the axis label font size
    scatter_plot.xaxis.axis_label_text_font_size = scatter_plot_x_axis_label_text_font_size
    scatter_plot.yaxis.axis_label_text_font_size = scatter_plot_y_axis_label_text_font_size

    # Container to display the static_images
    div_width = int(figure_width * image_plot_image_size_scale_factor)
    div_height = int(figure_height * image_plot_image_size_scale_factor)
    image_plot_div = Div(width=div_width, height=div_height)

    # Build the layout for plots
    layout = row(scatter_plot, image_plot_div)

    # Display the corresponding image when mouse is near a point in scatter plot
    display_image_plot_fn = display_image_plot_using_image_source_location(image_plot_image_source_location, 
                                                        image_plot_div, 
                                                        latent_variable_1, 
                                                        latent_variable_2, 
                                                        image_plot_image_brightness, 
                                                        image_plot_index_label_text_font_size, 
                                                        image_plot_slac_username=image_plot_slac_username, 
                                                        image_plot_slac_dataset_name=image_plot_slac_dataset_name, 
                                                        dataset_file=dataset_file, 
                                                        image_type=image_type)
    scatter_plot.js_on_event(events.MouseMove, display_image_plot_fn)

    # Display the plots
    tic = time.time()
    show(layout)
    toc = time.time()
    print("It takes {:.2f} seconds to display the plots.".format(toc-tic))

def visualize_latent_space_for_incremental_principal_component_analysis(
    dataset_file,
    latent_space_file,
    image_type, 
    latent_idx_1=0, 
    latent_idx_2=1,
    figure_height = 450, 
    figure_width = 450, 
    scatter_plot_x_axis_label_text_font_size='15pt',
    scatter_plot_y_axis_label_text_font_size='15pt', 
    scatter_plot_color_bar_height = 400,
    scatter_plot_color_bar_width = 120,
    scatter_plot_type = "hexbin",
    image_plot_image_size_scale_factor = 0.9,
    image_plot_image_brightness=1.0,
    image_plot_image_source_location=None, 
    image_plot_slac_username=None,
    image_plot_slac_dataset_name=None,
    image_plot_index_label_text_font_size='20px'
    ):
    
    """
    Visualize the latent space from an HDF5 file
    
    :param dataset_file: The path to the HDF5 file, as a str
    :param latent_space_file: The path to the HDF5 file containing the latent space built by incremental_principal_component_analysis, as a str
    :param image_type: A key in the HDF5 file that represents the type of image, as a str
    :param latent_method: A key in the HDF5 file that represents the latent method used to build the latent space, as a str
    :param figure_height: height of the figure containing the plots, as an int
    :param figure_width: width of the figure containing the plots, as an int
    :param scatter_plot_x_axis_label_text_font_size: Font size of the x axis label for the scatter plot, as a str
    :param scatter_plot_y_axis_label_text_font_size: Font size of the y axis label for the scatter plot, as a str
    :param scatter_plot_color_bar_height: height of the color bar, as an int
    :param scatter_plot_color_bar_width: width of the color bar, as an int
    :param image_plot_image_size_scale_factor: scale the image size according to a fraction of the figure size, as a float
    :param image_plot_image_brightness: Brightness of the image displayed in the image plot, as a float
    :param image_plot_image_source_location: An alias for the determining where to display images from, as a str
    :param image_plot_slac_username: A valid SLAC username, as a str
    :param image_plot_slac_dataset_name: The name of the dataset in SLAC Compute Cluster, as a str
    :param image_plot_index_label_text_font_size: Font size of the label for the index below the image plot, as a float
    """
    
    # Load metadata from the HDF5 file
    with h5.File(dataset_file, "r") as dataset_file_handle:        
        if scatter_plot_type == "hexbin_colored_by_single_hit_flag" or scatter_plot_type == "colored_by_single_hit_flag":
            # Load the single-hit mask from the HDF5 file
            single_hit_mask_key = "single_hits_mask"
            single_hit_mask = dataset_file_handle[single_hit_mask_key][:].astype(np.bool)
        
        elif scatter_plot_type == "colored_by_training_mask":
            # Load the training set mask from the HDF5 file
            training_set_mask_key = "training_set_mask"
            training_set_mask = dataset_file_handle[training_set_mask_key][:].astype(np.bool)

        # unclear on how to plot targets
        # labels = np.zeros(len(images))
   
    # Load latent space from the HDF5 file
    tic = time.time()
    with h5.File(latent_space_file, "r") as latent_space_file_handle: 
        latent_space_projection_for_downsampled_diffraction_patterns_seen_thus_far_h5_key = "latent_space_projection_for_downsampled_diffraction_patterns_seen_thus_far"
        latent_variable_1 = latent_space_file_handle[latent_space_projection_for_downsampled_diffraction_patterns_seen_thus_far_h5_key][:, latent_idx_1 - 1]
        latent_variable_2 = latent_space_file_handle[latent_space_projection_for_downsampled_diffraction_patterns_seen_thus_far_h5_key][:, latent_idx_2 - 1] 
#         print(latent_variable_1)
#         print(latent_variable_2)
#         print(latent_variable_1.shape)
#         print(latent_variable_2.shape)
#         latent_variable_1 = np.random.rand(200,)
#         latent_variable_2 = np.random.rand(200,)
        
    toc = time.time()
    print("It takes {:.2f} seconds to load the {} latent vectors and metadata from the HDF5 file.".format(toc-tic, len(latent_variable_1)))
    
    # unclear on how to plot targets
    # n_labels = len(np.unique(labels))
    
    if scatter_plot_type == "hexbin":
        # Scatter plot for the latent vectors
        scatter_plot = figure(match_aspect=True, width=figure_width, height=figure_height, tools="pan,wheel_zoom,box_zoom,reset")
#         scatter_plot = figure(match_aspect=True, width=figure_width, height=figure_height, tools="pan,wheel_zoom,box_zoom,reset", background_fill_color="#440154")

        # Make scatter plot grid invisible
        scatter_plot.grid.visible = False
        
        # Data source for the scatter plot
        scatter_plot_data_source = ColumnDataSource(data=dict(latent_variable_1=latent_variable_1, latent_variable_2=latent_variable_2))

        # Hexbin the scatter plot
        scatter_plot_hexbin_renderer, scatter_plot_hexbin_bins = scatter_plot.hexbin(latent_variable_1, latent_variable_2, size=0.5, hover_color="pink", hover_alpha=0.8)

        # Populate the scatter plot
        #scatter_plot.circle('latent_variable_1', 'latent_variable_2', source=scatter_plot_data_source, fill_alpha=0.6, line_color=None)
        scatter_plot.circle('latent_variable_1', 'latent_variable_2', source=scatter_plot_data_source, color='white', size=0.1)
        
        # Display data point count for each hex tile in the hexbin scatter plot
        scatter_plot.add_tools(HoverTool(
            tooltips=[("count", "@c")],
            mode="mouse", point_policy="follow_mouse", renderers=[scatter_plot_hexbin_renderer]
        ))
    
    elif scatter_plot_type == "colored_by_single_hit_flag":
        #
        single_hit_idx = np.where(single_hit_mask == True)[0]
        outlier_idx = np.where(single_hit_mask == False)[0]
        
        #
        single_hit_labels = np.empty((len(latent_variable_1),), dtype=np.int)
        single_hit_labels[single_hit_idx] = 0
        single_hit_labels[outlier_idx] = 1
        
        # Scatter plot for the latent vectors
#         scatter_plot = figure(match_aspect=True, width=figure_width, height=figure_height, tools="pan,wheel_zoom,box_zoom,reset", background_fill_color="#440154")
        scatter_plot = figure(match_aspect=True, width=figure_width, height=figure_height, tools="pan,wheel_zoom,box_zoom,reset")
        
        # Color the points in the scatter plot according to the single-hit mask
        scatter_plot_colors = get_colors_from_discrete_values(single_hit_labels, [bokeh.palettes.Set1[3][1], bokeh.palettes.Set1[3][0]])

        # Define the legend 
        data_point_type = np.empty((len(latent_variable_1),), dtype=np.object)
        data_point_type[single_hit_idx] = "Single-hit"
        data_point_type[outlier_idx] = "Outlier"

        # Data source for the scatter plot
#         scatter_plot_data_source = ColumnDataSource(data=dict(latent_variable_1=latent_variable_1, latent_variable_2=latent_variable_2, data_point_type=data_point_type, scatter_plot_colors=scatter_plot_colors))
        
        # Make scatter plot grid invisible
        scatter_plot.grid.visible = False
        
        # Data source for the scatter plot
        scatter_plot_data_source = ColumnDataSource(data=dict(latent_variable_1=latent_variable_1, latent_variable_2=latent_variable_2, data_point_type=data_point_type, scatter_plot_colors=scatter_plot_colors))

        # Hexbin the scatter plot
#         scatter_plot_hexbin_renderer, scatter_plot_hexbin_bins = scatter_plot.hexbin(latent_variable_1, latent_variable_2, size=0.5, hover_color="pink", hover_alpha=0.8)

        # Populate the scatter plot
        #scatter_plot.circle('latent_variable_1', 'latent_variable_2', source=scatter_plot_data_source, fill_alpha=0.6, line_color=None)
        scatter_plot.circle('latent_variable_1', 'latent_variable_2', fill_color='scatter_plot_colors', source=scatter_plot_data_source, fill_alpha=0.6, line_color=None, legend_field="data_point_type", size=3.0)
        
#         scatter_plot.circle('latent_variable_1', 'latent_variable_2', fill_color='scatter_plot_colors', source=scatter_plot_data_source, fill_alpha=0.6, legend_field="data_point_type", line_color=None)
        
        # Display data point count for each hex tile in the hexbin scatter plot
#         scatter_plot.add_tools(HoverTool(
#             tooltips=[("count", "@c")],
#             mode="mouse", point_policy="follow_mouse", renderers=[scatter_plot_hexbin_renderer]
#         ))
    
    elif scatter_plot_type == "hexbin_colored_by_single_hit_flag":
        # Scatter plot for the latent vectors
#         scatter_plot = figure(match_aspect=True, width=figure_width, height=figure_height, tools="pan,wheel_zoom,box_zoom,reset", background_fill_color="#440154")
        scatter_plot = figure(match_aspect=True, width=figure_width, height=figure_height, tools="pan,wheel_zoom,box_zoom,reset")
        
        # Color the points in the scatter plot according to the single-hit mask
        scatter_plot_colors = get_colors_from_discrete_values(single_hit_mask, bokeh.palettes.Set1[3][:2])

        # Define the legend 
        data_point_type = np.empty((len(latent_variable_1),), dtype=np.object)
        data_point_type[single_hit_mask] = "Single-hit"
        data_point_type[np.invert(single_hit_mask)] = "Outlier"

        # Data source for the scatter plot
#         scatter_plot_data_source = ColumnDataSource(data=dict(latent_variable_1=latent_variable_1, latent_variable_2=latent_variable_2, data_point_type=data_point_type, scatter_plot_colors=scatter_plot_colors))
        
        # Make scatter plot grid invisible
        scatter_plot.grid.visible = False
        
        # Data source for the scatter plot
        scatter_plot_data_source = ColumnDataSource(data=dict(latent_variable_1=latent_variable_1, latent_variable_2=latent_variable_2, data_point_type=data_point_type, scatter_plot_colors=scatter_plot_colors))

        # Hexbin the scatter plot
        scatter_plot_hexbin_renderer, scatter_plot_hexbin_bins = scatter_plot.hexbin(latent_variable_1, latent_variable_2, size=0.5, hover_color="pink", hover_alpha=0.8)

        # Populate the scatter plot
        #scatter_plot.circle('latent_variable_1', 'latent_variable_2', source=scatter_plot_data_source, fill_alpha=0.6, line_color=None)
        scatter_plot.circle('latent_variable_1', 'latent_variable_2', fill_color='scatter_plot_colors', source=scatter_plot_data_source, fill_alpha=0.6, legend_field="data_point_type", line_color=None,  size=0.5)
        
#         scatter_plot.circle('latent_variable_1', 'latent_variable_2', fill_color='scatter_plot_colors', source=scatter_plot_data_source, fill_alpha=0.6, legend_field="data_point_type", line_color=None)
        
        # Display data point count for each hex tile in the hexbin scatter plot
        scatter_plot.add_tools(HoverTool(
            tooltips=[("count", "@c")],
            mode="mouse", point_policy="follow_mouse", renderers=[scatter_plot_hexbin_renderer]
        ))
    
    elif scatter_plot_type == "colored_by_training_mask":
        # Scatter plot for the latent vectors
        scatter_plot = figure(match_aspect=True, width=figure_width, height=figure_height, tools="pan,wheel_zoom,box_zoom,reset", background_fill_color="#440154")
        
        # Color the points in the scatter plot according to the training set mask    
        scatter_plot_colors = get_colors_from_discrete_values(training_set_mask, bokeh.palettes.Set1[3][:2])

        # Define the legend 
        data_point_type = np.empty((len(latent_variable_1),), dtype=np.object)
        data_point_type[training_set_mask] = "Train"
        data_point_type[np.invert(training_set_mask)] = "Test"

        # Data source for the scatter plot
        scatter_plot_data_source = ColumnDataSource(data=dict(latent_variable_1=latent_variable_1, latent_variable_2=latent_variable_2, data_point_type=data_point_type, scatter_plot_colors=scatter_plot_colors))

        # Populate the scatter plot
        # Adapted from: https://stackoverflow.com/questions/50083062/how-to-add-legend-inside-pythons-bokeh-circle-plot
        scatter_plot.circle('latent_variable_1', 'latent_variable_2', fill_color='scatter_plot_colors', source=scatter_plot_data_source, fill_alpha=0.6, legend_field="data_point_type", line_color=None)
    
    else:
        # Scatter plot for the latent vectors
        scatter_plot = figure(match_aspect=True, width=figure_width, height=figure_height, tools="pan,wheel_zoom,box_zoom,reset")

        # Data source for the scatter plot
        scatter_plot_data_source = ColumnDataSource(data=dict(latent_variable_1=latent_variable_1, latent_variable_2=latent_variable_2))
        
        # Populate the scatter plot
        scatter_plot.circle('latent_variable_1', 'latent_variable_2', source=scatter_plot_data_source, fill_alpha=0.6, line_color=None)        

    # Add axis labels
    scatter_plot.xaxis.axis_label = "PC {}".format(latent_idx_1)
    scatter_plot.yaxis.axis_label = "PC {}".format(latent_idx_2)
    
    # Change the axis label font size
    scatter_plot.xaxis.axis_label_text_font_size = scatter_plot_x_axis_label_text_font_size
    scatter_plot.yaxis.axis_label_text_font_size = scatter_plot_y_axis_label_text_font_size

    # Container to display the static_images
    div_width = int(figure_width * image_plot_image_size_scale_factor)
    div_height = int(figure_height * image_plot_image_size_scale_factor)
    image_plot_div = Div(width=div_width, height=div_height)

    # Build the layout for plots
    layout = row(scatter_plot, image_plot_div)

    # Display the corresponding image when mouse is near a point in scatter plot
    display_image_plot_fn = display_image_plot_using_image_source_location(image_plot_image_source_location, 
                                                        image_plot_div, 
                                                        latent_variable_1, 
                                                        latent_variable_2, 
                                                        image_plot_image_brightness, 
                                                        image_plot_index_label_text_font_size, 
                                                        image_plot_slac_username=image_plot_slac_username, 
                                                        image_plot_slac_dataset_name=image_plot_slac_dataset_name, 
                                                        dataset_file=dataset_file, 
                                                        image_type=image_type)
    scatter_plot.js_on_event(events.MouseMove, display_image_plot_fn)

    # Display the plots
    tic = time.time()
    show(layout)
    toc = time.time()
    print("It takes {:.2f} seconds to display the plots.".format(toc-tic))

def visualize_latent_space_for_incremental_principal_component_analysis_predicted_by_elliptic_envelope_outlier_prediction(
    dataset_file,
    latent_space_file,
    outlier_prediction_mask_file,
    image_type, 
    latent_idx_1=0, 
    latent_idx_2=1,
    figure_height = 450, 
    figure_width = 450, 
    scatter_plot_x_axis_label_text_font_size='15pt',
    scatter_plot_y_axis_label_text_font_size='15pt', 
    scatter_plot_color_bar_height = 400,
    scatter_plot_color_bar_width = 120,
    scatter_plot_type = "hexbin",
    image_plot_image_size_scale_factor = 0.9,
    image_plot_image_brightness=1.0,
    image_plot_image_source_location=None, 
    image_plot_slac_username=None,
    image_plot_slac_dataset_name=None,
    image_plot_index_label_text_font_size='20px'
    ):
    
    """
    Visualize the latent space from an HDF5 file
    
    :param dataset_file: The path to the HDF5 file, as a str
    :param latent_space_file: The path to the HDF5 file containing the latent space built by incremental_principal_component_analysis, as a str
    :param outlier_prediction_mask_file: The path to the HDF5 file containing the outlier prediction mask obtained from elliptic_envelope, as a str
    :param image_type: A key in the HDF5 file that represents the type of image, as a str
    :param latent_method: A key in the HDF5 file that represents the latent method used to build the latent space, as a str
    :param figure_height: height of the figure containing the plots, as an int
    :param figure_width: width of the figure containing the plots, as an int
    :param scatter_plot_x_axis_label_text_font_size: Font size of the x axis label for the scatter plot, as a str
    :param scatter_plot_y_axis_label_text_font_size: Font size of the y axis label for the scatter plot, as a str
    :param scatter_plot_color_bar_height: height of the color bar, as an int
    :param scatter_plot_color_bar_width: width of the color bar, as an int
    :param image_plot_image_size_scale_factor: scale the image size according to a fraction of the figure size, as a float
    :param image_plot_image_brightness: Brightness of the image displayed in the image plot, as a float
    :param image_plot_image_source_location: An alias for the determining where to display images from, as a str
    :param image_plot_slac_username: A valid SLAC username, as a str
    :param image_plot_slac_dataset_name: The name of the dataset in SLAC Compute Cluster, as a str
    :param image_plot_index_label_text_font_size: Font size of the label for the index below the image plot, as a float
    """
        
    # Load the single-hit mask from the HDF5 file
    with h5.File(dataset_file, "r") as dataset_file_handle:        
        if scatter_plot_type == "colored_by_outlier_prediction_label" or scatter_plot_type == "hexbin_colored_by_outlier_prediction_label":
            # Load the single-hit mask from the HDF5 file
            single_hit_mask_key = "single_hits_mask"
            single_hit_mask = dataset_file_handle[single_hit_mask_key][:].astype(np.bool)
   
    # Load latent space from the HDF5 file
    tic = time.time()
    with h5.File(latent_space_file, "r") as latent_space_file_handle: 
        latent_space_projection_for_downsampled_diffraction_patterns_seen_thus_far_h5_key = "latent_space_projection_for_downsampled_diffraction_patterns_seen_thus_far"
        latent_variable_1 = latent_space_file_handle[latent_space_projection_for_downsampled_diffraction_patterns_seen_thus_far_h5_key][:, latent_idx_1 - 1]
        latent_variable_2 = latent_space_file_handle[latent_space_projection_for_downsampled_diffraction_patterns_seen_thus_far_h5_key][:, latent_idx_2 - 1] 
        
    toc = time.time()
    print("It takes {:.2f} seconds to load the {} latent vectors and metadata from the HDF5 file.".format(toc-tic, len(latent_variable_1)))
    
    # Load the outlier prediction mask from the HDF5 file
    with h5.File(outlier_prediction_mask_file, "r") as outlier_prediction_mask_file_handle: 
        elliptic_envelope_outlier_prediction_mask_for_downsampled_diffraction_patterns_seen_thus_far_h5_key = "elliptic_envelope_outlier_prediction_mask_for_downsampled_diffraction_patterns_seen_thus_far"
        outlier_prediction_mask = outlier_prediction_mask_file_handle[elliptic_envelope_outlier_prediction_mask_for_downsampled_diffraction_patterns_seen_thus_far_h5_key][:]
    
    if scatter_plot_type == "colored_by_outlier_prediction_label":
        
        # Prepare the outlier prediction labels for the scatter plot colors and legend labels
        true_positive_mask = np.bitwise_and(single_hit_mask, outlier_prediction_mask)
        false_positive_mask = np.bitwise_and(np.bitwise_not(single_hit_mask), outlier_prediction_mask)
        true_negative_mask = np.bitwise_and(np.bitwise_not(single_hit_mask), np.bitwise_not(outlier_prediction_mask))
        false_negative_mask = np.bitwise_and(single_hit_mask, np.bitwise_not(outlier_prediction_mask))
        
        true_positive_idx = np.where(true_positive_mask == True)[0]
        false_positive_idx = np.where(false_positive_mask == True)[0]
        true_negative_idx = np.where(true_negative_mask == True)[0]
        false_negative_idx = np.where(false_negative_mask == True)[0]
        
        # Scatter plot for the latent vectors
        scatter_plot = figure(match_aspect=True, width=figure_width, height=figure_height, tools="pan,wheel_zoom,box_zoom,reset")
        
        # Color the points in the scatter plot according to the single-hit mask
        outlier_prediction_labels = np.empty((len(latent_variable_1),), dtype=np.int)
        outlier_prediction_labels[true_positive_idx] = 0
        outlier_prediction_labels[false_positive_idx] = 1
        outlier_prediction_labels[true_negative_idx] = 2
        outlier_prediction_labels[false_negative_idx] = 3
        
        scatter_plot_colors = get_colors_from_discrete_values(outlier_prediction_labels, bokeh.palettes.Set1[4])

        # Define the legend 
        data_point_type = np.empty((len(latent_variable_1),), dtype=np.object)
        data_point_type[true_positive_idx] = "True positive"        
        data_point_type[false_positive_idx] = "False positive"
        data_point_type[true_negative_idx] = "True negative"
        data_point_type[false_negative_idx] = "False negative"

        # Make scatter plot grid invisible
        scatter_plot.grid.visible = False
        
        # Data source for the scatter plot
        scatter_plot_data_source = ColumnDataSource(data=dict(latent_variable_1=latent_variable_1, latent_variable_2=latent_variable_2, data_point_type=data_point_type, scatter_plot_colors=scatter_plot_colors))

        # Populate the scatter plot
        scatter_plot.circle('latent_variable_1', 'latent_variable_2', fill_color='scatter_plot_colors', source=scatter_plot_data_source, fill_alpha=0.6, legend_field="data_point_type", line_color=None,  size=0.1)
    
    elif scatter_plot_type == "hexbin_colored_by_outlier_prediction_label":
        
        # Prepare the outlier prediction labels for the scatter plot colors and legend labels
        true_positive_mask = np.bitwise_and(single_hit_mask, outlier_prediction_mask)
        false_positive_mask = np.bitwise_and(np.bitwise_not(single_hit_mask), outlier_prediction_mask)
        true_negative_mask = np.bitwise_and(np.bitwise_not(single_hit_mask), np.bitwise_not(outlier_prediction_mask))
        false_negative_mask = np.bitwise_and(single_hit_mask, np.bitwise_not(outlier_prediction_mask))
        
        true_positive_idx = np.where(true_positive_mask == True)[0]
        false_positive_idx = np.where(false_positive_mask == True)[0]
        true_negative_idx = np.where(true_negative_mask == True)[0]
        false_negative_idx = np.where(false_negative_mask == True)[0]
        
        # Scatter plot for the latent vectors
        scatter_plot = figure(match_aspect=True, width=figure_width, height=figure_height, tools="pan,wheel_zoom,box_zoom,reset")
        
        # Color the points in the scatter plot according to the single-hit mask
        outlier_prediction_labels = np.empty((len(latent_variable_1),), dtype=np.int)
        outlier_prediction_labels[true_positive_idx] = 0
        outlier_prediction_labels[false_positive_idx] = 1
        outlier_prediction_labels[true_negative_idx] = 2
        outlier_prediction_labels[false_negative_idx] = 3
        
        scatter_plot_colors = get_colors_from_discrete_values(outlier_prediction_labels, bokeh.palettes.Set1[4])

        # Define the legend 
        data_point_type = np.empty((len(latent_variable_1),), dtype=np.object)
        data_point_type[true_positive_idx] = "True positive"        
        data_point_type[false_positive_idx] = "False positive"
        data_point_type[true_negative_idx] = "True negative"
        data_point_type[false_negative_idx] = "False negative"

        # Make scatter plot grid invisible
        scatter_plot.grid.visible = False
        
        # Data source for the scatter plot
        scatter_plot_data_source = ColumnDataSource(data=dict(latent_variable_1=latent_variable_1, latent_variable_2=latent_variable_2, data_point_type=data_point_type, scatter_plot_colors=scatter_plot_colors))

        # Hexbin the scatter plot
        scatter_plot_hexbin_renderer, scatter_plot_hexbin_bins = scatter_plot.hexbin(latent_variable_1, latent_variable_2, size=0.5, hover_color="pink", hover_alpha=0.8)

        # Populate the scatter plot
        scatter_plot.circle('latent_variable_1', 'latent_variable_2', fill_color='scatter_plot_colors', source=scatter_plot_data_source, fill_alpha=0.6, legend_field="data_point_type", line_color=None,  size=0.1)
        
        # Display data point count for each hex tile in the hexbin scatter plot
        scatter_plot.add_tools(HoverTool(
            tooltips=[("count", "@c")],
            mode="mouse", point_policy="follow_mouse", renderers=[scatter_plot_hexbin_renderer]
        ))

    # Add axis labels
    scatter_plot.xaxis.axis_label = "PC {}".format(latent_idx_1)
    scatter_plot.yaxis.axis_label = "PC {}".format(latent_idx_2)
    
    # Change the axis label font size
    scatter_plot.xaxis.axis_label_text_font_size = scatter_plot_x_axis_label_text_font_size
    scatter_plot.yaxis.axis_label_text_font_size = scatter_plot_y_axis_label_text_font_size

    # Container to display the static_images
    div_width = int(figure_width * image_plot_image_size_scale_factor)
    div_height = int(figure_height * image_plot_image_size_scale_factor)
    image_plot_div = Div(width=div_width, height=div_height)

    # Build the layout for plots
    layout = row(scatter_plot, image_plot_div)

    # Display the corresponding image when mouse is near a point in scatter plot
    display_image_plot_fn = display_image_plot_using_image_source_location(image_plot_image_source_location, 
                                                        image_plot_div, 
                                                        latent_variable_1, 
                                                        latent_variable_2, 
                                                        image_plot_image_brightness, 
                                                        image_plot_index_label_text_font_size, 
                                                        image_plot_slac_username=image_plot_slac_username, 
                                                        image_plot_slac_dataset_name=image_plot_slac_dataset_name, 
                                                        dataset_file=dataset_file, 
                                                        image_type=image_type)
    scatter_plot.js_on_event(events.MouseMove, display_image_plot_fn)

    # Display the plots
    tic = time.time()
    show(layout)
    toc = time.time()
    print("It takes {:.2f} seconds to display the plots.".format(toc-tic))
