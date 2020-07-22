import time

import numpy as np

from bokeh import events
import bokeh.io
from bokeh.io import output_notebook
from bokeh.io import show
from bokeh.layouts import row
from bokeh.models import CustomJS, Div, LinearColorMapper, ColorBar, Range1d

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


def get_color(x, color_bar_palette, vmin, vmax):
    n = len(color_bar_palette)
    return color_bar_palette[int((x - vmin) / (vmax - vmin) * n)]

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

def get_elevation_azimuth_rotation_angles_from_orientations(orientations):
    x = np.zeros((orientations.shape[0],))
    y = np.zeros((orientations.shape[0],))
    z = np.zeros((orientations.shape[0],))
    
    for orientation_idx, orientation in enumerate(orientations):
        axis, theta = angle_axis_representation(orientation)
        azimuth, elevation = azimuth_elevation_representation(axis)

        x[orientation_idx] = azimuth
        y[orientation_idx] = elevation
        z[orientation_idx] = theta
    
    return x, y, z

# https://sscc.nimh.nih.gov/pub/dist/bin/linux_gcc32/meica.libs/nibabel/quaternions.py
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
    s = 2.0/Nq
    X = x*s
    Y = y*s
    Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z
    return np.array(
           [[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
            [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
            [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]])

def get_3d_rotation_matrices_from_quaternions(quats):    
    rotation_matrices_3d = np.zeros((quats.shape[0], 3, 3))
    
    for idx, quat in enumerate(quats):
        rotation_matrix_3d = quat2mat(quat)
        """
        Antoine:
        
        “Orientations” are weirdly defined because they are the relative orientation of the particle with regard to the beam.

        In a sense, the beam is always on the same Z axis and the particles have different orientations.

        But, when work with a diffraction volume that you want to slice, it is more convenient to think as the particle as the base of your coordinate system, and the beam comes with some orientation.

        So, when you are slicing, are you giving the orientation of the particle or of the beam?

        If you only use one, it doesn’t really matter.

        But it matters if you want your data to be consistent…
        New

        So, to come back to the question. The slicing function takes one orientation. This orientation is assumed to be the beam orientation. Now, if you want it to represent the particle orientation, you need to inverse it.

        That way, if you orient the particle a certain way, that corresponds to taking the slices with the same orientation (inversed).
        """
        
        rotation_matrices_3d[idx] = rotation_matrix_3d
    
    return rotation_matrices_3d

# # https://github.com/fredericpoitevin/pysingfel/blob/master/pysingfel/geometry/convert.py
# # Converters between different descriptions of 3D rotation.
# def angle_axis_to_rot3d(axis, theta):
#     """
#     Convert rotation with angle theta around a certain axis to a rotation matrix in 3D.
#     :param axis: A numpy array for the rotation axis.
#         Axis names 'x', 'y', and 'z' are also accepted.
#     :param theta: Rotation angle.
#     :return:
#     """
#     if isinstance(axis, string_types):
#         axis = axis.lower()
#         if axis == 'x':
#             axis = np.array([1., 0., 0.])
#         elif axis == 'y':
#             axis = np.array([0., 1., 0.])
#         elif axis == 'z':
#             axis = np.array([0., 0., 1.])
#         else:
#             raise ValueError("Axis should be 'x', 'y', 'z' or a 3D vector.")
#     elif len(axis) != 3:
#         raise ValueError("Axis should be 'x', 'y', 'z' or a 3D vector.")
#     axis = axis.astype(float)
#     axis /= np.linalg.norm(axis)
#     a = axis[0]
#     b = axis[1]
#     c = axis[2]
#     cos_theta = np.cos(theta)
#     bracket = 1 - cos_theta
#     a_bracket = a * bracket
#     b_bracket = b * bracket
#     c_bracket = c * bracket
#     sin_theta = np.sin(theta)
#     a_sin_theta = a * sin_theta
#     b_sin_theta = b * sin_theta
#     c_sin_theta = c * sin_theta
#     rot3d = np.array(
#         [[a * a_bracket + cos_theta, a * b_bracket - c_sin_theta, a * c_bracket + b_sin_theta],
#          [b * a_bracket + c_sin_theta, b * b_bracket + cos_theta, b * c_bracket - a_sin_theta],
#          [c * a_bracket - b_sin_theta, c * b_bracket + a_sin_theta, c * c_bracket + cos_theta]])
#     return rot3d

# # /reg/neh/home/dujardin/pysingfel/examples/scripts/gui.py
# def build_3d_rotation_matrix_from_azimuth_elevation(azim, elev):
#     axis_azim = np.array([0., 0., 1.]) # +z
#     axis_elev = np.array([0., -1., 0.]) # -y
#     rot_azim = angle_axis_to_rot3d(axis_azim, azim) # counter-clockwise about +z
#     rot_elev = angle_axis_to_rot3d(axis_elev, elev) # counter-clockwise about -y
#     rot = np.matmul(rot_elev, rot_azim)
#     return rot

# def get_3d_rotation_matrices_from_azimuth_elevation_coordinates(azims, elevs):
#     rotation_matrices_3d = np.zeros((azims.shape[0], 3, 3))
    
#     for idx, (azim, elev) in enumerate(zip(azims, elevs)):
#         rotation_matrix_3d = build_3d_rotation_matrix_from_azimuth_elevation(azim, elev)
#         rotation_matrices_3d[idx] = rotation_matrix_3d
    
#     return rotation_matrices_3d

def get_colors_from_rotation_angles(rotation_angles, color_bar_palette=bokeh.palettes.plasma(256)):
    color_bar_vmin = 0.0
    color_bar_vmax = 2*np.pi
        
    colors = []
    for rotation_angle in rotation_angles:
        color = get_color(rotation_angle, color_bar_palette, color_bar_vmin, color_bar_vmax)
        colors.append(color)
    
    color_mapper = LinearColorMapper(palette=color_bar_palette, low=color_bar_vmin, high=color_bar_vmax)
    return colors, color_mapper

def gnp2im(image_np, bit_depth_scale_factor):
    """
    Converts an image stored as a 2-D grayscale Numpy array into a PIL image.
    
    Assumes values in image_np are between [0, 1].
    """
    return Image.fromarray((image_np * bit_depth_scale_factor).astype(np.uint8), mode='L')

def to_base64(png):
    return "data:image/png;base64," + base64.b64encode(png).decode("utf-8")

def get_images(data, bit_depth_scale_factor=255):
    images = []
    for gnp in data:
        im = gnp2im(gnp, bit_depth_scale_factor)
        memout = BytesIO()
        im.save(memout, format='png')
        images.append(to_base64(memout.getvalue()))
    return images

def display_event(div, x, y, static_images, image_brightness, attributes=[], style = 'font-size:20px;text-align:center'):
    "Build a suitable CustomJS to display the current event in the div model."
    return CustomJS(args=dict(div=div, x=x, y=y, static_images=static_images, image_brightness=image_brightness), code="""
        var attrs = %s; var args = []; var n = x.length;
        
        var test_x;
        var test_y;
        for (var i = 0; i < attrs.length; i++) {
            if (attrs[i] == 'x') {
                test_x = Number(cb_obj[attrs[i]]);
            }
            
            if (attrs[i] == 'y') {
                test_y = Number(cb_obj[attrs[i]]);
            }
        }
    
        var minDiffIndex = -1;
        var minDiff = 99999;
        var squareDiff;
        for (var i = 0; i < n; i++) {
            squareDiff = (test_x - x[i]) ** 2 + (test_y - y[i]) ** 2;
            if (squareDiff < minDiff) {
                minDiff = squareDiff;
                minDiffIndex = i;
            }
        }
        
        var img_tag_attrs = "style='filter: brightness(" + image_brightness + ");'";
        var img_tag = "<div><img src='" + static_images[minDiffIndex] + "' " + img_tag_attrs + "></img></div>";
        //var line = img_tag + "\\n";
        var line = img_tag + "<p style=%r>" + (minDiffIndex+1) + "</p>" + "\\n";
        div.text = "";
        var text = div.text.concat(line);
        var lines = text.split("\\n")
        if (lines.length > 35)
            lines.shift();
        div.text = lines.join("\\n");
    """ % (attributes, style))

# optimize image plotting
def display_event2(img, x, y, images, attributes=[]):
    "Build a suitable CustomJS to display the current event in the image plot."
    return CustomJS(args=dict(img=img, x=x, y=y, images=images, attributes=attributes), code="""
        var attrs = %s; var args = []; var n = x.length;
        
        var test_x;
        var test_y;
        for (var i = 0; i < attrs.length; i++) {
            if (attrs[i] == 'x') {
                test_x = Number(cb_obj[attrs[i]]);
            }
            
            if (attrs[i] == 'y') {
                test_y = Number(cb_obj[attrs[i]]);
            }
        }
    
        var minDiffIndex = -1;
        var minDiff = 99999;
        var squareDiff;
        for (var i = 0; i < n; i++) {
            squareDiff = (test_x - x[i]) ** 2 + (test_y - y[i]) ** 2;
            if (squareDiff < minDiff) {
                minDiff = squareDiff;
                minDiffIndex = i;
            }
        }
        
        if (minDiffIndex > -1) {
            // identify the image that corresponds to the nearest (x, y) point
            var image = images[minDiffIndex];

            // plot the image
            im.data['values'][0] = image;
            im.change.emit();
        }
    """ % (attributes,))

def display_real2d_plot(real2d, x, y, rotation_matrices, atomic_coordinates, attributes=[]):
    "Build a suitable CustomJS to display the current event in the real2d_plot scatter plot."
    return CustomJS(args=dict(real2d=real2d, x=x, y=y, rotation_matrices=rotation_matrices, atomic_coordinates=atomic_coordinates), code="""
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
    
        var attrs = %s; var args = []; var n = x.length;
        
        var test_x;
        var test_y;
        for (var i = 0; i < attrs.length; i++) {
            if (attrs[i] == 'x') {
                test_x = Number(cb_obj[attrs[i]]);
            }
            
            if (attrs[i] == 'y') {
                test_y = Number(cb_obj[attrs[i]]);
            }
        }
    
        var minDiffIndex = -1;
        var minDiff = 99999;
        var squareDiff;
        for (var i = 0; i < n; i++) {
            squareDiff = (test_x - x[i]) ** 2 + (test_y - y[i]) ** 2;
            if (squareDiff < minDiff) {
                minDiff = squareDiff;
                minDiffIndex = i;
            }
        }
                
        if (minDiffIndex > -1) {
            // identify the rotation matrix that corresponds to the nearest (azimuth, elevation) point
            var rotation_matrix = rotation_matrices[minDiffIndex];

            // rotate atomic_coordinates using rotation_matrix
            // Adapted from: /reg/neh/home/dujardin/pysingfel/examples/scripts/gui.py
            var rotated_atomic_coordinates = transposeSecondArgThenMultiplyMatrices(rotation_matrix, atomic_coordinates);

            // scatter plot the rotated_atomic_coordinates
            // Adapted from: /reg/neh/home/dujardin/pysingfel/examples/scripts/gui.py
            real2d.data['x'] = rotated_atomic_coordinates[1];
            real2d.data['y'] = rotated_atomic_coordinates[0];
            real2d.change.emit();
        }
    """ % (attributes))    

# optimize orientation plotting
def display_real2d_plot2(real2d, x, y, quaternions, atomic_coordinates, attributes=[]):
    "Build a suitable CustomJS to display the current event in the real2d_plot scatter plot."
    return CustomJS(args=dict(real2d=real2d, x=x, y=y, quaternions=quaternions, atomic_coordinates=atomic_coordinates), code="""
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
    
        var attrs = %s; var args = []; var n = x.length;
        
        var test_x;
        var test_y;
        for (var i = 0; i < attrs.length; i++) {
            if (attrs[i] == 'x') {
                test_x = Number(cb_obj[attrs[i]]);
            }
            
            if (attrs[i] == 'y') {
                test_y = Number(cb_obj[attrs[i]]);
            }
        }
    
        var minDiffIndex = -1;
        var minDiff = 99999;
        var squareDiff;
        for (var i = 0; i < n; i++) {
            squareDiff = (test_x - x[i]) ** 2 + (test_y - y[i]) ** 2;
            if (squareDiff < minDiff) {
                minDiff = squareDiff;
                minDiffIndex = i;
            }
        }
                
        if (minDiffIndex > -1) {
            // identify the quaternion that corresponds to the nearest (azimuth, elevation) point
            var quaternion = quaternions[minDiffIndex];
            
            // compute the rotation matrix that corresponds to the quaternion
            var rotation_matrix = quat2mat(quaternion);

            // rotate atomic_coordinates using rotation_matrix
            // Adapted from: /reg/neh/home/dujardin/pysingfel/examples/scripts/gui.py
            var rotated_atomic_coordinates = transposeSecondArgThenMultiplyMatrices(rotation_matrix, atomic_coordinates);

            // scatter plot the rotated_atomic_coordinates
            // Adapted from: /reg/neh/home/dujardin/pysingfel/examples/scripts/gui.py
            real2d.data['x'] = rotated_atomic_coordinates[1];
            real2d.data['y'] = rotated_atomic_coordinates[0];
            real2d.change.emit();
        }
    """ % (attributes)) 

def visualize(dataset_file, image_type, latent_method, 
              latent_idx_1=None, latent_idx_2=None, 
              particle_property=None,
              particle_plot_x_axis_label_text_font_size='20pt', particle_plot_y_axis_label_text_font_size='20pt',
              real2d_plot_x_lower=-2e-9, real2d_plot_x_upper=2e-9, real2d_plot_y_lower=-2e-9, real2d_plot_y_upper=2e-9,
              x_axis_label_text_font_size='20pt', y_axis_label_text_font_size='20pt', 
              index_label_text_font_size='20px',
              image_brightness=1.0, 
              figure_width = 450, figure_height = 450, 
              image_size_scale_factor = 0.9, 
              color_bar_height = 400, color_bar_width = 120):
    
    tic = time.time()
    with h5.File(dataset_file, "r") as dataset_file_handle:
        images = dataset_file_handle[image_type][:]
        latent = dataset_file_handle[latent_method][:]
        
        if latent_method == "orientations":
            atomic_coordinates = dataset_file_handle[particle_property][:]
        
        # unclear on how to plot targets
        # labels = np.zeros(len(images)) 

    toc = time.time()
    print("It takes {:.2f} seconds to load the data.".format(toc-tic))

    # unclear on how to plot targets
    # n_labels = len(np.unique(labels))
        
    tic = time.time()
    static_images = get_images(images)
    toc = time.time()
    print("It takes {:.2f} seconds to generate static images in memory.".format(toc-tic))
    
    scatter_plot = figure(width=figure_width, height=figure_height, tools="pan,wheel_zoom,box_zoom,reset")
    scatter_plot.xaxis.axis_label_text_font_size = x_axis_label_text_font_size
    scatter_plot.yaxis.axis_label_text_font_size = y_axis_label_text_font_size

    # Container to display the static_images
    div_width = int(figure_width * image_size_scale_factor)
    div_height = int(figure_height * image_size_scale_factor)
    div = Div(width=div_width, height=div_height)
    
    # optimize image plotting
    # # https://stackoverflow.com/questions/33789011/bokeh-implementing-custom-javascript-in-an-image-plot
    # img_plot = figure(width=figure_width, height=figure_height, tools="pan,wheel_zoom,box_zoom,reset")
    # x = np.linspace(0, 10, 1024)
    # y = np.linspace(0, 10, 1040)
    # xx, yy = np.meshgrid(x, y)
    # d = np.sin(xx) * np.cos(yy)
    # img_plot_data_source = ColumnDataSource(data=dict(values=[d], x_vals=[x], y_vals=[y]))
    # img_plot.image(image='values', x=0, y=0, dw=10, dh=10, palette="Spectral11", source=img_plot_data_source)
    
    point_attributes = ['x', 'y']

    if latent_method == "principal_component_analysis":
        x = latent[:, latent_idx_1]
        y = latent[:, latent_idx_2]   
        
        scatter_plot.scatter(x, y, fill_alpha=0.6)
        scatter_plot.xaxis.axis_label = "PC {}".format(latent_idx_1 + 1)
        scatter_plot.yaxis.axis_label = "PC {}".format(latent_idx_2 + 1)

        layout = row(scatter_plot, div)

        # optimize image plotting
        # layout = row(p, img_plot)
    elif latent_method == "diffusion_map":  
        x = latent[:, latent_idx_1]
        y = latent[:, latent_idx_2]   
        
        scatter_plot.scatter(x, y, fill_alpha=0.6)
        scatter_plot.xaxis.axis_label = "DC {}".format(latent_idx_1 + 1)
        scatter_plot.yaxis.axis_label = "DC {}".format(latent_idx_2 + 1)

        layout = row(scatter_plot, div)

        # optimize image plotting
        # layout = row(p, img_plot)
    elif latent_method == "orientations":   
        # quaternion -> angle-axis -> (azimuth, elevation), rotation angle about axis
        x, y, rotation_angles = get_elevation_azimuth_rotation_angles_from_orientations(latent)

        # testing whether the following modification will speed up the code
        
        # quaternion -> 3d rotation matrix
        # rotation_matrices = get_3d_rotation_matrices_from_quaternions(latent)
        
        # previous method
        #rotation_matrices = get_3d_rotation_matrices_from_azimuth_elevation_coordinates(x, y)
        
        colors, color_mapper = get_colors_from_rotation_angles(rotation_angles)
        
        real2d_plot = figure(width=figure_width, height=figure_height, tools="pan,wheel_zoom,box_zoom,reset")
        real2d_plot.xaxis.axis_label_text_font_size = particle_plot_x_axis_label_text_font_size
        real2d_plot.yaxis.axis_label_text_font_size = particle_plot_y_axis_label_text_font_size
        
        real2d_plot_data_source = ColumnDataSource({'x': [], 'y': []})
        
        real2d_plot.scatter('x', 'y', source=real2d_plot_data_source)
        
        real2d_plot.xaxis.axis_label = "Y"
        real2d_plot.yaxis.axis_label = "X"
        
        real2d_plot.x_range = Range1d(real2d_plot_x_lower, real2d_plot_x_upper)
        real2d_plot.y_range = Range1d(real2d_plot_y_lower, real2d_plot_y_upper)
                
        scatter_plot.scatter(x, y, fill_alpha=0.6, fill_color=colors, line_color=None)
        
        scatter_plot.xaxis.axis_label = "Azimuth"
        scatter_plot.yaxis.axis_label = "Elevation"
        
        scatter_plot_x_lower, scatter_plot_x_upper, scatter_plot_y_lower, scatter_plot_y_upper = -np.pi, np.pi, -np.pi / 2, np.pi / 2
        scatter_plot.x_range = Range1d(scatter_plot_x_lower, scatter_plot_x_upper)
        scatter_plot.y_range = Range1d(scatter_plot_y_lower, scatter_plot_y_upper)
        
        color_bar_plot = figure(title="Rotation", title_location="right", 
                                height=color_bar_height, width=color_bar_width, 
                                min_border=0, 
                                outline_line_color=None,
                                toolbar_location=None)
        
        color_bar_plot.title.align = "center"
        color_bar_plot.title.text_font_size = "12pt"
        color_bar_plot.scatter([], []) # removes Bokeh warning 1000 (MISSING_RENDERERS)
        
        color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12, border_line_color=None, location=(0,0))
        color_bar_plot.add_layout(color_bar, "right")
        
        layout = row(real2d_plot, scatter_plot, color_bar_plot, img_plot)
        
        # scatter_plot.js_on_event(events.MouseMove, display_real2d_plot(real2d_plot_data_source, x, y, rotation_matrices, atomic_coordinates, attributes=point_attributes))
        
        # optimize orientation plotting
        # testing whether the following modification will speed up the code
        scatter_plot.js_on_event(events.MouseMove, display_real2d_plot2(real2d_plot_data_source, x, y, latent, atomic_coordinates, attributes=point_attributes))
    else:
        raise Exception("Unrecognized latent method. Please choose from: principal_component_analysis, diffusion_map")
        
    scatter_plot.js_on_event(events.MouseMove, display_event(div, x, y, static_images, image_brightness, attributes=point_attributes, style='font-size:{};text-align:center'.format(index_label_text_font_size)))
    
    # optimize image plotting
    # scatter_plot.js_on_event(events.MouseMove, display_event2(img_plot_data_source, x, y, images, attributes=point_attributes))

    tic = time.time()
    show(layout)
    toc = time.time()
    print("It takes {:.2f} seconds to display the data.".format(toc-tic))
        
def output_notebook():
    bokeh.io.output_notebook()
