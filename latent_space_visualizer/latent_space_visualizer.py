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


"""
Function for displaying Bokeh plots in Jupyter notebook
"""
    
def output_notebook():
    bokeh.io.output_notebook()


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

"""
Functions for coloring points in the scatter plot according to rotation angle
"""

def get_color(x, color_bar_palette, vmin, vmax):
    n = len(color_bar_palette)
    return color_bar_palette[int((x - vmin) / (vmax - vmin) * n)]

def get_colors_from_rotation_angles(rotation_angles, color_bar_palette=bokeh.palettes.plasma(256)):
    color_bar_vmin = 0.0
    color_bar_vmax = 2*np.pi
        
    colors = []
    for rotation_angle in rotation_angles:
        color = get_color(rotation_angle, color_bar_palette, color_bar_vmin, color_bar_vmax)
        colors.append(color)
    
    color_mapper = LinearColorMapper(palette=color_bar_palette, low=color_bar_vmin, high=color_bar_vmax)
    return colors, color_mapper

"""
Function for displaying real-space XY projection plot using quaternions and atomic coordinates
"""

def display_real2d_plot_using_quaternions(real2d, x, y, quaternions, atomic_coordinates, attributes=[]):
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
            real2d.data['x'] = rotated_atomic_coordinates[1]; // x-axis represents the y-coordinate
            real2d.data['y'] = rotated_atomic_coordinates[0]; // y-axis represents the x-coordinate
            real2d.change.emit();
        }
    """ % (attributes)) 

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
Function for displaying image plot using PNG image byte strings
"""

def display_image_plot(div, x, y, static_images, image_brightness, attributes=[], style = 'font-size:20px;text-align:center'):
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

"""
Function for displaying image plot using images loaded from SLAC PSWWW
"""

def display_image_plot_on_slac_pswww(div, x, y, slac_username, slac_dataset_name, image_brightness, attributes=[], style = 'font-size:20px;text-align:center'):
    "Build a suitable CustomJS to display the current event in the div model."
    return CustomJS(args=dict(div=div, x=x, y=y, slac_username=slac_username, slac_dataset_name=slac_dataset_name, image_brightness=image_brightness), code="""
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
                
        var img_src = "https://pswww.slac.stanford.edu/jupyterhub/user/" + slac_username + "/files/" + slac_dataset_name + "/images/diffraction-pattern-" + minDiffIndex + ".png";
        var img_tag_attrs = "style='filter: brightness(" + image_brightness + "); transform: scaleY(-1);'";
        var img_tag = "<div><img src='" + img_src + "' " + img_tag_attrs + "></img></div>";
        //var line = img_tag + "\\n";
        var line = img_tag + "<p style=%r>" + (minDiffIndex+1) + "</p>" + "\\n";
        div.text = "";
        var text = div.text.concat(line);
        var lines = text.split("\\n")
        if (lines.length > 35)
            lines.shift();
        div.text = lines.join("\\n");
    """ % (attributes, style))

"""
Function for visualizing data from an HDF5 file
"""

def visualize(dataset_file, 
              image_type, latent_method, 
              image_location=None, 
              slac_username=None, slac_dataset_name=None, 
              latent_idx_1=None, latent_idx_2=None, 
              particle_property=None,
              particle_plot_x_axis_label_text_font_size='20pt', particle_plot_y_axis_label_text_font_size='20pt',
              real2d_plot_x_lower=-2e-9, real2d_plot_x_upper=2e-9, real2d_plot_y_lower=-2e-9, real2d_plot_y_upper=2e-9,
              x_axis_label_text_font_size='20pt', y_axis_label_text_font_size='20pt', 
              index_label_text_font_size='20px',
              image_brightness=1.0, 
              figure_width = 450, figure_height = 450, 
              image_size_scale_factor = 0.9, 
              color_bar_height = 400, color_bar_width = 120,
              particle_atoms_random_sample_size=1000):
    
    tic = time.time()
    with h5.File(dataset_file, "r") as dataset_file_handle:
        latent = dataset_file_handle[latent_method][:]
        
        if latent_method == "orientations":
            atomic_coordinates = dataset_file_handle[particle_property][:]            
        
        # unclear on how to plot targets
        # labels = np.zeros(len(images)) 

    toc = time.time()
    print("It takes {:.2f} seconds to load the latent vectors and metadata from the HDF5 file.".format(toc-tic))
    
    # unclear on how to plot targets
    # n_labels = len(np.unique(labels))

    # Keep track of the mouse x and y positions in the scatter plot
    point_attributes = ['x', 'y']
    
    # Container to display the static_images
    div_width = int(figure_width * image_size_scale_factor)
    div_height = int(figure_height * image_size_scale_factor)
    div = Div(width=div_width, height=div_height)

    if latent_method == "principal_component_analysis":
        x = latent[:, latent_idx_1]
        y = latent[:, latent_idx_2]   
        
        # Scatter plot for the latent vectors
        source = ColumnDataSource(data=dict(
            x=x,
            y=y,
            z=rotation_angles
        ))
        #         tooltips = [
#             ("(x,y,z)", "($x, $y, $z)"),
#         ]
#         scatter_plot = figure(width=figure_width, height=figure_height, tools="pan,wheel_zoom,box_zoom,reset,hover", tooltips=tooltips)
        scatter_plot = figure(width=figure_width, height=figure_height, tools="pan,wheel_zoom,box_zoom,reset")
        scatter_plot.xaxis.axis_label_text_font_size = x_axis_label_text_font_size
        scatter_plot.yaxis.axis_label_text_font_size = y_axis_label_text_font_size
        
        scatter_plot.scatter('x', 'y', source=source, fill_alpha=0.6)
        scatter_plot.xaxis.axis_label = "PC {}".format(latent_idx_1 + 1)
        scatter_plot.yaxis.axis_label = "PC {}".format(latent_idx_2 + 1)

        # Build layout for plots
        layout = row(scatter_plot, div)
        
    elif latent_method == "diffusion_map":          
        x = latent[:, latent_idx_1]
        y = latent[:, latent_idx_2]   
        
        # Scatter plot for the latent vectors
        source = ColumnDataSource(data=dict(
            x=x,
            y=y,
            z=rotation_angles
        ))
#         tooltips = [
#             ("(x,y,z)", "($x, $y, $z)"),
#         ]
#         scatter_plot = figure(width=figure_width, height=figure_height, tools="pan,wheel_zoom,box_zoom,reset,hover", tooltips=tooltips)
        scatter_plot = figure(width=figure_width, height=figure_height, tools="pan,wheel_zoom,box_zoom,reset")
        scatter_plot.xaxis.axis_label_text_font_size = x_axis_label_text_font_size
        scatter_plot.yaxis.axis_label_text_font_size = y_axis_label_text_font_size
        
        scatter_plot.scatter('x', 'y', source=source, fill_alpha=0.6)
        scatter_plot.xaxis.axis_label = "DC {}".format(latent_idx_1 + 1)
        scatter_plot.yaxis.axis_label = "DC {}".format(latent_idx_2 + 1)

        # Build layout for plots
        layout = row(scatter_plot, div)
        
    elif latent_method == "orientations":   
        # Quaternion -> angle-axis -> (azimuth, elevation), rotation angle about axis
        x, y, rotation_angles = get_elevation_azimuth_rotation_angles_from_orientations(latent)
        
        # Use rotation_angles to color the points on the scatter plots
        colors, color_mapper = get_colors_from_rotation_angles(rotation_angles)
        
        # Scatter plot for the latent vectors
        source = ColumnDataSource(data=dict(
            x=x,
            y=y,
            z=rotation_angles,
            colors=colors,
            azimuth=x / np.pi * 180,
            elevation=y / np.pi * 180,
            rotation=rotation_angles / np.pi * 180
        ))
        #         tooltips = [
#             ("(x,y,z)", "($x, $y, $z)"),
#         ]
#         scatter_plot = figure(width=figure_width, height=figure_height, tools="pan,wheel_zoom,box_zoom,reset,hover", tooltips=tooltips)
        scatter_plot = figure(width=figure_width, height=figure_height, tools="pan,wheel_zoom,box_zoom,reset")
        scatter_plot.xaxis.axis_label_text_font_size = x_axis_label_text_font_size
        scatter_plot.yaxis.axis_label_text_font_size = y_axis_label_text_font_size
                
        scatter_plot.circle('x', 'y', fill_alpha=0.6, fill_color='colors', line_color=None, source=source)
        
        scatter_plot.xaxis.axis_label = "Azimuth"
        scatter_plot.yaxis.axis_label = "Elevation"
        
        scatter_plot_x_lower, scatter_plot_x_upper, scatter_plot_y_lower, scatter_plot_y_upper = -np.pi, np.pi, -np.pi / 2, np.pi / 2
        scatter_plot.x_range = Range1d(scatter_plot_x_lower, scatter_plot_x_upper)
        scatter_plot.y_range = Range1d(scatter_plot_y_lower, scatter_plot_y_upper)
        
        # Real-space XY projection plots
        real2d_plot = figure(width=figure_width, height=figure_height, tools="pan,wheel_zoom,box_zoom,reset")
        real2d_plot.xaxis.axis_label_text_font_size = particle_plot_x_axis_label_text_font_size
        real2d_plot.yaxis.axis_label_text_font_size = particle_plot_y_axis_label_text_font_size
        
        real2d_plot_data_source = ColumnDataSource({'x': [], 'y': []})
        
        real2d_plot.scatter('x', 'y', source=real2d_plot_data_source)
        
        # Assuming the beam travels in the +Z direction, the real-space XY projection plot faces the beam.
        real2d_plot.xaxis.axis_label = "Y"
        real2d_plot.yaxis.axis_label = "X"
        
        real2d_plot.x_range = Range1d(real2d_plot_x_lower, real2d_plot_x_upper)
        real2d_plot.y_range = Range1d(real2d_plot_y_lower, real2d_plot_y_upper)
        
        # Color bar
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
        
        # Build layout for plots
        layout = row(real2d_plot, scatter_plot, color_bar_plot, div)
        
        # To prevent a lag when displaying particle with several atoms, randomly select particle_random_sample_size atoms
        particle_atoms_random_sample_idx = np.random.choice(len(atomic_coordinates), particle_atoms_random_sample_size, replace=False)
        atomic_coordinates = atomic_coordinates[particle_atoms_random_sample_idx]
        
        # Display real-space XY projection of particle when mouse is near a point in scatter plot
        scatter_plot.js_on_event(events.MouseMove, display_real2d_plot_using_quaternions(
                                                            real2d_plot_data_source, x, y, 
                                                            latent, 
                                                            atomic_coordinates, 
                                                            attributes=point_attributes))
        
    else:
        raise Exception("Unrecognized latent method. Please choose from: principal_component_analysis, diffusion_map")
    
    # Directly display images from SLAC PSWWW
    if image_location == "slac-pswww":
        
        if slac_username is None:
            raise Exception("Please provide: slac_username")
        
        if slac_dataset_name is None:
            raise Exception("Please provide: slac_dataset_name")
        
        # Display corresponding image when mouse is near a point in scatter plot
        scatter_plot.js_on_event(events.MouseMove, display_image_plot_on_slac_pswww(
                                                             div, x, y, 
                                                             slac_username, slac_dataset_name, 
                                                             image_brightness, 
                                                             attributes=point_attributes, 
                                                             style='font-size:{};text-align:center'.format(index_label_text_font_size)))
    
    # Display PNG image byte strings generated from image arrays in HDF5 file
    elif image_location is None:
        
        # Load images from HDF5 file
        tic = time.time()
        with h5.File(dataset_file, "r") as dataset_file_handle:
            images = dataset_file_handle[image_type][:]
        
        toc = time.time()
        print("It takes {:.2f} seconds to load images from the HDF5 file.".format(toc-tic))
        
        # Convert loaded images to PNG image byte strings
        tic = time.time()
        static_images = get_images(images)
        toc = time.time()
        print("It takes {:.2f} seconds to generate static images in memory.".format(toc-tic))

        # Display corresponding image when mouse is near a point in scatter plot
        scatter_plot.js_on_event(events.MouseMove, display_image_plot(
                                                             div, x, y, 
                                                             static_images,
                                                             image_brightness, 
                                                             attributes=point_attributes, 
                                                             style='font-size:{};text-align:center'.format(index_label_text_font_size)))
    
    else:
        raise Exception("Unknown image location.")
    
    # Display the plots
    tic = time.time()
    show(layout)
    toc = time.time()
    print("It takes {:.2f} seconds to display the plots.".format(toc-tic))
