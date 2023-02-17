import numpy as np
import pandas as pd
import PIL as pil

# Constant values
c_val = 299792458
h_val = 6.62607015e-34
k_val = 1.380649e-23

# Calculate XYZ to linear sRGB conversion matrix according to https://www.fourmilab.ch/documents/specrend/
XYZr = [0.64, 0.33, 0.03]
XYZg = [0.3, 0.6, 0.1]
XYZb = [0.15, 0.06, 0.79]
XYZw = [0.3127, 0.329, 0.3583]

rXYZ = np.cross(XYZg, XYZb)
gXYZ = np.cross(XYZb, XYZr)
bXYZ = np.cross(XYZr, XYZg)

rw = np.dot(rXYZ, XYZw) / XYZw[1]
rg = np.dot(gXYZ, XYZw) / XYZw[1]
rb = np.dot(bXYZ, XYZw) / XYZw[1]

rXYZ = rXYZ / rw
gXYZ = gXYZ / rw
bXYZ = bXYZ / rw

#lsRGB_conversion_matrix = np.array([rXYZ, gXYZ, bXYZ])
lsRGB_conversion_matrix = np.array([[3.241,-1.5374,-0.4986],[-0.9692,1.876,0.0416],[0.0556,-0.204,1.057]])

nano_meter_conversion = 10e-9

"""Returns function for planck curve at given temperature
:param T: temperature
:return : function for intensity at given wavelenght in nanometers
"""
def planck_spectrum_function(T):
    return lambda x: planck_spectrum(T, x)


def planck_spectrum(T, wavelength):
    wlm = wavelength * 1e-9
    specfic_intensity = (3.74183e-16 * np.power(wlm, -5.0)) / (np.exp(1.4388e-2 / (wlm * T)) - 1.0)
    if(np.isnan(specfic_intensity)):
        return 0
    else:
        return specfic_intensity




"""Generates numerical approximation for planck spectrum at temperature T on interval [start,end)
:param T: temperature
:param start: first sample point of the spectrum
:param end: last point (exclusive) of the spectrum
:count : samples taken from spectrum
:return spectrum from black body radiator at temperature T on interval [start,end) with count samples
"""


def generate_planck_spectrum(T, start, end, count):
    # Generate wavelengths we need to evaluate
    step_size = (end - start) / count
    wavelenghts = np.arange(start, end, step_size);

    # Return the intensity for each wavelength
    return np.array(list(map(planck_spectrum_function(T), wavelenghts)))


"""Gamma corrects the give color value to sRGB
:param x: color value
:return : gamma corrected value
"""


def gamma_correct(x):
    if x <= 0.0031308:
        return 12.92 * x
    else:
        return 1.055 * (x ** (1 / 2.4)) - 0.055


def get_rgb_color(T):
    # Integrate matching fucntions multiplied by planck spectrum for given temperature and
    XYZ = np.array(list(
        map(lambda x: np.sum(matching_functions[x] * generate_planck_spectrum(T, start, (end + 1), sample_count)),
            matching_functions.keys()[1:])))

    XYZ_Normalized = XYZ / max(np.sum(XYZ),1e-4)

    lsRGB = lsRGB_conversion_matrix @ XYZ_Normalized

    lsRGB = np.clip(lsRGB / (max(1e-4,np.max(lsRGB))), 0, 1)

    return np.array(list(map(gamma_correct, lsRGB))) * min(XYZ[1], 1)

# Read matching functions
matching_functions = pd.read_csv("./Resources/CIE_xyz_1931_2deg.csv")

sample_count = matching_functions.shape[0]
start = matching_functions["wavelength"][0];
end = matching_functions["wavelength"][sample_count - 1]

T_colors = {}
for T in range(100,10000,100):
    T_colors[T] = get_rgb_color(T)*255

for T in range(10000, 30000,1000):
    T_colors[T] = get_rgb_color(T)*255

for T in T_colors.keys():
    rgb_color = T_colors[T]
    print("{" + "{0:.4f},{1:.4f},{2:.4f}".format(rgb_color[0], rgb_color[1], rgb_color[2]) + "},")

band_image = [];
for T in range(100,10000,100):
    for f in range(0,10):
        a = f/10
        band_image.append(np.array([T_colors[T]] * 40) * (1 - a) +  a * np.array([T_colors[T+100]] * 40))


band_image = np.array(band_image,dtype="uint8")
band_image = np.transpose(band_image,axes=(1,0,2))
im = pil.Image.fromarray(band_image,mode="RGB")
im.save("blackbody_range.jpg")