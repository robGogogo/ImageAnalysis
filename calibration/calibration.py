import numpy as np

# -----------------------------------------------------------------------------
# Scene Configuration
# -----------------------------------------------------------------------------

# Maximum valid depth threshold (metres). Points beyond this are discarded.
maxDepth = 10


# -----------------------------------------------------------------------------
# RGB Camera — Intrinsic Parameters
# -----------------------------------------------------------------------------

fx_rgb = 5.1885790117450188e+02   # Focal length, x-axis (pixels)
fy_rgb = 5.1946961112127485e+02   # Focal length, y-axis (pixels)
cx_rgb = 3.2558244941119034e+02   # Principal point, x (pixels)
cy_rgb = 2.5373616633400465e+02   # Principal point, y (pixels)


# -----------------------------------------------------------------------------
# RGB Camera — Lens Distortion Coefficients (Brown-Conrady Model)
# -----------------------------------------------------------------------------

k1_rgb =  2.0796615318809061e-01  # Radial distortion, 1st order
k2_rgb = -5.8613825163911781e-01  # Radial distortion, 2nd order
p1_rgb =  7.2231363135888329e-04  # Tangential distortion, x
p2_rgb =  1.0479627195765181e-03  # Tangential distortion, y
k3_rgb =  4.9856986684705107e-01  # Radial distortion, 3rd order


# -----------------------------------------------------------------------------
# Depth Camera — Intrinsic Parameters
# -----------------------------------------------------------------------------

fx_d = 5.8262448167737955e+02     # Focal length, x-axis (pixels)
fy_d = 5.8269103270988637e+02     # Focal length, y-axis (pixels)
cx_d = 3.1304475870804731e+02     # Principal point, x (pixels)
cy_d = 2.3844389626620386e+02     # Principal point, y (pixels)


# -----------------------------------------------------------------------------
# Depth Camera — Lens Distortion Coefficients (Brown-Conrady Model)
# -----------------------------------------------------------------------------

k1_d = -9.9897236553084481e-02    # Radial distortion, 1st order
k2_d =  3.9065324602765344e-01    # Radial distortion, 2nd order
p1_d =  1.9290592870229277e-03    # Tangential distortion, x
p2_d = -1.9422022475975055e-03    # Tangential distortion, y
k3_d = -5.1031725053400578e-01    # Radial distortion, 3rd order


# -----------------------------------------------------------------------------
# Extrinsic Parameters — Depth-to-RGB Camera Transform
# -----------------------------------------------------------------------------
# Rotation matrix derived from the calibration file (MATLAB convention).
# Original MATLAB ops: R = -reshape(R, [3,3]); R = inv(R')
# Reproduced here in NumPy for identical results.

R_raw = -np.array([
     9.9997798940829263e-01,  5.0518419386157446e-03,  4.3011152014118693e-03,
    -5.0359919480810989e-03,  9.9998051861143999e-01, -3.6879781309514218e-03,
    -4.3196624923060242e-03,  3.6662365748484798e-03,  9.9998394948385538e-01,
]).reshape(3, 3)
R = np.linalg.inv(R_raw.T)

# Translation vector: depth camera origin relative to RGB camera origin (metres)
t_x =  2.5031875059141302e-02
t_y =  6.6238747008330102e-04
t_z = -2.9342312935846411e-04


# -----------------------------------------------------------------------------
# Precomputed Camera Matrices
# -----------------------------------------------------------------------------

K_d   = np.array([
    [fx_d,   0, cx_d],  
    [0, fy_d,   cy_d],  
    [0, 0, 1]
])  # Depth intrinsic matrix
K_rgb = np.array([
    [fx_rgb, 0, cx_rgb],
    [0, fy_rgb, cy_rgb], 
    [0, 0, 1]
])  # RGB intrinsic matrix
