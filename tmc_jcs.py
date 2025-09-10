import csv
import argparse
import numpy as np
import pyvista as pv
import SimpleITK as sitk

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


def load_mesh(mesh_path):
    """Load a VTK PolyData surface using PyVista and return both its points and the mesh.

    Parameters
    ----------
    mesh_path : str or os.PathLike
        Path to a VTK PolyData file representing the joint surface.

    Returns
    -------
    points : (N, 3) ndarray of float
        XYZ vertex coordinates from the surface mesh in world units.
    mesh : pyvista.PolyData
        The loaded PyVista mesh object.
    """
    mesh = pv.read(mesh_path)
    return mesh.points, mesh


def get_center_point_and_normal(polydata):
    """
    Find the point on the mesh surface closest to the center of mass,
    and return the point and its surface normal.

    Parameters
    ----------
    polydata : pv.PolyData
        A surface mesh with or without precomputed normals.

    Returns
    -------
    center_point : np.ndarray, shape (3,)
        Closest point on the mesh to the center of the bounding box.
    normal : np.ndarray, shape (3,)
        Normal vector at that point (unit length).
    """
    # Ensure normals exist
    if "Normals" not in polydata.point_data:
        polydata = polydata.compute_normals(
            point_normals=True, auto_orient_normals=True, inplace=False
        )

    # Find center of bounding box (not necessarily on the mesh)
    center_bbox = np.array(polydata.bounds).reshape(3, 2).mean(axis=1)

    # Find closest point on the mesh to the bounding box center
    point_id = polydata.find_closest_point(center_bbox)
    center_point = polydata.points[point_id]
    normal = polydata.point_data["Normals"][point_id]

    # Normalize to unit length
    normal /= np.linalg.norm(normal)

    return center_point, normal


def poly_features(x, y, degree=5):
    """Create 2D polynomial feature matrix up to a given total degree.

    Constructs the design matrix containing all monomials :math:`x^i y^j`
    where :math:`i \\ge 0, j \\ge 0, i + j \\le \\text{degree}`.

    Parameters
    ----------
    x : (N,) array_like
        X-coordinates.
    y : (N,) array_like
        Y-coordinates, same length/order as ``x``.
    degree : int, optional
        Maximum total polynomial degree (default is 5).

    Returns
    -------
    X : (N, M) ndarray
        Design matrix of polynomial features, where
        :math:`M = (\\text{degree}+1)(\\text{degree}+2)/2`.
    """
    return np.vstack(
        [(x**i) * (y**j) for i in range(degree + 1) for j in range(degree + 1 - i)]
    ).T


def compute_saddle_and_principal_directions(
    points, degree=5, resolution=200, radius=3.0
):
    """Estimate the saddle point (central, interior) and principal curvature directions.

    A polynomial surface :math:`z = f(x, y)` is fit to the PCA-aligned input
    points. The saddle point is chosen as the grid location with minimal
    gradient magnitude that is both:
      1. Closest to the geometric centre of the surface, and
      2. Located away from the grid boundaries (default: interior >10% margin).

    Within a circular neighborhood around this saddle point (default 3 mm,
    following Halilaj et al. 2013), the Hessian of :math:`f` is computed at
    each grid cell. The eigenvectors of these Hessians are averaged and mapped
    back to world space to produce two orthonormal tangent directions,
    approximating the principal curvature directions.

    Parameters
    ----------
    points : (N, 3) array_like
        Surface point cloud (XYZ) in world coordinates (e.g., mm).
    degree : int, optional
        Degree of the fitted polynomial surface (default=5).
    resolution : int, optional
        Number of grid samples per axis for surface evaluation (default=200).
    radius : float, optional
        Neighborhood radius (in PCA x-y units) around the saddle point used
        when averaging Hessian eigenvectors (default=3.0).

    Returns
    -------
    saddle_world : (3,) ndarray
        Estimated saddle point location in world coordinates.
    i_world : (3,) ndarray
        First principal curvature direction in world coordinates.
    k_world : (3,) ndarray
        Second principal curvature direction in world coordinates, orthogonal
        to ``i_world`` in the tangent plane.

    Raises
    ------
    ValueError
        If no valid interior saddle point can be found (e.g., all candidates
        fall within the boundary margin).

    Notes
    -----
    - The corresponding surface normal can be obtained as
      ``np.cross(i_world, k_world)`` followed by normalization.
    - Boundary candidates are rejected by enforcing an interior margin
      (10% of grid size by default). This reduces the chance of selecting
      spurious saddles on the rim of the fitted surface.
    """
    # PCA alignment
    pca = PCA(n_components=3)
    aligned = pca.fit_transform(points)
    x, y, z = aligned[:, 0], aligned[:, 1], aligned[:, 2]

    # Polynomial fit
    X_design = poly_features(x, y, degree)
    model = LinearRegression(fit_intercept=False).fit(X_design, z)

    # Evaluation grid
    xi = np.linspace(x.min(), x.max(), resolution)
    yi = np.linspace(y.min(), y.max(), resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    Xg = poly_features(xi_grid.ravel(), yi_grid.ravel(), degree)
    zi = Xg @ model.coef_
    zi_grid = zi.reshape(resolution, resolution)

    # Gradient magnitude
    zx = np.gradient(zi_grid, axis=1)
    zy = np.gradient(zi_grid, axis=0)
    gradient_mag = np.sqrt(zx**2 + zy**2)

    # "closest to centre" + "not boundary"
    centre = np.array([np.mean(x), np.mean(y)])
    distances = (xi_grid - centre[0]) ** 2 + (yi_grid - centre[1]) ** 2
    combined_metric = gradient_mag + 1e-6 * distances

    flat_idx = np.argsort(combined_metric.ravel())
    candidate_indices = np.column_stack(np.unravel_index(flat_idx, gradient_mag.shape))

    # Interior margin (10% of grid)
    margin = int(0.1 * resolution)

    min_idx = None
    for i, j in candidate_indices:
        if margin < i < resolution - margin and margin < j < resolution - margin:
            min_idx = (i, j)
            break

    if min_idx is None:
        raise ValueError(
            "No valid interior saddle point found (all candidates near boundary)."
        )

    # Saddle location
    saddle_point = np.array([xi[min_idx[1]], yi[min_idx[0]]])
    saddle_height = zi_grid[min_idx]

    # Hessian components
    zxx = np.gradient(zx, axis=1)
    zyy = np.gradient(zy, axis=0)
    zxy = np.gradient(zx, axis=0)

    dist_grid = np.sqrt(
        (xi_grid - saddle_point[0]) ** 2 + (yi_grid - saddle_point[1]) ** 2
    )
    mask = dist_grid < radius

    eigvecs = []
    for i, j in zip(*np.where(mask)):
        H = np.array([[zxx[i, j], zxy[i, j]], [zxy[i, j], zyy[i, j]]])
        _, vecs = np.linalg.eigh(H)
        eigvecs.append(vecs)

    eigvecs = np.stack(eigvecs, axis=0)
    i_dir = np.mean(eigvecs[:, :, 0], axis=0)
    k_dir = np.mean(eigvecs[:, :, 1], axis=0)

    i_dir /= np.linalg.norm(i_dir)
    k_dir /= np.linalg.norm(k_dir)

    # Back to world space
    saddle_world = pca.inverse_transform(
        [[saddle_point[0], saddle_point[1], saddle_height]]
    )[0]
    i_world = (
        pca.inverse_transform([[*i_dir, 0]])[0] - pca.inverse_transform([[0, 0, 0]])[0]
    )
    k_world = (
        pca.inverse_transform([[*k_dir, 0]])[0] - pca.inverse_transform([[0, 0, 0]])[0]
    )

    return saddle_world, i_world, k_world


def sitk_to_pyvista_surface(image):
    """Convert a SimpleITK binary mask to a PyVista surface via isocontouring.

    Parameters
    ----------
    image : SimpleITK.Image
        Binary volume in SimpleITK space. Spacing, origin, and direction
        are used to place the surface in physical coordinates.

    Returns
    -------
    surface : pyvista.PolyData
        Extracted isosurface (iso-value 0.5) in world coordinates.

    Notes
    -----
    - The SITK array (Z, Y, X) is transposed to (X, Y, Z) for PyVista.
    - Values are flattened in Fortran order to match VTK expectations.
    """
    bone_array = sitk.GetArrayFromImage(image)
    spacing = np.array(image.GetSpacing())
    origin = np.array(image.GetOrigin())

    # Transform the binary mask to physical space
    # Reverse ZYX to XYZ for PyVista
    bone_array = np.transpose(bone_array, (2, 1, 0))

    # Create a PyVista structured grid in physical space
    nx, ny, nz = bone_array.shape
    x = np.arange(nx) * spacing[0]
    y = np.arange(ny) * spacing[1]
    z = np.arange(nz) * spacing[2]
    grid = pv.RectilinearGrid(x + origin[0], y + origin[1], z + origin[2])

    # Voxel data must be flattened in Fortran order
    grid["values"] = bone_array.flatten(order="F")

    bone_surface = grid.contour([0.5])

    return bone_surface


def save_coordinate_system_to_csv(filename, origin, x_axis, y_axis, z_axis):
    """Write a 3D coordinate system (origin + axes) to a CSV file.

    Parameters
    ----------
    filename : str or os.PathLike
        Output CSV path.
    origin : array_like, shape (3,)
        XYZ coordinates of the origin (e.g., saddle point).
    x_axis : array_like, shape (3,)
        Unit vector for the X axis in world coordinates.
    y_axis : array_like, shape (3,)
        Unit vector for the Y axis in world coordinates.
    z_axis : array_like, shape (3,)
        Unit vector for the Z axis in world coordinates.

    Returns
    -------
    None
        Writes a CSV with header ``Name,X,Y,Z`` and rows for the origin and axes.
    """
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "X", "Y", "Z"])
        writer.writerow(["Saddle_point"] + list(origin))
        writer.writerow(["X_axis"] + list(x_axis))
        writer.writerow(["Y_axis"] + list(y_axis))
        writer.writerow(["Z_axis"] + list(z_axis))
    print(f"Saved coordinate system to {filename}")


def compute_bone_cs(bone_label, joint_surface_path):
    """Compute an anatomical coordinate system from a joint surface.

    Loads the joint surface, estimates a saddle point and two principal directions of curvature,
    then assembles an anatomical right-handed frame for MC1 and TRP according to Halilaj, et al. 2013:

    "Following the ISB convention (Wu and Cavanagh, 1995; Wu et al., 2005), the zaxis of the trapezial 
    coordinate system was defined by i_TRP, running in a ulnar-to-radial direction, the y-axis (Y_TRP) 
    by the cross product of Z_TRP and k_TRP, oriented in a distal-to-proximal direction, and the 
    x-axis (X_TRP) by the cross product of Y_TRP and Z_TRP, running in a dorsal-to-volar direction. 
    Similarly, X_MC1 in the metacarpal coordinate system was defined by i_MC1, Y_MC1 by the cross product
    of k_MC1 and X_MC1, and Z_MC1 as the cross product of the X_MC1 and the Y_MC1. The saddle point, 
    served as the origin of each segment coordinate system."

    Parameters
    ----------
    bone_label : {'TRP', 'MC1'}
        Bone identifier. Use 'MC1' (first metacarpal) or 'TRP' (trapezium).
    joint_surface_path : str or os.PathLike
        Path to the joint surface mesh file (VTK).

    Returns
    -------
    origin : (3,) ndarray
        Saddle point coordinates in world space.
    x_axis : (3,) ndarray
        Unit X axis (world coordinates).
    y_axis : (3,) ndarray
        Unit Y axis (world coordinates).
    z_axis : (3,) ndarray
        Unit Z axis (world coordinates).

    Raises
    ------
    ValueError
        If ``bone_label`` is not 'TRP' or 'MC1'.
    """
    joint_points, _ = load_mesh(joint_surface_path)
    saddle, i_axis, k_axis = compute_saddle_and_principal_directions(joint_points)

    if bone_label.upper() == "TRP":
        # Z axis, ulnar -> radial
        z_axis = i_axis / np.linalg.norm(i_axis)

        if z_axis[0] < 0:
            z_axis *= -1

        # Y axis, distal -> proximal
        y_axis = np.cross(z_axis, k_axis)
        y_axis /= np.linalg.norm(y_axis)

        if y_axis[1] < 0:
            y_axis *= -1

        # X axis = Y x Z
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)

    elif bone_label.upper() == "MC1":
        # X axis, ulnar to radial
        x_axis = i_axis / np.linalg.norm(i_axis)

        if x_axis[0] < 0:
            x_axis *= -1

        # Y axis, distal to proximal
        y_axis = np.cross(k_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)

        if y_axis[1] < 0:
            y_axis *= -1

        # Z axis = X x Y, dorsal to volar
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis)  # Z = X × Y

    else:
        raise ValueError("bone_label must be 'TRP' or 'MC1'")

    return saddle, x_axis, y_axis, z_axis


def compute_and_plot(
    bone_label, joint_surface_path, bone_mask_path, output_csv, visualize=True
):
    """Compute a bone coordinate system, save it to CSV, and visualize (optional).

    Parameters
    ----------
    bone_label : {'TRP', 'MC1'}
        Bone identifier controlling anatomical axis construction.
    joint_surface_path : str or os.PathLike
        Path to the joint surface mesh (VTK).
    bone_mask_path : str or os.PathLike
        Path to a binary bone mask volume (NIfTI) used for visualization.
    output_csv : str or os.PathLike
        Output CSV path for the computed coordinate system.
    visualize : bool, optional
        If True, render a 3D scene with the joint surface, bone surface,
        saddle point, and axes using PyVista (default is True).

    Returns
    -------
    None
        Side effects: writes ``output_csv`` and (optionally) opens a PyVista window.
    """
    saddle, x_axis, y_axis, z_axis = compute_bone_cs(bone_label, joint_surface_path)
    save_coordinate_system_to_csv(output_csv, saddle, x_axis, y_axis, z_axis)

    if visualize:
        bone_image = sitk.ReadImage(bone_mask_path)
        _, joint_mesh = load_mesh(joint_surface_path)

        bone_surf = sitk_to_pyvista_surface(bone_image)
        plotter = pv.Plotter()
        plotter.add_mesh(joint_mesh, color="yellow", opacity=0.4, label="Joint Surface")
        plotter.add_mesh(bone_surf, color="white", opacity=0.3, label="Bone Surface")
        plotter.add_mesh(
            pv.Sphere(radius=0.5, center=saddle), color="red", label="Saddle Point"
        )
        plotter.add_mesh(
            pv.Arrow(start=saddle, direction=x_axis, scale=5.0),
            color="orange",
            label="X",
        )
        plotter.add_mesh(
            pv.Arrow(start=saddle, direction=y_axis, scale=5.0),
            color="purple",
            label="Y",
        )
        plotter.add_mesh(
            pv.Arrow(start=saddle, direction=z_axis, scale=5.0),
            color="green",
            label="Z",
        )
        plotter.add_legend()
        plotter.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute and export coordinate systems for MC1 and TRP based on Halilaj, et al. 2013."
    )
    parser.add_argument("mc1_surface", help="Path to MC1 joint surface VTK file")
    parser.add_argument("mc1_mask", help="Path to MC1 bone NIFTI file")
    parser.add_argument("trp_surface", help="Path to TRP joint surface VTK file")
    parser.add_argument("trp_mask", help="Path to TRP bone NIFTI file")
    parser.add_argument(
        "--mc1_csv",
        default="mc1_coordinate_system.csv",
        help="Output CSV for MC1 coordinate system",
    )
    parser.add_argument(
        "--trp_csv",
        default="trp_coordinate_system.csv",
        help="Output CSV for TRP coordinate system",
    )
    parser.add_argument(
        "--no_viz", action="store_true", help="Disable 3D visualization"
    )
    args = parser.parse_args()

    compute_and_plot(
        "MC1", args.mc1_surface, args.mc1_mask, args.mc1_csv, visualize=not args.no_viz
    )
    compute_and_plot(
        "TRP", args.trp_surface, args.trp_mask, args.trp_csv, visualize=not args.no_viz
    )
