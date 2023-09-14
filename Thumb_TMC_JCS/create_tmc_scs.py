import os
import sys
import argparse

import numpy as np
import scipy, scipy.optimize
import sympy as sp

import SimpleITK as sitk

from stl import mesh
import vtk
import pyvista as pv
from pyacvd import Clustering

import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import math
import progressbar
from timeit import default_timer as timer 

def find_critical_points(crit, minimizer_bounds, data):
    crit_points = []

    widgets = [' [',
         progressbar.Timer(format= 'elapsed time: %(elapsed)s'),
         '] ',
           progressbar.Bar('*'),' (',
           progressbar.ETA(), ') ',
          ]
    bar = progressbar.ProgressBar(max_value=len(data),
                                widgets=widgets).start()
    i = 0

    for x_guess, y_guess in data:
        try:
            solution = scipy.optimize.minimize(crit, x0=(x_guess, y_guess), method = 'Nelder-Mead', bounds=minimizer_bounds,
                                            tol=5e-3, options={'maxiter':50000}).x
            solution = np.array([solution[0], solution[1]], dtype=float)

            for point in crit_points:
                if(np.allclose(solution, point, atol=1e-03)):
                    # skip to next point
                    raise ValueError()
            crit_points.append(solution)
        except:
            pass

        bar.update(i)
        i += 1

    return crit_points

def find_saddle_point(Hessian_det, crit_points, data):
    # compute saddle point
    saddle_point = []
    min_diff = float("inf")
    x, y = sp.symbols('x y')

    widgets = [' [',
         progressbar.Timer(format= 'elapsed time: %(elapsed)s'),
         '] ',
           progressbar.Bar('*'),' (',
           progressbar.ETA(), ') ',
          ]
    bar = progressbar.ProgressBar(max_value=len(crit_points),
                                widgets=widgets).start()
    i = 0

    for crit in crit_points:
        det = Hessian_det.subs([(x, crit[0]), (y, crit[1])])
        if(det < 0):
            # find point with Hessain determinant that is closes to 0.1 and
            # within the dataset, might need to play around with tolerance.
            # print("Hessian solution for point {} is  = {}".format(crit, det))
            is_within_points = [np.allclose(crit, point, atol=1e-2) for point in data]
            diff = abs(-det-0.1)
            if diff<min_diff and np.any(is_within_points):
                min_diff = diff
                saddle_point = crit

        bar.update(i)
        i += 1
    
    return saddle_point

def crop_with_radius_dijkstra(origin, mesh, points, radius=3):
    # crop around origin and return points within given radius
    points_cropped = []
    origin_index = mesh.find_closest_point(origin)

    widgets = [' [',
         progressbar.Timer(format= 'elapsed time: %(elapsed)s'),
         '] ',
           progressbar.Bar('*'),' (',
           progressbar.ETA(), ') ',
          ]
    bar = progressbar.ProgressBar(max_value=len(points),
                                widgets=widgets).start()
    i = 0

    for point in points:
        # find distance to all points using dijkstras algorithm
        # and filter out points further than 3mm
        point_index = mesh.find_closest_point(point)
        if(point_index == origin_index):
            points_cropped.append(point)
            continue
        try:
            # fails for unknown reason for 1 or 2 points out of the 1000s of points
            # reason is unknown therefore this was put in a try-except block
            distance = mesh.geodesic_distance(origin_index, point_index)
            if (distance < radius):
                points_cropped.append(point)
        except:
            continue

        bar.update(i)
        i += 1

    return points_cropped

def crop_with_radius_euclidean(origin, points, radius=3):
    # crop around origin and return points within given radius
    points_cropped = []

    widgets = [' [',
         progressbar.Timer(format= 'elapsed time: %(elapsed)s'),
         '] ',
           progressbar.Bar('*'),' (',
           progressbar.ETA(), ') ',
          ]
    bar = progressbar.ProgressBar(max_value=len(points),
                                widgets=widgets).start()
    i = 0

    for point in points:
        # find distance to all points using euclidean measurements
        # and filter out points further than 3mm
        difference = origin - point
        distance = np.sqrt(np.dot(difference, difference))
        if (distance < radius):
            points_cropped.append(point)

        bar.update(i)
        i += 1

    return points_cropped

def compute_gradient_fields_parametrized(Gaussian_curvature, parametrized_points):
    # compute curvature accross fitted polynomial surface
    curvatures = []
    s, t = sp.symbols('s t')

    widgets = [' [',
         progressbar.Timer(format= 'elapsed time: %(elapsed)s'),
         '] ',
           progressbar.Bar('*'),' (',
           progressbar.ETA(), ') ',
          ]
    bar = progressbar.ProgressBar(max_value=len(parametrized_points),
                                widgets=widgets).start()
    i = 0

    for point in parametrized_points:
        # det = Hessian_det.subs([(x, point[0]), (y, point[1])])
        # x_grad = fx.subs([(x, point[0]), (y, point[1])])
        # y_grad = fy.subs([(x, point[0]), (y, point[1])])
        # curvature = abs(det) /((1+(x_grad**2)+(y_grad**2))**3)
        curvature = Gaussian_curvature.subs([(s, point[0]), (t, point[1])])
        curvatures.append(curvature)
        print(curvature)

        bar.update(i)
        i += 1
    
    return curvatures

def compute_gradient_fields(Hessian_det, fx, fy, points):
    # compute curvature accross fitted polynomial surface
    curvatures = []
    x, y = sp.symbols('x y')

    widgets = [' [',
         progressbar.Timer(format= 'elapsed time: %(elapsed)s'),
         '] ',
           progressbar.Bar('*'),' (',
           progressbar.ETA(), ') ',
          ]
    bar = progressbar.ProgressBar(max_value=len(x_guess),
                                widgets=widgets).start()
    i = 0

    for point in points:
        det = Hessian_det.subs([(x, point[0]), (y, point[1])])
        x_grad = fx.subs([(x, point[0]), (y, point[1])])
        y_grad = fy.subs([(x, point[0]), (y, point[1])])
        curvature = det /((1+(x_grad**2)+(y_grad**2))**2)
        # curvature = Gaussian_curvature.subs([(s, point[0]), (t, point[1])])
        curvatures.append(curvature)
        print(curvature)

        bar.update(i)
        i += 1
        
    
    return curvatures

def compute_gradient_fields_normal_method(normal_func, saddle_normal, parametrized_points, plotter, points, shift_vector):
    # compute curvature accross fitted polynomial surface
    angles = []
    s, t = sp.symbols('s t')
    idx = 0

    widgets = [' [',
         progressbar.Timer(format= 'elapsed time: %(elapsed)s'),
         '] ',
           progressbar.Bar('*'),' (',
           progressbar.ETA(), ') ',
          ]
    bar = progressbar.ProgressBar(max_value=len(parametrized_points),
                                widgets=widgets).start()
    i = 0

    for point in parametrized_points:
        # det = Hessian_det.subs([(x, point[0]), (y, point[1])])
        # x_grad = fx.subs([(x, point[0]), (y, point[1])])
        # y_grad = fy.subs([(x, point[0]), (y, point[1])])
        # curvature = abs(det) /((1+(x_grad**2)+(y_grad**2))**3)
        normal = np.array([normal_func[0].subs([(s, point[0]), (t, point[1])]),
                  normal_func[1].subs([(s, point[0]), (t, point[1])]),
                  normal_func[2].subs([(s, point[0]), (t, point[1])])], dtype=float)
        # print(np.dot(normal, saddle_normal))
        arc = points[idx] - saddle_normal
        arc_len = np.sqrt(np.dot(arc, arc))
        arc /= arc_len

        dot_prod = float(np.dot(arc, saddle_normal))
        angle = np.arccos(dot_prod)

        # if (angle>((np.pi)/2)):
        #     angle = np.pi - angle
        #     normal = -1 * normal

        
        dist_normal_moved = (points[idx] + normal) - saddle_normal
        dist_normal_moved = np.sqrt(np.dot(dist_normal_moved, dist_normal_moved))

        # change if normal was found on bottom of surface (180 deg - angle)
        if arc_len < dist_normal_moved:
            angle *= -1
        # if (angle>((np.pi)/2)):
        #     angle = np.pi - angle
        #     normal = -1 * normal
        #     if dist_normal > dist_saddle:
        #         angle = -1*angle
        # else:
        #     if dist_normal < dist_saddle:
        #         angle = -1*angle
        angles.append(angle)

        if(i % 2 == 0):
            plotter.add_arrows(points[idx]+shift_vector, normal, mag=0.2, color='orange')
        idx += 1
        bar.update(i)
        i += 1
    
    return angles

# def func(data, a,b,c,d,e,f,alpha):
#     # ,g,h,i,j,k,l,m,n,o,h,i,j,k,l,m,n,o,p,q,r,s,t,u,
#     # 5th order polynomial
#     x = data[0]
#     y = data[1]
#     return a + b*(np.cos(alpha)*x-np.sin(alpha)*y) + \
#         c*(np.sin(alpha)*x-np.cos(alpha)*y) + d*(np.cos(alpha)*x-np.sin(alpha)*y)**2 +\
#         e*(np.cos(alpha)*x-np.sin(alpha))*(np.sin(alpha)*x-np.cos(alpha)*y) + \
#         f*(np.sin(alpha)*x-np.cos(alpha)*y)**2 
#             # g*(np.cos(alpha)*x-np.sin(alpha)*y)**3 +\
#         # h*((np.cos(alpha)*x-np.sin(alpha)*y)**2)*(np.sin(alpha)*x-np.cos(alpha)*y) + \
#         # i*((np.cos(alpha)*x-np.sin(alpha)*y))*((np.sin(alpha)*x-np.cos(alpha)*y)**2) + j*((np.sin(alpha)*x-np.cos(alpha)*y)**3) + \
#         # k*((np.cos(alpha)*x-np.sin(alpha)*y)**4) + l*((np.cos(alpha)*x-np.sin(alpha)*y)**3)*((np.sin(alpha)*x-np.cos(alpha)*y)) + \
#         # m*((np.cos(alpha)*x-np.sin(alpha)*y)**2)*((np.sin(alpha)*x-np.cos(alpha)*y)**2) + \
#         # n*((np.cos(alpha)*x-np.sin(alpha)*y))*((np.sin(alpha)*x-np.cos(alpha)*y)**3) + o*((np.sin(alpha)*x-np.cos(alpha)*y)**4) 
#         # p*((np.cos(alpha)*x-np.sin(alpha)*y)**5) + q*((np.cos(alpha)*x-np.sin(alpha)*y)**4)*((np.sin(alpha)*x+np.cos(alpha)*y)) + \
#         # r*((np.cos(alpha)*x-np.sin(alpha)*y)**3)*((np.sin(alpha)*x+np.cos(alpha)*y)**2) + \
#         # s*((np.cos(alpha)*x-np.sin(alpha)*y)**2)*((np.sin(alpha)*x+np.cos(alpha)*y)**3) + \
#         # t*((np.cos(alpha)*x-np.sin(alpha)*y))*((np.sin(alpha)*x+np.cos(alpha)*y)**4) + u*((np.sin(alpha)*x+np.cos(alpha)*y)**5)

def func(data, c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20
         ,c21,c22,c23,c24,c25,c26,c27,c28,c29,c30,c31,c32,c33,c34,c35,c36,c37,c38,c39,c40,
         c41,c42,c43,c44,c45,c46,c47,c48,c49,c50,c51,c52,c53,c54,c55,c56,c57,c58,c59,c60,
          c61,c62,c63,c64,c65,c66,c67,c68,c69,c70,c71,c72,c73,c74,c75,c76):
    # ,c55,c56,c57,c58,c59,c60,
    #      c61,c62,c63,c64,c65,c66,c67,c68,c69,c70,c71,c72,c73,c74,c75,c76
    # 11th order polynomial
    x = data[0]
    y = data[1]
    return c0 + c1*x + c2*y + c3*x**2 + c4*x*y + c5*y**2 + c6*x**3 + c7*y*x**2 + \
            c8*x*y**2 + c9*y**3 + c10*x**4 + c11*y*x**3 + c12*(x**2)*(y**2) + \
            c13*x*y**3 + c14*y**4 + c15*x**5 + c16*y*x**4 + c17*(x**3)*(y**2) + \
            c18*(x**2)*(y**3) + c19*x*y**4 + c20*y**5 + c21*x**6 + c22*y*x**5 +\
            c23*(x**4)*(y**2) + c24*(x**3)*(y**3) + c25*(x**2)*(y**4) + c26*x*(y**5) +\
            c27*y**6 + c28*x**7 + c29*y*x**6 + c30*(x**5)*(y**2) + c31*(x**4)*(y**3) +\
            c32*(x**3)*(y**4) + c33*(x**2)*(y**5) + c34*x*y**6 + c35*y**7 + c36*x**8 +\
            c37*y*x**7 + c38*(x**6)*(y**2) + c39*(x**5)*(y**3) + c40*(x**4)*(y**4) + \
            c41*(x**3)*(y**5) + c42*(x**2)*(y**6) + c43*x*y**7 + c44*y**8 + c45*x**9 +\
            c46*y*x**8 + c47*(x**7)*(y**2) + c48*(x**6)*(y**3) + c49*(x**5)*(y**4) + \
            c50*(x**4)*(y**5) + c51*(x**3)*(y**6) + c52*(x**2)*(y**7) + c53*x*y**8 +\
            c54*y**9 + c55*x**10 + c56*y*x**9 + c57*(x**8)*(y**2) + c58*(x**7)*(y**3) +\
            c59*(x**6)*(y**4) + c60*(x**5)*(y**5) + c61*(x**4)*(y**6) + c62*(x**3)*(y**7) +\
            c63*(x**2)*(y**8) + c64*y**10 + c65*x**11 + c66*y*x**10 + c67*(x**9)*(y**2) + \
            c68*(x**8)*(y**3) + c69*(x**7)*(y**4) + c70*(x**6)*(y**5) + c71*(x**5)*(y**6) +\
            c72*(x**4)*(y**7) + c73*(x**3)*(y**8) + c74*(x**2)*(y**9) + c75*x*y**10 + c76*y**11

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("seg_path", type=str, help="Bone segmentation (STL file)")
    parser.add_argument("show_figure", type=int, nargs='?', default=0, help="Show figures after finishing computation.")

    args = parser.parse_args()
    seg_path = args.seg_path
    show_figure_flag = args.show_figure

    widgets = [' [',
         progressbar.Timer(format= 'elapsed time: %(elapsed)s'),
         '] ',
           progressbar.Bar('*'),' (',
           progressbar.ETA(), ') ',
          ]

    # 1. VTK NII to VTKPOLYDATA
    
    # 2. extract articular surface (3D Slicer)

    # Load the STL files and add the vectors to the plot
    your_mesh = mesh.Mesh.from_file(seg_path)

    # Remove duplicate/overlapping points
    points = your_mesh.points.reshape([-1, 3])
    point_list = np.unique(points, axis=0)

    x_data = point_list[:, 0]
    y_data = point_list[:, 1]
    z_data = point_list[:, 2]
    points = np.c_[x_data, y_data, z_data]

    x_shift = np.median(x_data)
    y_shift = np.median(y_data)
    z_shift = np.median(z_data)
    shift_vector = [x_shift, y_shift, z_shift]

    x_data = x_data - x_shift
    y_data = y_data - y_shift
    z_data = z_data - z_shift

    print("Number of points in input mesh: {}".format(points.shape))
    print("Number of points in input mesh: {}".format(x_data.size))

    # here a non-linear surface fit is made with scipy's curve_fit()
    fittedParameters, pcov = scipy.optimize.curve_fit(func, [x_data, y_data], z_data, method='lm', maxfev = 50000)
    fitted_z_data = func([x_data, y_data], *fittedParameters)
    error = np.sqrt(np.mean(np.square(z_data-fitted_z_data)))
    print("RMSE between original surface and fitted polynomial: {}".format(error))

    # Sympy differential equations
    start_time = timer()
    print("Computing up partial derivatives and parametrizing surface...")
    x, y = sp.symbols('x y')
    expression = func([x, y], *fittedParameters)
    fx = sp.diff(expression, x)
    fy = sp.diff(expression, y)
    fxx = sp.diff(fx, x)
    fyy = sp.diff(fy, y)
    fxy = sp.diff(fx, y)
    print('Time taken: {}s'.format(timer()-start_time))

    # parametrize for curvature computations later
    s,t = sp.symbols('s t')
    x_parametrized = s*sp.cos(t)
    y_parametrized = s*sp.sin(t)
    z_parametrized = func([x_parametrized, y_parametrized], *fittedParameters)

    # partial derivatives of parametized surface
    rs = [sp.diff(x_parametrized, s), sp.diff(y_parametrized, s), sp.diff(z_parametrized, s)]
    rt = [sp.diff(x_parametrized, t), sp.diff(y_parametrized, t), sp.diff(z_parametrized, t)]
    rss = [sp.diff(rs[0], s), sp.diff(rs[1], s), sp.diff(rs[2], s)]
    rtt = [sp.diff(rt[0], t), sp.diff(rt[1], t), sp.diff(rt[2], t)]
    rst = [sp.diff(rs[0], t), sp.diff(rs[1], t), sp.diff(rs[2], t)]

    # First fundamental form coefficients
    E = np.dot(rs, rs)
    F = np.dot(rs, rt)
    G = np.dot(rt, rt)

    # surface unit normal vector
    normal = np.cross(rs, rt)
    normal = normal/sp.sqrt(np.dot(normal, normal))

    # Second fundamental form
    L = np.dot(rss, normal)
    M = np.dot(rst, normal)
    N = np.dot(rtt, normal)
    Gaussian_Curvature = (L*N - M**2)/(E*G - F**2)

    # Equation to find critical points
    crit_eq = sp.Eq(fx-fy, 0)
    crit_eq = sp.lambdify([x, y], fx-fy)
    def crit(coord):
        return crit_eq(coord[0], coord[1])
    
    # x_range = np.linspace(min(x_data), max(x_data), num=100)
    # y_range = np.linspace(min(y_data), max(y_data), num=100)
    start_time = timer()
    print("Computing crtical points across surface...")
    minimizer_bounds = [(min(x_data), max(x_data)), (min(y_data), max(y_data))]
    crit_points = find_critical_points(crit, minimizer_bounds, np.c_[x_data, y_data])

    Hessian = sp.Matrix([[fxx, fxy], [fxy, fyy]])
    Hessian_det = Hessian.det()
    print('Time taken: {}s'.format(timer()-start_time))

    # x_mid = (max(x_data)+min(x_data))/2
    # y_mid = (max(y_data)+min(y_data))/2
    # z_mid = (max(z_data)+min(z_data))/2

    start_time = timer()
    print("Finding saddle point on surface...")
    saddle_point = find_saddle_point(Hessian_det, crit_points, np.c_[x_data, y_data])
    if(len(saddle_point) > 0):
        saddle_point = np.append(saddle_point, func(saddle_point, *fittedParameters))
        print("Saddle point is {}.".format(saddle_point))
    else:
        # exit if no saddle point was found
        raise ValueError("Saddle point could not be computed on given surface.")
    print('Time taken: {}s'.format(timer()-start_time))

    plotter = pv.Plotter()
    plotter.show_axes()

    # Plot fitted polynomial
    points2 = np.c_[x_data, y_data, func([x_data, y_data], *fittedParameters)]
    cloud = pv.PolyData(points2)
    mesh_fitted = cloud.delaunay_2d()
    c = Clustering(mesh_fitted)
    c.subdivide(3)
    c.cluster(10000)
    mesh_fitted = c.create_mesh()
    plotter.add_mesh(mesh_fitted, show_edges=True, color="blue")
    plotter.add_points(np.array([saddle_point]), render_points_as_spheres=True, point_size = 30, color="orange")

    points = np.c_[x_data, y_data, z_data]

    # Create mesh from clustered points.
    # Clustering allows for more accurate distance measurement but adds to computation complexity.
    cloud = pv.PolyData(points)
    mesh_original = cloud.delaunay_2d()
    d = Clustering(mesh_original)
    d.subdivide(3)
    d.cluster(10000)
    mesh_original = d.create_mesh()
    plotter.add_mesh(mesh_original, show_edges=True, color="red")

    start_time = timer()
    print("Cropping surface around saddle point...")
    points_3mm_cropped = crop_with_radius_euclidean(saddle_point, points, radius=3)
    reoriented_surface = np.array([point+shift_vector for point in points_3mm_cropped])
    print('Time taken: {}s'.format(timer()-start_time))
    print(shift_vector)
    print(points_3mm_cropped[0])
    print(reoriented_surface[0])
    
    # Create mesh from 3mm points
    cloud = pv.PolyData(points_3mm_cropped)
    mesh_3mm = cloud.delaunay_2d()
    points_3mm_cropped = (pv._vtk.vtk_to_numpy(mesh_3mm.GetPoints().GetData())).astype('f')
    points_3mm_cropped = np.array([np.array(point) for point in points_3mm_cropped])

    start_time = timer()
    print("Converting cropped points to parametrized coordinate system....")
    s_data = []
    t_data = []
    bar = progressbar.ProgressBar(max_value=len(points_3mm_cropped),
                                widgets=widgets).start()
    i = 0
    for i in range(len(points_3mm_cropped)):
        x_eq = sp.Eq(x_parametrized, points_3mm_cropped[i][0])
        y_eq = sp.Eq(y_parametrized, points_3mm_cropped[i][1])
        solution = sp.solve((x_eq, y_eq), (s, t))
        s_data.append(solution[0][0])
        t_data.append(solution[0][1])
        bar.update(i)
        i += 1
    parametrized_data = np.c_[s_data, t_data]
    print('Time taken: {}s'.format(timer()-start_time))

    # Get saddle point on original surface rather than polynomial.
    saddle_index = mesh_3mm.find_closest_point(saddle_point)
    saddle_point = points_3mm_cropped[saddle_index]

    start_time = timer()
    print("Computing curvatures across surface...")
    saddle_normal = np.array([normal[0].subs([(s, parametrized_data[saddle_index][0]), (t, parametrized_data[saddle_index][1])]),
                     normal[1].subs([(s, parametrized_data[saddle_index][0]), (t, parametrized_data[saddle_index][1])]),
                     normal[2].subs([(s, parametrized_data[saddle_index][0]), (t, parametrized_data[saddle_index][1])])], dtype=float)
    
    curvatures_3mm = compute_gradient_fields_normal_method(normal, saddle_normal, parametrized_data, plotter, points_3mm_cropped, shift_vector)

    curvatures_3mm_sorted = sorted(curvatures_3mm)
    print('Time taken: {}s'.format(timer()-start_time))
    
    # min_curvature_point = np.array([0, 0, 0])
    max_curvature_vector = np.array([0, 0, 0])
    min_curvature_vector = np.array([0, 0, 0])

    # Find search sites to search for maximum and minimum curvature directions
    start_time = timer()
    print("Finding relevant search sites...")
    min_search_sites = []
    prev_index = -1
    bar = progressbar.ProgressBar(max_value=len(curvatures_3mm_sorted),
                                widgets=widgets).start()
    i = 0
    for curvature in curvatures_3mm_sorted:
        if len(min_search_sites) > 2:
            break
        index = curvatures_3mm.index(curvature)
        if prev_index == -1:
            prev_index = index
            min_search_sites.append(curvatures_3mm_sorted.index(curvature))
        dist = points_3mm_cropped[index] - points_3mm_cropped[prev_index]
        dist = dist.flatten()
        dist = np.dot(dist, dist)
        if dist > 0.0625:
            min_search_sites.append(curvatures_3mm_sorted.index(curvature))
        bar.update(i)
        i += 1

    max_search_sites = []
    prev_index = -1
    bar = progressbar.ProgressBar(max_value=len(curvatures_3mm_sorted),
                                widgets=widgets).start()
    i = 0
    for curvature in reversed(curvatures_3mm_sorted):
        if len(max_search_sites) > 2:
            break
        index = curvatures_3mm.index(curvature)
        if prev_index == -1:
            prev_index = index
            max_search_sites.append(curvatures_3mm_sorted.index(curvature))
        dist = points_3mm_cropped[index] - points_3mm_cropped[prev_index]
        dist = dist.flatten()
        dist = np.dot(dist, dist)
        if dist > 0.0625:
            max_search_sites.append(curvatures_3mm_sorted.index(curvature))
        bar.update(i)
        i += 1
    print('Time taken: {}s'.format(timer()-start_time))

    # Compute minimum curvature
    print("Computing direction of minimum curvature...")
    start_time = timer()
    min_curvature = 9999
    bar = progressbar.ProgressBar(max_value=len(min_search_sites),
                                widgets=widgets).start()
    i = 0
    for search_site in min_search_sites:
        for idx in range(10):
            # find minimum average curvature along a direction.
            average_curvature = 0
            min_index = curvatures_3mm.index(curvatures_3mm_sorted[idx + search_site])
            path2min = pv._vtk.vtk_to_numpy(mesh_3mm.geodesic(saddle_index, min_index).GetPoints().GetData())
            if len(path2min) < 5:
                continue
            direction_vector = path2min[1] - path2min[-2]
            num_points_on_path = len(path2min[1:-1])
            for point in range(num_points_on_path):
                searched_index = mesh_3mm.find_closest_point(path2min[point+1])
                if point > 0.9*num_points_on_path:
                    print(point)
                    average_curvature += curvatures_3mm[searched_index]
            average_curvature = average_curvature/num_points_on_path*0.1
            if(average_curvature<min_curvature):
                min_curvature = average_curvature
                min_curvature_vector = direction_vector
        bar.update(i)
        i += 1
    print('Time taken: {}s'.format(timer()-start_time))

    # Compute maximum curvature
    start_time = timer()
    print("Computing direction of maximum curvature...")
    max_curvature = -9999
    bar = progressbar.ProgressBar(max_value=len(max_search_sites),
                                widgets=widgets).start()
    i = 0
    for search_site in max_search_sites:
        for idx in range(10):
            # find maximum average curvature along a direction.
            average_curvature = 0
            max_index = curvatures_3mm.index(curvatures_3mm_sorted[search_site-idx])
            path2max = pv._vtk.vtk_to_numpy(mesh_3mm.geodesic(saddle_index, max_index).GetPoints().GetData())
            if len(path2max) < 5:
                continue
            direction_vector = path2max[1] - path2max[-2]
            num_points_on_path = len(path2max[1:-1])
            for point in range(num_points_on_path):
                searched_index = mesh_3mm.find_closest_point(path2max[point+1])
                if point > 0.9*num_points_on_path:
                    average_curvature += curvatures_3mm[searched_index]
            average_curvature = average_curvature/num_points_on_path*0.1
            if(average_curvature>max_curvature):
                max_curvature = average_curvature
                max_curvature_vector = direction_vector
        bar.update(i)
        i += 1
    print('Time taken: {}s'.format(timer()-start_time))

    # close estimate of max curvature pricipal direction (along surface)
    direction_k = np.array(max_curvature_vector)
    mag_k = np.sqrt(np.dot(direction_k, direction_k))
    direction_k = direction_k/mag_k
    
    # orthogonality check
    direction_j = saddle_normal
    dot_prod = np.dot(direction_j, direction_k)
    angle = np.arccos(dot_prod)
    print("Estimated angle between j and k vectors: {}".format(np.degrees(angle)))

    direction_i = np.array(min_curvature_vector)
    mag_i = np.sqrt(np.dot(direction_i, direction_i))
    direction_i = direction_i/mag_i
    
    # orthogonality check
    dot_prod = np.dot(direction_k, direction_i)
    angle = np.arccos(dot_prod)
    print("Estimated angle between k and i vectors: {}".format(np.degrees(angle)))

    # rotate k vector towards saddle normal till orthogonal
    direction_k = np.cross(np.cross(direction_j, direction_k), direction_j)
    mag_k = np.sqrt(np.dot(direction_k, direction_k))
    direction_k = direction_k/mag_k

    # rotate k vector towards saddle normal till orthogonal
    direction_i = np.cross(np.cross(direction_j, direction_i), direction_j)
    mag_i = np.sqrt(np.dot(direction_i, direction_i))
    direction_i = direction_i/mag_i
    
    # # 2 direction i exist
    # direction_i = np.cross(direction_j, direction_k)
    # mag_i = np.sqrt(np.dot(direction_i, direction_i))
    # direction_i = direction_i/mag_i

    # find point on the end of surface in i direction and -i direction
    i_pos_idx = mesh_3mm.find_closest_point(3*direction_i + saddle_point)
    i_neg_idx = mesh_3mm.find_closest_point(-3*direction_i + saddle_point)

    if(curvatures_3mm[i_pos_idx] > curvatures_3mm[i_neg_idx]):
        direction_i = -1*direction_i

    # mag_j = np.sqrt(np.dot(direction_j, direction_j))
    # direction_j = direction_j/mag_j

    # verify angle
    angle_ik = np.degrees(np.arccos(np.dot(direction_i, direction_k)))
    angle_ij = np.degrees(np.arccos(np.dot(direction_i, direction_j)))
    angle_kj = np.degrees(np.arccos(np.dot(direction_k, direction_j)))

    print("Angles:")
    print("i to k: {}".format(angle_ik))
    print("i to j: {}".format(angle_ij))
    print("k to j: {}".format(angle_kj))

    # Re-Orient and Plot coordinates
    saddle_point = saddle_point + shift_vector

    print("Plotting mesh and coordinates...")
    plotter.add_arrows(saddle_point, direction_i, mag=2, color="green")
    plotter.add_arrows(saddle_point, direction_k, mag=2, color="blue")
    plotter.add_arrows(saddle_point, direction_j, mag=2, color="red")

    cloud = pv.PolyData(reoriented_surface)
    reoriented_surface = cloud.delaunay_2d()
    plotter.add_mesh(reoriented_surface, show_edges=False)
    plotter.add_mesh(reoriented_surface, scalars=curvatures_3mm, show_edges=False)

    plotter.add_mesh(mesh_original, show_edges=True, color="red")
    plotter.add_mesh(mesh_fitted, show_edges=True, color="blue")
    plotter.add_points(np.array([saddle_point]), render_points_as_spheres=True, point_size = 30, color="orange")
    # plotter.add_points(np.array([min_curvature_point]), render_points_as_spheres=True, point_size = 30, color="green")
    # plotter.add_points(np.array([max_curvature_point]), render_points_as_spheres=True, point_size = 30, color="yellow")

    if show_figure_flag:
        plotter.show()

if __name__ == "__main__":
    main()