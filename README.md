# TMC_JS

## Description
An open-source implementation of the automated trapeziometacarpal (TMC) joint coordinate system (JCS) definition by ***[Halilaj, et al. 2013](https://doi.org/10.1016/j.jbiomech.2012.12.002)***. The `tmc_js.py` script provides all functions necessary to compute bone coordinate systems for the first metacarpal (MC1) and trapezium (TRP). Then, a TMC joint coordinate system can be defined using the JCS in *Halilaj, et al. 2013*.

## How to Run:   

**1. Python and Packages:**  
This script was developed using Python v3.8.5. Ensure your Python environment includes the following packages:
- NumPy v1.23.3
- PyVista v0.44.0
- SimpleITK v2.0.2
- scikit-learn v1.3.2

**2. Bone Segmentation and Joint Surface Extraction:**  
Generate a binary bone segmentation mask for your MC1 and TRP bones. Then, use the masks to manually crop the MC1 and TRP joint surfaces. 
1. Using 3D Slicer (v5.6.2), load your bone masks as a segmentation.
2. Convert the segmentation to a model (right click and export visible segments to models)
3. Use the Surface Toolbox module to smooth the model with the following parameters:
    - Select "Clean"
    - Select "Smooth"
        - Taubin
        - Iterations = 15
        - Pass band = 0.5
    - Fill holes (default)
4. Use the Markups module to draw a curve representing the joint surface:
    - Use closed curve
    - Under curve settings:
        - Curve Type = Spline
        - Constrain curve to smoothed model
5. Use the Dynamic Modeler module to cut out the joint surface:
    - Select curve cut
    - Select the smoothed model and curve you created
    - The output node you want is “Inside model”
6. Use the Surface Toolbox again to smooth out the joint surface and fill holes:
    - Select "Clean"
    - Select "Smooth":
        - Taubin
        - Iterations = 15
        - Pass band = 0.1
    - Fill holes (default)
    - Extract largest component
7. Save the joint surface as a VTK PolyData (.vtk)

**3. Get the MC1 and TRP Coordinate Systems:**  
Finally, run the `tmc_js.py` script with the bone masks (NIfTI images) and joint surfaces (.vtk files). It will output a .CSV file for each bone with the saddle point (centre of the bone coordinate system) and bone coordinate system axes. There is an opptional flag to visualize the bone coordinate systems (default = On).

**4. Get the TMC JCS:**  
As per *Halilaj, et al. 2013*, the TMC JCS is composed as follows:
- $e1 = Z_{TRP}$, oriented ulnar to radial
- $e3 = X_{MC1}$, oriented dorsal to volar
- $e2 = {e1} \times {e3}$, oriented distal to proximal
