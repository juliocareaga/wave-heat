
# 📌 Codes paper:

# "Finite element discretization of nonlinear models of ultrasound heating"

This Git-Hub repository contains the codes to reproduce the numerical examples shown in Section 7 of the paper entitled "Finite element discretization of nonlinear models of ultrasound heating".

> **Authors:** [**J. Careaga**](https://scholar.google.com/citations?user=-SYWkN8AAAAJ&hl=es), [**B. Dörich**](https://scholar.google.com/citations?user=h9b6i00AAAAJ&hl=en), [**V. Nikolić**](https://scholar.google.com/citations?user=73kZ9csAAAAJ&hl=en)

The numerical schemes (Implicit Euler, BDF2 and Newmark in time) are programmed in Python 
<img src="https://raw.githubusercontent.com/marwin1991/profile-technology-icons/refs/heads/main/icons/python.png"  width="15" height="15" /> version 3.12.1, making use of the open source software [FEniCSx](https://fenicsproject.org/) version 0.7.3, installation of the latest version with [Anaconda](https://docs.anaconda.com/anaconda/install/) 
<img src="https://github.com/tandpfun/skill-icons/blob/main/icons/Anaconda-Dark.svg" width="15" height="15" />
via:

```console 
conda create -n fenicsx-env
conda activate fenicsx-env
conda install -c conda-forge fenics-dolfinx mpich pyvista
```

The following Python packages are need: **mpi4py**, **numpy** and **scipy**. The main python codes are 3: **wave-heat-error.py** for the error computations, **wave-heat-example2.py** for the Westervelt example, and **wave-heat-example3.py** for the Kuznetsov example.

# **Error computations** 

Code **wave-heat-error.py** requires setting $\color{purple}\texttt{modeltype}$ as $\color{teal}\texttt{"W"}$ for the Westervelt model or $\color{teal}\texttt{"K"}$ for the case of Kuznetsov's equation, $\color{purple}\texttt{timescheme}$ which defines the time approximation $\color{teal}\texttt{"Euler"}$, $\color{teal}\texttt{"BDF2"}$ or $\color{teal}\texttt{"Newmark"}$, and the polynomial degree $\color{purple}\texttt{ell}$ (integer). Then run:
```console
python3 wave-heat-error.py
```
The sequence of space discretizations are set through the variable Ms (array), and the time steps divisions are accounted with Ns (array). The output corresponds to a $\texttt{\color{brown}mat}$ file, which contains vector of $L^2$-errors at the final times for each time step. After computing the errors for ($\color{purple}\texttt{modeltype}$, $\color{purple}\texttt{timescheme}$, $\color{purple}\texttt{ell}$) equals to

```console
("W", "Euler", 1), ("W", "BDF2", 2), ("W", "BDF2", 3)
("K", "Euler", 1), ("K", "BDF2", 2), ("K", "BDF2", 3)
```
The error plots can be obtained with **Matlab** or **Octabe** making use of **ploterror.m** changing the variable $\color{purple}\texttt{modeltype}$ as before. Note that this script requires the mat files produced with the input values explained above.

# Example 2

The main script is **wave-heat-example2.py**, and the simulation corresponds to the Westervelt equation with a manufactured initial pressure. The outputs correspond to $\texttt{\color{brown}xdmf}$ and $\texttt{\color{brown}h5}$ files, which contain the pair of solutions (pressure and temperature) at the time points saved. The xdmf file can be loaded with **Paraview**, and the procedure to obtain the images for the pressure is described next:

```console
Use calculator:  coords + (0.025+0.6e-7*pressure)*kHat
Click on: Coordinates Result, Tcoords
Use clip: Origin: [0,0,0.002], Normal: [0,0,-2]
Center the image
Color bar: Horizontal, Any location, Position [0.3, 0.25], font 30
Lighting: Interpolation Flat, Speculator: 1, Specular Power: 4, Ambient: 0, Diffuse: 0.9
```
Then, set the angle of the camera using **camera-sims.pvcc** and the images are obtained taking screenshots for the desired time points. The plots for the temperature are obtained similarly without making use of the calculator and changing to a different colormap. The labels corresponding to the variable are simply added with Latex. The required mesh is also contained in this repository and corresponds to **mesh-new.xdmf**. Note that the file **mesh-new.h5** is also needed.


https://github.com/user-attachments/assets/ef8ced18-8d48-40e9-9371-390525bde43d



# Example 3

The main script is **wave-heat-example3.py**, and the simulation corresponds to the Kuznetsov equation with a manufactured source term. The outputs correspond to $\texttt{\color{brown}xdmf}$ and $\texttt{\color{brown}h5}$ files, which contain the pair of solutions (pressure and temperature) at the time points saved as in example 2. The plots are obtained following the same steps as in Example2.

https://github.com/user-attachments/assets/93022184-9963-46ea-9fa2-55a69b1f4e4f



