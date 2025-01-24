
# 📌 Codes paper: "Finite element discretization of nonlinear models of ultrasound heating"

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
The sequence of space discretizations are set through the variable Ms (array), and the time steps divisions are accounted with Ns (array). The output corresponds to a $\texttt{\color{brown}mat}$ file, which contains vector of $L^2$-errors at the final times for each time step. After computing the errors for ($\color{purple}\texttt{modeltype}$,$\color{purple}\texttt{timescheme}$,$\color{purple}\texttt{ell}$) equal to ($\color{teal}\texttt{"W"}$,$\color{teal}\texttt{"Euler"}$,$\color{teal}\texttt{1}$)



There are 3 scripts, one for each example: 



