
# 📌 Codes paper: "Finite element discretization of nonlinear models of ultrasound heating"

This Git-Hub repository contains the codes to reproduce the numerical examples shown in Section 7 of the paper entitled "Finite element discretization of nonlinear models of ultrasound heating".

> **Authors:** [**J. Careaga**](https://scholar.google.com/citations?user=-SYWkN8AAAAJ&hl=es), [**B. Dörich**](https://scholar.google.com/citations?user=h9b6i00AAAAJ&hl=en), [**V. Nikolić**](https://scholar.google.com/citations?user=73kZ9csAAAAJ&hl=en)

The numerical schemes are programmed in Python version 3.12.1, making use of the open source software [**FEniCSx**](https://fenicsproject.org/) version 0.7.3, instalation via

```python 
conda create -n fenicsx-env
conda activate fenicsx-env
conda install -c conda-forge fenics-dolfinx mpich pyvista
```

The following packages need to be instaled
There are 3 scripts, one for each example: 



