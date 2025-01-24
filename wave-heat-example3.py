#=====================================================================
# Code by Julio Careaga
# main code paper: FINITE ELEMENT DISCRETIZATION OF NONLINEAR MODELS OF ULTRASOUND HEATING
# authors: J. Careaga, B. Dörich and V. Nikolic
#---------------------------------------------------------------------
from dolfinx      import plot, mesh, fem, io, geometry
from dolfinx.io   import XDMFFile, gmshio
from dolfinx.mesh import create_unit_square, locate_entities, meshtags
import dolfinx.fem.petsc as dfpet
##--------------------------------------------------------------------
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import ufl
import sys
import os
import scipy.io

#====================================================================
modeltype     = 'K'     ## chose W: Westervelt, K: Kuznetsov
time_operator = "BDF2"  ## chose 'Euler', 'BDF2', 'Newmark'
ell           = 1
#--------------------------------------------------------------------
tau = 1e-7
T   = 400*tau

## Parameters and functions:
DOLFIN_EPS = 3.0e-16

#--------------------------------------------------------------------------
thetaa  = 37.0 # 37ºC or 310.15 K
betaac  = 6.0  # liver; Connor2002, table 3, variable beta
rho     = 1050.0 # taken as liver; 
rhoa    = 1050.0 # liver; Connor2002, table 3
rhob    = 1030.0 # blood; Connor2002, table 3
Ca      = 3600.0 # liver; Connor2002, table 3, variable C
Cb      = 3620.0 # blood; Connor2002, table 3, variable C
kappaa  = 0.512  # liver; Connor2002, table 3, variable k
frec    = 100000        # frequency function g(t)
wfrec   = 2*np.pi*frec  # omega function g(t)
gg0     = 1.0e+9        # amplitude function g(t)
omega   = 2*np.pi*frec  # angular frequency;
alpha   = (4.5/1e6)*frec    # liver; Connor2002, table 3; (alpha_0/1e6)*f
zeta    = 2.0               # constant in source term G at heat equation
kappa   = kappaa/( rhoa*Ca) # diffusion; heat equation
nu      = rhob*Cb/(rhoa*Ca) # perfusion factor; heat equation
gamma   = 0.85  # Newmark parameter
beta    = 0.45  # Newmark parameter
alphanl = 0.8   # nonlocal time derivative exponent
max_fpi = 50    # maximum number of linear iterations for each time step
tol     = 1e-8  # relative error allowed to approve convergence of a step


kappa = 1.0
#--------------------------------------------------------------------------
## Functions:
speed = lambda ss: 1529.3 + 1.6856*ss + 6.1131e-2*ss**2 - 2.2967e-3*ss**3 + 2.2657e-5*ss**4 - 7.1795e-8*ss**5
qq    = lambda ss: speed(ss)**2
bb    = lambda ss: 2.0*alpha*(speed(ss)**3)/(omega*omega)

#--------------------------------------------------------
omg_b = lambda ss: 0.0005 + 0.0001*ss
omg_b = lambda ss: 1.0 + 0*ss

##==================================================================
## Manufactured solutions
##------------------------------------------------------------------
activatelinear = 0 ## 1 linear;  0 : nolineal

if activatelinear == 1:
    lnr = 1e-30
    labelin = "L"
else:
    lnr = 1.0
    labelin = ""

def Ritz_projection(u_ex, V , V2, bc):
    u_Ritz = ufl.TrialFunction(V)
    v_Ritz = ufl.TestFunction(V)
           
    u_H1 = fem.Function(V2)
    u_H1.interpolate(u_ex)
    
    a_Ritz = ufl.dot(ufl.grad(u_Ritz), ufl.grad(v_Ritz))*ufl.dx
    L_Ritz = ufl.dot(ufl.grad(u_H1),   ufl.grad(v_Ritz))*ufl.dx    
    var_Ritz = dfpet.LinearProblem(a_Ritz, L_Ritz, bcs = [bc])
    
    u_proj = var_Ritz.solve()
    u_proj.x.scatter_forward()    
    return u_proj

if modeltype == 'W':
    kk  = lambda ss: -betaac/(rho*qq(ss))*lnr
    FF1 = lambda ss, p_ast, pt: (1.0 - 2.0*kk(ss + thetaa)*p_ast)
    FF  = lambda ss, p_ast, pt: 2.0*kk(ss + thetaa)*pt*pt                                
    GG  = lambda ps, pt, ss: .5*( (alpha/(rho*speed(ss)))*ps*ps  +  2.0*(bb(ss)/(rho*speed(ss)**4))*pt*pt )
elif modeltype == 'K':   
    kk  = lambda ss: -(betaac - 1.0)/(qq(ss))*lnr    
    FF1 = lambda ss, p_ast, pt:  (1.0 - 2.0*kk(ss + thetaa)*pt)
    FF  = lambda ss, p_ast, pt:   2.0*ufl.dot(ufl.grad(p_ast),ufl.grad(pt))*lnr
    GG = lambda ps, pt, ss: (rho**2)*( (alpha/(rho*speed(ss)))*ps*ps ) 

if activatelinear == 1:
    kk  = lambda ss: 0.0*ss    
    FF1 = lambda ss, p_ast, pt: 1.0 
    FF  = lambda ss, p_ast, pt: 2.0*ufl.dot(ufl.grad(p_ast),ufl.grad(pt))*lnr 


##==================================================================
## Define all functions
##----------------------
                     
def boundaryfunction(tt):
    # function g(t)
    wt = wfrec*tt
    SN, SN4 = np.sin(wt), np.sin(wt/4.0)
    CN, CN4 = np.cos(wt), np.cos(wt/4.0)
    if tt > 2.0*np.pi/wfrec:
        f_bdry   = SN*( 1.0  +  SN4 )
        df_bdry  = wfrec*( CN*(1.0 + SN4 ) + (1.0/4.0)*SN*CN4 )
        ddf_bdry = ((wfrec**2)/16.0)*(8.0*CN*CN4 - 16.0*SN*( 1.0 + SN4 ) - SN*SN4)                
    else:
        f_bdry   =  SN
        df_bdry  =  wfrec*CN
        ddf_bdry = -(wfrec**2)*SN
    return gg0*f_bdry, gg0*df_bdry, gg0*ddf_bdry

def boundaries(Hh,Qh,domain):
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)

    ppD    = fem.Function(Hh)
    thetaD = fem.Function(Qh)
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    boundary_dofs1  = fem.locate_dofs_topological(Hh, fdim, boundary_facets)
    boundary_dofs2  = fem.locate_dofs_topological(Qh, fdim, boundary_facets)
    bcWave = fem.dirichletbc(ppD,    boundary_dofs1)
    bcHeat = fem.dirichletbc(thetaD, boundary_dofs2)
    return bcWave, bcHeat
    
def boundarytag(mydomain,fdimension,conditions,mymarks):
    # function to mark pieces of boundary 
    myfacets, f_idx, f_mrk = [], [], []  
    for i in range(len(mymarks)):
        facet0 = mesh.locate_entities_boundary(mydomain, fdimension, marker = conditions[i])
        f_idx.append(facet0)
        f_mrk.append(np.full_like(facet0,  mymarks[i]))
        myfacets.append(facet0)
    f_idx = np.hstack(f_idx).astype(np.int32)
    f_mrk = np.hstack(f_mrk).astype(np.int32)
    f_sorted = np.argsort(f_idx)
    f_tags = meshtags(mydomain, fdimension, f_idx[f_sorted], f_mrk[f_sorted])
    return f_tags, myfacets

#-----------------------------------------------------------------------------

def subdomtag(mydomain,tdimension, conditions, mymarks):
    # function to mark pieces of boundary 
    mydomains, triangle_idx, triangle_mrk = [], [], []  
    for i in range(len(mymarks)):
        subdom0 = mesh.locate_entities(mydomain, tdimension, marker = conditions[i])
        triangle_idx.append(subdom0)
        triangle_mrk.append(np.full_like(subdom0,  mymarks[i]))
        mydomains.append(subdom0)
        
    triangle_idx    = np.hstack( triangle_idx).astype(np.int32)
    triangle_mrk    = np.hstack( triangle_mrk).astype(np.int32)
    triangle_sorted = np.argsort(triangle_idx)
    triangle_tags   = meshtags(mydomain, tdimension, triangle_idx[triangle_sorted], triangle_mrk[triangle_sorted])
    return triangle_tags, mydomains

#---------------------------------------------------------------------

def initial_p(x00,x11):
    valuepressure = 0.0*x00
    for ii in range(len(x00)):
        
        x0 = x00[ii]
        x1 = x11[ii]
 
        nro_ang = 3.5;
        lowery, uppery = -0.018, 0.03
        length0 = uppery - lowery;
        anglemin, anglemax = -np.pi/nro_ang, np.pi/nro_ang;
    
        pend1 = np.cos(anglemin)/np.sin(anglemin);
        pend2 = np.cos(anglemax)/np.sin(anglemax);    
        yval1 = -pend1*x0 + uppery;
        yval2 = -pend2*x0 + uppery;

        if x1 < yval1 and x1 < yval2:
            radio = np.sqrt(x0**2 + (x1 - uppery)**2)
            
            if  radio <= length0:
                angle = np.arctan(x0/(x1 - uppery))
                xx    = ( 3.0*np.pi/(uppery - lowery) )*(length0 - radio)
                ## Expresion made considering xx from 0 to 3*np.pi
                valuepressure[ii] = np.cos((nro_ang/2)*angle)*1e6*np.exp(-xx)*xx*np.sin(5*xx);
    return valuepressure  

def initial_dp(x00,x11):
    valuepressure = 0.0*x00
    for ii in range(len(x00)):
        
        x0 = x00[ii]
        x1 = x11[ii]

        nro_ang = 3.5;
        lowery, uppery = -0.018, 0.03
        length0 = uppery - lowery;
        anglemin, anglemax = -np.pi/nro_ang, np.pi/nro_ang;
    
        pend1 = np.cos(anglemin)/np.sin(anglemin);
        pend2 = np.cos(anglemax)/np.sin(anglemax);    
        yval1 = -pend1*x0 + uppery;
        yval2 = -pend2*x0 + uppery;

        if x1 < yval1 and x1 < yval2:    
            #radio = norm([x0, x1 - uppery])
            radio = np.sqrt(x0**2 + (x1 - uppery)**2)
            
            if  radio <= length0:
                angle = np.arctan(x0/(x1 - uppery))      
                aa    = uppery - length0*np.cos(angle)
                xx    = ( 3.0*np.pi/(uppery - lowery) )*(length0 - radio)
                ## Expresion made considering xx from 0 to 3*np.pi
                valuepressure[ii] = np.cos((nro_ang/2)*angle)*1e6*np.exp(-xx)*xx*np.cos(8*xx);
    return valuepressure

def fsource(tt,x00,x11):
    valuepressure = 0.0*x00
    nfun = 40.0
    nro_ang = 3.5;
    lowery, uppery = -0.018, 0.03
    length0 = uppery - lowery;
    anglemin, anglemax = -np.pi/nro_ang, np.pi/nro_ang;
    pend1 = np.cos(anglemin)/np.sin(anglemin);
    pend2 = np.cos(anglemax)/np.sin(anglemax);    

    for ii in range(len(x00)):
        
        x0 = x00[ii]
        x1 = x11[ii]    
        yval1 = -pend1*x0 + uppery;
        yval2 = -pend2*x0 + uppery;

        if x1 < yval1 and x1 < yval2:
            radio = np.sqrt(x0**2 + (x1 - uppery)**2)
            
            if  radio <= length0:
                angle = np.arctan(x0/(x1 - uppery))                
                xx   = (length0 - radio)/length0
                valuepressure[ii] = np.cos((nro_ang/2)*angle)*1e12*xx*(np.exp(-nfun*xx) - np.exp(-nfun))*np.cos(wfrec*tt)
    return valuepressure  
    


#------------------------------------------------------------------------------------------------
FF_Euler = lambda ss: (tau**2)*qq(ss + thetaa) + tau*bb(ss + thetaa)            # Euler
FF_BDF2  = lambda ss: (tau**2)*qq(ss + thetaa) + (3./2.)*tau*bb(ss + thetaa)    # BDF2
FF_NWMRK = lambda ss: (tau**2)*beta*qq(ss + thetaa) + gamma*tau*bb(ss + thetaa) # Newmark
Q        = lambda ss: qq(ss + thetaa)

###=============================================================================================
### Spatial operators:
def spaceop_wave_Newmark(tt, uu, v, theta, pp, dp, pp_pred, dp_pred):
    ff.interpolate(lambda x:  fsource(tt,x[0],x[1]))
    a_wave  = FF1(theta,pp,dp)*uu*v*ufl.dx + ufl.dot(ufl.grad(uu),ufl.grad(FF_NWMRK(theta)*v))*ufl.dx
    #--------------------------------------------------------------------
    f_wave  = FF(theta,pp,dp)*v*ufl.dx 
    f_wave += - ufl.dot(ufl.grad(pp_pred), ufl.grad(Q(theta)*v))*ufl.dx
    f_wave += - ufl.dot(ufl.grad(dp_pred),ufl.grad(bb(theta + thetaa)*v))*ufl.dx
    f_wave += ff*v*ufl.dx
    return a_wave, f_wave

def spaceop_wave_Euler(tt, uu, v, theta, pp, pp1, pp2, dp):
    ## pp1 = p^{n-1}; pp2 = p^{n-2}
    ff.interpolate(lambda x:  fsource(tt,x[0],x[1]))
    a_wave  = FF1(theta,pp,dp)*uu*v*ufl.dx + ufl.dot(ufl.grad(uu),ufl.grad(FF_Euler(theta)*v))*ufl.dx
    #--------------------------------------------------------------------
    f_wave  = - FF1(theta,pp,dp)*(-2.0*pp1 + pp2)*v*ufl.dx    
    f_wave += (tau**2)*FF(theta, pp, dp)*v*ufl.dx
    f_wave += tau*ufl.dot(ufl.grad(pp1), ufl.grad(bb(theta + thetaa)*v))*ufl.dx
    f_wave += (tau**2)*ff*v*ufl.dx
    return a_wave, f_wave

def spaceop_wave_BDF2(tt, uu, v, theta, pp, pp1, pp2, pp3, dp):
    ## pp1 = p^{n-1}; pp2 = p^{n-2}
    ff.interpolate(lambda x:  fsource(tt,x[0],x[1]))
    a_wave  = FF1(theta, pp, dp)*2.*uu*v*ufl.dx + ufl.dot(ufl.grad(uu),ufl.grad(FF_BDF2(theta)*v))*ufl.dx
    #--------------------------------------------------------------------
    f_wave  = - FF1(theta, pp, dp)*( -5.*pp1 + 4.*pp2 - pp3 )*v*ufl.dx    
    f_wave += (tau**2)*FF(theta, pp, dp)*v*ufl.dx
    f_wave += tau*ufl.dot(ufl.grad(2.*pp1 - 0.5*pp2), ufl.grad(bb(theta + thetaa)*v))*ufl.dx
    f_wave += (tau**2)*ff*v*ufl.dx
    return a_wave, f_wave


def spaceop_wave_BDF2_linear(tt, uu, v, theta, pp, pp1, pp2, pp3, dp):
    ## pp1 = p^{n-1}; pp2 = p^{n-2}
    ff.interpolate(lambda x:  fsource(tt,x[0],x[1]))
    a_wave  = 2.*uu*v*ufl.dx + ufl.dot(ufl.grad(uu),ufl.grad(FF_BDF2(theta)*v))*ufl.dx
    #--------------------------------------------------------------------
    f_wave  = - ( -5.*pp1 + 4.*pp2 - pp3 )*v*ufl.dx
    f_wave += tau*ufl.dot(ufl.grad(2.*pp1 - 0.5*pp2), ufl.grad(bb(theta + thetaa)*v))*ufl.dx
    f_wave += (tau**2)*ff*v*ufl.dx
    return a_wave, f_wave


###=============================================================================================
### Time discretizations
def timeop_wave_Newmark(t, ddp, dp, pp, pp_pred, dp_pred):
    error    = tol + 1.0
    iter_fpi = 0    
    
    pp_pred.x.array[:] = pp.x.array + tau*dp.x.array + (tau**2.0)*(0.5 - beta)*ddp.x.array
    dp_pred.x.array[:] = dp.x.array + (1 - gamma)*tau*ddp.x.array
    
    pp.x.array[:] = pp_pred.x.array
    dp.x.array[:] = dp_pred.x.array
    ddp_aux.x.array[:] = ddp.x.array
    
    while error > tol and iter_fpi < max_fpi:
        iter_fpi += 1
        L_wave, rhs_wave = spaceop_wave_Newmark(t, uu, v, theta, pp, dp, pp_pred, dp_pred)                
        wave_problem = dfpet.LinearProblem(L_wave, rhs_wave, bcs = [bcWave])
        ddp = wave_problem.solve()
        pp.x.array[:] = pp_pred.x.array + beta*(tau**2.0)*ddp.x.array
        dp.x.array[:] = dp_pred.x.array + gamma*tau*ddp.x.array
        normpp = np.sqrt(fem.assemble_scalar(fem.form(ddp*ddp*ufl.dx))) + DOLFIN_EPS        
        error  = np.sqrt(fem.assemble_scalar(fem.form((ddp_aux - ddp)**2*ufl.dx)))/normpp     
        ddp_aux.x.array[:] = ddp.x.array
    return ddp, dp, pp, pp_pred, dp_pred


def timeop_wave_Euler(t, pp, pp1, pp2, dp):
    error    = tol + 1.0
    iter_fpi = 0    
    pp_aux.x.array[:] = pp.x.array
    
    while error > tol and iter_fpi < max_fpi:
        iter_fpi += 1
        L_wave, rhs_wave = spaceop_wave_Euler(t, uu, v, theta, pp, pp1, pp2, dp)                 
        wave_problem = dfpet.LinearProblem(L_wave, rhs_wave, bcs = [bcWave])
        pp = wave_problem.solve()        
        normpp = np.sqrt(fem.assemble_scalar(fem.form(pp*pp*ufl.dx))) + DOLFIN_EPS        
        error  = np.sqrt(fem.assemble_scalar(fem.form((pp_aux - pp)**2*ufl.dx)))/normpp     
        pp_aux.x.array[:] = pp.x.array
        dp.x.array[:] = (pp.x.array - pp1.x.array)/tau
        
    ddp.x.array[:] = (pp.x.array - 2.0*pp1.x.array + pp2.x.array)/(tau**2.)
    pp2.x.array[:] = pp1.x.array
    pp1.x.array[:] = pp.x.array

    return pp, pp1, pp2, dp, ddp


def timeop_wave_BDF2(t, pp, pp1, pp2, pp3, dp):
    error    = tol + 1.0
    iter_fpi = 0    
    pp_aux.x.array[:] = pp.x.array    
    while error > tol and iter_fpi < max_fpi:
        iter_fpi += 1
        L_wave, rhs_wave = spaceop_wave_BDF2(t, uu, v, theta, pp, pp1, pp2, pp3, dp)               
        wave_problem = dfpet.LinearProblem(L_wave, rhs_wave, bcs = [bcWave])
        pp = wave_problem.solve()        
        normpp = np.sqrt(fem.assemble_scalar(fem.form(pp*pp*ufl.dx))) + DOLFIN_EPS        
        error  = np.sqrt(fem.assemble_scalar(fem.form((pp_aux - pp)**2*ufl.dx)))/normpp     
        pp_aux.x.array[:] = pp.x.array

        dp.x.array[:]   =   ((3./2.)*pp.x.array - 2.*pp1.x.array + .5*pp2.x.array)/tau
    
    ddp.x.array[:] = (2.*pp.x.array - 5.*pp1.x.array + 4.*pp2.x.array - pp3.x.array)/(tau**2.)
    pp3.x.array[:] = pp2.x.array
    pp2.x.array[:] = pp1.x.array
    pp1.x.array[:] = pp.x.array
    
    return pp, pp1, pp2, pp3, dp, ddp
    
    
def timeop_wave_BDF2_linear(t, pp, pp1, pp2, pp3, dp):  
      
    L_wave, rhs_wave = spaceop_wave_BDF2_linear(t, uu, v, theta, pp, pp1, pp2, pp3, dp)               
    wave_problem = dfpet.LinearProblem(L_wave, rhs_wave, bcs = [bcWave])
    pp = wave_problem.solve()        
    dp.x.array[:]   =   ((3./2.)*pp.x.array - 2.*pp1.x.array + .5*pp2.x.array)/tau
    
    ddp.x.array[:] = (2.*pp.x.array - 5.*pp1.x.array + 4.*pp2.x.array - pp3.x.array)/(tau**2.)
    pp3.x.array[:] = pp2.x.array
    pp2.x.array[:] = pp1.x.array
    pp1.x.array[:] = pp.x.array
    
    return pp, pp1, pp2, pp3, dp, ddp

## Heat ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def timeop_heat_Euler(tt, theta, theta1, auxpp, auxdp):
    a_heat  = zz*phi*ufl.dx 
    a_heat += tau*kappa*ufl.dot(ufl.grad(zz), ufl.grad(phi))*ufl.dx
    a_heat += tau*nu*omg_b(theta + thetaa)*zz*phi*ufl.dx
    
    f_heat  = theta1*phi*ufl.dx 
    f_heat += tau*GG(auxpp, auxdp, theta + thetaa)*phi*ufl.dx
    
    heat_problem = dfpet.LinearProblem(a_heat, f_heat, bcs = [bcHeat])
    theta = heat_problem.solve()
    
    dtheta.x.array[:] = (theta.x.array - theta1.x.array)/tau
    theta1.x.array[:] = theta.x.array
    return theta, theta1, dtheta

def timeop_heat_BDF2(tt, theta, theta1, theta2, auxpp, auxdp):

    a_heat  = (3.0/2.0)*zz*phi*ufl.dx
    a_heat += tau*kappa*ufl.dot(ufl.grad(zz), ufl.grad(phi))*ufl.dx
    a_heat += tau*nu*omg_b(theta + thetaa)*zz*phi*ufl.dx
    
    f_heat  = - (-2.0*theta1 + 0.5*theta2)*phi*ufl.dx
    f_heat += tau*GG(auxpp, auxdp, theta + thetaa)*phi*ufl.dx
    
    heat_problem = dfpet.LinearProblem(a_heat, f_heat, bcs = [bcHeat])
    theta = heat_problem.solve()
    
    dtheta.x.array[:] = ((3./2.)*theta.x.array - 2.*theta1.x.array + .5*theta2.x.array)/tau
        
    theta2.x.array[:] = theta1.x.array
    theta1.x.array[:] = theta.x.array
    return theta, theta1, theta2, dtheta

###=============================================================================================        
frec_save = 10
namesim   = "example3_" + modeltype + labelin

if True:

    # to change:
    with XDMFFile(MPI.COMM_WORLD, "mesh-new.xdmf", "r") as file0:
        domain = file0.read_mesh(name = "Grid")    
    
    # Vector spaces
    Hh = fem.FunctionSpace(domain, ("CG", ell))
    Qh = fem.FunctionSpace(domain, ("CG", ell))
        
    bcWave, bcHeat = boundaries(Hh,Qh,domain)
    #=================================================================
    # Trial functions
    uu = ufl.TrialFunction(Hh) 
    zz = ufl.TrialFunction(Qh) 
    #--------------------------
    # Test function
    v   = ufl.TestFunction(Hh)
    phi = ufl.TestFunction(Qh)
    #===================================================================================
    # Unknowns (pp,dp, ddp) and (theta) and auxiliar variable for fixed point iteration
    pp  = fem.Function(Hh)
    dp  = fem.Function(Hh)
    ddp = fem.Function(Hh)
    ddp_aux = fem.Function(Hh)
    pp_pred = fem.Function(Hh)
    dp_pred = fem.Function(Hh)
    
    # Other variables for Euler or BDFF2
    pp_aux = fem.Function(Hh)
    pp1 = fem.Function(Hh)
    pp2 = fem.Function(Hh)
    pp3 = fem.Function(Hh)
    
    theta  = fem.Function(Qh)
    theta1 = fem.Function(Qh)
    theta2 = fem.Function(Qh)

    dtheta  = fem.Function(Qh)    
    
    ### ex playing the role as initial:
    pp_ex = fem.Function(Hh)
    dp_ex = fem.Function(Hh)

    theta_ex = fem.Function(Qh)
    ff = fem.Function(Hh)

    ###======================================================================
    
    if True:
        t   = 0
        inc = 0
        
        theta.interpolate(lambda x:  0.0*x[0])        
        pp.interpolate(   lambda x:  0.0*x[0])
        dp.interpolate(   lambda x:  0.0*x[0])
                
        ##-----------------------------------------------------------------------
        ## at t=0: pp = pp1 = p_0;  pp2 = p_{-1} solves for pp at n+1
        pp1.x.array[:] = pp.x.array
        pp2.x.array[:] = pp1.x.array - tau*dp.x.array
        pp3.x.array[:] = pp2.x.array      #
        theta1.x.array[:] = theta.x.array
        theta2.x.array[:] = theta1.x.array                        

        if time_operator == "Euler":
            wave_strtimeop = "pp, pp1, pp2, dp, ddp = timeop_wave_Euler(tnp, pp, pp1, pp2, dp)"
            heat_strtimeop = "theta, theta1, dtheta = timeop_heat_Euler(tnp, theta, theta1, pp, dp)"
            
        elif time_operator == "BDF2" and activatelinear == 1:
            wave_strtimeop = "pp, pp1, pp2, pp3, dp, ddp    = timeop_wave_BDF2_linear(tnp, pp, pp1, pp2, pp3, dp)"
            heat_strtimeop = "theta, theta1, theta2, dtheta = timeop_heat_BDF2(tnp, theta, theta1, theta2, pp, dp)"
            pp, pp1, pp2, dp, ddp = timeop_wave_Euler(tau, pp, pp1, pp2, dp)
            theta, theta1, dtheta = timeop_heat_Euler(tau, theta, theta1, pp, dp)
        elif time_operator == "BDF2":
            wave_strtimeop = "pp, pp1, pp2, pp3, dp, ddp    = timeop_wave_BDF2(tnp, pp, pp1, pp2, pp3, dp)"
            heat_strtimeop = "theta, theta1, theta2, dtheta = timeop_heat_BDF2(tnp, theta, theta1, theta2, pp, dp)"
            pp, pp1, pp2, dp, ddp = timeop_wave_Euler(tau, pp, pp1, pp2, dp)
            theta, theta1, dtheta = timeop_heat_Euler(tau, theta, theta1, pp, dp)
        
        elif time_operator == "Newmark":
            wave_strtimeop = "ddp, dp, pp, pp_pred, dp_pred = timeop_wave_Newmark(tnp, ddp, dp, pp, pp_pred, dp_pred)"
            heat_strtimeop = "theta, theta1, theta2, dtheta = timeop_heat_BDF2(tnp, theta, theta1, theta2, pp, dp)"
            
            ddp.interpolate(lambda x: 0.0*x[0])
            pp_pred.x.array[:] = pp.x.array + tau*dp.x.array + (tau**2.0)*(0.5 - beta)*ddp.x.array
            dp_pred.x.array[:] = dp.x.array + (1 - gamma)*tau*ddp.x.array
         
        with io.XDMFFile(domain.comm, namesim + ".xdmf", "w") as xdmf:
            xdmf.write_mesh(domain)
            
            tk = 0
            while t <= T: # t = tn

                tn   = t
                tnp  = t + tau
                t    = tnp
                inc += 1
                tk  += 1
                #----------------------------------------------------------------------------------------
                ## Solve wave equation at t^(n+1)
                exec(wave_strtimeop)
                #----------------------------------------------------------------------------------------
                ## Solve heat equation at t^(n+1)        
                exec(heat_strtimeop)

                if inc % frec_save == 0: 
                    #===================================================
                    # Save to xmdf file:
                    pp.name    = "pressure";      xdmf.write_function(pp,inc)
                    theta.name = "temperature";   xdmf.write_function(theta,inc)

