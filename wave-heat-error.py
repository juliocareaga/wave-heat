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
import ufl
import sys
import os
import scipy.io

#====================================================================
modeltype  = 'W'      ## chose W: Westervelt, K: Kuznetsov
timescheme = "Euler"  ## chose 'Euler', 'BDF2', 'Newmark'
ell        = 1
#-----------------------------
Ms = np.array([2,4,8,16,32,64]) # space
Ns = np.array([2,4,8,16,32,64]) # time
T    = 0.8
has  = 1/Ms
taus = T/Ns
#====================================================================
num_space = Ms.size
num_time  = Ns.size

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
gamma   = 0.85 # Newmark parameter
beta    = 0.45 # Newmark parameter
alphanl = 0.8 # nonlocal time derivative exponent
max_fpi = 20    # maximum number of linear iterations for each time step
tol     = 1e-15 # relative error allowed to approve convergence of a step

#--------------------------------------------------------------------------
kappa = 1.0
## Functions:
speed = lambda ss: 1529.3 + 1.6856*ss + 6.1131e-2*ss**2 - 2.2967e-3*ss**3 + 2.2657e-5*ss**4 - 7.1795e-8*ss**5
qq    = lambda ss: speed(ss)**2
bb    = lambda ss: 2.0*alpha*(speed(ss)**3)/(omega*omega)
#--------------------------------------------------------
omg_b = lambda ss: 0.0005 + 0.0001*ss
GG    = lambda ps, pt, ss: zeta/(rhoa*Ca)*(bb(ss)/(qq(ss)*qq(ss)))*pt*pt

##==================================================================
## Manufactured solutions
##------------------------------------------------------------------
exact_soln = 1

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

if exact_soln == 1:
    nw1 = 1.0  # 1
    nw2 = 1.0  # 1
    csp1   = 1.0
    csp2   = 1.0
    ctime1 =  2.0    
    ctime2 = -2.0    
    #--------------------
    frec   = 1e-6
    kappa  = 1.0
    nu     = 1e-5
    #---------------------
    
    alpha = (4.5/1e6)*frec #
    wfrec = 2*np.pi*frec   #
    
    speed = lambda ss: 1529.3 + 1.6856*ss + 6.1131e-2*ss**2   #
    qq    = lambda ss: speed(ss)**2                           #
    bb    = lambda ss: 2.0*alpha*(speed(ss)**3)/(omega*omega) #
    #--------------------------------------------------------
    omg_b = lambda ss: 1.0 + 0*ss
    if modeltype == "K":
        # Kuznetsov:
        GG = lambda ps, pt, ss: (rho**2)*( (alpha/(rho*speed(ss)))*ps*ps)
    elif modeltype == "W":
        # Westervelt:    
        GG = lambda ps, pt, ss: .5*( (alpha/(rho*speed(ss)))*ps*ps  +  2.0*(bb(ss)/(rho*speed(ss)**4))*pt*pt )
    
    #-----------------------------------------------------------------------------------------
    u_exakt = lambda t_eval, x: csp1*np.exp(ctime1*t_eval) * np.sin(nw1*np.pi*x[0]) * np.sin(nw1*np.pi*x[1])    
    Delta_u = lambda t_eval, x: u_exakt(t_eval,x)*(-2*(nw1*np.pi)**2)
    dt_2_u  = lambda t_eval, x: u_exakt(t_eval,x)*(ctime1**2)   
    #-----------------------------------------------------------------------------------------
    # v playing the role of \partial_t u
    v_exakt = lambda t_eval, x: ctime1*u_exakt(t_eval,x)
    Delta_v = lambda t_eval, x: ctime1*u_exakt(t_eval,x)*(-2*(nw1*np.pi)**2)
    
    dotgrad_exakt = lambda t_eval, x: (ctime1*(nw1*np.pi*csp1*np.exp(ctime1*t_eval))**2)*\
                                      0.5*(1.0 - np.cos(2.0*nw1*np.pi*x[0]) * np.cos(2.0*nw1*np.pi*x[1]))
                                          
    ##==============================================================================================================================    
    theta_exakt = lambda t_eval, x: csp2*np.exp(ctime2*t_eval) * np.sin(nw2*np.pi*x[0]) * np.sin(nw2*np.pi*x[1])
    dt_theta    = lambda t_eval, x: ctime2*theta_exakt(t_eval,x)
    Delta_theta = lambda t_eval, x: theta_exakt(t_eval,x)*(-2*(nw2*np.pi)**2)  
    ##==============================================================================================================================  

if modeltype == 'W':
    kk  = lambda ss: -betaac/(rho*qq(ss))
    FF1 = lambda ss, p_ast, pt: (1.0 - 2.0*kk(ss + thetaa)*p_ast)
    FF  = lambda ss, p_ast, pt: 2.0*kk(ss + thetaa)*pt*pt
    #---------------------------------------------------------------------------------------------------------------------------
    ffwave_ex = lambda tt,x: (1.0 - 2.0*kk( theta_exakt(tt,x) + thetaa) * u_exakt(tt,x) ) * dt_2_u(tt,x)\
                            - qq( theta_exakt(tt,x) + thetaa ) * Delta_u(tt,x)\
                            - bb( theta_exakt(tt,x) + thetaa ) * Delta_v(tt, x)\
                            - 2*kk(theta_exakt(tt,x) + thetaa)*v_exakt(tt,x)**2.0
                                
elif modeltype == 'K':
    kk  = lambda ss: -5.0/(qq(ss))
    
    FF1 = lambda ss, p_ast, pt: (1.0 - 2.0*kk(ss + thetaa)*pt)
    FF  = lambda ss, p_ast, pt:  2.0*ufl.dot(ufl.grad(p_ast),ufl.grad(pt))
    #-----------------------------------------------------------------------------------------------------
    ffwave_ex = lambda tt,x: (1.0 - 2.0*kk( theta_exakt(tt,x) + thetaa) * v_exakt(tt,x) ) * dt_2_u(tt,x)\
                            - qq( theta_exakt(tt,x) + thetaa ) * Delta_u(tt,x)\
                            - bb( theta_exakt(tt,x) + thetaa ) * Delta_v(tt, x)\
                            - 2*kk(theta_exakt(tt,x) + thetaa)*(dotgrad_exakt(tt,x))

ggheat_ex = lambda tt, x: dt_theta(tt,x) - kappa*Delta_theta(tt,x)\
                          + nu*omg_b(theta_exakt(tt, x) + thetaa)*theta_exakt(tt, x)\
                          - GG(u_exakt(tt,x), v_exakt(tt,x), theta_exakt(tt, x) + thetaa)

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
    
#------------------------------------------------------------------------------------------------
FF_Euler = lambda ss: (tau**2)*qq(ss + thetaa) + tau*bb(ss + thetaa)            # Euler
FF_BDF2  = lambda ss: (tau**2)*qq(ss + thetaa) + (3./2.)*tau*bb(ss + thetaa)    # BDF2
FF_NWMRK = lambda ss: (tau**2)*beta*qq(ss + thetaa) + gamma*tau*bb(ss + thetaa) # Newmark
Q = lambda ss: qq(ss + thetaa)

###=============================================================================================
### Spatial operators:
def spaceop_wave_Newmark(tt, uu, v, theta, pp, dp, pp_pred, dp_pred):
    ff.interpolate(lambda x: ffwave_ex(tt,x))
    a_wave  = FF1(theta,pp,dp)*uu*v*ufl.dx + ufl.dot(ufl.grad(uu),ufl.grad(FF_NWMRK(theta)*v))*ufl.dx
    #--------------------------------------------------------------------
    f_wave  = FF(theta,pp,dp)*v*ufl.dx 
    f_wave += - ufl.dot(ufl.grad(pp_pred), ufl.grad(Q(theta)*v))*ufl.dx
    f_wave += - ufl.dot(ufl.grad(dp_pred),ufl.grad(bb(theta + thetaa)*v))*ufl.dx
    f_wave += ff*v*ufl.dx
    return a_wave, f_wave

def spaceop_wave_Euler(tt, uu, v, theta, pp, pp1, pp2, dp):
    ## pp1 = p^{n-1}; pp2 = p^{n-2}
    ff.interpolate(lambda x: ffwave_ex(tt,x))        
    a_wave  = FF1(theta,pp,dp)*uu*v*ufl.dx + ufl.dot(ufl.grad(uu),ufl.grad(FF_Euler(theta)*v))*ufl.dx
    #--------------------------------------------------------------------
    f_wave  = - FF1(theta,pp,dp)*(-2.0*pp1 + pp2)*v*ufl.dx    
    f_wave += (tau**2)*FF(theta, pp, dp)*v*ufl.dx
    f_wave += tau*ufl.dot(ufl.grad(pp1), ufl.grad(bb(theta + thetaa)*v))*ufl.dx
    f_wave += (tau**2)*ff*v*ufl.dx
    return a_wave, f_wave

def spaceop_wave_BDF2(tt, uu, v, theta, pp, pp1, pp2, pp3, dp):
    ## pp1 = p^{n-1}; pp2 = p^{n-2}
    ff.interpolate(lambda x: ffwave_ex(tt,x))
    a_wave  = FF1(theta, pp, dp)*2.*uu*v*ufl.dx + ufl.dot(ufl.grad(uu),ufl.grad(FF_BDF2(theta)*v))*ufl.dx
    #--------------------------------------------------------------------
    f_wave  = - FF1(theta, pp, dp)*( -5.*pp1 + 4.*pp2 - pp3 )*v*ufl.dx    
    f_wave += (tau**2)*FF(theta, pp, dp)*v*ufl.dx
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

## Heat ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def timeop_heat_Euler(tt, theta, theta1, auxpp, auxdp):

    gg.interpolate(lambda x: ggheat_ex(tt,x))
    a_heat  = zz*phi*ufl.dx 
    a_heat += tau*kappa*ufl.dot(ufl.grad(zz), ufl.grad(phi))*ufl.dx
    a_heat += tau*nu*omg_b(theta + thetaa)*zz*phi*ufl.dx
    
    f_heat  = theta1*phi*ufl.dx 
    f_heat += tau*GG(auxpp, auxdp, theta + thetaa)*phi*ufl.dx
    f_heat += tau*gg*phi*ufl.dx
    
    heat_problem = dfpet.LinearProblem(a_heat, f_heat, bcs = [bcHeat])
    theta = heat_problem.solve()
    
    dtheta.x.array[:] = (theta.x.array - theta1.x.array)/tau
    theta1.x.array[:] = theta.x.array
    return theta, theta1, dtheta

def timeop_heat_BDF2(tt, theta, theta1, theta2, auxpp, auxdp):
    
    gg.interpolate(lambda x: ggheat_ex(tt,x))
    a_heat  = (3.0/2.0)*zz*phi*ufl.dx
    a_heat += tau*kappa*ufl.dot(ufl.grad(zz), ufl.grad(phi))*ufl.dx
    a_heat += tau*nu*omg_b(theta + thetaa)*zz*phi*ufl.dx
    
    f_heat  = - (-2.0*theta1 + 0.5*theta2)*phi*ufl.dx
    f_heat += tau*GG(auxpp, auxdp, theta + thetaa)*phi*ufl.dx
    f_heat += tau*gg*phi*ufl.dx
    
    heat_problem = dfpet.LinearProblem(a_heat, f_heat, bcs = [bcHeat])
    theta = heat_problem.solve()
    
    dtheta.x.array[:] = ((3./2.)*theta.x.array - 2.*theta1.x.array + .5*theta2.x.array)/tau
        
    theta2.x.array[:] = theta1.x.array
    theta1.x.array[:] = theta.x.array
    return theta, theta1, theta2, dtheta
    
L2error_pp  = np.zeros([num_space, num_time]) 
L2error_dp  = np.zeros([num_space, num_time]) 
L2error_ddp = np.zeros([num_space, num_time]) 
L2error_theta  = np.zeros([num_space, num_time])
L2error_dtheta = np.zeros([num_space, num_time])

H1error_pp = np.zeros([num_space, num_time])
H1error_dp = np.zeros([num_space, num_time])
H1error_theta  = np.zeros([num_space, num_time])
H1error_dtheta = np.zeros([num_space, num_time])

for i in range(num_space):
    M = Ms[i]
    domain = mesh.create_unit_square(MPI.COMM_WORLD, M, M, mesh.CellType.triangle)
    print('mesh level ' + str(i + 1) + ' of ' + str(num_space))
    # Vector spaces
    Hh = fem.FunctionSpace(domain, ("CG", ell))
    Qh = fem.FunctionSpace(domain, ("CG", ell))
    
    
    Hh2 = fem.FunctionSpace(domain, ("CG", ell+2))
    Qh2 = fem.FunctionSpace(domain, ("CG", ell+2))
    
    
    bcWave, bcHeat = boundaries(Hh,Qh,domain)
    #=================================================================
    # Trial functions
    uu = ufl.TrialFunction(Hh) # pressure
    zz = ufl.TrialFunction(Qh) # temperature
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
    # Exact solutions:
    pp_ex     = fem.Function(Hh2)
    dp_ex     = fem.Function(Hh2)
    ddp_ex    = fem.Function(Hh2)
    theta_ex  = fem.Function(Qh2)
    dtheta_ex = fem.Function(Qh2)
    ff        = fem.Function(Hh2) # Right-hand side
    gg        = fem.Function(Qh2) # Right-hand side
    
    ###======================================================================
    
    for j in range(num_time):
        t   = 0
        inc = 0
        print('::::  meshsize' + str(i) + ' :: time level ' + str(j+1) + ' of '+str(num_time))
        N   = Ns[j]        
        tau = T/N
       
        theta_ex.interpolate(lambda x: theta_exakt(0.0,x))        
        pp_ex.interpolate(   lambda x:     u_exakt(0.0,x))
        dp_ex.interpolate(   lambda x:     v_exakt(0.0,x))
             
        theta = Ritz_projection(theta_ex, Qh, Qh2, bcHeat)
        pp    = Ritz_projection(pp_ex, Hh, Hh2, bcWave)
        dp    = Ritz_projection(dp_ex, Hh, Hh2, bcWave)
                  
        ##-----------------------------------------------------------------------
        ## at t=0: pp = pp1 = p_0;  pp2 = p_{-1} solves for pp at n+1
        pp1.x.array[:] = pp.x.array
        pp2.x.array[:] = pp1.x.array - tau*dp.x.array
        pp3.x.array[:] = pp2.x.array      ## = p_{-1}, when t = 2tau
        theta1.x.array[:] = theta.x.array
        theta2.x.array[:] = theta1.x.array                        

        if timescheme == "Euler":
            wave_strtimeop = "pp, pp1, pp2, dp, ddp = timeop_wave_Euler(tnp, pp, pp1, pp2, dp)"
            heat_strtimeop = "theta, theta1, dtheta = timeop_heat_Euler(tnp, theta, theta1, pp, dp)"
            
        elif timescheme == "BDF2":
            wave_strtimeop = "pp, pp1, pp2, pp3, dp, ddp    = timeop_wave_BDF2(tnp, pp, pp1, pp2, pp3, dp)"
            heat_strtimeop = "theta, theta1, theta2, dtheta = timeop_heat_BDF2(tnp, theta, theta1, theta2, pp, dp)"
            pp, pp1, pp2, dp, ddp = timeop_wave_Euler(tau, pp, pp1, pp2, dp)
            theta, theta1, dtheta = timeop_heat_Euler(tau, theta, theta1, pp, dp)
            
        elif timescheme == "Newmark":
            wave_strtimeop = "ddp, dp, pp, pp_pred, dp_pred = timeop_wave_Newmark(tnp, ddp, dp, pp, pp_pred, dp_pred)"
            heat_strtimeop = "theta, theta1, theta2, dtheta = timeop_heat_BDF2(tnp, theta, theta1, theta2, pp, dp)"
            
            dp.interpolate(lambda x:  v_exakt(0.0,x))
            ddp.interpolate(lambda x: dt_2_u(0.0,x))
            pp_pred.x.array[:] = pp.x.array + tau*dp.x.array + (tau**2.0)*(0.5 - beta)*ddp.x.array
            dp_pred.x.array[:] = dp.x.array + (1 - gamma)*tau*ddp.x.array
            
        ###=============================================================================================        
        frec_save = 1
        addfolder = ""
        listvecs  = []        
                
        for tk in range(1,N):
            tn   = tk*tau
            tnp  = (tk + 1)*tau
            inc += 1
            #----------------------------------------------------------------------------------------
            ## Solve wave equation at t^(n+1)
            exec(wave_strtimeop)
            #----------------------------------------------------------------------------------------
            ## Solve heat equation at t^(n+1)        
            exec(heat_strtimeop)
            #----------------------------------------------------------------------------------------
            pp_ex.interpolate(   lambda x:     u_exakt(tnp,x))
            dp_ex.interpolate(   lambda x:     v_exakt(tnp,x))
            ddp_ex.interpolate(  lambda x:      dt_2_u(tnp,x))
            theta_ex.interpolate(lambda x: theta_exakt(tnp,x)) 
            dtheta_ex.interpolate(lambda x:    dt_theta(tnp,x)) 
            
        #============================================================================================
        # error computation
        L2_pp     = fem.assemble_scalar(fem.form(((pp_ex - pp)**2)*ufl.dx))
        L2_dp     = fem.assemble_scalar(fem.form(((dp_ex - dp)**2)*ufl.dx))
        L2_ddp    = fem.assemble_scalar(fem.form(((ddp_ex - ddp)**2)*ufl.dx))
        
        L2_theta  = fem.assemble_scalar(fem.form(ufl.inner(theta_ex - theta,theta_ex - theta)*ufl.dx))
        L2_dtheta = fem.assemble_scalar(fem.form(ufl.inner(dtheta_ex - dtheta,dtheta_ex - dtheta)*ufl.dx))
        #---------------------------------------------------------------------------------------------------------------------------
        H1_pp     = fem.assemble_scalar(fem.form(ufl.dot(ufl.grad(pp_ex - pp), ufl.grad(pp_ex - pp)) * ufl.dx))
        H1_dp     = fem.assemble_scalar(fem.form(ufl.dot(ufl.grad(dp_ex - dp), ufl.grad(dp_ex - dp)) * ufl.dx))
        H1_theta  = fem.assemble_scalar(fem.form(ufl.dot(ufl.grad(theta_ex - theta), ufl.grad(theta_ex - theta)) * ufl.dx))
        H1_dtheta = fem.assemble_scalar(fem.form(ufl.dot(ufl.grad(dtheta_ex - dtheta), ufl.grad(dtheta_ex - dtheta)) * ufl.dx))
        #---------------------------------------------------------------------------------------------------------------------------
           
        L2error_pp[i,j]     = np.sqrt(domain.comm.allreduce(L2_pp,     op = MPI.SUM))
        L2error_dp[i,j]     = np.sqrt(domain.comm.allreduce(L2_dp,     op = MPI.SUM))
        L2error_ddp[i,j]    = np.sqrt(domain.comm.allreduce(L2_ddp,    op = MPI.SUM))
        L2error_theta[i,j]  = np.sqrt(domain.comm.allreduce(L2_theta,  op = MPI.SUM))
        L2error_dtheta[i,j] = np.sqrt(domain.comm.allreduce(L2_dtheta, op = MPI.SUM))
        
        H1error_pp[i,j]     = np.sqrt(domain.comm.allreduce(H1_pp,     op = MPI.SUM))
        H1error_dp[i,j]     = np.sqrt(domain.comm.allreduce(H1_dp,     op = MPI.SUM))
        H1error_theta[i,j]  = np.sqrt(domain.comm.allreduce(H1_theta,  op = MPI.SUM))
        H1error_dtheta[i,j] = np.sqrt(domain.comm.allreduce(H1_dtheta, op = MPI.SUM))
            
###======================================
### save all error vectors in one mat file:       
scipy.io.savemat('errors_deg' + str(ell) + '_' + timescheme + '_'+ modeltype + '.mat', 
                 {'L2error_pp': L2error_pp, 'L2error_dp': L2error_dp, 'L2error_theta': L2error_theta, 'L2error_dtheta': L2error_dtheta,
                  'H1error_pp': H1error_pp, 'H1error_dp': H1error_dp, 'H1error_theta': H1error_theta, 'H1error_dtheta': H1error_dtheta,
                  'L2error_ddp': L2error_ddp, 'hh': has, 'taus': taus, 'T': T, 'degree': ell})

