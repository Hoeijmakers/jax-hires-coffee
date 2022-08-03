#!/usr/bin/env python
# coding: utf-8


import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro import distributions as dist

cpu_cores = 30
numpyro.set_host_device_count(cpu_cores)


import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
from jax import numpy as jnp
from jax import jit
from jax.scipy.optimize import minimize
from jax.random import PRNGKey, split
from functools import partial
import arviz
from corner import corner

import astropy.io.fits as fits
import astropy.constants as const
import astropy.units as u
import h5py

import tayph.util as ut
import tayph.system_parameters as sp
import tayph.functions as fun
import tayph.util as ut
from tayph.vartests import typetest,notnegativetest,nantest,postest,typetest_array,dimtest
from tayph.vartests import lentest
import tayph.operations as ops
import tayph.masking as masking
import copy
from pathlib import Path
from collections import OrderedDict
import pdb
import jax



class species:#Species
  def __init__(self, label, tag):
    self.label = label
    self.tag = tag

@partial(jit, static_argnums=(6,7,8,9,10,11,12,13,14))
def model_jax(p,wl,wlk,kappa_grid,x_kernel,phase,c,gamma,k,m,g,P0,R0,Rs2,n_species):
    T = p[0]
    chi_fe = 10**p[1]
    logratios = p[2] #This is now a jnp.array().
    logk0 = p[3]
    c0 = p[4] #Constant offset to ensure continuum-degeneracy despite filtering.
    lw = p[5]
    vsys = p[6]
    Kp = p[7]
    #Then we compute kappa:
    chi_i = chi_fe * 10 ** logratios
    K = chi_fe * kappa_grid[0] + jnp.dot(chi_i,kappa_grid[1:]) + 10**logk0
    #Then we do the magic:
    H = k*T/m/g
    R = R0 + H*(gamma+jnp.log(P0 * K / g * jnp.sqrt(2*np.pi*R0/H) ) )
    RT = c0-R**2 / Rs2
    #Then we convolve:
    kernel = jnp.exp(-0.5 * x_kernel**2 / lw**2)
    RT_b = jnp.convolve(RT,kernel/jnp.sum(kernel),mode='same')
    #Then we populate the 2D time series:
    rvp = jnp.sin(phase*2*np.pi)*Kp + vsys #Radial velocity of the planet as a function of the orbital phase.
    shifted_wl = jnp.outer(1-rvp/c,wl)#This populates a 2D matrix containing a row of shifted wavelengths for each of the spectra.
    spec2D = jnp.interp(shifted_wl,wlk,RT_b) # * filters
    return(spec2D)


def numpyro_model(y,y_e,*args):
    n_species = args[-1]
    T_prior = numpyro.sample('T', dist.Uniform(low=2100, high=2900))
    chi_fe_prior = numpyro.sample('log($\chi_{Fe}$)', dist.Uniform(low=-5.0, high=-3.0))
    chi_species_priors = numpyro.sample('log($\chi$ / $\chi_{Fe}$)', dist.Uniform(low=-3.0, high=0.0),sample_shape=(n_species-1,))
    k0_prior = numpyro.sample('log($\kappa_0$)', dist.Uniform(low=-3.0, high=-1.0))
    c_prior = numpyro.sample('c0', dist.Uniform(low=0.998, high=1.002))
    lw_prior = numpyro.sample('lw', dist.Uniform(low=3.0, high=5.0))
    vsys_prior = numpyro.sample('$v_{sys}$', dist.Uniform(low=19.0, high=21.0))
    Kp_prior = numpyro.sample('$K_p$', dist.Uniform(low=140, high=160))
    # beta = numpyro.sample('$\\beta$', dist.Uniform(low=0.5, high=2.0))
    priors = [T_prior,chi_fe_prior,chi_species_priors,k0_prior,c_prior,lw_prior,vsys_prior,Kp_prior]
    # Normally distributed likelihood
    numpyro.sample("obs", dist.Normal(loc=model_jax(priors,*args),scale=y_e), obs=y)






def run_retrieval(rng_seed=42,order_start=10,order_end=15,n_phases = 10, n_warmup=35,n_samples=150):


    rng_keys = split(PRNGKey(rng_seed),cpu_cores)
    print(f'Starting script in batch mode on {cpu_cores} cores.')
    # order_start = 10
    # order_end = 15
    # n_phases = 3
    # n_warmup = 50
    # n_samples = 150
    print(f'Running with {order_end-order_start} orders, {n_phases} phases, {n_warmup} & {n_samples} samples.')
    reread=False
    # labels = ['Ca', 'Ti', 'V', 'Cr', 'Fe']
    # tags = [2000,2200,2300,2400,2600]

    labels=['Fe','Ti','V','Cr']
    tags=[2600,2200,2300,2400]
    S = OrderedDict()#This will hold all my species objects.
    for i in range(len(labels)):
        S[labels[i]] = species(labels[i],tags[i])


    t1 = ut.start()
    for i in list(S.keys()):
        binpath = Path(f'opacity/VALD_{S[i].tag}e2/Out_00000_60000_02500_n800.bin',exists=True)
        fitspath = Path(f'opacity/VALD_{S[i].tag}e2/Out_00000_60000_02500_n800.fits')
        if fitspath.exists()==False or reread == True:
            print(f'Reading opacity {i} from binary.')
            kappa = np.array(ut.read_binary_kitzmann(binpath,double=False))
            ut.writefits(fitspath,kappa)

        S[i].path = ut.check_path(fitspath,exists=True)
        S[i].kappa = jnp.array(np.array(fits.getdata(fitspath),dtype='f8'))
    t2 = np.round(ut.end(t1,silent=True),1)
    print(f'{t2} seconds spent reading opacities.')

    k_wn = jnp.arange(len(S['Fe'].kappa))*1e-2#Wavenumbers
    k_wl = 1e7/k_wn#Wavelength in nm; common to all the opacity functions.

    dp = ut.check_path('data/KELT-9/night1/',exists=True)#This follows the file structure of tayph.
    list_of_wls=[]#This will store all the data.
    list_of_orders=[]
    list_of_sigmas=[]
    filelist_orders= [str(i) for i in Path(dp).glob('order_*.fits')]
    if len(filelist_orders) == 0:#If no order FITS files are found:
        raise Exception(f'Runtime error: No orders_*.fits files were found in {dp}.')
    try:
        order_numbers = [int(i.split('order_')[1].split('.')[0]) for i in filelist_orders]
    except:
        raise Exception('Runtime error: Failed at casting fits filename numerals to ints. Are the '
        'filenames of all of the spectral orders correctly formatted (e.g. order_5.fits)?')
        order_numbers.sort()#This is the ordered list of numerical order IDs.
        n_orders = len(order_numbers)


    for i in order_numbers:
        wavepath = dp/f'wave_{i}.fits'
        orderpath= dp/f'order_{i}.fits'
        ut.check_path(wavepath,exists=True)
        ut.check_path(orderpath,exists=True)
        wave_order = ut.readfits(wavepath)#2D or 1D?
        order_i = ut.readfits(orderpath)
        list_of_wls.append(ops.airtovac(wave_order))# deal with air wavelengths:
        #Test for negatives, set them to NaN.
        order_i[order_i <= 0] = np.nan #This is very important for later when we are computing
        #average spectra and the like, to avoid divide-by-zero cases.
        list_of_orders.append(order_i)
        list_of_sigmas.append(np.sqrt(order_i))


    rv_cor = sp.berv(dp)-sp.RV_star(dp)
    gamma = 1.0+(rv_cor*u.km/u.s/const.c)#Doppler factor.

    list_of_orders_cor = []
    list_of_sigmas_cor = []
    list_of_wls_cor = []

    for i in range(len(list_of_wls)):
        order = list_of_orders[i]
        sigma = list_of_sigmas[i]
        order_cor = order*0.0
        sigma_cor = sigma*0.0
        wl_cor = list_of_wls[i]

        for j in range(len(list_of_orders[0])):
            order_cor[j] = interp.interp1d(list_of_wls[i]*gamma[j],order[j],bounds_error=False)(wl_cor)
            sigma_cor[j] = interp.interp1d(list_of_wls[i]*gamma[j],sigma[j],bounds_error=False)(wl_cor)

        list_of_orders_cor.append(order_cor)
        list_of_sigmas_cor.append(sigma_cor)
        list_of_wls_cor.append(wl_cor)

    min_wl = np.inf #These are initialised and modified below to determine the extreme wavelengths.
    max_wl = 0


    mask = sp.transit(dp)
    mask[mask<1]=0 #We don't want in-transit spectra

    list_of_orders_oot = []
    list_of_wls_oot = []
    list_of_sigmas_oot = []
    for i in range(order_start,np.min([order_end,len(list_of_orders)])):
        list_of_orders_oot.append(list_of_orders_cor[i][mask==1])
        list_of_wls_oot.append(list_of_wls_cor[i])
        list_of_sigmas_oot.append(list_of_sigmas_cor[i][mask==1])
        min_wl = np.min([np.min(list_of_wls_cor[i]),min_wl])
        max_wl = np.max([np.max(list_of_wls_cor[i]),max_wl])

    list_of_wld = copy.deepcopy(list_of_wls_oot)


    n_exp = len(list_of_orders_oot[0])
    phase = np.linspace(-0.05,0.05,n_exp)#This will be used later to shift the model.

    meanfluxes = []#These are the time-dependent average fluxes that we divide out of each order.
    meanspecs = []#These are the average spectra that we divide out of each order.
    list_of_res = []
    list_of_res_e = []

    for i in range(len(list_of_orders_oot)):
        order = list_of_orders_oot[i]
        sigma = list_of_sigmas_oot[i]
        meanflux = np.nanmean(order,axis=1)
        meanfluxes.append(meanflux)
        order_norm = (order.T/meanflux).T
        sigma_norm = (sigma.T/meanflux).T
        meanspec = np.nanmean(order_norm,axis=0)
        meanspecs.append(meanspec)
        order_clean = order_norm/meanspec
        sigma_clean = sigma_norm/meanspec
        #I'm also going to set NaNs to 1.0 and then set sigma to infinite there.
        sigma_clean[np.isfinite(order_clean)==False]=np.inf
        order_clean[np.isfinite(order_clean)==False]=1.0
        list_of_res.append(order_clean)
        list_of_res_e.append(sigma_clean)


    list_of_filters = []
    list_of_res_clean = []
    list_of_res_clean_e = []
    deg = 1
    for i in range(len(list_of_res)):
        order = list_of_res[i]
        xfit = np.arange(len(order[0]))
        polyfilter = order*0.0
        fit2d = np.polyfit(xfit,order.T,deg).T

        for j in range(len(order)):
            polyfilter[j] = np.poly1d(fit2d[j])(xfit)
        list_of_filters.append(polyfilter)
        list_of_res_clean.append(list_of_res[i]/polyfilter)
        list_of_res_clean_e.append(list_of_res_e[i]/polyfilter)

    fxd = np.hstack(list_of_res_clean) #This is the data.
    err = np.hstack(list_of_res_clean_e)#This is the uncertainty
    fxf = np.hstack(list_of_filters)#This is the filter
    wld = np.hstack(list_of_wld)#This is the wavelength axis.

    #We are going to bracket the minimum and maximum wavelengths of the intermediate wavelength array by 500 km/s in velocity.
    doppler_factor = (500*u.km/u.s / const.c).decompose().value
    min_wl*=(1-doppler_factor)
    max_wl*=(1+doppler_factor)


    for i in list(S.keys()):
        wli,kappa_i,dv = ops.constant_velocity_wl_grid(np.array(k_wl[1:]),np.array(S[i].kappa[1:]),
                                                       oversampling=1.0,minmax=[min_wl,max_wl]) # The intermediate wavelength grid.
                                                        #Index it from 1 onwards because the first value is np.inf.
        S[i].kappa_i = copy.deepcopy(jnp.array(kappa_i))

    wli = jnp.array(copy.deepcopy(wli))


    kappa_grid = np.vstack([S[i].kappa_i for i in labels])
    n_species = len(kappa_grid)#The first species should be Fe.



    gamma = 0.57721
    RJ = const.R_jup.cgs.value
    MJ = const.M_jup.cgs.value
    G = const.G.cgs.value
    Rsun = const.R_sun.cgs.value
    P0 = (1.0*u.bar).cgs.value#bar
    R0 = 1.8*RJ
    M0 = 1.2*MJ
    k = const.k_B.cgs.value
    m = 2.33*const.u.cgs.value
    Rs = 1.4*Rsun
    g = G*M0 / R0**2
    c = const.c.to('km/s').value


    nwli = len(wli)
    nwld = len(wld)
    nexp = len(fxf)
    sfwhm = 2*jnp.sqrt(2*np.log(2))#2.355...


    #We calculate what range we want for the convolution kernel.
    mf=5.0#max_fwhm_expected
    nf=4.0#How many times wider the kernel is compared to the fwhm of the lsf?
    k_size = int(mf/dv*nf)
    if k_size%2 == 0: k_size+=1#Make sure that it is odd.
    x_kernel = (jnp.arange(k_size)-(k_size-1)/2)*dv #This places 0 directly in the middle; and serves as the x axis of our convolution. On-the-fly, this will be used to calculate a gaussian with which to convolve.




    #The next step would be to inject the model into the data before cleaning takes place.
    print('Making data with model as:')
    true_p = [2500.0,-4.0,jnp.array([-1.5,-1.5,-1.5]),-2.0,1.0,4.0,20.0,150.0]
    print(true_p)
    # phase = jnp.array([-0.1,-0.05,0,0.05,0.1])
    phase = jnp.array(phase[0:n_phases])
    true_model = model_jax(true_p,wld,wli,kappa_grid,x_kernel,jnp.array(phase),c,gamma,k,m,g,P0,R0,Rs**2,n_species)


    DATA_3_E = np.array(err[0:len(phase)])
    DATA_3_E[~np.isfinite(DATA_3_E)] = 5.0 # A large error.
    DATA_3_E = jnp.array(DATA_3_E)

    #A model injected into real data.
    DATA_3 = np.array(fxd[0:len(phase)] * true_model)
    DATA_3[~np.isfinite(DATA_3)] = 1.0
    DATA_3 = jnp.array(DATA_3)

    sampler = NUTS(
        numpyro_model,
        dense_mass=True#,
        #max_tree_depth=6
        # target_accept_prob = 0.5,
    )

# Monte Carlo sampling for a number of steps and parallel chains:
    mcmc = MCMC(
        sampler,
        num_warmup=n_warmup,
        num_samples=n_samples,
        num_chains=cpu_cores,
        chain_method = 'parallel'
    )

    # Run the MCMC
    t1 = ut.start()
    mcmc.run(rng_keys,DATA_3,DATA_3_E,wld,wli,kappa_grid, x_kernel,phase,c,gamma,k,m,g,P0,R0, Rs**2,n_species)
    duration = ut.end(t1)
    duration_minutes = np.round(duration/60.0,2)
    print('')
    print('')
    print('')
    print('')
    print('')
    print('')
    print('')

    mcmc.print_summary()
    mcmc.execution_duration = duration
    mcmc.n_phases = n_phases
    mcmc.order_start = order_start
    mcmc.order_end = order_end
    mcmc.random_seed = rng_seed

    print(f'{duration_minutes} minutes spent in MCMC.')
    import pickle as pickle
    with open(f'tests/MCMC-result-{n_phases}_phases-{order_end-order_start}_orders-{n_warmup}_{n_samples}_{cpu_cores}-{duration_minutes}_min-seed_{rng_seed}.pkl','wb') as f:
        pickle.dump(mcmc, f, pickle.HIGHEST_PROTOCOL)
    print('Going into debug mode to test integrity of mcmc object.')



for n_p in [3,5,8,10,15,20]:
    for n_o in [13,15,20,25]:
        run_retrieval(rng_seed=42,order_start=10,order_end=n_o,n_phases = n_p, n_warmup=50,n_samples=150)
