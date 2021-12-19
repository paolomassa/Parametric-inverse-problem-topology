import torch
import torch.nn as nn
import numpy as np
from numpy import load, savez
from torch.utils.data import Dataset
from torch.nn.modules import loss
import matplotlib.pyplot as plt
import os

from scipy.io.idl import readsav
from scipy import optimize

torch.pi= torch.acos(torch.zeros(1)).item() * 2.

###############################################################################################

# Needed for inverting the function that represents the length of a parabolic arc
def ff(z, c, x):
    return (2*z*c* np.sqrt(4*(z*c)**2 + 1) + np.log(2*z*c + np.sqrt(4*(z*c)**2 + 1)))/(4*c) - x

###############################################################################################

def VisGenerator_simple(folder, n_train, n_valid, n_test, name):
    
    # Read the (u,v) point coordinates
    s = readsav(folder + '/uv.sav')
    u = s.u
    u = u.byteswap().newbyteorder()
    v = s.v
    v = v.byteswap().newbyteorder()
    
    # Read the relative flux of each circular Gaussian that forms the loop
    s = readsav(folder + '/relflux.sav')
    relflux = s.relflux
    relflux = relflux.byteswap().newbyteorder()
    
    dim = len(u)

    # Training set
    vis_train       = np.zeros((n_train, 2*dim))
    xc_train        = np.zeros(n_train)
    yc_train        = np.zeros(n_train)
    fwhm_train      = np.zeros(n_train)
    flux_train      = np.zeros(n_train)
    ecc_train       = np.zeros(n_train)
    alpha_train     = np.zeros(n_train)
    c_train         = np.zeros(n_train)

    # Validation set
    vis_valid       = np.zeros((n_valid, 2*dim))
    xc_valid        = np.zeros(n_valid)
    yc_valid        = np.zeros(n_valid)
    fwhm_valid      = np.zeros(n_valid)
    flux_valid      = np.zeros(n_valid)
    ecc_valid       = np.zeros(n_valid)
    alpha_valid     = np.zeros(n_valid)
    c_valid         = np.zeros(n_valid)
    
    #Test set
    vis_test       = np.zeros((n_test, 2*dim))
    xc_test        = np.zeros(n_test)
    yc_test        = np.zeros(n_test)
    fwhm_test      = np.zeros(n_test)
    flux_test      = np.zeros(n_test)
    ecc_test       = np.zeros(n_test)
    alpha_test     = np.zeros(n_test)
    c_test         = np.zeros(n_test)

    

    ncirc = len(relflux)
    t = np.linspace(-(ncirc - 1.)/2., (ncirc - 1.)/2., num = ncirc)

    for i in range(n_train):
        
        xc_train[i]    = 0.
        yc_train[i]    = 0.
        fwhm_train[i]  = 8.
        ecc_train[i]   = 5.
        flux_train[i]  = 1000.
        alpha_train[i] = np.random.uniform(low=0., high=180.)
        c_train[i]     = np.random.uniform(low=-0.1, high=0.1)/2.

        eta = np.zeros(t.shape)    
   
        if np.abs(c_train[i]) <= 0.001:
            eta[5:ncirc] = eta[5:ncirc] + t[5:ncirc]* ecc_train[i]
        else:
            for k in range(int((ncirc - 1)/2 + 1)):
                eta[5+k] = optimize.bisect(ff, -60, 60, args=(c_train[i], t[5+k]* ecc_train[i], ))
 
        eta[0:6] = -np.flip(eta[5:ncirc])
        
        # Loop over the 11 circular sources
        for j in range(ncirc):

            c_eta_2 = c_train[i]*eta[j]**2
            posx = xc_train[i] + eta[j] * np.cos(alpha_train[i]/180.*np.pi) - c_eta_2 * np.sin(alpha_train[i]/180.*np.pi)
            posy = yc_train[i] + eta[j] * np.sin(alpha_train[i]/180.*np.pi) + c_eta_2 * np.cos(alpha_train[i]/180.*np.pi)

            phase = 2. * np.pi * (posx * u + posy * v)
            flux = flux_train[i] * relflux[j]
            
            # Real part Fourier transform
            vis_train[i, 0:dim] += flux * np.exp(-(np.pi * fwhm_train[i])**2 / (4. * np.log(2.)) * \
                                   (u**2 + v**2)) * np.cos(phase)
            
            # Imaginary part Fourier transform
            vis_train[i, dim:2*dim] += flux * np.exp(-(np.pi * fwhm_train[i])**2 / (4. * np.log(2.)) * \
                                       (u**2 + v**2)) * np.sin(phase)
            
            
            
    for i in range(n_valid):
        
        xc_valid[i]    = 0.
        yc_valid[i]    = 0.
        fwhm_valid[i]  = 8.
        ecc_valid[i]   = 5.
        flux_valid[i]  = 1000.
        alpha_valid[i] = np.random.uniform(low=0., high=180.)
        c_valid[i]     = np.random.uniform(low=-0.1, high=0.1)/2.

        eta = np.zeros(t.shape)    
   
        if np.abs(c_valid[i]) <= 0.001:
            eta[5:ncirc] = eta[5:ncirc] + t[5:ncirc]* ecc_valid[i]
        else:
            for k in range(int((ncirc - 1)/2 + 1)):
                eta[5+k] = optimize.bisect(ff, -60, 60, args=(c_valid[i], t[5+k]* ecc_valid[i], ))
 
        eta[0:6] = -np.flip(eta[5:ncirc])
        
        # Loop over the 11 circular sources
        for j in range(ncirc):

            c_eta_2 = c_valid[i]*eta[j]**2
            posx = xc_valid[i] + eta[j] * np.cos(alpha_valid[i]/180.*np.pi) - c_eta_2 * np.sin(alpha_valid[i]/180.*np.pi)
            posy = yc_valid[i] + eta[j] * np.sin(alpha_valid[i]/180.*np.pi) + c_eta_2 * np.cos(alpha_valid[i]/180.*np.pi)

            phase = 2. * np.pi * (posx * u + posy * v)
            flux = flux_valid[i] * relflux[j]
            
            # Real part Fourier transform
            vis_valid[i, 0:dim] += flux * np.exp(-(np.pi * fwhm_valid[i])**2 / (4. * np.log(2.)) * \
                                   (u**2 + v**2)) * np.cos(phase)
            
            # Imaginary part Fourier transform
            vis_valid[i, dim:2*dim] += flux * np.exp(-(np.pi * fwhm_valid[i])**2 / (4. * np.log(2.)) * \
                                       (u**2 + v**2)) * np.sin(phase)
    
    
    for i in range(n_test):
        
        xc_test[i]    = 0.
        yc_test[i]    = 0. 
        fwhm_test[i]  = 8.
        ecc_test[i]   = 5.
        flux_test[i]  = 1000.
        alpha_test[i] = np.random.uniform(low=0., high=180.)
        c_test[i]     = np.random.uniform(low=-0.1, high=0.1)/2.

        eta = np.zeros(t.shape)    
   
        if np.abs(c_test[i]) <= 0.001:
            eta[5:ncirc] = eta[5:ncirc] + t[5:ncirc]* ecc_test[i]
        else:
            for k in range(int((ncirc - 1)/2 + 1)):
                eta[5+k] = optimize.bisect(ff, -60, 60, args=(c_test[i], t[5+k]* ecc_test[i], ))
 
        eta[0:6] = -np.flip(eta[5:ncirc])
        
        # Loop over the 11 circular sources
        for j in range(ncirc):

            c_eta_2 = c_test[i]*eta[j]**2
            posx = xc_test[i] + eta[j] * np.cos(alpha_test[i]/180.*np.pi) - c_eta_2 * np.sin(alpha_test[i]/180.*np.pi)
            posy = yc_test[i] + eta[j] * np.sin(alpha_test[i]/180.*np.pi) + c_eta_2 * np.cos(alpha_test[i]/180.*np.pi)

            phase = 2. * np.pi * (posx * u + posy * v)
            flux = flux_test[i] * relflux[j]
            
            # Real part Fourier transform
            vis_test[i, 0:dim] += flux * np.exp(-(np.pi * fwhm_test[i])**2 / (4. * np.log(2.)) * \
                                   (u**2 + v**2)) * np.cos(phase)
            
            # Imaginary part Fourier transform
            vis_test[i, dim:2*dim] += flux * np.exp(-(np.pi * fwhm_test[i])**2 / (4. * np.log(2.)) * \
                                       (u**2 + v**2)) * np.sin(phase)
            
            
    savez(folder + name + '.npz', \
          xc_train=xc_train, xc_valid=xc_valid, xc_test=xc_test, \
          yc_train=yc_train, yc_valid=yc_valid, yc_test=yc_test, \
          fwhm_train=fwhm_train, fwhm_valid=fwhm_valid, fwhm_test=fwhm_test, \
          flux_train=flux_train, flux_valid=flux_valid, flux_test=flux_test, \
          ecc_train=ecc_train, ecc_test=ecc_test, ecc_valid=ecc_valid, \
          alpha_train=alpha_train, alpha_test=alpha_test, alpha_valid=alpha_valid, \
          vis_train=vis_train, vis_test=vis_test, vis_valid=vis_valid, \
          c_train=c_train, c_test=c_test, c_valid=c_valid, \
          u=u, v=v)

###############################################################################################

def VisGenerator_complete(folder, n_train, n_valid, n_test, name):
    
    # Read the (u,v) point coordinates
    s = readsav(folder + '/uv.sav')
    u = s.u
    u = u.byteswap().newbyteorder()
    v = s.v
    v = v.byteswap().newbyteorder()
    
    # Read the relative flux of each circular Gaussian that forms the loop
    s = readsav(folder + '/relflux.sav')
    relflux = s.relflux
    relflux = relflux.byteswap().newbyteorder()
    
    dim = len(u)

    # Training set
    vis_train       = np.zeros((n_train, 2*dim))
    vis_train_noisy = np.zeros((n_train, 2*dim))
    xc_train        = np.zeros(n_train)
    yc_train        = np.zeros(n_train)
    fwhm_train      = np.zeros(n_train)
    flux_train      = np.zeros(n_train)
    ecc_train       = np.zeros(n_train)
    alpha_train     = np.zeros(n_train)
    c_train         = np.zeros(n_train)

    # Validation set
    vis_valid       = np.zeros((n_valid, 2*dim))
    vis_valid_noisy = np.zeros((n_valid, 2*dim))
    xc_valid        = np.zeros(n_valid)
    yc_valid        = np.zeros(n_valid)
    fwhm_valid      = np.zeros(n_valid)
    flux_valid      = np.zeros(n_valid)
    ecc_valid       = np.zeros(n_valid)
    alpha_valid     = np.zeros(n_valid)
    c_valid         = np.zeros(n_valid)
    
    #Test set
    vis_test       = np.zeros((n_test, 2*dim))
    vis_test_noisy = np.zeros((n_test, 2*dim))
    xc_test        = np.zeros(n_test)
    yc_test        = np.zeros(n_test)
    fwhm_test      = np.zeros(n_test)
    flux_test      = np.zeros(n_test)
    ecc_test       = np.zeros(n_test)
    alpha_test     = np.zeros(n_test)
    c_test         = np.zeros(n_test)

    

    ncirc = len(relflux)
    t = np.linspace(-(ncirc - 1.)/2., (ncirc - 1.)/2., num = ncirc)

    for i in range(n_train):
        
        xc_train[i]    = np.random.uniform(low=-25., high=25.)
        yc_train[i]    = np.random.uniform(low=-25., high=25.) 
        fwhm_train[i]  = np.random.uniform(low=8., high=15.)#np.random.uniform(low=8., high=20.)
        ecc_train[i]   = np.random.uniform(low=0., high=5.)
        flux_train[i]  = np.random.uniform(low=1000., high=15000.)
        alpha_train[i] = np.random.uniform(low=0., high=180.)
        c_train[i]     = np.random.uniform(low=-0.1, high=0.1)

        eta = np.zeros(t.shape)    
   
        if np.abs(c_train[i]) <= 0.001:
            eta[5:ncirc] = eta[5:ncirc] + t[5:ncirc]* ecc_train[i]
        else:
            for k in range(int((ncirc - 1)/2 + 1)):
                eta[5+k] = optimize.bisect(ff, -60, 60, args=(c_train[i], t[5+k]* ecc_train[i], ))
 
        eta[0:6] = -np.flip(eta[5:ncirc])
        
        # Loop over the 11 circular sources
        for j in range(ncirc):

            c_eta_2 = c_train[i]*eta[j]**2
            posx = xc_train[i] + eta[j] * np.cos(alpha_train[i]/180.*np.pi) - c_eta_2 * np.sin(alpha_train[i]/180.*np.pi)
            posy = yc_train[i] + eta[j] * np.sin(alpha_train[i]/180.*np.pi) + c_eta_2 * np.cos(alpha_train[i]/180.*np.pi)

            phase = 2. * np.pi * (posx * u + posy * v)
            flux = flux_train[i] * relflux[j]
            
            # Real part Fourier transform
            vis_train[i, 0:dim] += flux * np.exp(-(np.pi * fwhm_train[i])**2 / (4. * np.log(2.)) * \
                                   (u**2 + v**2)) * np.cos(phase)
            
            # Imaginary part Fourier transform
            vis_train[i, dim:2*dim] += flux * np.exp(-(np.pi * fwhm_train[i])**2 / (4. * np.log(2.)) * \
                                       (u**2 + v**2)) * np.sin(phase)
            
            # Additive Gaussian noise
            noise = 2. * np.sqrt(flux_train[i])
            vis_train_noisy[i, :] = vis_train[i, :] + np.random.normal(loc=0, scale=1, size=(2*dim)) * noise
            
            
    for i in range(n_valid):
        
        xc_valid[i]    = np.random.uniform(low=-25., high=25.)
        yc_valid[i]    = np.random.uniform(low=-25., high=25.) 
        fwhm_valid[i]  = np.random.uniform(low=8., high=15.)#np.random.uniform(low=8., high=20.)
        ecc_valid[i]   = np.random.uniform(low=0., high=5.)
        flux_valid[i]  = np.random.uniform(low=1000., high=15000.)
        alpha_valid[i] = np.random.uniform(low=0., high=180.)
        c_valid[i]     = np.random.uniform(low=-0.1, high=0.1)

        eta = np.zeros(t.shape)    
   
        if np.abs(c_valid[i]) <= 0.001:
            eta[5:ncirc] = eta[5:ncirc] + t[5:ncirc]* ecc_valid[i]
        else:
            for k in range(int((ncirc - 1)/2 + 1)):
                eta[5+k] = optimize.bisect(ff, -60, 60, args=(c_valid[i], t[5+k]* ecc_valid[i], ))
 
        eta[0:6] = -np.flip(eta[5:ncirc])
        
        # Loop over the 11 circular sources
        for j in range(ncirc):

            c_eta_2 = c_valid[i]*eta[j]**2
            posx = xc_valid[i] + eta[j] * np.cos(alpha_valid[i]/180.*np.pi) - c_eta_2 * np.sin(alpha_valid[i]/180.*np.pi)
            posy = yc_valid[i] + eta[j] * np.sin(alpha_valid[i]/180.*np.pi) + c_eta_2 * np.cos(alpha_valid[i]/180.*np.pi)

            phase = 2. * np.pi * (posx * u + posy * v)
            flux = flux_valid[i] * relflux[j]
            
            # Real part Fourier transform
            vis_valid[i, 0:dim] += flux * np.exp(-(np.pi * fwhm_valid[i])**2 / (4. * np.log(2.)) * \
                                   (u**2 + v**2)) * np.cos(phase)
            
            # Imaginary part Fourier transform
            vis_valid[i, dim:2*dim] += flux * np.exp(-(np.pi * fwhm_valid[i])**2 / (4. * np.log(2.)) * \
                                       (u**2 + v**2)) * np.sin(phase)
            
            # Additive Gaussian noise
            noise = 2. * np.sqrt(flux_valid[i])
            vis_valid_noisy[i, :] = vis_valid[i, :] + np.random.normal(loc=0, scale=1, size=(2*dim)) * noise 
    
    
    for i in range(n_test):
        
        xc_test[i]    = np.random.uniform(low=-25., high=25.)
        yc_test[i]    = np.random.uniform(low=-25., high=25.) 
        fwhm_test[i]  = np.random.uniform(low=8., high=15.)#np.random.uniform(low=8., high=20.)
        ecc_test[i]   = np.random.uniform(low=0., high=5.)
        flux_test[i]  = np.random.uniform(low=1000., high=15000.)
        alpha_test[i] = np.random.uniform(low=0., high=180.)
        c_test[i]     = np.random.uniform(low=-0.1, high=0.1)

        eta = np.zeros(t.shape)    
   
        if np.abs(c_test[i]) <= 0.001:
            eta[5:ncirc] = eta[5:ncirc] + t[5:ncirc]* ecc_test[i]
        else:
            for k in range(int((ncirc - 1)/2 + 1)):
                eta[5+k] = optimize.bisect(ff, -60, 60, args=(c_test[i], t[5+k]* ecc_test[i], ))
 
        eta[0:6] = -np.flip(eta[5:ncirc])
        
        # Loop over the 11 circular sources
        for j in range(ncirc):

            c_eta_2 = c_test[i]*eta[j]**2
            posx = xc_test[i] + eta[j] * np.cos(alpha_test[i]/180.*np.pi) - c_eta_2 * np.sin(alpha_test[i]/180.*np.pi)
            posy = yc_test[i] + eta[j] * np.sin(alpha_test[i]/180.*np.pi) + c_eta_2 * np.cos(alpha_test[i]/180.*np.pi)

            phase = 2. * np.pi * (posx * u + posy * v)
            flux = flux_test[i] * relflux[j]
            
            # Real part Fourier transform
            vis_test[i, 0:dim] += flux * np.exp(-(np.pi * fwhm_test[i])**2 / (4. * np.log(2.)) * \
                                   (u**2 + v**2)) * np.cos(phase)
            
            # Imaginary part Fourier transform
            vis_test[i, dim:2*dim] += flux * np.exp(-(np.pi * fwhm_test[i])**2 / (4. * np.log(2.)) * \
                                       (u**2 + v**2)) * np.sin(phase)
            
            # Additive Gaussian noise
            noise = 2. * np.sqrt(flux_test[i])
            vis_test_noisy[i, :] = vis_test[i, :] + np.random.normal(loc=0, scale=1, size=(2*dim)) * noise
            
    savez(folder + name + '.npz', \
          xc_train=xc_train, xc_valid=xc_valid, xc_test=xc_test, \
          yc_train=yc_train, yc_valid=yc_valid, yc_test=yc_test, \
          fwhm_train=fwhm_train, fwhm_valid=fwhm_valid, fwhm_test=fwhm_test, \
          flux_train=flux_train, flux_valid=flux_valid, flux_test=flux_test, \
          ecc_train=ecc_train, ecc_test=ecc_test, ecc_valid=ecc_valid, \
          alpha_train=alpha_train, alpha_test=alpha_test, alpha_valid=alpha_valid, \
          vis_train=vis_train, vis_test=vis_test, vis_valid=vis_valid, \
          vis_train_noisy=vis_train_noisy, vis_test_noisy=vis_test_noisy, vis_valid_noisy=vis_valid_noisy, \
          c_train=c_train, c_test=c_test, c_valid=c_valid, \
          u=u, v=v)
    
###############################################################################################   

def build_loop_source(relflux, xc, yc, fwhm, ecc, flux, pa, a, npix, pix):

    x = np.linspace(-(npix - 1)/2, (npix - 1)/2, num=npix)*pix
    x = np.expand_dims(x, axis=0)
    x = np.repeat(x, npix, axis=0)

    y = np.linspace((npix - 1)/2, -(npix - 1)/2, num=npix)*pix
    y = np.expand_dims(y, axis=1)
    y = np.repeat(y, npix, axis=1)

    loop_source = np.zeros((npix, npix))

    ncirc = len(relflux)
    t = np.linspace(-(ncirc - 1.)/2., (ncirc - 1.)/2., num = ncirc)
    b = np.zeros(t.shape)
    for i in range(int((ncirc - 1)/2 + 1)):
        l = t[5+i] * ecc
        if np.abs(a) <=0.001:
            b[5+i] = l
        else:
            b[5+i] = optimize.bisect(ff, -60, 60, args=(a, l, ))

    b[0:6] = -np.flip(b[5:12])
    normfactor = (np.sqrt(4 * np.log(2.))/fwhm)**2
    for i in range(ncirc):

        ab = a*b[i]**2
        posx = xc + b[i] * np.cos(pa/180.*np.pi) - ab * np.sin(pa/180.*np.pi)
        posy = yc + b[i] * np.sin(pa/180.*np.pi) + ab * np.cos(pa/180.*np.pi)
        
        loop_source += flux*relflux[i]*normfactor/np.pi*np.exp(-((x - posx)**2 + (y - posy)**2)*normfactor)
    
    return loop_source

###############################################################################################

def Fourier_stix(u,v,npix,pix):
    
    x = np.linspace(-(npix - 1)/2, (npix - 1)/2, num=npix)*pix
    x = np.expand_dims(x, axis=0)
    x = np.repeat(x, npix, axis=0)

    y = np.linspace((npix - 1)/2, -(npix - 1)/2, num=npix)*pix
    y = np.expand_dims(y, axis=1)
    y = np.repeat(y, npix, axis=1)
    
    dim = len(u)
    F = np.zeros((2*dim, npix*npix))
    
    for i in range(dim):
        
        phase = 2*np.pi*(x * u[i] + y * v[i])
        F[i, :]    = np.reshape(np.cos(phase), (npix*npix,))
        F[i+dim, :] = np.reshape(np.sin(phase), (npix*npix,))
        
    return F * pix**2

###############################################################################################

class layer(nn.Module):
    def __init__(self, dim_in, dim_out, if_act=True):
        super(layer, self).__init__()
        
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.if_act = if_act
        
        self.linear = nn.Linear(self.dim_in, self.dim_out)
        if self.if_act:
            self.activation = nn.ReLU()


    def forward(self, x):
        if self.if_act:
            x = self.activation(self.linear(x))
        else:
            x = self.linear(x)
        return x
    
############################################################################

class net_loop_simple_embedd(nn.Module):
    def __init__(self, dimensions):
        super(net_loop_simple_embedd, self).__init__()
        
        n_el = len(dimensions)
        self.enc_sizes = dimensions[0:n_el-1]
        self.dim_out = dimensions[-1]
        
        layers = [layer(in_f, out_f)
                       for in_f, out_f in zip(self.enc_sizes, self.enc_sizes[1:])]

        self.layers = nn.Sequential(*layers)
        
        self.xx = layer(self.enc_sizes[-1], self.dim_out, if_act = False)
        self.yy = layer(self.enc_sizes[-1], self.dim_out, if_act = False)
        self.zz = layer(self.enc_sizes[-1], self.dim_out, if_act = False)
        
    def forward(self, x):
        
        layers = self.layers(x)
        
        xx = self.xx(layers)
        yy = self.yy(layers)
        zz = self.zz(layers)

        return torch.cat((xx, yy, zz), dim=1)
    
############################################################################

class net_loop_simple_direct(nn.Module):
    def __init__(self, dimensions):
        super(net_loop_simple_direct, self).__init__()
        
        n_el = len(dimensions)
        self.enc_sizes = dimensions[0:n_el-1]
        self.dim_out = dimensions[-1]
        
        layers = [layer(in_f, out_f)
                       for in_f, out_f in zip(self.enc_sizes, self.enc_sizes[1:])]

        self.layers = nn.Sequential(*layers)
        
        self.alpha = layer(self.enc_sizes[-1], self.dim_out, if_act = False)
        self.c     = layer(self.enc_sizes[-1], self.dim_out, if_act = False)
        
    def forward(self, x):
        
        layers = self.layers(x)
        
        alpha = self.alpha(layers)
        c     = self.c(layers)

        return torch.cat((alpha, c), dim=1)

############################################################################

class layer_drop(nn.Module):
    def __init__(self, dim_in, dim_out, if_act=True):
        super(layer_drop, self).__init__()
        
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.if_act = if_act
        
        self.linear = nn.Linear(self.dim_in, self.dim_out)
        if self.if_act:
            self.activation = nn.ReLU()
        self.drop = nn.Dropout()

    def forward(self, x):
        if self.if_act:
            x = self.activation(self.linear(self.drop(x)))
        else:
            x = self.linear(self.drop(x))
        return x
   
############################################################################

class net_loop_complete(nn.Module):
    def __init__(self, dimensions):
        super(net_loop_complete, self).__init__()
        
        n_el = len(dimensions)
        self.enc_sizes = dimensions[1:n_el-1]
        self.dim_out = dimensions[-1]
        
        self.input = layer(dimensions[0], dimensions[1])
        
        layers = [layer_drop(in_f, out_f)
                       for in_f, out_f in zip(self.enc_sizes, self.enc_sizes[1:])]

        self.layers = nn.Sequential(*layers)
        
        self.xc   = layer_drop(self.enc_sizes[-1], self.dim_out, if_act = False)
        self.yc   = layer_drop(self.enc_sizes[-1], self.dim_out, if_act = False)
        self.flux = layer_drop(self.enc_sizes[-1], self.dim_out, if_act = False)
        self.fwhm = layer_drop(self.enc_sizes[-1], self.dim_out, if_act = False)
        self.ecc  = layer_drop(self.enc_sizes[-1], self.dim_out, if_act = False)
        self.xx   = layer_drop(self.enc_sizes[-1], self.dim_out, if_act = False)
        self.yy   = layer_drop(self.enc_sizes[-1], self.dim_out, if_act = False)
        self.zz   = layer_drop(self.enc_sizes[-1], self.dim_out, if_act = False)
        
    def forward(self, x):
        
        layers = self.layers(self.input(x))
        
        xc   = self.xc(layers)
        yc   = self.yc(layers)
        flux = self.flux(layers)
        fwhm = self.fwhm(layers)
        ecc  = self.ecc(layers)
        xx   = self.xx(layers)
        yy   = self.yy(layers)
        zz   = self.zz(layers)

        return torch.cat((xc, yc, flux, fwhm, ecc, xx, yy, zz), dim=1)

############################################################################

class VisDatasetTrain(Dataset):
    def __init__(self, data, target):
        
        self.data = data
        self.target = target
        self.len = self.data.shape[0]
        
        self.x_data = self.data
        self.y_data = self.target

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
    
############################################################################

def ComputeVis(relflux, u, v, xc, yc, flux, fwhm, ecc, alpha, c):
    
    ncirc = len(relflux)
    t = np.linspace(-(ncirc - 1.)/2., (ncirc - 1.)/2., num = ncirc)
    
    eta = np.zeros(t.shape)    
   
    if np.abs(c) <= 0.001:
        eta[5:ncirc] = eta[5:ncirc] + t[5:ncirc]* ecc
    else:
        for k in range(int((ncirc - 1)/2 + 1)):
            eta[5+k] = optimize.bisect(ff, -60, 60, args=(c, t[5+k]* ecc, ))

    eta[0:6] = -np.flip(eta[5:ncirc])

    dim = len(u)
    vis = np.zeros((2*dim,))
    
    # Loop over the 11 circular sources
    for j in range(ncirc):

        c_eta_2 = c*eta[j]**2
        posx = xc + eta[j] * np.cos(alpha/180.*np.pi) - c_eta_2 * np.sin(alpha/180.*np.pi)
        posy = yc + eta[j] * np.sin(alpha/180.*np.pi) + c_eta_2 * np.cos(alpha/180.*np.pi)

        phase = 2. * np.pi * (posx * u + posy * v)
        fflux = flux * relflux[j]

        # Real part Fourier transform
        vis[0:dim]     += fflux * np.exp(-(np.pi * fwhm)**2 / (4. * np.log(2.)) * \
                               (u**2 + v**2)) * np.cos(phase)

        # Imaginary part Fourier transform
        vis[dim:2*dim] += fflux * np.exp(-(np.pi * fwhm)**2 / (4. * np.log(2.)) * \
                                   (u**2 + v**2)) * np.sin(phase)

    return vis


