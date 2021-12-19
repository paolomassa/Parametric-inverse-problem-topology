import nets
import torch
#import torch.nn as nn
import numpy as np
from numpy import load, savez
#from torch.utils.data import Dataset
#from torch.nn.modules import loss
import matplotlib
import matplotlib.pyplot as plt
import os

from scipy.io.idl import readsav
from scipy import optimize

#torch.pi= torch.acos(torch.zeros(1)).item() * 2.

###############################################################################################

def Plot_loss(results_folder, save=False):
    
    history = load(results_folder + '/history.npz')
    train_loss = history['train_loss']
    valid_loss = history['valid_loss']
    
    plt.figure(figsize=(10, 8))
    SMALL_SIZE = 18
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 22
    plt.rc('font', size=SMALL_SIZE)        # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)   # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=30)
    plt.plot(train_loss, label="Training loss")
    plt.plot(valid_loss, label="Validation loss")
    plt.legend(loc="upper right")
    plt.title("Loss function")
    plt.xlabel("Iterations")
    plt.xscale('log')
    plt.yscale('log')

    if save:
        plt.savefig(save + '/plot_losses.png')
        plt.close()
    else:
        plt.show()
        
    ind_min_valid = np.argmin(valid_loss)
    print("Epoch minimum loss value on validation set: ", ind_min_valid)
        
###############################################################################################

def Plot_predictions_direct(model, data, dataset, device, save=False):
    
    vis   = data['vis_' + dataset]
    alpha = data['alpha_' + dataset]
    c   = data['c_' + dataset]
    
    max_vis = np.max(vis, axis = 1)
    mmax_vis = np.expand_dims(max_vis, axis = 1)
    mmax_vis = np.repeat(mmax_vis, 60, axis=1)
    vis = vis / mmax_vis
    vis = torch.from_numpy(vis).to(device)

    pred = model(vis.float())
    pred = pred.cpu().detach().numpy()
    
    alpha_pred = pred[:, 0] * 180.
    c_pred     = pred[:, 1]

    plt.rcParams["figure.figsize"] = [17, 5]
    
    f, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1]})

    a0.scatter(alpha, alpha_pred)
    a0.set(title='Orientation angle')
    a0.title.set_size(20)
    a0.xaxis.label.set_size(18)
    a0.yaxis.label.set_size(18)
    a0.tick_params(axis="x", labelsize=15)
    a0.tick_params(axis="y", labelsize=15)
    a0.set(xlabel='Ground truth')
    a0.set(ylabel='Predicted')

    a1.scatter(c, c_pred)
    a1.set(title='Curvature')
    a1.title.set_size(20)
    a1.xaxis.label.set_size(18)
    a1.yaxis.label.set_size(18)
    a1.tick_params(axis="x", labelsize=15)
    a1.tick_params(axis="y", labelsize=15)
    a1.set(xlabel='Ground truth')
    a1.set(ylabel='Predicted')
    
    if save:
        if not(os.path.exists(save)):
            os.mkdir(save)
        plt.savefig(save + '/pred_loop_simple_direct_' + dataset +'.png')
        plt.close()
    else:
        plt.show()
    
###############################################################################################

def Plot_predictions_embedd(model, data, dataset, device, save=False):
    
    vis   = data['vis_' + dataset]
    alpha = data['alpha_' + dataset]
    c   = data['c_' + dataset]
    
    max_vis = np.max(vis, axis = 1)
    mmax_vis = np.expand_dims(max_vis, axis = 1)
    mmax_vis = np.repeat(mmax_vis, 60, axis=1)
    vis = vis / mmax_vis
    vis = torch.from_numpy(vis).to(device)

    pred = model(vis.float())
    pred = pred.cpu().detach().numpy()

    alpha_pred = np.arctan2(pred[:, 1], pred[:, 0]) * 180 / np.pi /2.
    ind = np.where(alpha_pred < 0.)
    alpha_pred[ind] += 180.
    
    ind1 = np.where(np.abs(alpha_pred - 90.) >= 10.)
    ind2 = np.where(np.abs(alpha_pred - 90.) < 10.)

    c_pred = np.zeros(len(alpha_pred))
    c_pred[ind1] = pred[ind1, 2] / np.cos(alpha_pred[ind1] / 180. * np.pi)
    c_pred[ind2] = (np.sqrt(pred[ind2, 0]**2 + pred[ind2, 1]**2) - 1.) / \
                   np.sin(alpha_pred[ind2] / 180. * np.pi)
    
    plt.rcParams["figure.figsize"] = [17, 5]
    
    f, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1]})

    a0.scatter(alpha, alpha_pred)
    a0.set(title='Orientation angle')
    a0.title.set_size(20)
    a0.xaxis.label.set_size(18)
    a0.yaxis.label.set_size(18)
    a0.tick_params(axis="x", labelsize=15)
    a0.tick_params(axis="y", labelsize=15)
    a0.set(xlabel='Ground truth')
    a0.set(ylabel='Predicted')

    a1.scatter(c, c_pred)
    a1.set(title='Curvature')
    a1.title.set_size(20)
    a1.xaxis.label.set_size(18)
    a1.yaxis.label.set_size(18)
    a1.tick_params(axis="x", labelsize=15)
    a1.tick_params(axis="y", labelsize=15)
    a1.set(xlabel='Ground truth')
    a1.set(ylabel='Predicted')
    
    if save:
        if not(os.path.exists(save)):
            os.mkdir(save)
        plt.savefig(save + '/pred_loop_simple_embedd_' + dataset +'.png')
        plt.close()
    else:
        plt.show()

###############################################################################################
        
def Plot_predictions_overlapped(model_direct, model_embedd, data, dataset, device, save=False):
    
    vis   = data['vis_' + dataset]
    alpha = data['alpha_' + dataset]
    c   = data['c_' + dataset]
    
    max_vis = np.max(vis, axis = 1)
    mmax_vis = np.expand_dims(max_vis, axis = 1)
    mmax_vis = np.repeat(mmax_vis, 60, axis=1)
    vis = vis / mmax_vis
    vis = torch.from_numpy(vis).to(device)

    # Proposed approach
    pred_embedd = model_embedd(vis.float())
    pred_embedd = pred_embedd.cpu().detach().numpy()

    alpha_pred_embedd = np.arctan2(pred_embedd[:, 1], pred_embedd[:, 0]) * 180 / np.pi /2.
    ind = np.where(alpha_pred_embedd < 0.)
    alpha_pred_embedd[ind] += 180.
    
    ind1 = np.where(np.abs(alpha_pred_embedd - 90.) >= 10.)
    ind2 = np.where(np.abs(alpha_pred_embedd - 90.) < 10.)

    c_pred_embedd = np.zeros(len(alpha_pred_embedd))
    c_pred_embedd[ind1] = pred_embedd[ind1, 2] / np.cos(alpha_pred_embedd[ind1] / 180. * np.pi)
    c_pred_embedd[ind2] = (np.sqrt(pred_embedd[ind2, 0]**2 + pred_embedd[ind2, 1]**2) - 1.) / \
                   np.sin(alpha_pred_embedd[ind2] / 180. * np.pi)
    
    # Direct approach
    pred_direct = model_direct(vis.float())
    pred_direct = pred_direct.cpu().detach().numpy()
    
    alpha_pred_direct = pred_direct[:, 0] * 180.
    c_pred_direct     = pred_direct[:, 1]
    
    plt.rcParams["figure.figsize"] = [17, 5]
    
    f, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1]})

    a0.scatter(alpha, alpha_pred_direct, s=70, label="Direct approach")
    a0.scatter(alpha, alpha_pred_embedd, c='red', marker='+', label="Proposed method")
    a0.set(title='Orientation angle')
    a0.title.set_size(20)
    a0.xaxis.label.set_size(18)
    a0.yaxis.label.set_size(18)
    a0.tick_params(axis="x", labelsize=15)
    a0.tick_params(axis="y", labelsize=15)
    a0.set(xlabel='Ground truth')
    a0.set(ylabel='Predicted')
    a0.legend(loc="upper left", fontsize='x-large')

    a1.scatter(c, c_pred_direct, s=70, label="Direct approach")
    a1.scatter(c, c_pred_embedd, c='red', marker='+', label="Proposed method")
    a1.set(title='Curvature')
    a1.title.set_size(20)
    a1.xaxis.label.set_size(18)
    a1.yaxis.label.set_size(18)
    a1.tick_params(axis="x", labelsize=15)
    a1.tick_params(axis="y", labelsize=15)
    a1.set(xlabel='Ground truth')
    a1.set(ylabel='Predicted')
    a1.legend(loc="upper left", fontsize='x-large')
    
    if save:
        if not(os.path.exists(save)):
            os.mkdir(save)
        plt.savefig(save + '/pred_loop_simple_over_' + dataset +'.png')
        plt.close()
    else:
        plt.show()

    plt.show()
    
###############################################################################################

def Plot_test_curvature(model_direct, model_embedd, folder, device, save=False):
    
    s = readsav(folder + 'data/uv.sav')
    u = s.u
    u = u.byteswap().newbyteorder()
    v = s.v
    v = v.byteswap().newbyteorder()

    # Read the relative flux of each circular Gaussian that forms the loop
    s = readsav(folder + 'data/relflux.sav')
    relflux = s.relflux
    relflux = relflux.byteswap().newbyteorder()

    vis = np.zeros((5, 60))

    ncirc = len(relflux)
    t = np.linspace(-(ncirc - 1.)/2., (ncirc - 1.)/2., num = ncirc)

    xc    = 0.
    yc    = 0.
    fwhm  = 8.
    ecc   = 5.
    flux  = 1000.
    alpha = 0.
    c     = np.zeros((5,))

    for i in range(5):

        c[i] = -0.05 + i * 0.025
        eta  = np.zeros(t.shape)

        if np.abs(c[i]) <= 0.001:
            eta[5:ncirc] = eta[5:ncirc] + t[5:ncirc]* ecc
        else:
            for k in range(int((ncirc - 1)/2 + 1)):
                eta[5+k] = optimize.bisect(nets.ff, -60, 60, args=(c[i], t[5+k]* ecc, ))

        eta[0:6] = -np.flip(eta[5:ncirc])

        # Loop over the 11 circular sources
        for j in range(ncirc):

            c_eta_2 = c[i]*eta[j]**2
            posx = xc + eta[j] * np.cos(alpha/180.*np.pi) - c_eta_2 * np.sin(alpha/180.*np.pi)
            posy = yc + eta[j] * np.sin(alpha/180.*np.pi) + c_eta_2 * np.cos(alpha/180.*np.pi)

            phase = 2. * np.pi * (posx * u + posy * v)
            fflux = flux * relflux[j]

            # Real part Fourier transform
            vis[i,0:30] += fflux * np.exp(-(np.pi * fwhm)**2 / (4. * np.log(2.)) * \
                                   (u**2 + v**2)) * np.cos(phase)

            # Imaginary part Fourier transform
            vis[i,30:60] += fflux * np.exp(-(np.pi * fwhm)**2 / (4. * np.log(2.)) * \
                                       (u**2 + v**2)) * np.sin(phase)

    max_vis  = np.max(vis, axis = 1)
    mmax_vis = np.expand_dims(max_vis, axis = 1)
    mmax_vis = np.repeat(mmax_vis, 60, axis=1)
    vis      = vis / mmax_vis
    vis      = torch.from_numpy(vis)
    vis      = vis.to(device)        

    param_pred_direct = model_direct(vis.float())
    param_pred_direct = param_pred_direct.cpu()
    param_pred_direct = param_pred_direct.detach().numpy()
    alpha_pred_direct = param_pred_direct[:,0]*180.
    c_pred_direct     = param_pred_direct[:,1]

    param_pred_embedd = model_embedd(vis.float())
    param_pred_embedd = param_pred_embedd.cpu().detach().numpy()
    alpha_pred_embedd = np.arctan2(param_pred_embedd[:, 1], param_pred_embedd[:, 0]) * 180 / np.pi /2.
    ind = np.where(alpha_pred_embedd < 0.)
    alpha_pred_embedd[ind] += 180.

    ind1 = np.where(np.abs(alpha_pred_embedd - 90.) >= 10.)
    ind2 = np.where(np.abs(alpha_pred_embedd - 90.) < 10.)

    c_pred_embedd = np.zeros(len(alpha_pred_embedd))
    c_pred_embedd[ind1] = param_pred_embedd[ind1, 2] / np.cos(alpha_pred_embedd[ind1] / 180. * np.pi)
    c_pred_embedd[ind2] = (np.sqrt(param_pred_embedd[ind2, 0]**2 + param_pred_embedd[ind2, 1]**2) - 1.) / \
                           np.sin(alpha_pred_embedd[ind2] / 180. * np.pi)

    npix = 129
    pix  = 0.5

    plt.rcParams["figure.figsize"] = [13, 8]
    fig, axs = plt.subplots(3, 5)

    for i in range(5):

        loop_gt = nets.build_loop_source(relflux, xc, yc, fwhm, ecc, flux, alpha, c[i], npix, pix)

        loop_pred_direct = nets.build_loop_source(relflux, xc, yc, fwhm, ecc, flux, \
                                             alpha_pred_direct[i], c_pred_direct[i], npix, pix)

        loop_pred_embedd = nets.build_loop_source(relflux, xc, yc, fwhm, ecc, flux, \
                                             alpha_pred_embedd[i], c_pred_embedd[i], npix, pix)


        axs[0, i].imshow(loop_gt, cmap='jet')
        axs[0, i].xaxis.set_ticks([])
        axs[0, i].yaxis.set_ticks([])
        axs[0, i].set(title='c='+str(round(c[i],3)))
        axs[0, i].title.set_size(20)
        axs[1, i].imshow(loop_pred_direct, cmap='jet')
        axs[1, i].xaxis.set_ticks([])
        axs[1, i].yaxis.set_ticks([])
        axs[2, i].imshow(loop_pred_embedd, cmap='jet')
        axs[2, i].xaxis.set_ticks([])
        axs[2, i].yaxis.set_ticks([])
        
        if i==0:
            axs[0, i].set(ylabel='Ground truth')
            axs[0, i].yaxis.label.set_size(20)
            axs[1, i].set(ylabel='Naive approach')
            axs[1, i].yaxis.label.set_size(20)
            axs[2, i].set(ylabel='Proposed method')
            axs[2, i].yaxis.label.set_size(20)

    fig.tight_layout()
    if save:
        if not(os.path.exists(save)):
            os.mkdir(save)
        plt.savefig(save + '/curvature_test.png')
        plt.close()
    else:
        plt.show()
    
    print(" ")
    print("Orientation angle (ground truth): 0")
    print("Orientation angle (predicted by the direct approach): " + str(alpha_pred_direct))
    print("Orientation angle (predicted by the proposed method): " + str(alpha_pred_embedd))
    print(" ")
    print("Curvature (ground truth): " + str(c))
    print("Curvature (predicted by the direct approach): " + str(c_pred_direct))
    print("Curvature (predicted by the proposed method): " + str(c_pred_embedd))
        
###############################################################################################

def Plot_results_complete(model, data, dataset, device, save=False):

    vis = data['vis_' + dataset + '_noisy']
    max_vis = np.max(vis, axis = 1)
    mmax_vis = np.expand_dims(max_vis, axis = 1)
    mmax_vis = np.repeat(mmax_vis, 60, axis=1)
    vis = vis / mmax_vis
    vis = torch.from_numpy(vis).to(device)

    pred = model(vis.float())
    pred = pred.cpu().detach().numpy()

    xc_pred   = 60. * pred[:, 0] - 30.
    yc_pred   = 60. * pred[:, 1] - 30.
    flux_pred = max_vis * pred[:, 2]
    fwhm_pred = 15. * pred[:, 3]
    ecc_pred  = 5.*pred[:,4]
    xx_pred = 13.*pred[:, 5]-6.
    yy_pred = 13.*pred[:, 6]-6.
    zz_pred = pred[:, 7]-0.5

    alpha_pred      = np.arctan2(yy_pred, xx_pred)/np.pi * 180./2.
    ind             = np.where(alpha_pred < 0.)
    alpha_pred[ind] = alpha_pred[ind] + 180

    ind1 = np.where(np.abs(alpha_pred - 90.) >= 10.)
    ind2 = np.where(np.abs(alpha_pred - 90.) < 10.)

    c_pred = np.zeros(len(alpha_pred))
    c_pred[ind1] = zz_pred[ind1] / (np.cos(alpha_pred[ind1] / 180. * np.pi) * ecc_pred[ind1])
    c_pred[ind2] = (np.sqrt(xx_pred[ind2]**2 + yy_pred[ind2]**2)/ecc_pred[ind2] - 1.) / \
                   np.sin(alpha_pred[ind2] / 180. * np.pi)



    xc    = data['xc_' + dataset]
    yc    = data['yc_' + dataset]
    flux  = data['flux_' + dataset]
    fwhm  = data['fwhm_' + dataset]
    ecc   = data['ecc_' + dataset]
    alpha = data['alpha_' + dataset]
    c     = data['c_' + dataset]


    range_x = np.max(xc) - np.min(xc)
    range_y = np.max(yc) - np.min(yc)
    range_fwhm = np.max(fwhm) - np.min(fwhm)
    range_flux = np.max(flux) - np.min(flux)
    range_ecc = np.max(ecc) - np.min(ecc)


    param_dict = {'X': np.abs(xc_pred - xc) / range_x, \
                  'Y': np.abs(yc_pred - yc) / range_y, \
                  'Flux': np.abs(flux_pred - flux) / range_flux, \
                  'FWHM': np.abs(fwhm_pred - fwhm) / range_fwhm, \
                  'Eccen.': np.abs(ecc_pred - ecc) / range_ecc, \
                  }


    plt.rcParams["figure.figsize"] = [24, 7]
    #plt.rcParams['text.usetex'] = True

    f, (a0, a1, a2) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1.25, 1.25]})

    a0.boxplot(param_dict.values())
    a0.set_xticklabels(param_dict.keys())
    a0.set(ylabel='Normalized absolute error')
    a0.xaxis.label.set_size(18)
    a0.yaxis.label.set_size(18)
    a0.tick_params(axis="x", labelsize=15)
    a0.tick_params(axis="y", labelsize=15)


    scatter1 = a1.scatter(alpha, alpha_pred, c=ecc, cmap=plt.cm.jet, norm=matplotlib.colors.PowerNorm(0.5))
    a1.set(title='Orientation angle')
    a1.title.set_size(20)
    a1.xaxis.label.set_size(18)
    a1.yaxis.label.set_size(18)
    a1.tick_params(axis="x", labelsize=15)
    a1.tick_params(axis="y", labelsize=15)
    a1.set(xlabel='Ground truth')
    a1.set(ylabel='Predicted')
    cbar1 = f.colorbar(scatter1,ax=a1)
    cbar1.ax.tick_params(labelsize=15) 

    scatter2 = a2.scatter(c, c_pred, c=ecc, cmap=plt.cm.jet, norm=matplotlib.colors.PowerNorm(0.5))
    a2.set(title='Curvature')
    a2.title.set_size(20)
    a2.xaxis.label.set_size(18)
    a2.yaxis.label.set_size(18)
    a2.tick_params(axis="x", labelsize=15)
    a2.tick_params(axis="y", labelsize=15)
    a2.set(xlabel='Ground truth')
    a2.set(ylabel='Predicted')
    a2.set_ylim([-0.12, 0.12])
    cbar2 = f.colorbar(scatter2,ax=a2)
    cbar2.ax.tick_params(labelsize=15) 

    if save:
        plt.savefig(save + '/pred_loop_complete_' + dataset +'.png')
    
    plt.show()    
    
    

