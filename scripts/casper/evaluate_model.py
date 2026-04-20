"""
writes out netcdf files of analyzed data for different ML models
"""

import os
import sys
import torch
# import logging
import torch.nn as nn
# import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
# from torch.autograd import Variable
# import torchvision.models as models

import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import datetime

from scipy.special import gamma, gammaln

import copy
import datetime
import yaml

import importlib

import tqdm

dirP_str = os.path.join(os.environ['HOME'], 
                    'Python',
                    'aerosol-scattering',
                    'library')
if dirP_str not in sys.path:
    sys.path.append(dirP_str)
    
import evid_nn
import data

save_path = '/glade/derecho/scratch/mhayman/aerosol_poly_nn/output_analysis/'

model_str_lst = [
    # "20260310T065310",  # 6 beta, 2 alpha all refractive indices
    "20260311T110300",  # 3 beta, 2 alpha all refractive indices
    # "20260324T070054",  # 1 beta, 1 alpha 355 nm, all refractive indices
    # "20260327T080000",  # 1 beta, 1 alpha 532 nm, all refractive indices
]
batch_size = 256
ensemble_size = 25
ensemble_width = 0.20 # (observation uncertainty in fraction.  e.g. 0.05 is 5% error)

is_cuda = torch.cuda.is_available()
device = torch.device(torch.cuda.current_device()) if is_cuda else torch.device("cpu")

if is_cuda:
    torch.backends.cudnn.benchmark = True

print(f'Preparing to use device {device}')

current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def batched_interpolation_with_batched_x(
    a_tnsr: torch.Tensor,
    X_old: torch.Tensor,
    x_new: torch.Tensor
) -> torch.Tensor:
    """
    Interpolates a single 1D tensor (a_tnsr) onto a fixed set of new x-coordinates (x_new),
    using a batch of different original x-coordinate definitions (X_old).

    Args:
        a_tnsr (torch.Tensor): The 1D tensor of y-values to be interpolated. Shape (N,).
        X_old (torch.Tensor): The 2D tensor of original x-coordinates. Shape (B, N).
                              Each row X_old[b, :] defines the x-axis for a_tnsr
                              for that batch item.
        x_new (torch.Tensor): The 1D tensor of new x-coordinates (target points)
                              to interpolate onto. Shape (N_new,).

    Returns:
        torch.Tensor: The interpolated tensor of shape (B, N_new).
    """
    B, N = X_old.shape
    N_new = x_new.shape[0]

    # Ensure all tensors are on the same device
    a_tnsr = a_tnsr.to(X_old.device)
    x_new = x_new.to(X_old.device)

    # Step 1: Find the indices of the two bounding points in X_old for each x_new point.
    # torch.searchsorted needs a sorted_sequence (X_old) and an input (x_new).
    # X_old is (B, N), x_new is (N_new,).
    # searchsorted will broadcast x_new to (B, N_new) and perform the search for each batch.
    # 'right=False' means it finds the index of the first element >= value,
    # so it gives us the index of the right bounding point.
    # FIX: Expand x_new to match the batch dimension of X_old for searchsorted.
    # indices = torch.searchsorted(X_old, x_new.unsqueeze(0).expand(B, -1), right=False) # commented for .continguous fix
    # Resulting 'indices' shape: (B, N_new)

    # FIX: Expand x_new to match the batch dimension of X_old and make it contiguous.
    # We also call .contiguous() on X_old just in case the upstream operations 
    # (like sum and cumsum) fragmented its memory layout.
    
    X_old_contig = X_old.contiguous()
    x_new_contig = x_new.unsqueeze(0).expand(B, -1).contiguous()
    
    indices = torch.searchsorted(X_old_contig, x_new_contig, right=False)

    # Step 2: Clamp indices to handle extrapolation and boundary conditions.
    # idx_left will be the index of the point to the left of x_new.
    # idx_right will be the index of the point to the right of x_new.
    # Clamping ensures indices are within [0, N-1] for valid lookups.
    idx_left = torch.clamp(indices - 1, 0, N - 1)
    idx_right = torch.clamp(indices, 0, N - 1)

    # For points exactly at x_old[i], searchsorted returns i.
    # So idx_left = i-1, idx_right = i.
    # For points < x_old[0], indices = 0, so idx_left = 0, idx_right = 0.
    # For points >= x_old[N-1], indices = N (if N_new is large enough),
    # so idx_left = N-1, idx_right = N-1.

    # Step 3: Get the x-coordinates of the bounding points from X_old.
    # Use torch.gather to select elements from X_old using the computed indices.
    x_left = torch.gather(X_old, 1, idx_left)   # Shape (B, N_new)
    x_right = torch.gather(X_old, 1, idx_right) # Shape (B, N_new)

    # Step 4: Get the corresponding y-values from a_tnsr.
    # a_tnsr is 1D (N,), but idx_left/idx_right are 2D (B, N_new).
    # PyTorch's advanced indexing handles this by broadcasting a_tnsr across the batch.
    y_left = a_tnsr[idx_left]   # Shape (B, N_new)
    y_right = a_tnsr[idx_right] # Shape (B, N_new)

    # Step 5: Calculate the interpolation factor (alpha).
    # Handle potential division by zero if x_left == x_right (e.g., N=1 or clamped points).
    # In such cases, alpha should effectively be 0 or 1, picking one of the y-values.
    denominator = (x_right - x_left)
    alpha = torch.zeros_like(denominator) # Initialize alpha to zeros
    # Only calculate alpha where denominator is not zero
    non_zero_denom_mask = (denominator != 0)
    alpha[non_zero_denom_mask] = (x_new.unsqueeze(0) - x_left)[non_zero_denom_mask] / denominator[non_zero_denom_mask]

    # Step 6: Perform the linear interpolation.
    # This formula naturally handles both interpolation (alpha between 0 and 1)
    # and extrapolation (alpha outside 0-1 range).
    A_interpolated = y_left * (1 - alpha) + y_right * alpha

    return A_interpolated



percentile_lst = [0.01, 0.05, 0.1, 0.2, 0.8, 0.9, 0.95, 0.99]
hist_width = 0.01
example_count = 10

# idx_arr = np.array([18334])
# idx_arr = np.array([60308, 11462, 44852,  6123, 41998, 50674, 87012, 87725, 87707, 17718])
idx_arr = np.array([60308, 11462, 44852,  6123, 41998, 50674, 87012, 87725, 87707, 18334, 44305, 44535, 49700, 55717, 55721, 55744,
        55875, 55960, 56042, 56201, 56414, 56527, 56740, 56798, 56810,
        56822, 56823, 62837, 65213, 53289])


conf_lst = []

r_centroid_arr_lst = []
n_centroid_arr_lst = []
r2_centroid_arr_lst = []
n2_centroid_arr_lst = []
r_act_arr_lst = []
n_act_arr_lst = []
p_model_arr_lst = []

x_rescale_lst = []
y_rescale_lst = []

label_param_lst_lst = []
label_wl_lst_lst = []

r_interval_arr_lst = []
n_interval_arr_lst = []

r_sim_hist_lst_lst = []
n_sim_hist_lst_lst = []

sim_hist_in_data_lst_lst = []

f_pdf_arr_lst = []
f_pdf_ens_arr_lst = []

r_ens_centroid_arr_lst = []
n_ens_centroid_arr_lst = []
r2_ens_centroid_arr_lst = []
n2_ens_centroid_arr_lst = []
r_ens_interval_arr_lst = []
n_ens_interval_arr_lst = []


for m_idx, model_str in enumerate(model_str_lst):
    print(f"loading model {m_idx+1} of {len(model_str_lst)}")
    conf_reload, model_reload = evid_nn.load_poly_nn_model(model_str,name_str='hyper_best',device=device)
    x_scaler = evid_nn.rescale(0,1)
    x_scaler.set_config(**conf_reload['save_model']['x_scaler'])
    y_scaler = evid_nn.rescale(0,1)
    y_scaler.set_config(**conf_reload['save_model']['y_scaler'])

    x_rescale_lst.append(x_scaler)
    y_rescale_lst.append(y_scaler)

    conf_lst.append(conf_reload)
    conf = conf_reload
    dtype = model_reload.dtype

    nc_attrs = {'model':model_str}

    if m_idx == 0:
        intrp_tnsr = torch.tensor(percentile_lst,dtype=dtype,device=device)

    
    label_param_lst = []
    label_wl_lst = []
    
    if 'extinction_coefficient' in conf_reload['data']['wavelength_inputs']:
        label_param_lst.append(r"$\alpha$")
        wl_key='extinction_coefficient'
    if 'backscatter_coefficient' in conf_reload['data']['wavelength_inputs']:
        label_param_lst.append(r"$\beta$")
        wl_key='backscatter_coefficient'

    for wl in conf_reload['data']['wavelength_inputs'][wl_key]['wavelength_lst']:
        label_wl_lst.append(str(int(wl*1e9)))

    label_param_lst_lst.append(label_param_lst)
    label_wl_lst_lst.append(label_wl_lst)


    ### Load Data ###
    # this is repeated because rescaling could theoretically be different between models
    
    override_validation = False  # forces the validation data to be split from training

    rank = 0
    trial = False
    
    
    # device = torch.device(f"cuda:{rank % torch.cuda.device_count()}") if torch.cuda.is_available() else torch.device("cpu")
    # torch.cuda.set_device(rank % torch.cuda.device_count())
    # if conf['model']['dtype'] == 'float32':
    #     dtype = torch.float32
    # elif conf['model']['dtype'] == 'float64':
    #     dtype = torch.float64
    # else:
    #     dtype = torch.float
    
    # Config settings
    # seed = 1000 if "seed" not in conf else conf["seed"]
    # seed_everything(seed)
    
    
    ds = xr.open_dataset(os.path.join(conf['data']['data_path'],conf['data']['data_file']))
    
    train_batch_size = conf['data']['batch_size']
    valid_batch_size = conf['data']['batch_size']
    test_batch_size = conf['data']['batch_size']
    
    # train_idx = int(np.round(ds.sizes['data_index']*conf['data']['train_fraction']))
    # valid_idx = train_idx+int(np.round(ds.sizes['data_index']*conf['data']['valid_fraction']))
    # test_idx = train_idx+int(np.round(ds.sizes['data_index']*conf['data']['test_fraction']))
    
    train_input_arr, train_label_arr, input_str_lst, output_str_lst = data.build_training_array(ds,conf)
    
    if not override_validation:
        valid_ds = xr.open_dataset(os.path.join(conf['data']['valid_data_path'],conf['data']['valid_data_file']))
        
        valid_input_arr, valid_label_arr, _, _ = data.build_training_array(valid_ds,conf)
        
        x_train = x_scaler.scale(np.log(train_input_arr.values))
        x_valid = x_scaler.scale(np.log(valid_input_arr.values))
        
        y_train = y_scaler.scale(np.log(train_label_arr.values))
        y_valid = y_scaler.scale(np.log(valid_label_arr.values))
    else:
        print("validation dataset encountered an error:")
        print("override_validation=True, so validation is split from training data")
        # print(str(E))
        x_train = x_scaler.scale(np.log(train_input_arr.values[:train_idx,:]))
        x_valid = x_scaler.scale(np.log(train_input_arr.values[train_idx:valid_idx,:]))
        
        y_train = y_scaler.scale(np.log(train_label_arr.values[:train_idx,:]))
        y_valid = y_scaler.scale(np.log(train_label_arr.values[train_idx:valid_idx,:]))
    
    
    cond_args = conf['data']['cond_args']
    
    # train_dataset = evid_nn.EvidDataset(x_train,y_train,
    #                             dtype=dtype,device=device,
    #                             **cond_args)
    # valid_dataset = evid_nn.EvidDataset(x_valid,y_valid,
    #                             dtype=dtype,device=device,
    #                             **cond_args)
    # # test_dataset = evid_nn.EvidDataset(x_test,y_test,
    # #                             dtype=dtype,device=device,
    # #                             **cond_args)
    
    # train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    # valid_dataloader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=True)
    
    test_data_file = os.path.join(conf['data']['test_data_path'],conf['data']['test_data_file'])
    print("Evaluting output using data:")
    print(test_data_file)

    test_ds = xr.open_dataset(test_data_file)
    
    test_input_arr, test_label_arr, _, _ = data.build_training_array(test_ds,conf)
    
    x_test = x_scaler.scale(np.log(test_input_arr.values))
    
    y_test = y_scaler.scale(np.log(test_label_arr.values))
    
    test_dataset = evid_nn.EvidDataset(x_test,y_test,
                                dtype=dtype,device=device,
                                **cond_args)
    
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    ### End Data Loader ###

    # in_data = x_train
    # out_label = y_train
    
    # in_data = x_valid
    # out_label = y_valid

    ### Perform example analysis ###
    # actual sample data
    # in_data = x_test
    # out_label = y_test

    # data for histogram analysis
    hist_in_data = np.concatenate([x_train,x_valid,x_test],axis=0)
    hist_out_label = np.concatenate([y_train,y_valid,y_test],axis=0)

    sample_interval = 20
    in_data = hist_in_data[::sample_interval,...]
    out_label = hist_out_label[::sample_interval,...]

    # # data for histogram analysis
    # hist_in_data = np.concatenate([x_train,x_valid,x_test],axis=0)
    # hist_out_label = np.concatenate([y_train,y_valid,y_test],axis=0)

    # if m_idx == 0:
    #     # idx_arr = (np.random.rand(example_count)*in_data.shape[0]).astype(int)  # get random input
    #     r_act = out_label[:,1][idx_arr]
    #     n_act = out_label[:,0][idx_arr]
        
    # sim_idx_lst = []
    # r_sim_hist_lst = []
    # n_sim_hist_lst = []
    # sim_hist_in_data_lst = []
    # for sidx in range(idx_arr.size):
    #     idx = idx_arr[sidx]
    #     sim_idx = np.where(np.prod(np.abs(hist_in_data - in_data[idx,:].reshape(1,-1)) < hist_width,axis=1))
    #     sim_idx_lst.append(sim_idx)

    #     sim_hist_in_data_lst.append(hist_in_data[sim_idx])

    #     r_sim_hist = hist_out_label[:,1][sim_idx]
    #     n_sim_hist = hist_out_label[:,0][sim_idx]

    #     r_sim_hist_lst.append(r_sim_hist)
    #     n_sim_hist_lst.append(n_sim_hist)

    # r_sim_hist_lst_lst.append(r_sim_hist_lst)
    # n_sim_hist_lst_lst.append(n_sim_hist_lst)
    # sim_hist_in_data_lst_lst.append(sim_hist_in_data_lst)

    # # pdf_ax = np.exp(y_scaler.unscale(np.stack([x_plt,x_plt],axis=1)))
    # # hist_ax = np.exp(y_scaler.unscale(np.stack([h_part[1],h_part[2]],axis=1)))
    # # act_ax = np.exp(y_scaler.unscale(np.stack([np.array([n_act]),np.array([r_act])],axis=1))).squeeze()

    # # pdf_ax_lst.append(pdf_ax)
    # # hist_ax_lst.append(hist_ax)
    # # act_ax_lst.append(act_ax)
        

    # f_pdf_lst = []
    # f_pdf_ens_lst = []
    # with torch.no_grad():
    #     for sidx, sim_idx in enumerate(sim_idx_lst):
    #         idx = idx_arr[sidx]
    #         f_pdf_tnsr = model_reload.output_pdf(torch.tensor(in_data[idx:(idx+1),:],dtype=dtype,device=device))
    #         f_pdf = f_pdf_tnsr.detach().cpu().numpy()
    #         x_plt = model_reload.x_int.detach().cpu().numpy()
            
    #         ensemble_size = 100
    #         ensemble_width = hist_width # (observation uncertainty)
    #         loop_length = 4
    #         f_pdf_ens = np.zeros((f_pdf.shape[1],f_pdf.shape[2]))
    #         for ens_group_idx in range(loop_length):
    #             f_pdf_ens_tnsr = model_reload.output_pdf(torch.tensor(in_data[idx:(idx+1),:]+(np.random.rand(ensemble_size,in_data.shape[1])*2-1)*ensemble_width,dtype=dtype,device=device))
    #             f_pdf_ens += np.sum(f_pdf_ens_tnsr.detach().cpu().numpy(),axis=0)
    #             # print(ens_group_idx)
            
    #         f_pdf_ens = f_pdf_ens/(loop_length*ensemble_size)

    #         f_pdf_lst.append(f_pdf)
    #         f_pdf_ens_lst.append(f_pdf_ens)

    # f_pdf_arr_lst.append(np.concatenate(f_pdf_lst,axis=0))
    # f_pdf_ens_arr_lst.append(np.stack(f_pdf_ens_lst,axis=0))

    
        

    ### perform aggregate analysis ###

    # batch_size = 256
    # ensemble_size = 25
    # ensemble_width = 0.10 # (observation uncertainty in fraction.  e.g. 0.05 is 5% error)
    batch_count = in_data.shape[0]//batch_size # hardcoded for testing
    
    r_centroid_lst = []
    n_centroid_lst = []
    r2_centroid_lst = []
    n2_centroid_lst = []
    r_centroid_ens_lst = []
    n_centroid_ens_lst = []
    r_act_lst = []
    n_act_lst = []
    p_model_lst = []
    r_interval_lst = []
    n_interval_lst = []

    r_ens_interval_lst = []
    n_ens_interval_lst = []

    r_ens_centroid_lst = []
    n_ens_centroid_lst = []
    r2_ens_centroid_lst = []
    n2_ens_centroid_lst = []
    
    with torch.no_grad():
        for batch in tqdm.tqdm(range(batch_count),miniters=100):
            input_batch = torch.tensor(in_data[batch*batch_size:((batch+1)*batch_size),:], dtype=dtype, device=device)
            
            f_pdf_tnsr = model_reload.output_pdf(input_batch)
            # f_pdf = f_pdf_tnsr.detach().cpu().numpy()
            # x_plt = model_reload.x_int.detach().cpu().numpy()
            # dx2 = (model_reload.dx_int**2).detach().cpu().numpy()
            cdf_r = torch.cumsum(torch.sum(f_pdf_tnsr,dim=1),dim=1)*model_reload.dx_int**2
            cdf_n = torch.cumsum(torch.sum(f_pdf_tnsr,dim=2),dim=1)*model_reload.dx_int**2
            r_interval_lst.append(batched_interpolation_with_batched_x(model_reload.x_int,cdf_r,intrp_tnsr).detach().cpu().numpy())
            n_interval_lst.append(batched_interpolation_with_batched_x(model_reload.x_int,cdf_n,intrp_tnsr,).detach().cpu().numpy())
            # print(r_interval_lst[-1].shape)
        
            r_centroid_lst.append(torch.sum(model_reload.x_int.detach().reshape(1,1,-1)*f_pdf_tnsr*model_reload.dx_int**2,dim=(1,2)).detach().cpu().numpy())
            n_centroid_lst.append(torch.sum(model_reload.x_int.detach().reshape(1,-1,1)*f_pdf_tnsr*model_reload.dx_int**2,dim=(1,2)).detach().cpu().numpy())
            r2_centroid_lst.append(torch.sum(model_reload.x_int.detach().reshape(1,1,-1)**2*f_pdf_tnsr*model_reload.dx_int**2,dim=(1,2)).detach().cpu().numpy())
            n2_centroid_lst.append(torch.sum(model_reload.x_int.detach().reshape(1,-1,1)**2*f_pdf_tnsr*model_reload.dx_int**2,dim=(1,2)).detach().cpu().numpy())
        
            r_act_lst.append(out_label[batch*batch_size:((batch+1)*batch_size),1])
            n_act_lst.append(out_label[batch*batch_size:((batch+1)*batch_size),0])
        
            fit_coef = model_reload.net(input_batch)
            p_model = model_reload.fpdf(fit_coef, input_batch)
            p_model_lst.append(p_model.detach().cpu().numpy())

            # for ensemble analysis (which will slow things down
            # ensemble_size = 100
            # ensemble_width = 0.05 # (observation uncertainty in fraction.  e.g. 0.05 is 5% error)
            hist_width_arr = torch.tensor(np.log10(1+ensemble_width)/x_scaler.gain,dtype=dtype,device=device)
            # loop_length = 4
            # f_pdf_ens = np.zeros((f_pdf_tnsr.shape[1],f_pdf_tnsr.shape[2]))
            f_pdf_ens = torch.zeros(f_pdf_tnsr.shape,dtype=dtype,device=device)
            for ens_group_idx in range(ensemble_size):
                f_pdf_ens_tnsr = model_reload.output_pdf(input_batch+torch.randn(batch_size,input_batch.shape[1],device=device)*hist_width_arr)
                f_pdf_ens += f_pdf_ens_tnsr
                # f_pdf_ens += f_pdf_ens_tnsr.detach().cpu().numpy()
            # for ens_group_idx in range(loop_length):
                # f_pdf_ens_tnsr = model_reload.output_pdf(torch.tensor(in_data[idx:(idx+1),:]+np.random.randn(ensemble_size,in_data.shape[1])*ensemble_width,dtype=dtype,device=device))
                # f_pdf_ens += np.sum(f_pdf_ens_tnsr.detach().cpu().numpy(),axis=0)
                # print(ens_group_idx)
            
            # f_pdf_ens = f_pdf_ens/(loop_length*ensemble_size)
            f_pdf_ens = f_pdf_ens/ensemble_size

            cdf_ens_r = torch.cumsum(torch.sum(f_pdf_ens,dim=1),dim=1)*model_reload.dx_int**2
            cdf_ens_n = torch.cumsum(torch.sum(f_pdf_ens,dim=2),dim=1)*model_reload.dx_int**2
            r_ens_interval_lst.append(batched_interpolation_with_batched_x(model_reload.x_int,cdf_ens_r,intrp_tnsr).detach().cpu().numpy())
            n_ens_interval_lst.append(batched_interpolation_with_batched_x(model_reload.x_int,cdf_ens_n,intrp_tnsr,).detach().cpu().numpy())
        
            r_ens_centroid_lst.append(torch.sum(model_reload.x_int.detach().reshape(1,1,-1)*f_pdf_ens*model_reload.dx_int**2,dim=(1,2)).detach().cpu().numpy())
            n_ens_centroid_lst.append(torch.sum(model_reload.x_int.detach().reshape(1,-1,1)*f_pdf_ens*model_reload.dx_int**2,dim=(1,2)).detach().cpu().numpy())
            r2_ens_centroid_lst.append(torch.sum(model_reload.x_int.detach().reshape(1,1,-1)**2*f_pdf_ens*model_reload.dx_int**2,dim=(1,2)).detach().cpu().numpy())
            n2_ens_centroid_lst.append(torch.sum(model_reload.x_int.detach().reshape(1,-1,1)**2*f_pdf_ens*model_reload.dx_int**2,dim=(1,2)).detach().cpu().numpy())
    
    
        r_centroid_arr_lst.append(np.concatenate(r_centroid_lst))
        n_centroid_arr_lst.append(np.concatenate(n_centroid_lst))
        r2_centroid_arr_lst.append(np.concatenate(r2_centroid_lst))
        n2_centroid_arr_lst.append(np.concatenate(n2_centroid_lst))
        r_act_arr_lst.append(np.concatenate(r_act_lst))
        n_act_arr_lst.append(np.concatenate(n_act_lst))
        p_model_arr_lst.append(np.concatenate(p_model_lst))
        r_interval_arr_lst.append(np.concatenate(r_interval_lst,axis=0))
        n_interval_arr_lst.append(np.concatenate(n_interval_lst,axis=0))

        r_ens_centroid_arr_lst.append(np.concatenate(r_ens_centroid_lst))
        n_ens_centroid_arr_lst.append(np.concatenate(n_ens_centroid_lst))
        r2_ens_centroid_arr_lst.append(np.concatenate(r2_ens_centroid_lst))
        n2_ens_centroid_arr_lst.append(np.concatenate(n2_ens_centroid_lst))
        r_ens_interval_arr_lst.append(np.concatenate(r_ens_interval_lst,axis=0))
        n_ens_interval_arr_lst.append(np.concatenate(n_ens_interval_lst,axis=0))

    save_output_file_name = "full_aggregate_analysis_"+model_str+f"_{int(100*ensemble_width)}error.nc"
    save_aggr_ds = xr.Dataset({},attrs=nc_attrs)
    save_aggr_ds.attrs['ensemble_width'] = ensemble_width
    save_aggr_ds.attrs['ensemble_size'] = ensemble_size
    save_aggr_ds.attrs['model'] = model_str
    save_aggr_ds.attrs['sample_interval']= sample_interval
    save_aggr_ds.attrs['file_creation_date'] = current_time_str

    save_aggr_ds['r_centroid'] = xr.DataArray(r_centroid_arr_lst[-1],dims=('data_index',),
                                              attrs={'description':'centroid of effective radius of PDF'})
    save_aggr_ds['n_centroid'] = xr.DataArray(n_centroid_arr_lst[-1],dims=('data_index',),
                                              attrs={'description':'centroid of number density'})
    
    save_aggr_ds['r_act'] = xr.DataArray(r_act_arr_lst[-1],dims=('data_index',),
                                              attrs={'description':'actual radius of aerosol'})
    # ValueError: cannot reindex or align along dimension 'data_index' because of conflicting dimension sizes: {3283200, 256}
    save_aggr_ds['n_act'] = xr.DataArray(n_act_arr_lst[-1],dims=('data_index',),
                                              attrs={'description':'ensemble actual number density of aerosol'})
    
    save_aggr_ds['r_ens_centroid'] = xr.DataArray(r_ens_centroid_arr_lst[-1],dims=('data_index',),
                                              attrs={'description':'centroid of effective radius of PDF',
                                                     'ensemble_width': ensemble_width})
    save_aggr_ds['n_ens_centroid'] = xr.DataArray(n_ens_centroid_arr_lst[-1],dims=('data_index',),
                                              attrs={'description':'ensemble centroid of number density',
                                                     'ensemble_width': ensemble_width})
    save_aggr_ds['ens_std_width'] = xr.DataArray(hist_width_arr.cpu().numpy().squeeze(),dims=('inputs',),
                                                 attrs={'description':'standard deviation of random number added to inputs'})
    save_aggr_ds['percentiles'] = xr.DataArray(percentile_lst,dims=('percentiles',),
                                               attrs={'description':'percentiles used to evaluate intervals'})
    save_aggr_ds['r_interval'] = xr.DataArray(r_interval_arr_lst[-1],dims=('data_index','percentiles'),
                                              coords={'percentiles':save_aggr_ds['percentiles']},
                                              attrs={'descrption':'intervals where distribution of radius crosses the percentile threshold'})   
    save_aggr_ds['r_ens_interval'] = xr.DataArray(r_ens_interval_arr_lst[-1],dims=('data_index','percentiles'),
                                              coords={'percentiles':save_aggr_ds['percentiles']},
                                              attrs={'descrption':'intervals where ensemble distribution of radius crosses the percentile threshold'})
    
    save_aggr_ds['n_interval'] = xr.DataArray(n_interval_arr_lst[-1],dims=('data_index','percentiles'),
                                              coords={'percentiles':save_aggr_ds['percentiles']},
                                              attrs={'descrption':'intervals where distribution of number density crosses the percentile threshold'})   
    save_aggr_ds['n_ens_interval'] = xr.DataArray(n_ens_interval_arr_lst[-1],dims=('data_index','percentiles'),
                                              coords={'percentiles':save_aggr_ds['percentiles']},
                                              attrs={'descrption':'intervals where ensemble distribution of number density crosses the percentile threshold'})

    save_aggr_ds['hist_in_data'] = xr.DataArray(hist_in_data,dims=('total_data_points','input_vars',),
                                                attrs={'description':'all NN input variables across train, validation and test datasets'})
    save_aggr_ds['hist_out_label'] = xr.DataArray(hist_out_label,dims=('total_data_points','output_vars',),
                                                attrs={'description':'all NN output labels across train, validation and test datasets'})
    
    save_file_full = os.path.join(save_path,save_output_file_name)
    save_aggr_ds.to_netcdf(save_file_full)
    print("Saving analysis result for model {model_str} to:")
    print(save_file_full)
    # save_aggr_ds['pdf_probability'] = xr.DataArray(p_model_lst[-1],dims=('data_index',),
    #                                                )

