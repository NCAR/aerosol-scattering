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

from scipy.special import gamma, gammaln

import copy
import datetime
import yaml

import importlib

# import tqdm
from tqdm import tqdm


dirP_str = os.path.join(os.environ['HOME'], 
                    'Python',
                    'aerosol-scattering',
                    'library')
if dirP_str not in sys.path:
    sys.path.append(dirP_str)
    
import evid_nn
import data


is_cuda = torch.cuda.is_available()
device = torch.device(torch.cuda.current_device()) if is_cuda else torch.device("cpu")

# device= torch.device("cpu")

if is_cuda:
    torch.backends.cudnn.benchmark = True

print(f'Preparing to use device {device}')

model_path = "$HOME/Python/aerosol-scattering/models/"

save_data_path = "/glade/derecho/scratch/mhayman/aerosol_poly_nn/output_analysis/"

# train model on refractive index from 1.3-1.5 and 0.0-0.06
model_str_lst = [
    # "20260310T065310",  # 6 beta, 2 alpha all refractive indices
    # "20260311T110300",  # 3 beta, 2 alpha all refractive indices
    # "20260324T070054",  # 1 beta, 1 alpha 355 nm, all refractive indices
    # "20260327T080000",  # 1 beta, 1 alpha 532 nm, all refractive indices
    "20260507T072534",  # 3 beta, 0 alpha, all refractive indices
]

run_example_only = True

example_count = 10

# idx_arr = np.array([18334])
# idx_arr = np.array([60308, 11462, 44852,  6123, 41998, 50674, 87012, 87725, 87707, 17718])
# idx_arr = np.array([60308, 11462, 44852,  6123, 41998, 50674, 87012, 87725, 87707, 18334, 44305, 44535, 49700, 55717, 55721, 55744,
#         55875, 55960, 56042, 56201, 56414, 56527, 56740, 56798, 56810,
#         56822, 56823, 62837, 65213, 53289])

# produced using idx_arr = (np.random.rand(150)*test_ds.sizes['data_index']).astype(int)
idx_arr = np.array([ 378522,  593729, 1041362,  717119,   48150,  635869,  926980,
        137710,  303504,  894220,  653744,  109183,  838026,   72911,
        791867,   91345,  479735, 1074203,   68518, 1085864,  664193,
        430965, 1078857,  975595,  721214,  824936,  477459,  103382,
       1002878,  990327,  557148,  820133, 1056277,  777505,  835633,
        792242,  940092,  817533,  564972,  252295,   69655, 1121670,
         14823,  601163,  773041,  231152, 1026252,  429214,  669218,
        857646,  684716,  882426,  531865,  728651,  543995,  830339,
        356460,  134732,  472545, 1033691,  571859,  187639, 1039548,
       1070737,  709717,  821860, 1134148,  455711,  545662, 1065300,
        559200,  692848,  311547,  727411, 1128726, 1096914,  218639,
         42582,  812045,  640661,   98643,   89202,  976381,   43838,
        871662,  403116,  950898,    6915,  772075,  154701,  519375,
       1095816,  600820,  295350,  946051, 1073333,  120368,    6384,
        758016,  181696, 1085045,  230645,  805596,  952608,  342586,
        287426,  673814,   11072,  787742,  695491, 1123782,  202502,
        843743, 1018509, 1119713,  384031,  280654,  904014,  353650,
        430253,  366978,    9715,  163843,  185899,  588775,  423068,
        155131,  429416,  586119,  868653,  413911,  601467,  762260,
        938064,  698400,  546354,  725411, 1143689,  197693,  234949,
        924755,  686778,  344789,  913148,  365489,  704516,  216246,
        288377,  344584,  515697])


# store results from optimizing the histogram bins for each model
model_opt_hist_dct = {
    "20260310T065310":{
        "input_bins": 53,
        "output_bins": 43
    },
    "20260311T110300":{
        "input_bins": 114,
        "output_bins": 43
    },
    "20260324T070054":{
        "input_bins": 190,
        "output_bins": 79
    },
    "20260327T080000":{
        "input_bins": 190,
        "output_bins": 108
    },
    "20260507T072534":{
        "input_bins": 147,
        "output_bins": 58
    },
}

# model: 20260310T065310
# optimal input bins: 53
# optimal output bins: 43

# model: 20260311T110300
# optimal input bins: 114
# optimal output bins: 43

# model: 20260324T070054
# optimal input bins: 190
# optimal output bins: 79

# model: 20260327T080000
# optimal input bins: 190
# optimal output bins: 108


conf_lst = []

# r_centroid_arr_lst = []
# n_centroid_arr_lst = []
# r2_centroid_arr_lst = []
# n2_centroid_arr_lst = []
# r_act_arr_lst = []
# n_act_arr_lst = []
# p_model_arr_lst = []

x_rescale_lst = []
y_rescale_lst = []

label_param_lst_lst = []
label_wl_lst_lst = []

# r_interval_arr_lst = []
# n_interval_arr_lst = []

# r_sim_hist_lst_lst = []
# n_sim_hist_lst_lst = []

# sim_hist_in_data_lst_lst = []

f_pdf_arr_lst = []
f_pdf_ens_arr_lst = []
h_pdf_arr_lst = []

# r_ens_centroid_arr_lst = []
# n_ens_centroid_arr_lst = []
# r2_ens_centroid_arr_lst = []
# n2_ens_centroid_arr_lst = []
# r_ens_interval_arr_lst = []
# n_ens_interval_arr_lst = []

hist_nll_lst = []
hist_eval_idx_lst = []
f_pdf_nll_lst = []


for m_idx, model_str in enumerate(model_str_lst):
    print(f"loading model {m_idx+1} of {len(model_str_lst)}")
    conf_reload, model_reload = evid_nn.load_poly_nn_model(model_str,name_str='hyper_best',device=device) # path=model_path

    # conf_reload['data']['data_path'] = "/export/breeze1/mhayman/aerosol_poly_nn/datasets/"
    # conf_reload['data']['valid_data_path'] = "/export/breeze1/mhayman/aerosol_poly_nn/datasets/"
    # conf_reload['data']['test_data_path'] = "/export/breeze1/mhayman/aerosol_poly_nn/datasets/"

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

    # if m_idx == 0:
    #     intrp_tnsr = torch.tensor(percentile_lst,dtype=dtype,device=device)

    
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
    
    
    ds = xr.open_dataset(os.path.join(conf['data']['data_path'],conf['data']['data_file']))
    
    train_batch_size = conf['data']['batch_size']
    valid_batch_size = conf['data']['batch_size']
    test_batch_size = 1024 # conf['data']['batch_size']
    
    train_idx = int(np.round(ds.sizes['data_index']*conf['data']['train_fraction']))
    valid_idx = train_idx+int(np.round(ds.sizes['data_index']*conf['data']['valid_fraction']))
    test_idx = train_idx+int(np.round(ds.sizes['data_index']*conf['data']['test_fraction']))
    
    
    train_input_arr, train_label_arr, input_str_lst, output_str_lst, input_frac_uncertainty = data.build_training_array(ds,conf)
    
    if not override_validation:
        # valid_ds = xr.open_dataset(os.path.join(conf['data']['valid_data_path'],conf['data']['valid_data_file']))
        
        
        # valid_input_arr, valid_label_arr, _, _, _ = data.build_training_array(valid_ds,conf)
        
        x_train = x_scaler.scale(np.log(train_input_arr.values))
        # x_valid = x_scaler.scale(np.log(valid_input_arr.values))
        
        y_train = y_scaler.scale(np.log(train_label_arr.values))
        # y_valid = y_scaler.scale(np.log(valid_label_arr.values))
    else:
        print("validation dataset encountered an error:")
        print("override_validation=True, so validation is split from training data")
        # print(str(E))
        x_train = x_scaler.scale(np.log(train_input_arr.values[:train_idx,:]))
        # x_valid = x_scaler.scale(np.log(train_input_arr.values[train_idx:valid_idx,:]))
        
        y_train = y_scaler.scale(np.log(train_label_arr.values[:train_idx,:]))
        # y_valid = y_scaler.scale(np.log(train_label_arr.values[train_idx:valid_idx,:]))
    
    
    cond_args = conf['data']['cond_args']
    
    train_dataset = evid_nn.EvidDataset(x_train,y_train,
                                dtype=dtype,device=device,
                                **cond_args)
    # valid_dataset = evid_nn.EvidDataset(x_valid,y_valid,
    #                             dtype=dtype,device=device,
    #                             **cond_args)
    # test_dataset = evid_nn.EvidDataset(x_test,y_test,
    #                             dtype=dtype,device=device,
    #                             **cond_args)
    
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    # valid_dataloader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=True)

    # use the training data to estimate histgrams with bin widths optimized using validation data
    # then evaluate the NLL of those PDFs on the test data.  Compare the result to the ML solution.
    # building the histograms requires we do a grid search over all possible inputs
    
    
    test_ds = xr.open_dataset(os.path.join(conf['data']['test_data_path'],conf['data']['test_data_file']))
    
    test_input_arr, test_label_arr, _, _, _ = data.build_training_array(test_ds,conf)
    
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
    in_data = x_test
    out_label = y_test

    ds_output = xr.Dataset({},
                    attrs={
                        'model':model_str,
                        'description':'analysis of NN model performance for aerosol size/concentration inversions',
                        'label_params':', '.join(label_param_lst),
                        'label_wl':', '.join(label_wl_lst)
                    })


    if not run_example_only:
        y_pred_lst = []
        with torch.no_grad():
            # calculate the loss for all test points
            # for k, (x,label_data) in enumerate(test_dataloader):
            for (x,label_data) in tqdm(test_dataloader, desc="NN Test Progress"):
                # y_pred_batch = model_reload(x,label_data).detach().cpu().numpy()
                fit_coef = model_reload.net(x)
                loss = model_reload.fpdf_nll(fit_coef,label_data)
                y_pred_lst.append(loss.cpu()) # Move to CPU here
                # y_pred_lst.append(loss)
        y_pred = torch.concat(y_pred_lst).detach().cpu().numpy()
    




    ### Calculate the histogram solution 
    hist_in_pt  = model_opt_hist_dct[model_str]["input_bins"]
    hist_out_pt = model_opt_hist_dct[model_str]["output_bins"]

    N_samples = x_train.shape[0]
    D_in = x_train.shape[1]
    D_out = y_train.shape[1]
    
    x_bins = np.clip(np.floor(x_train * hist_in_pt).astype(int), 0, hist_in_pt - 1)
            
    # 2. Digitize Outputs
    y_bins = np.clip(np.floor(y_train * hist_out_pt).astype(int), 0, hist_out_pt - 1)

    
    # 3. Stack inputs and outputs into a single array of shape (N_samples, D_in + D_out)
    combined_bins = np.hstack((x_bins, y_bins))
    
    # 4. Find all unique populated bins and count their frequencies
    # This completely skips any empty bins in your high-dimensional grid.
    unique_combinations, counts = np.unique(combined_bins, axis=0, return_counts=True)

    if not run_example_only:
        # 1. Digitize validation inputs and outputs
        x_valid_bins = np.clip(np.floor(in_data * hist_in_pt).astype(int), 0, hist_in_pt - 1)
        y_valid_bins = np.clip(np.floor(out_label * hist_out_pt).astype(int), 0, hist_out_pt - 1)


        # 2. Stack them into a single array of shape (N_samples, D_in + D_out)
        valid_combined_bins = np.hstack((x_valid_bins, y_valid_bins))
        
        # Helper function to view a 2D array of coordinates as a 1D array of bytes
        def view1d(a):
            a = np.ascontiguousarray(a)
            return a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()
        
        # Create 1D views of our joint bins
        train_1d = view1d(unique_combinations)
        valid_1d = view1d(valid_combined_bins)
        
        # np.unique(..., axis=0) naturally sorts its output, so train_1d is already sorted!
        # We find where each validation point would be inserted into the training bins
        insert_idx = np.searchsorted(train_1d, valid_1d)
        
        # Protect against validation points that fall completely outside the training range
        insert_idx = np.clip(insert_idx, 0, len(train_1d) - 1)
        
        # Check which validation points actually perfectly matched a populated training bin
        match_mask = (train_1d[insert_idx] == valid_1d)
        
        # Extract the joint counts! Validation points that land in empty bins get 0.
        valid_joint_counts = np.zeros(len(valid_combined_bins), dtype=counts.dtype)
        valid_joint_counts[match_mask] = counts[insert_idx[match_mask]]
        
        # Extract just the input coordinates from the unique training bins
        train_x_bins = unique_combinations[:, :D_in]
        
        # Find the unique input bins and map every original row to its unique input index
        _, x_inv_idx = np.unique(train_x_bins, axis=0, return_inverse=True)
        
        # Sum the counts for all outputs sharing the same input bin
        marginal_x_counts = np.bincount(x_inv_idx, weights=counts)
        
        # Now apply the same 1D lookup trick to map marginal_x_counts to the validation points
        train_x_1d = view1d(unique_combinations[np.unique(x_inv_idx, return_index=True)[1], :D_in])
        valid_x_1d = view1d(x_valid_bins)
        
        idx_x = np.searchsorted(train_x_1d, valid_x_1d)
        idx_x = np.clip(idx_x, 0, len(train_x_1d) - 1)
        match_x_mask = (train_x_1d[idx_x] == valid_x_1d)
        
        valid_marginal_counts = np.zeros(len(x_valid_bins), dtype=float)
        valid_marginal_counts[match_x_mask] = marginal_x_counts[idx_x[match_x_mask]]
        
        # Calculate the conditional probability P(Y_bin | X_bin)
        # Avoid division by zero for inputs that were never seen in training
        with np.errstate(divide='ignore', invalid='ignore'):
            conditional_prob = np.where(
                valid_marginal_counts > 0, 
                valid_joint_counts / valid_marginal_counts, 
                0.0
            )
        # when the histogram is zero at the target location, this presents a problem for using the histogram to evaluate the NLL
        eval_idx = np.where(conditional_prob > 0)
        hist_nll = np.ones(conditional_prob.size)*1e10
        hist_nll[eval_idx] = -np.log(conditional_prob[eval_idx]/(1.0/hist_out_pt)**2)  # use sum (instead of mean) to penalize missing data points that aren't included?
        print(f"input bins {hist_in_pt}, output bins {hist_out_pt}: {np.sum(hist_nll[eval_idx])}")


        print(f"NN: {np.sum(y_pred[eval_idx])}")
        print(f"NN unfiltered: {np.sum(y_pred)}")
        print()

        hist_nll_lst.append(hist_nll)
        hist_eval_idx_lst.append(eval_idx)
        f_pdf_nll_lst.append(y_pred)

        # ds_output = xr.Dataset({},
        #                     attrs={
        #                         'model':model_str,
        #                         'description':'analysis of NN model performance for aerosol size/concentration inversions',
        #                         'label_params':', '.join(label_param_lst),
        #                         'label_wl':', '.join(label_wl_lst)
        #                     })
        ds_output['test_hist_nll'] = xr.DataArray(hist_nll,
                                                dims=('test_data',),
                                                attrs={'description':'Negative log-likelihood of test data evaluated against a histogram estimate of the PDF.  Use test_eval_idx to filter out cases where no histogram data exists',
                                                        'input_bins':hist_in_pt,
                                                        'output_bins':hist_out_pt,})
        ds_output['test_eval_idx'] = xr.DataArray(conditional_prob > 0,
                                                dims=('test_data',),
                                                attrs={'description':'Set to 1 where histogame data is valid.  0 where histogram data is not valid due to lack of data points.',
                                                        })
        ds_output['test_nn_nll'] = xr.DataArray(y_pred,
                                                dims=('test_data',),
                                                attrs={'description':'Negative log-likelihood of test data evaluated against a NN estimate of the PDF',
                                                        })

    # h_in_cnt_lst.append(hist_in_pt)
    # h_out_cnt_lst.append(hist_out_pt)
    # h_vld_lst.append(hist_nll)

    

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

    # pdf_ax = np.exp(y_scaler.unscale(np.stack([x_plt,x_plt],axis=1)))
    # hist_ax = np.exp(y_scaler.unscale(np.stack([h_part[1],h_part[2]],axis=1)))
    # act_ax = np.exp(y_scaler.unscale(np.stack([np.array([n_act]),np.array([r_act])],axis=1))).squeeze()

    # pdf_ax_lst.append(pdf_ax)
    # hist_ax_lst.append(hist_ax)
    # act_ax_lst.append(act_ax)
        

    f_pdf_lst = []
    f_pdf_ens_lst = []
    with torch.no_grad():
        # for sidx, sim_idx in enumerate(sim_idx_lst):
        # for sidx in range(idx_arr.size):
        for sidx in tqdm(range(idx_arr.size), desc="Example case progress"):
            idx = idx_arr[sidx]
            f_pdf_tnsr = model_reload.output_pdf(torch.tensor(in_data[idx:(idx+1),:],dtype=dtype,device=device))
            f_pdf = f_pdf_tnsr.detach().cpu().numpy()
            x_plt = model_reload.x_int.detach().cpu().numpy()
            
            ensemble_size = 100
            # ensemble_width = hist_width # (observation uncertainty)
            ensemble_width = 1.0/hist_in_pt # (observation uncertainty) set by optimized histogram analysis
            loop_length = 4
            f_pdf_ens = np.zeros((f_pdf.shape[1],f_pdf.shape[2]))
            for ens_group_idx in range(loop_length):
                f_pdf_ens_tnsr = model_reload.output_pdf(torch.tensor(in_data[idx:(idx+1),:]+(np.random.rand(ensemble_size,in_data.shape[1])*2-1)*ensemble_width,dtype=dtype,device=device))
                f_pdf_ens += np.sum(f_pdf_ens_tnsr.detach().cpu().numpy(),axis=0)
                # print(ens_group_idx)
            
            f_pdf_ens = f_pdf_ens/(loop_length*ensemble_size)
            f_pdf_ens_lst.append(f_pdf_ens)
            
            f_pdf_lst.append(f_pdf)

    f_pdf_arr_lst.append(np.concatenate(f_pdf_lst,axis=0))
    f_pdf_ens_arr_lst.append(np.stack(f_pdf_ens_lst,axis=0))

    pdf_ax = np.exp(y_rescale_lst[m_idx].unscale(np.stack([x_plt,x_plt],axis=1)))
    label_examples =  np.exp(y_rescale_lst[m_idx].unscale(y_test[idx_arr,:]))
    
    ds_output['example_idx'] = xr.DataArray(idx_arr,dims=('example_idx',),
                                            attrs = {
                                                'description':'example input/output cases for testing and visualization'
                                            }
                                           )
    ds_output['nn_conc_bins'] = xr.DataArray(pdf_ax[:,0],dims=('nn_conc_bins',),
                                             attrs = {
                                                 'description':'concentration axis of nn output examples',
                                                 'units':'m^(-3)',
                                             }
                                            )
    ds_output['nn_rad_bins'] = xr.DataArray(pdf_ax[:,1],dims=('nn_rad_bins',),
                                             attrs = {
                                                 'description':'effective radius axis of nn output examples',
                                                 'units':'m',
                                             }
                                            )
    
    ds_output['effective_radius'] = xr.DataArray(label_examples[:,1],
                                                 dims=('example_idx'),
                                                 coords={'example_idx':ds_output['example_idx'],},
                                                 attrs={'description':'actual effective radius of data point used for each example case',
                                                        'units':'m'})
    
    ds_output['number_concentration'] = xr.DataArray(label_examples[:,0],
                                                 dims=('example_idx'),
                                                 coords={'example_idx':ds_output['example_idx'],},
                                                 attrs={'description':'actual number concentration of data point used for each example case',
                                                        'units':'m^(-3)'})
    
    ds_output['nn_pdf_examples'] = xr.DataArray(f_pdf_arr_lst[-1],
                                                dims=('example_idx','nn_conc_bins','nn_rad_bins',),
                                                coords={'example_idx':ds_output['example_idx'],
                                                       'nn_conc_bins':ds_output['nn_conc_bins'],
                                                       'nn_rad_bins':ds_output['nn_rad_bins']},
                                                attrs={
                                                    'description':'NN estimated PDF outputs for example cases',
                                                }
                                               )
    ds_output['nn_ens_pdf_examples'] = xr.DataArray(f_pdf_ens_arr_lst[-1],
                                                dims=('example_idx','nn_conc_bins','nn_rad_bins',),
                                                coords={'example_idx':ds_output['example_idx'],
                                                       'nn_conc_bins':ds_output['nn_conc_bins'],
                                                       'nn_rad_bins':ds_output['nn_rad_bins']},
                                                attrs={
                                                    'description':'NN estimated PDF outputs for example cases with ensemble input variation for better comparison to histograms with finite input bins',
                                                    'ensemble_width':ensemble_width,
                                                    'ensemble_size':ensemble_size,
                                                }
                                               )
                                            

    h_pdf_lst = []
    for sidx in range(idx_arr.size):
        idx = idx_arr[sidx]
        
        # store the histogram version of the PDF
        # Example: We want the 2D output distribution for input bin [5, 2, 8, 1]
        # target_input_bin = np.array([5, 5, 8, 1, 4])
        target_input_bin = np.clip(np.floor(in_data[idx,:] * hist_in_pt).astype(int), 0, hist_in_pt - 1)
        
        # Create a boolean mask where the input columns match our target
        mask = np.all(unique_combinations[:, :D_in] == target_input_bin, axis=1)
        
        # Filter the arrays using the mask
        output_coords = unique_combinations[mask, D_in:]
        output_counts = counts[mask]
        
        # Reconstruct the 2D histogram for this specific input
        # Initialize an empty 2D grid
        hist_2d = np.zeros((hist_out_pt, hist_out_pt))
        
        # Populate the grid with the counts we found
        if len(output_counts) > 0:
            hist_2d[output_coords[:, 0], output_coords[:, 1]] = output_counts
            hist_2d = hist_2d/np.sum(output_counts)/(1.0/hist_out_pt)**2

        h_pdf_lst.append(hist_2d[np.newaxis,...])
    h_pdf_arr_lst.append(np.concatenate(h_pdf_lst,axis=0))

    x_h_plt = np.arange(model_opt_hist_dct[model_str]['output_bins'])/model_opt_hist_dct[model_str]['output_bins']
    hist_ax = np.exp(y_rescale_lst[m_idx].unscale(np.stack([x_h_plt,x_h_plt],axis=1)))

    
    ds_output['hist_conc_bins'] = xr.DataArray(hist_ax[:,0],dims=('hist_conc_bins',),
                                                 attrs = {
                                                     'description':'concentration axis of histogram output examples',
                                                     'units':'#/m^3',
                                                 }
                                                )
    ds_output['hist_rad_bins'] = xr.DataArray(hist_ax[:,1],dims=('hist_rad_bins',),
                                             attrs = {
                                                 'description':'effective radius axis of histogram output examples',
                                                 'units':'m',
                                             }
                                            )
    ds_output['hist_pdf_examples'] = xr.DataArray(h_pdf_arr_lst[-1],
                                                dims=('example_idx','hist_conc_bins','hist_rad_bins',),
                                                coords={'example_idx':ds_output['example_idx'],
                                                       'hist_conc_bins':ds_output['hist_conc_bins'],
                                                       'hist_rad_bins':ds_output['hist_rad_bins']},
                                                attrs={
                                                    'description':'Histogram estimated PDF outputs for example cases',
                                                }
                                               )
    if run_example_only:
        save_file_name = model_str+"_Model_Eval_PostAnalysis_Examples.nc"
    else:
        save_file_name = model_str+"_Model_Eval_PostAnalysis.nc"
    print("Saving file "+save_file_name+" to the path")
    print("   "+save_data_path)
    ds_output.to_netcdf(os.path.join(save_data_path,save_file_name))
    print("done")