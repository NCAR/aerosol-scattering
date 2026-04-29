from argparse import ArgumentParser
from pathlib import Path
from echo.src.base_objective import BaseObjective
from torch.cuda.amp import GradScaler

import logging
import shutil
import os
import sys
import yaml

import optuna

import torch
import xarray as xr
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader

 # add path to libraries in this repo
lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__,)),'..','..','library')
if lib_dir not in sys.path:
    sys.path.append(lib_dir)
from pbs import launch_script, launch_script_mpi
from seed import seed_everything
import evid_nn
# import loss as losslib
# import polynomial as p
from trainerlib import Trainer
import data

# def transpose_and_flatten_data_array(da: xr.DataArray,flattened_dim_name:str = None) -> xr.DataArray:
#     """
#     Transposes an xarray DataArray so the first dimension is 'data_index'
#     and all other dimensions are flattened into a single second dimension
#     named 'flattened_dims'.

#     Args:
#         da (xr.DataArray): The input DataArray.

#     Returns:
#         xr.DataArray: The reshaped DataArray with dimensions ('data_index', 'flattened_dims').
#                       The 'flattened_dims' dimension will have a MultiIndex if multiple
#                       dimensions were stacked, preserving original coordinate information.
#     """

#     # --- Handle Scalar DataArray ---
#     # If the DataArray is a scalar (no dimensions), expand it to a (1,1) array
#     # with the desired dimension names.
#     if not da.dims:
#         return da.expand_dims({'data_index': 1, 'flattened_dims': 1})

#     # --- Prepare Dimensions ---
#     original_dims = list(da.dims)
#     data_index_dim_name = 'data_index'
#     if flattened_dim_name is None:
#         flattened_dim_name = 'input_vars'

#     # Step 1: Ensure 'data_index' is a dimension and is the first dimension.
#     if data_index_dim_name not in original_dims:
#         # If 'data_index' is not present, assume the current first dimension
#         # should be renamed to 'data_index'.
#         first_dim_original_name = original_dims[0]
#         da = da.rename({first_dim_original_name: data_index_dim_name})
#         # Update the list of original_dims to reflect the rename
#         original_dims = list(da.dims)
    
#     # Now, 'data_index' is guaranteed to be a dimension in `da`.
#     # Ensure 'data_index' is the very first dimension.
#     if da.dims[0] != data_index_dim_name:
#         # Use .transpose() to move 'data_index' to the first position,
#         # keeping the relative order of other dimensions.
#         da = da.transpose(data_index_dim_name, *[d for d in da.dims if d != data_index_dim_name])

#     # --- Flatten Other Dimensions ---
#     # Identify all dimensions that are NOT 'data_index'. These will be flattened.
#     dims_to_flatten = [d for d in da.dims if d != data_index_dim_name]

#     if not dims_to_flatten:
#         # If there are no other dimensions (e.g., the original array was already
#         # just ('data_index',)), we need to add a 'flattened_dims' dimension
#         # of size 1 to achieve the target 2D structure.
#         # We use axis=1 to add it as the second dimension.
#         return da.expand_dims(flattened_dim_name, axis=1)

#     # Use .stack() to combine the identified dimensions into a single new dimension.
#     # xarray's .stack() method automatically creates a MultiIndex for the new dimension,
#     # which is excellent for preserving the original coordinate information.
#     # The new stacked dimension will be added as the last dimension by default.
#     # Since 'data_index' was already moved to the first position, the final order
#     # will be (data_index, flattened_dims), which is the desired outcome.
#     stacked_da = da.stack({flattened_dim_name: dims_to_flatten})

#     return stacked_da

def trainer(rank, conf, trial=False):
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}") if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.set_device(rank % torch.cuda.device_count())
    if conf['model']['dtype'] == 'float32':
        dtype = torch.float32
    elif conf['model']['dtype'] == 'float64':
        dtype = torch.float64
    else:
        dtype = torch.float

    # Config settings
    seed = 1000 if "seed" not in conf else conf["seed"]
    seed_everything(seed)


    # train_ds = xr.open_dataset(os.path.join(conf['data']['data_path'],conf['data']['data_file']))
    ds = xr.open_dataset(os.path.join(conf['data']['data_path'],conf['data']['data_file']))

    train_batch_size = conf['data']['batch_size']
    valid_batch_size = conf['data']['batch_size']
    test_batch_size = conf['data']['batch_size']

    train_idx = int(np.round(ds.sizes['data_index']*conf['data']['train_fraction']))
    valid_idx = train_idx+int(np.round(ds.sizes['data_index']*conf['data']['valid_fraction']))
    test_idx = train_idx+int(np.round(ds.sizes['data_index']*conf['data']['test_fraction']))

    # ds = train_ds

    # input_lst = []
    # input_str_lst = []
    # for var in conf['data']['wavelength_inputs']:
    #     for wl_idx, wl in enumerate(conf['data']['wavelength_inputs'][var]['wavelength_lst']):
    #         data_idx_dct = {'wavelength':wl,}
    #         if 'real_index_lst' in conf['data']['wavelength_inputs'][var]:
    #             data_idx_dct['real_index_refraction'] = conf['data']['wavelength_inputs'][var]['real_index_lst'][wl_idx]
    #         if 'imag_index_lst' in conf['data']['wavelength_inputs'][var]:
    #             data_idx_dct['imag_index_refraction'] = conf['data']['wavelength_inputs'][var]['imag_index_lst'][wl_idx]
    #         # data_idx_dct['method'] = 'nearest'
    #         # print(data_idx_dct)
    #         input_lst.append(transpose_and_flatten_data_array(ds[var].sel(indexers=data_idx_dct)))
    #         # input_lst.append(transpose_and_flatten_data_array(ds[var].sel(**data_idx_dct)))
    #         input_str_lst.append(var+f"_{int(wl*1e9)}")
    # for var in conf['data']['input_cols']:
    #     input_lst.append(transpose_and_flatten_data_array(ds[var],flattened_dim_name='input_vars'))
    #     input_str_lst.append(var)

    # # train_input_arr = xr.concat(input_lst,'input_vars')
    # train_input_arr = xr.concat(input_lst,pd.Index(input_str_lst,name='input_vars'))

    # output_lst = []
    # output_str_lst = []
    # for var in conf['data']['output_cols']:
    #     output_lst.append(transpose_and_flatten_data_array(ds[var],flattened_dim_name='label_vars'))
    #     output_str_lst.append(var)

    # train_label_arr = xr.concat(output_lst,pd.Index(conf['data']['output_cols'],name='label_vars'))

    train_input_arr, train_label_arr, input_str_lst, output_str_lst, input_frac_uncertainty = data.build_training_array(ds,conf)

    x_scaler = evid_nn.rescale(0,1,axes=(0,))
    y_scaler = evid_nn.rescale(0,1,axes=(0,))

    x_scaler.set_scale(np.log(train_input_arr.values))
    y_scaler.set_scale(np.log(train_label_arr.values))

    # TODO set the noise scale for the inputs and store it in the config
    training_input_noise = np.log10(1+np.array(input_frac_uncertainty)[np.newaxis,:])/x_scaler.gain 
    # print(training_input_noise)
    # print(training_input_noise.shape)
    # print(training_input_noise)
    # if np.sum(np.isnan(training_input_noise)) > 0:
    #     print("Found Nan in noise:")
    #     print(training_input_noise)
    #     print(input_frac_uncertainty)
    #     raise ValueError
    conf['data']['training_input_noise'] = torch.tensor(training_input_noise,dtype=dtype,device=device)

    # x_train = x_scaler.scale(np.log(train_input_arr.values[:train_idx,:]))
    # x_valid = x_scaler.scale(np.log(train_input_arr.values[train_idx:valid_idx,:]))
    # x_test = x_scaler.scale(np.log(train_input_arr.values[valid_idx:,:]))


    # y_train = y_scaler.scale(np.log(train_label_arr.values[:train_idx,:]))
    # y_valid = y_scaler.scale(np.log(train_label_arr.values[train_idx:valid_idx,:]))
    # y_test = y_scaler.scale(np.log(train_label_arr.values[valid_idx:,:]))

    try:
        valid_ds = xr.open_dataset(os.path.join(conf['data']['valid_data_path'],conf['data']['valid_data_file']))
        # input_lst = []
        # # input_str_lst = []
        # for var in conf['data']['wavelength_inputs']:
        #     for wl_idx, wl in enumerate(conf['data']['wavelength_inputs'][var]['wavelength_lst']):
        #         data_idx_dct = {'wavelength':wl}
        #         if 'real_index_lst' in conf['data']['wavelength_inputs'][var]:
        #             data_idx_dct['real_index_refraction'] = conf['data']['wavelength_inputs'][var]['real_index_lst'][wl_idx]
        #         if 'imag_index_lst' in conf['data']['wavelength_inputs'][var]:
        #             data_idx_dct['imag_index_refraction'] = conf['data']['wavelength_inputs'][var]['imag_index_lst'][wl_idx]
        #         # print(data_idx_dct)
        #         # data_idx_dct['method'] = 'nearest'
        #         input_lst.append(transpose_and_flatten_data_array(valid_ds[var].sel(indexers=data_idx_dct)))
        #         # input_lst.append(transpose_and_flatten_data_array(valid_ds[var].sel(**data_idx_dct)))
        #         # input_str_lst.append(var+f"_{int(wl*1e9)}")
        # for var in conf['data']['input_cols']:
        #     input_lst.append(transpose_and_flatten_data_array(valid_ds[var],flattened_dim_name='input_vars'))
        #     # input_str_lst.append(var)

        # # train_input_arr = xr.concat(input_lst,'input_vars')
        # valid_input_arr = xr.concat(input_lst,pd.Index(input_str_lst,name='input_vars'))

        # output_lst = []
        # # output_str_lst = []
        # for var in conf['data']['output_cols']:
        #     output_lst.append(transpose_and_flatten_data_array(valid_ds[var],flattened_dim_name='label_vars'))
        #     # output_str_lst.append(var)

        # valid_label_arr = xr.concat(output_lst,pd.Index(conf['data']['output_cols'],name='label_vars'))
        valid_input_arr, valid_label_arr, _, _, _ = data.build_training_array(valid_ds,conf)
        x_train = x_scaler.scale(np.log(train_input_arr.values))
        x_valid = x_scaler.scale(np.log(valid_input_arr.values))
        
        y_train = y_scaler.scale(np.log(train_label_arr.values))
        y_valid = y_scaler.scale(np.log(valid_label_arr.values))
    except Exception as E:
        logging.warning(f"validation dataset encountered an error: {str(E)}")
        logging.warning("Splitting the training data to obtain validation data")
        x_train = x_scaler.scale(np.log(train_input_arr.values[:train_idx,:]))
        x_valid = x_scaler.scale(np.log(train_input_arr.values[train_idx:valid_idx,:]))
        
        y_train = y_scaler.scale(np.log(train_label_arr.values[:train_idx,:]))
        y_valid = y_scaler.scale(np.log(train_label_arr.values[train_idx:valid_idx,:]))

    # x_scaler = evid_nn.rescale(0,1,axes=(0,))
    # y_scaler = evid_nn.rescale(0,1,axes=(0,))

    # x_scaler.set_scale(np.log(train_input_arr.values))
    # y_scaler.set_scale(np.log(train_label_arr.values))

    # x_train = x_scaler.scale(np.log(train_input_arr.values[:train_idx,:]))
    # x_valid = x_scaler.scale(np.log(train_input_arr.values[train_idx:valid_idx,:]))
    # x_test = x_scaler.scale(np.log(train_input_arr.values[valid_idx:,:]))


    # y_train = y_scaler.scale(np.log(train_label_arr.values[:train_idx,:]))
    # y_valid = y_scaler.scale(np.log(train_label_arr.values[train_idx:valid_idx,:]))
    # y_test = y_scaler.scale(np.log(train_label_arr.values[valid_idx:,:]))

    # cond_args = {
    #     'in_norm_arr':1,
    #     'in_bias_arr':0,
    #     'out_norm_arr':1,
    #     'out_bias_arr':0,
    # }

    cond_args = conf['data']['cond_args']

    train_dataset = evid_nn.EvidDataset(x_train,y_train,
                                dtype=dtype,device=device,
                                **cond_args)
    valid_dataset = evid_nn.EvidDataset(x_valid,y_valid,
                                dtype=dtype,device=device,
                                **cond_args)
    # test_dataset = evid_nn.EvidDataset(x_test,y_test,
    #                             dtype=dtype,device=device,
    #                             **cond_args)
    
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    # build the model
    # layer_lst = conf['model']['layer_lst'] # [128,512,512,]# [512,512,]
    layer_lst = [conf['model']['layer_nodes'],]*conf['model']['layer_count']
    polynomial_order = [conf['model']['polynomial_order_1'], conf['model']['polynomial_order_2']]

    # note the factor of 2 on output for evidential output
    model = evid_nn.PolyPDF_dense_net(input_channels=x_train.shape[1],
                                    layer_lst=layer_lst,
                                    output_channels=y_train.shape[1],
                                    polynomial_order=polynomial_order,
                                    dtype=dtype,device=device,
                                    int_count=conf['model']['int_count'])

    # learning_rate = conf['trainer']['learning_rate'] # 1e-4 trainer:learning_rate
    max_epochs = conf['trainer']['max_epochs'] # 1500

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=conf['trainer']['learning_rate'], amsgrad=False)
    scaler = GradScaler(enabled=conf['trainer']['amp'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                    factor=conf['trainer']['scheduler']['factor'], 
                    patience=conf['trainer']['scheduler']['patience'], 
                    threshold=conf['trainer']['scheduler']['threshold'], 
                    threshold_mode='abs',min_lr=conf['trainer']['scheduler']['min_lr'])
    
    # Initialize a trainer object

    trainer = Trainer(model, rank, module=False)

    # Fit the model

    result = trainer.fit(
        conf,
        train_dataloader,
        valid_dataloader,
        optimizer,
        # train_criterion,
        # valid_criterion,
        scaler,
        scheduler,
        # metrics,
        trial=trial
    )

    return result

    # model.train()


    # loss_dct = {'train_loss':[],'vld_loss':[],}
    # for idx in range(max_epochs):
    #     (in_data,label_data) = next(iter(train_dataloader))
        
    #     optimizer.zero_grad()
        
    #     est = model(in_data,label_data)
        
    #     total_loss = est # loss_fnc(est,label_data,reg_coef)
        
        
    #     loss = total_loss
    #     loss_dct['train_loss'].append(total_loss.item())

        
    #     loss.backward()

    #     optimizer.step()
    #     scheduler.step(loss)
        
    #     with torch.no_grad():
    #         (in_data,label_data) = next(iter(valid_dataloader))
    #         est = model(in_data,label_data)

    #         total_loss = est # val_loss_fnc(est,label_data)


    #         loss_dct['vld_loss'].append(total_loss.item())

    #         scheduler.step(total_loss)
        
        
        
    #     if (idx+1) % 50 == 0:
    #         print('%d iterations' % (idx+1), end=" ")
    #         print('loss: %.3e'%loss_dct['train_loss'][-1], end=", ")
    #         print('validation loss: %.3e'%loss_dct['vld_loss'][-1])

class Objective(BaseObjective):
    def __init__(self, config, metric="val_loss", device="cpu"):

        # Initialize the base class
        BaseObjective.__init__(self, config, metric, device)

    def train(self, trial, conf):
        try:
            return trainer(0, conf, trial=trial)

        except Exception as E:
            if "CUDA" in str(E):
                logging.warning(
                    f"Pruning trial {trial.number} due to CUDA memory overflow: {str(E)}."
                )
                # raise E
                raise optuna.TrialPruned()
            elif "list index out of range" in str(E).lower():
                logging.warning(
                    f"Pruning trial {trial.number} due to list index error: {str(E)}."
                )
                raise optuna.TrialPruned()
            # elif "Groups" in str(E):
            #     logging.warning(
            #         f"Pruning trial {trial.number} due to groups error: {str(E)}."
            #     )
            #     raise optuna.TrialPruned()
            else:
                logging.warning(f"Trial {trial.number} failed due to error: {str(E)}.")
                raise E

if __name__ == "__main__":

    description = "Train Polynomial PDF fitting NN to estimate aerosol properties from lidar data"
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-c",
        dest="model_config",
        type=str,
        default=False,
        help="Path to the model configuration (yml) containing your inputs.",
    )
    parser.add_argument(
        "-l",
        dest="launch",
        type=int,
        default=0,
        help="Submit workers to PBS.",
    )
    parser.add_argument(
        "-w",
        "--world-size",
        type=int,
        default=4,
        help="Number of processes (world size) for multiprocessing"
    )
    args = parser.parse_args()
    args_dict = vars(args)
    config = args_dict.pop("model_config")
    launch = int(args_dict.pop("launch"))

    # Set up logger to print stuff
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

    # Stream output to stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # Load the configuration and get the relevant variables
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    # Create directories if they do not exist and copy yml file
    os.makedirs(conf["save_loc"], exist_ok=True)
    if not os.path.exists(os.path.join(conf["save_loc"], "model.yml")):
        shutil.copy(config, os.path.join(conf["save_loc"], "model.yml"))

    # Launch PBS jobs
    if launch:
        # Where does this script live?
        script_path = Path(__file__).absolute()
        if conf['pbs']['queue'] == 'casper':
            logging.info("Launching to PBS on Casper")
            launch_script(config, script_path)
        else:
            logging.info("Launching to PBS on Derecho")
            launch_script_mpi(config, script_path)
        sys.exit()

#     wandb.init(
#         # set the wandb project where this run will be logged
#         project="Derecho parallelism",
#         name=f"Worker {os.environ["RANK"]} {os.environ["WORLD_SIZE"]}"
#         # track hyperparameters and run metadata
#         config=conf
#     )

    seed = 1000 if "seed" not in conf else conf["seed"]
    seed_everything(seed)

    # if conf["trainer"]["mode"] in ["fsdp", "ddp"]:
    #     trainer(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), conf)
    # else:
    trainer(0,conf)