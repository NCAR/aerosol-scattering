import numpy as np
import torch
# import logging
import torch.nn as nn
# import torch.nn.functional as F
from torch.utils.data import Dataset

import datetime
import yaml
import os
import copy

class rescale:
    def __init__(self,min_value,max_value,axes=None,nobias=False):
        self.min_value = min_value
        self.max_value = max_value
        self.gain = 1
        self.bias = 0
        self.nobias = nobias
        if axes is None:
            axes = (0,1)
        self.axes = axes
    def set_scale(self,in_arr):
        min_val = np.min(in_arr,axis=self.axes,keepdims=True)
        max_val = np.max(in_arr,axis=self.axes,keepdims=True)
        if self.nobias:
            self.gain = (max_val-min_val)/(self.max_value - self.min_value)
            self.bias = 0
        else:
            self.gain = (max_val-min_val)/(self.max_value - self.min_value)
            self.bias = 0.5*(min_val+max_val)-0.5*(self.min_value+self.max_value)*self.gain

    def set_config(self,**config_kwargs):
        if 'min_value' in config_kwargs:
            self.min_value = config_kwargs['min_value']
        if 'max_value' in config_kwargs:
            self.max_value = config_kwargs['max_value']
        if 'bias' in config_kwargs:
            self.bias = np.array(config_kwargs['bias'])
        if 'gain' in config_kwargs:
            self.gain = np.array(config_kwargs['gain'])
        if 'nobias' in config_kwargs:
            self.nobias = config_kwargs['nobias']

    def scale(self,in_arr):
        return (in_arr - self.bias)/self.gain

    def unscale(self,in_arr):
        return in_arr*self.gain + self.bias

def shuffle_data(x_arr,y_arr):
    arr_idx = np.arange(x_arr.shape[1])
    arr_idx = np.random.shuffle(arr_idx)
    return x_arr[arr_idx,...], y_arr[arr_idx,...]

class EvidDataset(Dataset):
    def __init__(self, in_data_arr, out_data_arr,
                in_norm_arr=1, out_norm_arr=1,
                in_bias_arr = 0, out_bias_arr = 0,
                dtype=None,device=None):
        """
        a temperature dataset for use in training temperature prediction NN
        """

        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cpu")

        if dtype is not None:
            self.dtype = dtype
        else:
            self.dtype = torch.float

        self.in_data_arr = torch.tensor((in_data_arr-in_bias_arr)/in_norm_arr,dtype=dtype,device=self.device)
        self.out_data_arr = torch.tensor((out_data_arr-out_bias_arr)/out_norm_arr,dtype=dtype,device=self.device).reshape(out_data_arr.shape[0],-1)
        
        self.in_bias_arr = in_bias_arr
        self.out_bias_arr = out_bias_arr
        
        self.in_norm_arr = in_norm_arr
        self.out_norm_arr = out_norm_arr
        

    def __len__(self):
        return self.in_data_arr.shape[0]

    def __getitem__(self, idx):
        
        in_data = self.in_data_arr[idx,...]
        label_data = self.out_data_arr[idx,...]
        
        return in_data, label_data


class Evid_dense_net(nn.Module):
    def __init__(self,
                 input_channels=1,
                 layer_lst=None,
                 output_channels=1,
                 output_activation=None,
                 activation = None,
                 device=None,
                 dtype=None,
                 output_bias_channel_cnt=0,
                 ):
        """
        input_channels: int
            number of input channels
        layer_lst: List[int]
            list of the number of dense nodes at each layer
            this effectively sets the number of layers as 
            1 + len(layer_lst) where the additional layer is the
            output layer
        output_channels: int
            number of outputs channels
        output_activation: func
            activation function for the output layer
        activation: func
            activation function for all non-output layers
        device: torch.device
            device the model is run on
        dtype: torch.dtype
            data type used in the model
        output_bias_channel_cnt: int
            sets the number of additional output channels beyond
            what the NN outputs.  This creates parameters for these
            channels which can be updated in the training process.
            The concept is that the mean (and associated epistemic priors)
            might just be a constant across the training set, not inferred
            for each profile.

        """

        super(Evid_dense_net, self).__init__()
        
        if layer_lst is None:
            layer_lst = [128,512,128]
        if output_activation is None:
            output_activation = nn.Identity()
        if activation is None:
            activation = nn.ReLU()

        self.input_channels = input_channels
        self.num_layers = len(layer_lst)
        self.layer_lst = layer_lst
        self.in_layer_lst = [input_channels] + layer_lst
        self.out_layer_lst = layer_lst + [output_channels]
        self.output_channels=output_channels
        self.activation = activation
        self.output_activation = output_activation
        self.output_bias_channel_cnt = output_bias_channel_cnt

        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cpu")

        if dtype is not None:
            self.dtype = dtype
        else:
            self.dtype = torch.float
        
        
        # setup dense net
        dense_lst = []
        for layer_idx,out_features in enumerate(self.out_layer_lst):
            in_features = self.in_layer_lst[layer_idx]
            layer_def = nn.Linear(in_features, out_features,dtype=self.dtype,device=self.device)
            dense_lst.append(layer_def)
            if layer_idx < self.num_layers - 1:
                dense_lst.append(self.activation)
            else:
                dense_lst.append(self.output_activation)
            
            
        self.net = nn.Sequential(*dense_lst)

        if output_bias_channel_cnt > 0:
            self.output_bias = torch.nn.Parameter(0+torch.zeros((1,self.output_channels*self.output_bias_channel_cnt),dtype=self.dtype,device=self.device))
            self.forward_fnc = self.forward_bias
        else:
            self.forward_fnc = self.forward_nobias
        
    def forward_nobias(self,input_data):
        return self.net(input_data)
    
    def forward_bias(self,input_data):
        net_out = self.net(input_data)
        const_out = self.output_bias.repeat(net_out.shape[0],1)
        return torch.cat([net_out,const_out],dim=-1)
    
    def forward(self,input_data):
        return self.forward_fnc(input_data)
    

class PolyPDF_dense_net(nn.Module):
    def __init__(self,
                 input_channels=1,
                 layer_lst=None,
                 output_channels=1,
                 polynomial_order=None,
                 output_activation=None,
                 activation = None,
                 device=None,
                 dtype=None,
                 int_count=1000,
                 ):
        """
        input_channels: int
            number of input channels
        layer_lst: List[int]
            list of the number of dense nodes at each layer
            this effectively sets the number of layers as 
            1 + len(layer_lst) where the additional layer is the
            output layer
        output_channels: int
            number of outputs channels
        polynomial_order: List[int]
            list of PDF polynomial order for each output channel
        output_activation: func
            activation function for the output layer
        activation: func
            activation function for all non-output layers
        device: torch.device
            device the model is run on
        dtype: torch.dtype
            data type used in the model
        int_count: int
            number of points to use in integral calculation

        """

        import polynomial as p

        super(PolyPDF_dense_net, self).__init__()
        
        if layer_lst is None:
            layer_lst = [128,512,128]
        if output_activation is None:
            output_activation = nn.Identity()
        if activation is None:
            activation = nn.ReLU()
        if polynomial_order is None:
            polynomial_order = output_channels*[5,]

        

        self.input_channels = input_channels
        self.num_layers = len(layer_lst)
        self.layer_lst = layer_lst
        self.in_layer_lst = [input_channels] + layer_lst
        # self.out_layer_lst = layer_lst + [output_channels]
        self.output_channels=output_channels
        self.activation = activation
        self.output_activation = output_activation

        self.int_count = int_count

        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cpu")

        if dtype is not None:
            self.dtype = dtype
        else:
            self.dtype = torch.float
        
        self.order1 = polynomial_order[0]
        self.order2 = polynomial_order[1]
        
        self.pow1 = torch.arange(0,self.order1+1,device=device,dtype=dtype).reshape(1,-1)
        self.pow2 = torch.arange(0,self.order2+1,device=device,dtype=dtype).reshape(1,-1)
        self.chmat1 = p.chebyshev_poly(self.order1,device=device,dtype=dtype)
        self.chmat2 = p.chebyshev_poly(self.order2,device=device,dtype=dtype)

        self.x_int = torch.linspace(0,1,self.int_count,dtype=dtype,device=device) # integration axis in 1 dimension
        self.dx_int = torch.mean(torch.diff(self.x_int))
        self.x_mesh_int1, self.x_mesh_int2 = torch.meshgrid(self.x_int,self.x_int,indexing='ij')
        self.x_mesh_int = torch.stack([self.x_mesh_int1.flatten(), self.x_mesh_int2.flatten()],axis=1)
        self.x_mesh_ch_poly = self.ch_poly_vector(self.x_mesh_int)

        self.out_layer_lst = layer_lst + [self.x_mesh_ch_poly.shape[1]]

        
        # setup dense net
        dense_lst = []
        for layer_idx,out_features in enumerate(self.out_layer_lst):
            in_features = self.in_layer_lst[layer_idx]
            layer_def = nn.Linear(in_features, out_features,dtype=self.dtype,device=self.device)
            dense_lst.append(layer_def)
            if layer_idx < self.num_layers - 1:
                dense_lst.append(self.activation)
            else:
                dense_lst.append(self.output_activation)
            
            
        self.net = nn.Sequential(*dense_lst)
        
    def fpoly(self,fit_coef,x):
        """
        calculate the polynomial function for
        fit_coef and array of values of x
        """
        xc = self.ch_poly_vector(x)
        
        return torch.sum(fit_coef*xc,dim=1)
    
    def ch_poly_vector(self,x):
        """
        calculates the chebyshev polynomial vector
        for a set of x values
        """

        x1 = x[:,0].reshape(-1,1)
        x2 = x[:,1].reshape(-1,1)

        x1p = x1**self.pow1
        x2p = x2**self.pow2

        x1c = x1p @ self.chmat1
        x2c = x2p @ self.chmat2

        xc = torch.einsum('bi,bj->bij', x1c, x2c).view(x.shape[0], -1)
        return xc
    
    def fpdf(self,fit_coef,x):
        fpdf0 = torch.exp(self.fpoly(fit_coef,x))
        fpdf_int = self.fpdf_integral(fit_coef)
        return fpdf0/fpdf_int
    
    def fpdf_nll(self,fit_coef,x):
        return torch.log(self.fpdf_integral(fit_coef)) - self.fpoly(fit_coef,x)
    
    def output_pdf(self,input_data):
        fit_coef = self.net(input_data)
        f_pdf_0 = torch.exp(fit_coef @ self.x_mesh_ch_poly.T)
        f_pdf_tnsr = f_pdf_0.reshape(fit_coef.shape[0],self.x_int.shape[0],self.x_int.shape[0])
        f_pdf_int = self.fpdf_integral(fit_coef).reshape(-1,1,1)
        return f_pdf_tnsr/f_pdf_int

    def fpdf_integral(self,fit_coef):
        """
        numerically estimate the integral of the
        exp-polynomial fitting function
        for batch dimension B, polynomial dimension N (includes cross terms) and mesh dimension M
        fit_coef is (B x N)
        x_mesh_int is (N x M)
        """
        
        y_integral = torch.sum(torch.exp(fit_coef @ self.x_mesh_ch_poly.T),dim=1)*self.dx_int*self.dx_int

        # TODO check if trapazoidal integration works better
        # y_int_1 = torch.trapz(torch.exp(fit_coef @ self.x_mesh_ch_poly.T).reshape(-1,self.int_count,self.int_count),x=self.x_int,dim=1)
        # y_integral = torch.trapz(y_int_1,x=self.x_int,dim=2)

        return y_integral
    
    def forward_coef(self,input_data):
        """
        this only calculates the fit coefficients
        """
        fit_coef = self.net(input_data)

        return fit_coef
    
    def forward(self,input_data,y):
        """
        use this to calculate the loss
        by calculating the polynomial
        for explicit values of y.
        It is assumed that the y values
        correspond with the input_data
        """
        fit_coef = self.net(input_data)

        # loss = -torch.log(self.fpdf(fit_coef,y))
        loss = self.fpdf_nll(fit_coef,y)

        return torch.sum(loss)
    

def save_poly_nn_model(model,path=None,
                          x_scaler=None,
                          y_scaler=None,
                          conf=None,
                          input_str_lst=None,
                          output_str_lst=None,
                          initial_model="None",
                          name_str="",
                          ):
    """
    saves the model state for 
    polynomial fitting dense net.  The state is saved
    to a .pt file while the model config
    is saved to a yaml
    """
    time_str = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    
    save_yaml_file = "_".join([name_str,time_str])+"_poly_nn_model_config.yaml"
    save_model_file = "_".join([name_str,time_str])+"_model_state.pt"
    
    if path is None:
        path = os.path.abspath(__file__+'/../../models/')
    save_dct = copy.deepcopy(conf)
    save_dct['save_model'] = {}
    save_dct['save_model']['output_channels'] = model.output_channels
    save_dct['save_model']['input_channels'] = model.input_channels
    save_dct['layer_lst'] = model.layer_lst
    save_dct['save_model']['initial_model'] = initial_model  # the pre-trained model version used to train this one

    if input_str_lst is not None:
        save_dct['save_model']['input_str_lst'] = input_str_lst

    if output_str_lst is not None:
        save_dct['save_model']['output_str_lst'] = output_str_lst

    if 'training_input_noise' in conf['data']:
        if isinstance(conf['data']['training_input_noise'],np.ndarray) or torch.is_tensor(conf['data']['training_input_noise']):
            save_dct['data']['training_input_noise'] = conf['data']['training_input_noise'].tolist()


    save_dct['save_model']['x_scaler'] = {}
    if x_scaler is None:
        save_dct['save_model']['x_scaler']['gain'] = [1,]
        save_dct['save_model']['x_scaler']['bias'] = [0,]
        save_dct['save_model']['x_scaler']['min_value'] = 0
        save_dct['save_model']['x_scaler']['max_value'] = 1
        save_dct['save_model']['x_scaler']['nobias'] = False
    else:
        if isinstance(x_scaler.gain,np.ndarray):
            save_dct['save_model']['x_scaler']['gain'] = x_scaler.gain.tolist()
        else:
            save_dct['save_model']['x_scaler']['gain'] = [x_scaler.gain,]
        if isinstance(x_scaler.bias,np.ndarray):
            save_dct['save_model']['x_scaler']['bias'] = x_scaler.bias.tolist()
        else:
            save_dct['save_model']['x_scaler']['bias'] = [x_scaler.bias,]

        save_dct['save_model']['x_scaler']['min_value'] = x_scaler.min_value
        save_dct['save_model']['x_scaler']['max_value'] = x_scaler.max_value
        save_dct['save_model']['x_scaler']['nobias'] = x_scaler.nobias

    save_dct['save_model']['y_scaler'] = {}
    if y_scaler is None:
        save_dct['save_model']['y_scaler']['gain'] = [1,]
        save_dct['save_model']['y_scaler']['bias'] = [0,]
        save_dct['save_model']['y_scaler']['min_value'] = 0
        save_dct['save_model']['y_scaler']['max_value'] = 1
        save_dct['save_model']['y_scaler']['nobias'] = False
    else:
        if isinstance(y_scaler.gain,np.ndarray):
            save_dct['save_model']['y_scaler']['gain'] = y_scaler.gain.tolist()
        else:
            save_dct['save_model']['y_scaler']['gain'] = [y_scaler.gain,]
        if isinstance(y_scaler.bias,np.ndarray):
            save_dct['save_model']['y_scaler']['bias'] = y_scaler.bias.tolist()  # this should always be zero
        else:
            save_dct['save_model']['y_scaler']['bias'] = [y_scaler.bias,]
        save_dct['save_model']['y_scaler']['min_value'] = y_scaler.min_value
        save_dct['save_model']['y_scaler']['max_value'] = y_scaler.max_value
        save_dct['save_model']['y_scaler']['nobias'] = y_scaler.nobias
    
    
    # save configuration information
    with open(os.path.join(path,save_yaml_file), 'w') as outfile:
        yaml.dump(save_dct, outfile, sort_keys=False)
        
    print("saving "+save_model_file+" to")
    print(path)
    # save the model state
    torch.save(model.state_dict(), os.path.join(path,save_model_file))
    print("done")

def load_poly_nn_model(time_str,name_str="",path=None,dtype=None,device=None):
    
    if path is None:
        path = os.path.abspath(__file__+'/../../models/')

    if device is None:
        device = torch.device("cpu")

    print("loading model from ")
    print(path)

    save_yaml_file = "_".join([name_str,time_str])+"_poly_nn_model_config.yaml"
    save_model_file = "_".join([name_str,time_str])+"_model_state.pt"
    
    save_dct = {}
    with open(os.path.join(path,save_yaml_file), "r") as r:
        save_dct = yaml.safe_load(r)

    if dtype is None:
        if save_dct['model']['dtype'] == 'float64':
            dtype = torch.float64
        elif save_dct['model']['dtype'] == 'float32':
            dtype = torch.float32
        else:
            dtype = torch.float
        
    layer_lst = [save_dct['model']['layer_nodes'],]*save_dct['model']['layer_count']
    polynomial_order = [save_dct['model']['polynomial_order_1'], save_dct['model']['polynomial_order_2']]
    # I have skipped dealing with activation functions for now
    # TODO activations need to be implemented
    model = PolyPDF_dense_net(input_channels=save_dct['save_model']['input_channels'],
                                layer_lst=layer_lst,
                                output_channels=save_dct['save_model']['output_channels'],
                                polynomial_order=polynomial_order,
                                dtype=dtype,device=device,
                                int_count=save_dct['model']['int_count'])
    
    model.load_state_dict(torch.load(os.path.join(path,save_model_file),map_location=device))
    model.eval()
    return save_dct, model