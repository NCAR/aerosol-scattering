import pandas as pd
import xarray as xr

def transpose_and_flatten_data_array(da: xr.DataArray,flattened_dim_name:str = None) -> xr.DataArray:
    """
    Transposes an xarray DataArray so the first dimension is 'data_index'
    and all other dimensions are flattened into a single second dimension
    named 'flattened_dims'.

    Args:
        da (xr.DataArray): The input DataArray.

    Returns:
        xr.DataArray: The reshaped DataArray with dimensions ('data_index', 'flattened_dims').
                      The 'flattened_dims' dimension will have a MultiIndex if multiple
                      dimensions were stacked, preserving original coordinate information.
    """

    # --- Handle Scalar DataArray ---
    # If the DataArray is a scalar (no dimensions), expand it to a (1,1) array
    # with the desired dimension names.
    if not da.dims:
        return da.expand_dims({'data_index': 1, 'flattened_dims': 1})

    # --- Prepare Dimensions ---
    original_dims = list(da.dims)
    data_index_dim_name = 'data_index'
    if flattened_dim_name is None:
        flattened_dim_name = 'input_vars'

    # Step 1: Ensure 'data_index' is a dimension and is the first dimension.
    if data_index_dim_name not in original_dims:
        # If 'data_index' is not present, assume the current first dimension
        # should be renamed to 'data_index'.
        first_dim_original_name = original_dims[0]
        da = da.rename({first_dim_original_name: data_index_dim_name})
        # Update the list of original_dims to reflect the rename
        original_dims = list(da.dims)
    
    # Now, 'data_index' is guaranteed to be a dimension in `da`.
    # Ensure 'data_index' is the very first dimension.
    if da.dims[0] != data_index_dim_name:
        # Use .transpose() to move 'data_index' to the first position,
        # keeping the relative order of other dimensions.
        da = da.transpose(data_index_dim_name, *[d for d in da.dims if d != data_index_dim_name])

    # --- Flatten Other Dimensions ---
    # Identify all dimensions that are NOT 'data_index'. These will be flattened.
    dims_to_flatten = [d for d in da.dims if d != data_index_dim_name]

    if not dims_to_flatten:
        # If there are no other dimensions (e.g., the original array was already
        # just ('data_index',)), we need to add a 'flattened_dims' dimension
        # of size 1 to achieve the target 2D structure.
        # We use axis=1 to add it as the second dimension.
        return da.expand_dims(flattened_dim_name, axis=1)

    # Use .stack() to combine the identified dimensions into a single new dimension.
    # xarray's .stack() method automatically creates a MultiIndex for the new dimension,
    # which is excellent for preserving the original coordinate information.
    # The new stacked dimension will be added as the last dimension by default.
    # Since 'data_index' was already moved to the first position, the final order
    # will be (data_index, flattened_dims), which is the desired outcome.
    stacked_da = da.stack({flattened_dim_name: dims_to_flatten})

    return stacked_da


def build_training_array(ds, conf):
    """
    Inputs:
        ds: xarray dataset
        conf: dct
            configuration file for data processing
    Outputs:
        train_input_arr: xarray data array
            data contains input data obtained from the dataset ds organized with dimensions
            (data_index, input_vars).  The first dimension corresponds to the number of training 
            points and the second dimension corresponds to the number of inputs for the neural net.
        train_label_arr xarray data array
            data contains output labels from the dataset ds organized with dimensions
            (data_index, label_vars).  The second dimension corresponds to the output prediction from the
            neural net.

    This function reads in data from the dataset and organizes it for training a neural net where the
    input and output data arrays have the same first dimension (data_index).  

    Note that if the input variables have multiple dimensions
        when conf['data']['flatten_to_multiple_points'] is a list of dimensions contained in the
            wavelength variables, all instances in those dimensions will be included as separate
            training data points.  If the list is not provided, the specific coordinates must be
            indexed in a list corresponding to each wavelength. 
    """
    
    flatten_dims = conf['data'].get('flatten_to_multiple_points', [])
    is_flattening = len(flatten_dims) > 0
    flatten_stack = ['data_index'] + flatten_dims if is_flattening else []
    def _prepare_and_flatten(da, final_dim_name):
        """
        Helper to broadcast a DataArray to the required target dimensions, 
        stack it, and format it using transpose_and_flatten_data_array.
        """
        if is_flattening:
            # Broadcast the array to ensure it has all dimensions being flattened
            for d in flatten_stack:
                if d not in da.dims:
                    if d in ds.coords:
                        da = da.expand_dims({d: ds[d]})
                    elif d in ds.dims:
                        # Fallback for dimensions without coordinates
                        da = da.expand_dims({d: range(ds.sizes[d])})
                    else:
                        da = da.expand_dims({d: 1})
            
            # Stack into a single multi-dimensional index
            da = da.stack(new_data_index=flatten_stack)
            
            # Reset the MultiIndex to unpack the stacked coordinates
            da = da.reset_index("new_data_index")
            
            # Rename the old 'data_index' coordinate to prevent conflicts
            if 'data_index' in da.coords:
                da = da.rename({"data_index": "original_data_index"})
                
            # Create a new index array and attach it to the 'new_data_index' dimension
            new_idx = range(da.sizes["new_data_index"])
            da = da.assign_coords(data_index=("new_data_index", new_idx))
            
            # Swap the dimensions (this makes 'data_index' the actual dimension)
            da = da.swap_dims({"new_data_index": "data_index"})
            
            # Clean up the leftover 'new_data_index' coordinate
            da = da.drop_vars("new_data_index", errors="ignore")
        
        # Pass through the standard formatter to establish the second dimension
        return transpose_and_flatten_data_array(da, flattened_dim_name=final_dim_name)
    # def _prepare_and_flatten(da, final_dim_name):
    #     """
    #     Helper to broadcast a DataArray to the required target dimensions, 
    #     stack it, and format it using transpose_and_flatten_data_array.
    #     """
    #     if is_flattening:
    #         # Broadcast the array to ensure it has all dimensions being flattened
    #         for d in flatten_stack:
    #             if d not in da.dims:
    #                 if d in ds.coords:
    #                     da = da.expand_dims({d: ds[d]})
    #                 elif d in ds.dims:
    #                     # Fallback for dimensions without coordinates
    #                     da = da.expand_dims({d: range(ds.sizes[d])})
    #                 else:
    #                     da = da.expand_dims({d: 1})
            
    #         # Stack into a single multi-dimensional index
    #         da = da.stack(new_data_index=flatten_stack)
    #         # Reset the MultiIndex to keep standard coordinates and rename main dimension
    #         da = da.reset_index("new_data_index").rename({"new_data_index": "data_index"})
        
    #     # Pass through the standard formatter to establish the second dimension
    #     return transpose_and_flatten_data_array(da, flattened_dim_name=final_dim_name)

    input_lst = []
    input_str_lst = []
    input_uncert_frac_lst = []
    
    # --- Process Wavelength Inputs ---
    for var in conf['data']['wavelength_inputs']:
        for wl_idx, wl in enumerate(conf['data']['wavelength_inputs'][var]['wavelength_lst']):
            data_idx_dct = {'wavelength': wl}
            
            if is_flattening:
                da = ds[var].sel(indexers=data_idx_dct)
                processed_da = _prepare_and_flatten(da, 'input_vars')
                input_lst.append(processed_da)
            else:
                # Only use the index of refraction data as specified for the wavelength
                if 'real_index_lst' in conf['data']['wavelength_inputs'][var]:
                    data_idx_dct['real_index_refraction'] = conf['data']['wavelength_inputs'][var]['real_index_lst'][wl_idx]
                if 'imag_index_lst' in conf['data']['wavelength_inputs'][var]:
                    data_idx_dct['imag_index_refraction'] = conf['data']['wavelength_inputs'][var]['imag_index_lst'][wl_idx]
                
                da = ds[var].sel(indexers=data_idx_dct)
                input_lst.append(transpose_and_flatten_data_array(da, flattened_dim_name='input_vars'))
            
            # load the amount of noise to inject in the training inputs
            if 'train_noise_frac' in conf['data']['wavelength_inputs'][var]:
                input_frac_uncertainty = conf['data']['wavelength_inputs'][var]['train_noise_frac'][wl_idx]
            else:
                input_frac_uncertainty = 0.0
                
            input_str_lst.append(var + f"_{int(wl*1e9)}")
            input_uncert_frac_lst.append(input_frac_uncertainty)
            
    # --- Process Additional Input Columns ---
    for var in conf['data']['input_cols']:
        # This will automatically repeat/broadcast scalar inputs to match flattened dims
        da = ds[var]
        input_lst.append(_prepare_and_flatten(da, 'input_vars'))
        input_str_lst.append(var)

    train_input_arr = xr.concat(input_lst, pd.Index(input_str_lst, name='input_vars'))

    # --- Process Output Columns ---
    output_lst = []
    output_str_lst = []
    for var in conf['data']['output_cols']:
        # This will automatically repeat/broadcast output labels to match flattened dims
        da = ds[var]
        output_lst.append(_prepare_and_flatten(da, 'label_vars'))
        output_str_lst.append(var)

    train_label_arr = xr.concat(output_lst, pd.Index(conf['data']['output_cols'], name='label_vars'))

    return train_input_arr, train_label_arr, input_str_lst, output_str_lst, input_uncert_frac_lst


# def build_training_array(ds,conf):
#     """
#     Inputs:
#         ds: xarray dataset
#         conf: dct
#             configuration file for data processing
#     Outputs:
#         train_input_arr: xarray data array
#             data contains input data obtained from the dataset ds organized with dimensions
#             (data_index, input_vars).  The first dimension corresponds to the number of training 
#             points and the second dimension corresponds to the number of inputs for the neural net.
#         train_label_arr xarray data array
#             data contains output labels from the dataset ds organized with dimensions
#             (data_index, label_vars).  The second dimension corresponds to the output prediction from the
#             neural net.

#     This function reads in data from the dataset and organizes it for training a neural net where the
#     input and output data arrays have the same first dimension (data_index).  

#     Note that if the input variables have multiple dimensions
#         when conf['data']['flatten_to_multiple_points'] is a list of dimensions contained in the
#             wavelength variables, all instances in those dimensions will be included as separate
#             training data points.  If the list is not provided, the specific coordinates must be
#             indexed in a list corresponding to each wavelength. 
#     """

#     input_lst = []
#     input_str_lst = []
#     for var in conf['data']['wavelength_inputs']:
#         for wl_idx, wl in enumerate(conf['data']['wavelength_inputs'][var]['wavelength_lst']):
#             data_idx_dct = {'wavelength':wl,}
#             if len(conf['data'].get('flatten_to_multiple_points',[])) > 0:
#                 # TODO need to make sure other non-wavelength data and output data are repeated
#                 # to ensure the same number of points
#                 flatten_stack = ['data_index']+conf['data']['flatten_to_multiple_points']
#                 # flatten all index of refraction data into separate input/label pairs
#                 input_lst.append(ds[var].sel(indexers=data_idx_dct).stack(new_data_index=flatten_stack).reset_index("new_data_index").rename({"new_data_index":"data_index"}))
#             else:
#                 # only use the index of refraction data as specified for the wavlength
#                 if 'real_index_lst' in conf['data']['wavelength_inputs'][var]:
#                     data_idx_dct['real_index_refraction'] = conf['data']['wavelength_inputs'][var]['real_index_lst'][wl_idx]
#                 if 'imag_index_lst' in conf['data']['wavelength_inputs'][var]:
#                     data_idx_dct['imag_index_refraction'] = conf['data']['wavelength_inputs'][var]['imag_index_lst'][wl_idx]
#                 # data_idx_dct['method'] = 'nearest'
#                 # print(data_idx_dct)
#                 input_lst.append(transpose_and_flatten_data_array(ds[var].sel(indexers=data_idx_dct)))
#             # input_lst.append(transpose_and_flatten_data_array(ds[var].sel(**data_idx_dct)))
#             input_str_lst.append(var+f"_{int(wl*1e9)}")
#     for var in conf['data']['input_cols']:
#         # stack additional input data without wavelength dependence
#         # TODO when flatten_to_multiple_points is provided, need to repeat the inputs
#          # corresponding to the repeated inputs
#         input_lst.append(transpose_and_flatten_data_array(ds[var],flattened_dim_name='input_vars'))
#         input_str_lst.append(var)

#     # train_input_arr = xr.concat(input_lst,'input_vars')
#     train_input_arr = xr.concat(input_lst,pd.Index(input_str_lst,name='input_vars'))

#     # create output data
#     # TODO when flatten_to_multiple_points is provided, need to repeat the labels
#     # corresponding to the repeated inputs
#     output_lst = []
#     output_str_lst = []
#     for var in conf['data']['output_cols']:
#         output_lst.append(transpose_and_flatten_data_array(ds[var],flattened_dim_name='label_vars'))
#         output_str_lst.append(var)

#     train_label_arr = xr.concat(output_lst,pd.Index(conf['data']['output_cols'],name='label_vars'))

#     return train_input_arr, train_label_arr