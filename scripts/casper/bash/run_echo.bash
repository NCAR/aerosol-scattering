#!/usr/bin/env bash

module load conda
conda activate holodec

cd $HOME/Python/aerosol-scattering/

# echo-run scripts/casper/echo_config/cached_hyper.yml scripts/casper/echo_config/polynn_model_config.yml # run one instance from a gpu node
echo-opt scripts/casper/echo_config/cached_hyper.yml scripts/casper/echo_config/polynn_model_config.yml # run full optimization

# get report and best model
# echo-report scripts/casper/echo_config/cached_hyper.yml -m scripts/casper/echo_config/polynn_model_config.yml 
# get report and best model in the run output directory
# module load conda
# conda activate holodec
# echo-report cached_hyper.yml -m polynn_model_config.yml 

# cp best.yml ~/Python/aerosol-scattering/config/echo/best_beta1_alpha1_532_r_eff_all_index.yml