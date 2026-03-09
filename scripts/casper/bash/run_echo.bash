#!/usr/bin/env bash

module load conda
conda activate holodec

cd $HOME/Python/aerosol-scattering/

# echo-run scripts/casper/echo_config/cached_hyper.yml scripts/casper/echo_config/polynn_model_config.yml # run one instance from a gpu node
echo-opt scripts/casper/echo_config/cached_hyper.yml scripts/casper/echo_config/polynn_model_config.yml # run full optimization

# get report and best model
# echo-report scripts/casper/echo_config/cached_hyper.yml -m scripts/casper/echo_config/polynn_model_config.yml 

# cp best.yml ~/Python/aerosol-scattering/config/echo/best_beta3_alpha2_r_eff_13_02.yml