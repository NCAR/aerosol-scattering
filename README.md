# aerosol-scattering
Analysis of aerosol scattering characteristics and how they project onto Lidar data

# Hyper parameter optimize the model
To run echo-opt on casper use scripts in
`sh scripts/casper/bash/run_echo.bash`
Make sure config files for both the model and hyper parameter have the same save directories.
These config files are in the `run_echo.bash` script and are typically
```
polynn_model_config.yml
```
and
```
cached_hyper.yml
```
After the optimization is finished, I copy the best hyper parameter config from the echo save directory to this repo using something like this:
```
cp best.yml ~/Python/aerosol-scattering/config/echo/best_beta1_alpha1_532_r_eff_all_index.yml
```
The second file name has a tag that is typically the name of the echo directory.

I then train the model using the notebook and by referencing the `best*.yml` file mentioned above.  Skip all the way to the `Train using Trainer Class` section.  No need to run anything above it.
```
scripts/casper/TrainPolyPDFNN.ipynb
```
make sure to save the result under the `Save the trained model` subheading and make note of the save tag which can be referenced in other evaluation notebooks and python scripts.

# Model results analysis
To analyze data from models, set the desired analysis variables in
```
evaluate_model.py
```
then submit a job using
```
qsub scripts/casper/pbs/run_gpu_model_eval
```


# Transfer to linux share
Transfer post analysis:
```
rsync -avh -e ssh /glade/derecho/scratch/mhayman/aerosol_poly_nn/output_analysis/ mhayman@gurgle.eol.ucar.edu:/scr/tmp/mhayman/aerosol_poly_nn/output_analysis/
```

Tranfer model
```
rsync -avh -e ssh $HOME/Python/aerosol-scattering/models/ mhayman@gurgle.eol.ucar.edu:/scr/tmp/mhayman/aerosol_poly_nn/models/
```

Training data
```
rsync -avh -e ssh /glade/derecho/scratch/mhayman/aerosol/datasets/  mhayman@gurgle.eol.ucar.edu:/scr/tmp/mhayman/aerosol_poly_nn/datasets/
```