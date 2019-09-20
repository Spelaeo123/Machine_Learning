# A pipeline for applying state of the art machine learning to artefact sourcing


The pipeline applies state of the art machine learning to mass-spectrometry data for sourcing flint artefacts. The pipeline classifies which bedrock site the artefacts were originally sourced from based on the geochemical data. 

The pipeline includes data preproccessing, feature selection, dimensionality reduction, outlier detection, the evaluation of machine learning classifiers and final model building and prediction. 

***
### Notes
* Please set the configuration variables in config.py before running the pipeline, there are descriptions about the paramaters in the config.py file
* create conda environment by running below in terminal

```conda create --name my_env ```

* activate said environment

```conda activate my_env ```

* change directory to one containing requirements files

```cd config_and_dependencies```

* install python libraries unto environment

``` conda install --file requirements_conda.txt```
``` pip install -r requirements_pip.txt

* the notebooks should be run in the order that they are numbered
***

This pipeline was originally used for the study referenced below:

paper ref: