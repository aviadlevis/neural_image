# neural_image
An example of fitting a [neural image](https://www.matthewtancik.com/nerf) to pixel values written in [Jax](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html). 
The image is a Sagitarrius A* model from [eht-imaging](https://github.com/achael/eht-imaging/tree/main/models).
Usefull for playing around with basic concepts like [Fourier features](https://arxiv.org/pdf/2006.10739.pdf) and network architecture.

Installation
---
Start a conda virtual environment and add channels
```
conda config --add channels conda-forge
conda create -n jax python=3.9 numpy==1.23.1
conda activate jax
```
Install requirements 
```
pip install numpy scipy matplotlib jupyterlab nodejs tqdm ipympl ipyvolume mpmath scikit-image ruamel.yaml
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install flax optax diffrax scikit-learn tensorboardX tensorboard
```

Note: This list was not curated so there might be some unused packages.
