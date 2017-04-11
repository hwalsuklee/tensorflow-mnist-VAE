# Variational Auto-Encoder for MNIST
An implementation of variational auto-encoder (VAE) for MNIST descripbed in the paper:  
* [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114) by Kingma et al.

## Results
### Reproduce
Well trained VAE must be able to reproduce input image.  
Figure 5 in the paper shows reproduce performance of learned generative models for different dimensionalities.  
The following results can be reproduced with command:  
```
python run_main.py --dim_z <each value> --num_epochs 60
``` 

<table align='center'>
<tr align='center'>
<td> Input image </td>
<td> 2-D latent space </td>
<td> 5-D latent space </td>
<td> 10-D latent space </td>
<td> 20-D latent space </td>
</tr>
<tr>
<td><img src = 'results/input.jpg' height = '150px'>
<td><img src = 'results/dim_z_2.jpg' height = '150px'>
<td><img src = 'results/dim_z_5.jpg' height = '150px'>
<td><img src = 'results/dim_z_10.jpg' height = '150px'>
<td><img src = 'results/dim_z_20.jpg' height = '150px'>
</tr>
</table>

### Denoising

When training, salt & pepper noise is added to input image, so that VAE can reduce noise and restore original input image.  
The following results can be reproduced with command:  
```
python run_main.py --dim_z 20 --add_noise True --num_epochs 40
```
<table align='center'>
<tr align='center'>
<td> Original input image </td>
<td> Input image with noise </td>
<td> Restored image via VAE </td>
</tr>
<tr>
<td><img src = 'results/input.jpg' height = '300px'>
<td><img src = 'results/input_noise.jpg' height = '300px'>
<td><img src = 'results/denoising.jpg' height = '300px'>
</tr>
</table>

### Learned MNIST manifold
Visualizations of learned data manifold for generative models with 2-dim. latent space are given in Figure. 4 in the paper.  
The following results can be reproduced with command:  
```
python run_main.py --dim_z 2 --num_epochs 60 --PMLR True
```
<table align='center'>
<tr align='center'>
<td> Learned MNIST manifold </td>
<td> Distribution of labeled data  </td>
</tr>
<tr>
<td><img src = 'results/PMLR.jpg' height = '400px'>
<td><img src = 'results/PMLR_map.jpg' height = '400px'>
</tr>
</table>

## Usage
### Prerequisites
1. Tensorflow
2. Python packages : numpy, scipy, PIL(or Pillow), matplotlib

### Command
```
python run_main.py --dim_z <latent vector dimension>
```
*Example*:
`python run_main.py --dim_z 20`

### Arguments
*Required* :  
* `--dim_z`: Dimension of latent vector. *Default*: `20`

*Optional* :  
* `--results_path`: File path of output images. *Default*: `results`
* `--add_noise`: Boolean for adding salt & pepper noise to input image. *Default*: `False`
* `--n_hidden`: Number of hidden units in MLP. *Default*: `500`
* `--learn_rate`: Learning rate for Adam optimizer. *Default*: `1e-3`
* `--num_epochs`: The number of epochs to run. *Default*: `20`
* `--batch_size`: Batch size. *Default*: `128`
* `--PRR`: Boolean for plot-reproduce-result. *Default*: `True`
* `--PRR_n_img_x`: Number of images along x-axis. *Default*: `10`
* `--PRR_n_img_y`: Number of images along y-axis. *Default*: `10`
* `--PRR_resize_factor`: Resize factor for each displayed image. *Default*: `1.0`
* `--PMLR`: Boolean for plot-manifold-learning-result. *Default*: `False`
* `--PMLR_n_img_x`: Number of images along x-axis. *Default*: `20`
* `--PMLR_n_img_y`: Number of images along y-axis. *Default*: `20`
* `--PMLR_resize_factor`: Resize factor for each displayed image. *Default*: `1.0`
* `--PMLR_n_samples`: Number of samples in order to get distribution of labeled data. *Default*: `5000`

## References
The implementation is based on the projects:  
[1] https://github.com/oduerr/dl_tutorial/tree/master/tensorflow/vae  
[2] https://github.com/fastforwardlabs/vae-tf/tree/master  
[3] https://github.com/kvfrans/variational-autoencoder  
[4] https://github.com/altosaar/vae

## Acknowledgements
This implementation has been tested with Tensorflow r0.12 on Windows 10.
