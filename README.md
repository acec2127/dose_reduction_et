# Diffusion model 
## Presentation

This code implement the diffusion model based on the article "Elucidating the Design Space of Diffusion-Based Generative Models" (EDM) (Tero Karras, Miika Aittala, Timo Aila and Samuli Laine, 2022)
Link: https://arxiv.org/abs/2206.00364
It also (and most importantly!) implement a conditional diffusion model - also known as Denoising Diffusion for Posterior Sampling (Jeremy Heng, Valentin De Bortoli and Arnaud Doucet, 2023) - for dose enhancement in PET-scans. Checkout the report in the docs for more details!

The code is subdivided  in the following way :

- **configs** : This folder contains experiment parameters. Every training and sampling run corresponds to a single config yaml file. 
- **datasets** :  This folder stores the datasets. In github I have simply uploaded two npz files names "low_res_phantoms_train.npz" and "low_res_phantoms_infer.npz" which corresponds to phantoms processed from the Brainweb dataset. More details later.
- **utils** : This folder rassembles a collection of utility files.
     - distributed.py : Collection of functions to manage distributed data parallelism.
     - training_stats.py : File with global variables which collects statistics from the different processes and agregate them.
     - torch_utils.py: Pytorch utilities.
     - utils.py: Variate functions to manage I/O files, regex operations, convenience classes, etc...
- **tomography_utils** : This folder contains functions used specifically when dealing with PET data.
    - projector.py : Compute the projection matrix used to obtain sinograms from the phantoms. The matrix is computed once for all phantoms of a certain resolution and is stored in a subfolder "projector_matrices".
    - mlem.py : Implements a batched version of the Maximum Likelihood Estimator Method used to reconstruct images from sinograms.
- **training** : This folder contains alll of the files necessary to train the model. 
    - dataset.py: Imports dataset from the pâth specified in the config file and wrap it into a PyTorch Dataset subclass object.
    - sampler.py: Implements a distributed infinite sampler to pass in the dataloader that loops over the dataset indefinitely, shuffling items as it goes.
    - networks.py: Implementation of a UNet neural networks with PyTorch. The UNet is finally wrapped into a preconditioning neural networks as described in the EDM paper.
    - augment.py : Augmentation pipeline for training. Applies a series of geometrical transformations to the input which are activated with a probability given in the config file. When sampling, this unit is desactivated.
    -loss.py : Implemenation of the loss as a an object. Pass the input from the trinaing_loop to the augmentation pipleine and then to the neural network and compute the loss.
    - loss_pet.py : Loss class for the conditional diffusion model applied to PET images. Both the diffused and the conditional input undergo the same transformations when passed to the augmentation pipeline.
    - trainnig_loop.py : Training loop called by the train.py file. Trains the model with config specified by the config file. We use an Adam optimizer, together with a cosine annealing schedule and an Exponential Moving Average (EMA). 
    - training_loop_pet.py :Same as above but for the PET dose-enhancement conditional diffusion model.
- **train.py** :  Main training file. Preprocess information before passing it to the main training_loop file.
- **generate.py** : Sampling function. Implments both a deterministic and a stochastich sampler solving the probability flow ODE backward in time and the backward diffusion SDE respectivly. Both use a Gaussian as the input. For every sample generated we employ a new random number generator, specifier by the input seeds.  
- **environment.yml**: Dependencies to install as yaml file. 

Make sure to have at least one (NVidia) GPU on your machine! This code does not run on CPU only machines :(

## Preliminaries

First, you need to install the latest miniconda software. If your machine run on Linux OS, the easiest is to run in your terminal the following code snippet : 
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
```
It will create a directory to install miniconda into, download the latest python 3 based install script for Linux 64 bit, run the install script, delete the install script, then add a conda initialize to your bash or zsh shell. After doing this you can restart your shell and conda will be ready to go. Make sure first though to update the conda compiler by running the following command 
```
conda update -n base -c defaults conda
```
Now clone the github repository. Finally, run the following command from the repository to install dependencies :
```
conda env create -f environment.yml
```
Finally activate the conda environment : 
```
conda activate diffusion_cea_env
```
## Training 

Once edited the training configuration file "config\_phantoms\_0.yml" - this is just an example name file you can edit any config file name - supposing you have 2 GPUs, run the following command to start training a model:
```
torchrun --standalone --nproc_per_node=2 train.py -cn=config_phantoms_0
```
You can of course modify the nproc_per_node to match the total number of GPUs on your machine. The torchrun program manages multiprocessing specifications. It is important here to not run the python file directly as the whole program supports only multiprocessing (or single processing if nproc_per_node=1).

Configurations files are splitted into two sub config dictionaries : 'training' and 'sampling'. During training, the train.py file retains only the training dict config. You'll find in it the following sub-config dictonaries : 

- **options** : Options which are directly passed to the training_loop file. 
- **secondary_options**: Options used in the train.py file.
- **dataset** : Arguments to instantiate the training dataset together with the location of the class to instantiate in hydra-core compatible format (given by paramter \_target\_).
- **network**: Arguments to instantiate the network together with the location of the class to instantiate in hydra-core compatible format (given by paramter \_target\_).
- **loss**: Arguments to instantiate the loss class together with the location of the class to instantiate in hydra-core compatible format (given by paramter \_target\_).
- **augment**: Arguments to instantiate the augmentation pipeline together with the location of the class to instantiate in hydra-core compatible format (given by paramter \_target\_).
- **optimizer**: Optimizer parameters used in trainin loop file to set up the ADAM optimizer
- **dataloader**: Dataloader parameters used in trainin loop file to set up the infinite sampler on the dataset.

When running the train.py file, the program first selects which type of training we are applying, namely the unconditional diffusion model on images or the conditional diffusion model for PET dose-enhancement task on reconstructed images. Then, it launch the processes and formats all the configs to feed to the training\_loop file. For example it checks if we resume training or if we are runing a new training session. It also creates a new folder for the training run in the "results" folder (or the name of the folder you gave to "run_dir" in the config file). Even if the training run has same parameters as a preceding run, the program take care of creating a new running folder. Inside this folder you will find he following files :

- **stats.jsonl** : Collected statistics from processes between two consecutive ticks (which are a certain number of kilo-images input in your config file), including the the average loss and its standard deviation in json line format.
- **training-state-xxxxxx.pt**: Model and optimizer checkpoint taken at xxxxxx kimgs in pt format. Use only to resume training from that state. The frequency at which the model is saved is given in terms of ticks inside the config file.
- **network-snapshot-xxxxx.pkl**: Model checkpoint taken at xxxxx kimgs in pickle format. Use to launch new training or to generate samples from that state. The frequency at which the model is saved is given in terms of ticks inside the config file. 

## Networks structure 
The network is divided into the following components :

- An initial convolutional layer embeds the image into the model channel dimension (in the diagram we fixed C=128 channels)
- Four encoder and decoder stages interconnected with skip connections (see figure). Skip connections concatenate output from the encoder modules to the input of the decoder modules along the channel dimension. 
- Each of these stages performs convolutional and attention operations at a specific resolution. Resolution gets divided by two along increasing encoder stages and multiplied by two along decreasing decoder stages. In the diagram, the input image has a resolution of 64 by 64 pixels. The division of the resolution is applied by a convolutional layer located inside a UNet block (see figure) and is appended at the end of an encoder stage. The multiplication of the resolution is performed by a transposed convolutional layer and is placed at the beginning of a decoder stage. 
- Stages also perform operations at a specified channel dimension, which is a factor of the model channel base dimension. Here the channel multiplier is one for stage 1 and two for the other stages.
- An intermediate stage is placed between the encoder and the decoder stages and composed of two UNet blocks, one with attention operation appended at the end of it.
- A final decoder stage which outputs the image to the appropriate format.
- A time embedding is used as a conditional variable. 
\end{itemize}

Here are the diagrams : 

![Dhariwal Unet diagram](/docs/images/Dhariwal_UNet.drawio.png "Dhariwal Unet diagram")
![Dhariwal Unet stages diagram](/docs/images/UNet_Stages.drawio.png "Dhariwal Unet stages diagram")
![Dhariwal Unet block diagram](/docs/images/UNet_block.drawio.png "Dhariwal Unet block diagram")

There is also a preconditioning network applied on top of the UNet. The details about the inner mechanics are given in the report! You will also find all the informations regarding the sampling procedure, all training details, the diffusion process chosen and how a diffusion - both unconditional and conditional - works. Furthermore, the code has been extensively and heavily commented so checkout that too :)

Also the code implement classifier-free guidance on discrete class conditioning, but in order to work on our dataset, we would need to tweak the dataset file to output the class associated with the images and the training loop to adapt to this case.

## Sampling 
Once trained you want to generate samples. To do so, select a pickled snapshot from your favorite trained neural network, input the path in the sampling section of the config file, in the variable "network_pkl", add all the other configs and run the following code :
```
torchrun --standalone --nproc_per_node=2 generate.py -cn=config_phantoms_0
```
In the sampling sub-config dictionnary you'll need to input the seed numbers you'll want to use to generate samples. This will generate several random number generators, one for every generated samlples. It ensures samples are sampled from a different "reality". Seeds are given in parse-int list format, i.e. '0-63' for the first 64 seeds. You'll also find the option to make sampler stochastic or deterministic depending on the value of 'S\_churn' (see EDM paper for more details and the comments in the config file uploaded in the repository). Finally if you want the ouput file to be saved as images, use option "is\_image: True" in the config file and find the images in folder "out" (or the name specified in "outdir"). Otherwise the file will be saved in a npz file, with inside array "images" of dimensions N x H X W, where N is the total number of generated images and array "cond\_images" of same dimensions, which are the corresponding low-dose images, and H and W depends on the resolution you working off. When "is\_image"  is false we assumed images have only one channel dimension (in fact I think this option only work with the conditional diffusion model, need to check for the unconditional one sorry).
