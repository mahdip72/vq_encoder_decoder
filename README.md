# Vector Quantize Encoder-Decoder Model of 3D Structures

This is a research project regarding a vector quantize encoder-decoder model of protein 3D structure.
The vector quantize model is based on the original paper
"Neural Discrete Representation Learning" by Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. 
The paper can be found [here](https://arxiv.org/abs/1711.00937).

Here is a great repo of different pytorch implementation of VQ-VAE models: [repository](https://github.com/lucidrains/vector-quantize-pytorch)


## Installation
To use this project, follow the install.sh file to install the dependencies.

### Install using SH file
Create a conda environment and use this command to install the required packages inside the conda environment.
First, make the install.sh file executable by running the following command:
```commandline
chmod +x install.sh
```
Then, run the following command to install the required packages inside the conda environment:
```commandline
bash install.sh
```

## Training

To utilize the accelerator power in you training code such as distributed multi GPU training, 
you have to set the accelerator config by running accelerate config in the command line:
```commandline
accelerate config
```
This command will create an accelerate config file in your environment. Then, you have to set
the training settings and hyperparameters inside your target task `configs/config_{task}.yaml` file. Finally,
you can start your training using a config file from configs by running the following command:
```commandline
accelerate launch train.py --config_path configs/config_file.yaml
```

Example:
```commandline
accelerate launch train_cifar.py --config_path configs/config_cifar.yaml
```

You might not use accelerator to run the train.py script if you just want to debug your script on single GPU.
If so, simply after setting the config.yaml file, run the code like `python train.py`.
It should be noted that accelerate library supports both single gpu and distributed training.
So, you can use it for your final training.

## To Do
- [x] Connect the config file to the VQ model
- [x] Connect GVP model to the config file
- [x] Add the evaluation code to fashionMNIST dataset
- [x] Create the main VQ model with the GVP encoder and a decoder
- [x] Add contact map dataloader to the project
- [ ] Add and test the LFQ model as an option
- [x] Add other features of accelerate training to the project
