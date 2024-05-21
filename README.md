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

## To Do
- [x] Connect the config file to the VQ model
- [x] Connect GVP model to the config file
- [x] Add the evaluation code to fashionMNIST dataset
- [ ] Add contact map dataloader to the project
- [ ] Add and test the LFQ model as an option
- [ ] Add other features of accelerate training to the project
- [ ] Create the main VQ model with the GVP encoder and a decoder