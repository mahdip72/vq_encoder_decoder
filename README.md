# Vector Quantize Encoder-Decoder Model of 3D Structures

This is a research project regarding a vector quantize encoder-decoder model of protein 3D structure.
The vector quantize model is based on the original paper
"Neural Discrete Representation Learning" by Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. 
The paper can be found [here](https://arxiv.org/abs/1711.00937).

Here is a great repo of different pytorch implementation of VQ-VAE models: [repository](https://github.com/lucidrains/vector-quantize-pytorch)


## Installation
To use S-PLM project, install the corresponding environment.yaml file in your environment. Or you can follow the install.sh file to install the dependencies.

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
