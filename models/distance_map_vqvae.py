import torch.nn as nn
import torch
import numpy as np
from vector_quantize_pytorch import VectorQuantize
from utils.utils import print_trainable_parameters
import torch.nn.functional as F


class ConvNeXtBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ConvNeXtBlock, self).__init__()
        self.dw_conv = nn.Conv2d(input_dim, input_dim, kernel_size=7, padding=3, groups=input_dim)  # Depthwise convolution
        self.norm = nn.LayerNorm([input_dim, 1, 1])
        self.pw_conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)  # Pointwise convolution
        self.gelu = nn.GELU()
        self.pw_conv2 = nn.Conv2d(hidden_dim, input_dim, kernel_size=1)  # Pointwise convolution

    def forward(self, x):
        residual = x
        out = self.dw_conv(x)
        out = out.permute(0, 2, 3, 1)  # Change to (batch, height, width, channels) for LayerNorm
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2)  # Change back to (batch, channels, height, width)
        out = self.pw_conv1(out)
        out = self.gelu(out)
        out = self.pw_conv2(out)
        out += residual
        return out


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, input_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(input_dim)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class VQVAE3DResNet(nn.Module):
    def __init__(self, latent_dim, codebook_size, decay, configs):
        super(VQVAE3DResNet, self).__init__()

        self.max_length = configs.model.max_length

        # Define the number of residual blocks for encoder and decoder
        self.num_encoder_blocks = configs.model.vqvae.residual_encoder.num_blocks
        self.num_decoder_blocks = configs.model.vqvae.residual_decoder.num_blocks
        self.encoder_dim = configs.model.vqvae.residual_encoder.dimension
        self.decoder_dim = configs.model.vqvae.residual_decoder.dimension

        start_dim = 4
        # Encoder
        self.encoder_tail = nn.Sequential(
            nn.Conv2d(1, start_dim, kernel_size=1),
            nn.Conv2d(start_dim, start_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(start_dim),
            nn.ReLU()
        )

        dims = list(np.linspace(start_dim, self.decoder_dim, self.num_encoder_blocks).astype(int))
        encoder_blocks = []
        prev_dim = start_dim
        for i, dim in enumerate(dims):
            block = nn.Sequential(
                nn.Conv2d(prev_dim, dim, 3, padding=1),
                ResidualBlock(dim, dim),
                ResidualBlock(dim, dim),
            )
            encoder_blocks.append(block)
            if (i+1) % 4 == 0:
                pooling_block = nn.Sequential(
                    nn.Conv2d(dim, dim, 3, stride=2, padding=1),
                    nn.BatchNorm2d(dim),
                    nn.ReLU())
                encoder_blocks.append(pooling_block)
            prev_dim = dim
        self.encoder_blocks = nn.Sequential(*encoder_blocks)

        self.encoder_head = nn.Sequential(
            nn.Conv2d(self.encoder_dim, self.encoder_dim, 1),
            nn.BatchNorm2d(self.encoder_dim),
            nn.ReLU(),

            nn.Conv2d(self.encoder_dim, self.encoder_dim, 1),
            nn.BatchNorm2d(self.encoder_dim),
            nn.ReLU(),

            nn.Conv2d(self.encoder_dim, latent_dim, 1),
        )

        self.vector_quantizer = VectorQuantize(
            dim=latent_dim,
            codebook_size=codebook_size,
            decay=decay,
            commitment_weight=1.0,
            accept_image_fmap=True,
        )

        self.decoder_tail = nn.Sequential(
            nn.Conv2d(latent_dim, self.decoder_dim, 1),

            nn.Conv2d(self.decoder_dim, self.decoder_dim, 1),
            nn.BatchNorm2d(self.decoder_dim),
            nn.ReLU(),

            nn.Conv2d(self.decoder_dim, self.decoder_dim, 1),
            nn.BatchNorm2d(self.decoder_dim),
            nn.ReLU(),
        )

        dims = list(np.linspace(start_dim, self.decoder_dim, self.num_decoder_blocks).astype(int))[::-1]
        # Decoder
        decoder_blocks = []
        dims = dims + [dims[-1]]
        for i, dim in enumerate(dims[:-1]):
            if (i+1) % 4 == 0:
                pooling_block = nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(dim, dim, 3, padding=1),
                    nn.BatchNorm2d(dim),
                    nn.ReLU())
                decoder_blocks.append(pooling_block)
            block = nn.Sequential(
                ResidualBlock(dim, dim),
                ResidualBlock(dim, dim),
                nn.Conv2d(dim, dims[i + 1], 3, padding=1),
            )
            decoder_blocks.append(block)
        self.decoder_blocks = nn.Sequential(*decoder_blocks)

        self.decoder_head = nn.Sequential(
            nn.Conv2d(start_dim, 1, 1)
        )

    def forward(self, batch, return_vq_only=False):
        x = batch['input_distance_map']
        # keep the shape of intial_x
        # initial_x_shape = x.shape
        # x = x.reshape(initial_x_shape[0], 4, int(initial_x_shape[2]/2), int(initial_x_shape[3]/2))

        x = self.encoder_tail(x)
        x = self.encoder_blocks(x)
        x = self.encoder_head(x)

        x, indices, commit_loss = self.vector_quantizer(x)

        if return_vq_only:
            return x, indices, commit_loss

        x = self.decoder_tail(x)
        x = self.decoder_blocks(x)
        x = self.decoder_head(x)

        # Return x to its original shape
        # x = x.reshape(initial_x_shape[0], 1, int(initial_x_shape[2]), int(initial_x_shape[3]))
        return x, indices, commit_loss


class VQVAE3D(nn.Module):
    def __init__(self, input_dim, latent_dim, codebook_size, decay, configs):
        super(VQVAE3D, self).__init__()

        self.max_length = configs.model.max_length

        self.encoder_layers = nn.Sequential(
            nn.Conv1d(12, input_dim, 1),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),

            nn.Conv1d(input_dim, input_dim, 3, padding=1),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),

            nn.Conv1d(input_dim, input_dim, 3, padding=1),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),

            nn.Conv1d(input_dim, input_dim, 3, padding=1),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),

            nn.Conv1d(input_dim, latent_dim, 3, padding=1),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
        )

        self.vector_quantizer = VectorQuantize(
            dim=latent_dim,
            codebook_size=codebook_size,
            decay=decay,
            commitment_weight=1.0,
            # accept_image_fmap=True,
        )
        self.decoder_layers = nn.Sequential(
            nn.Conv1d(latent_dim, input_dim, 3, padding=1),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),

            nn.Conv1d(input_dim, input_dim, 3, padding=1),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),

            nn.Conv1d(input_dim, input_dim, 3, padding=1),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),

            nn.Conv1d(input_dim, input_dim, 3, padding=1),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),

            nn.Conv1d(input_dim, input_dim, 1),
            nn.BatchNorm1d(input_dim),
        )

        self.head = nn.Sequential(
            nn.Conv1d(input_dim, 12, 1),
            # nn.Tanh()
        )

    def forward(self, batch, return_vq_only=False):
        # change the shape of x from (batch, num_nodes, node_dim) to (batch, node_dim, num_nodes)
        x = batch['coords'].permute(0, 2, 1)

        for layer in self.encoder_layers:
            x = layer(x)

        x = x.permute(0, 2, 1)
        x, indices, commit_loss = self.vector_quantizer(x)
        x = x.permute(0, 2, 1)

        if return_vq_only:
            return x, indices, commit_loss

        for layer in self.decoder_layers:
            x = layer(x)

        for layer in self.head:
            x = layer(x)

        # make it to be (batch, num_nodes, 12)
        x = x.permute(0, 2, 1)
        return x, indices, commit_loss


class VQVAE3DTransformer(nn.Module):
    def __init__(self, codebook_size, decay, configs):
        super(VQVAE3DTransformer, self).__init__()

        self.max_length = configs.model.max_length
        self.encoder_dim = configs.model.vqvae.encoder.dimension
        self.decoder_dim = configs.model.vqvae.decoder.dimension

        # Projecting the input to the dimension expected by the Transformer
        # self.input_projection = nn.Linear(12, self.encoder_dim)

        self.input_projection = nn.Sequential(
            nn.Conv1d(3, int(self.encoder_dim / 2), 1),
            nn.Conv1d(int(self.encoder_dim / 2), int(self.encoder_dim / 2), 3, padding=1),
            nn.BatchNorm1d(int(self.encoder_dim / 2)),
            nn.ReLU(),
            nn.Conv1d(int(self.encoder_dim / 2), self.encoder_dim, 3, padding=1),
            nn.BatchNorm1d(self.encoder_dim),
            nn.ReLU()
        )

        self.pos_embed_encoder = nn.Parameter(torch.randn(1, self.max_length, self.encoder_dim) * .02)

        # Transformer Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.encoder_dim,
            nhead=configs.model.vqvae.encoder.num_heads,
            dim_feedforward=configs.model.vqvae.encoder.dim_feedforward,
            activation=configs.model.vqvae.encoder.activation_function
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=configs.model.vqvae.encoder.num_layers)

        # Projecting the output of the Transformer to the dimension expected by the VQ layer
        self.vq_in_projection = nn.Linear(self.encoder_dim, configs.model.vqvae.vector_quantization.dim)

        self.pos_embed_decoder = nn.Parameter(torch.randn(1, self.max_length, self.encoder_dim) * .02)

        # Vector Quantizer
        self.vector_quantizer = VectorQuantize(
            dim=configs.model.vqvae.vector_quantization.dim,
            codebook_size=codebook_size,
            decay=decay,
            commitment_weight=1.0,
        )

        # Projecting the output of the VQ layer back to the decoder dimension
        self.vq_out_projection = nn.Linear(configs.model.vqvae.vector_quantization.dim, self.decoder_dim)

        # Transformer Decoder
        self.decoder_layer = nn.TransformerEncoderLayer(
            d_model=self.decoder_dim,
            nhead=configs.model.vqvae.decoder.num_heads,
            dim_feedforward=configs.model.vqvae.decoder.dim_feedforward,
            activation=configs.model.vqvae.encoder.activation_function
        )
        self.decoder = nn.TransformerEncoder(self.decoder_layer, num_layers=configs.model.vqvae.decoder.num_layers)

        # Projecting the output back to the original dimension
        # self.output_projection = nn.Linear(self.decoder_dim, 12)

        self.output_projection = nn.Sequential(
            nn.Conv1d(self.decoder_dim, self.decoder_dim, 3, padding=1),
            nn.BatchNorm1d(self.decoder_dim),
            nn.ReLU(),
            nn.Conv1d(self.decoder_dim, int(self.decoder_dim / 2), 3, padding=1),
            nn.BatchNorm1d(int(self.decoder_dim / 2)),
            nn.ReLU(),
            nn.Conv1d(int(self.decoder_dim / 2), 3, 1),
        )

    @staticmethod
    def drop_positional_encoding(embedding, pos_embed):
        embedding = embedding + pos_embed
        return embedding

    def forward(self, batch, return_vq_only=False):
        x = batch['input_coords']

        # Apply input projection
        # x = self.input_projection(x)
        x = x.permute(0, 2, 1)
        for layer in self.input_projection:
            x = layer(x)

        # Permute for Transformer [batch, sequence, feature]
        x = x.permute(0, 2, 1)

        # Apply positional encoding to encoder
        x = self.drop_positional_encoding(x, self.pos_embed_encoder)

        # Encoder
        x = self.encoder(x)

        # Apply qv_in_projection
        x = self.vq_in_projection(x)

        x, indices, commit_loss = self.vector_quantizer(x)

        if return_vq_only:
            x = x.permute(0, 2, 1)
            return x, indices, commit_loss

        # Apply vq_out_projection
        x = self.vq_out_projection(x)

        # Apply positional encoding to decoder
        x = self.drop_positional_encoding(x, self.pos_embed_decoder)

        # Decoder
        x = self.decoder(x)

        # Permute back to [batch, feature, sequence]
        x = x.permute(0, 2, 1)

        # Apply output projection
        # x = self.output_projection(x)
        for layer in self.output_projection:
            x = layer(x)
        x = x.permute(0, 2, 1)

        return x, indices, commit_loss


def prepare_models_distance_map_vqvae(configs, logger, accelerator):
    vqvae = VQVAE3DResNet(
        latent_dim=configs.model.vqvae.vector_quantization.dim,
        codebook_size=configs.model.vqvae.vector_quantization.codebook_size,
        decay=configs.model.vqvae.vector_quantization.decay,
        configs=configs
    )

    # vqvae = VQVAE3DTransformer(
    #     codebook_size=configs.model.vqvae.vector_quantization.codebook_size,
    #     decay=configs.model.vqvae.vector_quantization.decay,
    #     configs=configs
    # )

    if accelerator.is_main_process:
        print_trainable_parameters(vqvae, logger, 'VQ-VAE')

    return vqvae


if __name__ == '__main__':
    import yaml
    import tqdm
    from utils.utils import load_configs, get_dummy_logger
    from torch.utils.data import DataLoader
    from accelerate import Accelerator
    from data.dataset import VQVAEDataset

    config_path = "../configs/config_vqvae.yaml"

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    test_configs = load_configs(config_file)

    test_logger = get_dummy_logger()
    accelerator = Accelerator()

    test_model = prepare_models_distance_map_vqvae(test_configs, test_logger, accelerator)

    dataset = VQVAEDataset(test_configs.valid_settings.data_path, configs=test_configs)

    test_loader = DataLoader(dataset, batch_size=test_configs.train_settings.batch_size,
                             num_workers=test_configs.train_settings.num_workers, pin_memory=True)
    struct_embeddings = []
    test_model.eval()
    for batch in tqdm.tqdm(test_loader, total=len(test_loader)):
        output, _, _ = test_model(batch)
        print(output.shape)
        break
