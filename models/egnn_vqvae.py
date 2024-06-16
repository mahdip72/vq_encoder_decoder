import torch.nn as nn
import torch
import numpy as np
from vector_quantize_pytorch import VectorQuantize
from utils.utils import print_trainable_parameters
from egnn_pytorch import EGNN_Network
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, input_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(input_dim)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class SE3VQVAE3DResNet(nn.Module):
    def __init__(self, latent_dim, codebook_size, decay, configs):
        super(SE3VQVAE3DResNet, self).__init__()

        self.max_length = configs.model.max_length

        self.embedding = nn.Embedding(num_embeddings=5, embedding_dim=4, padding_idx=0)
        # self.embedding = self.embedding.to('cpu')

        self.se3_model_encoder = net = EGNN_Network(
            num_tokens=21,
            num_positions=1024,
            # unless what you are passing in is an unordered set, set this to the maximum sequence length
            dim=32,
            depth=3,
            num_nearest_neighbors=8,
            coor_weights_clamp_value=2.
            # absolute clamped value for the coordinate weights, needed if you increase the num neareest neighbors
        )

        # Define the number of residual blocks for encoder and decoder
        self.num_encoder_blocks = configs.model.vqvae.residual_encoder.num_blocks
        self.num_decoder_blocks = configs.model.vqvae.residual_decoder.num_blocks
        self.encoder_dim = configs.model.vqvae.residual_encoder.dimension
        self.decoder_dim = configs.model.vqvae.residual_decoder.dimension

        start_dim = 128
        # Encoder
        self.encoder_tail = nn.Sequential(
            nn.Conv1d(12, start_dim, 1),
            nn.BatchNorm1d(start_dim),
            nn.ReLU()
        )

        dims = list(np.linspace(start_dim, self.encoder_dim, self.num_encoder_blocks).astype(int))
        encoder_blocks = []
        prev_dim = start_dim
        for i, dim in enumerate(dims):
            block = nn.Sequential(
                nn.Conv1d(prev_dim, dim, 3, padding=1),
                ResidualBlock(dim, dim),
                ResidualBlock(dim, dim),
            )
            encoder_blocks.append(block)
            prev_dim = dim
        self.encoder_blocks = nn.Sequential(*encoder_blocks)

        self.encoder_head = nn.Sequential(
            nn.Conv1d(self.encoder_dim, self.encoder_dim, 1),
            nn.BatchNorm1d(self.encoder_dim),
            nn.ReLU(),

            nn.Conv1d(self.encoder_dim, self.encoder_dim, 1),
            nn.BatchNorm1d(self.encoder_dim),
            nn.ReLU(),

            nn.Conv1d(self.encoder_dim, latent_dim, 1),
        )

        self.vector_quantizer = VectorQuantize(
            dim=latent_dim,
            codebook_size=codebook_size,
            decay=decay,
            commitment_weight=1.0,
        )

        self.decoder_tail = nn.Sequential(
            nn.Conv1d(latent_dim, self.decoder_dim, 1),

            nn.Conv1d(self.decoder_dim, self.decoder_dim, 1),
            nn.BatchNorm1d(self.decoder_dim),
            nn.ReLU(),

            nn.Conv1d(self.decoder_dim, self.decoder_dim, 1),
            nn.BatchNorm1d(self.decoder_dim),
            nn.ReLU(),
        )

        dims = list(np.linspace(self.decoder_dim, start_dim, self.num_encoder_blocks).astype(int))
        # Decoder
        decoder_blocks = []
        dims = dims + [dims[-1]]
        for i, dim in enumerate(dims[:-1]):
            block = nn.Sequential(
                ResidualBlock(dim, dim),
                ResidualBlock(dim, dim),
                nn.Conv1d(dim, dims[i + 1], 3, padding=1),
            )
            decoder_blocks.append(block)
        self.decoder_blocks = nn.Sequential(*decoder_blocks)

        self.decoder_head = nn.Sequential(
            nn.Conv1d(start_dim, 12, 1),
            nn.BatchNorm1d(12),
            nn.ReLU(),
        )

    @staticmethod
    def create_atoms_tensor_with_mask(x, mask):
        """
        Create an atom tensor and apply the mask to replace masked regions with 0.

        Parameters:
        - x (torch.Tensor): The input tensor with shape (batch_size, num_amino_acids, 12)
        - mask (torch.Tensor): The input mask tensor with shape (batch_size, num_amino_acids)

        Returns:
        - torch.Tensor: The resulting atom tensor with shape (batch_size, num_amino_acids * 4)
        """
        batch_size, num_amino_acids, _ = x.shape

        # Create a tensor of numbers 1, 2, 3, 4 repeated for each amino acid
        atoms_sequence = torch.tensor([1, 2, 3, 4])

        # Repeat this sequence for each amino acid and then expand it for the entire batch
        atoms_per_amino_tensor = atoms_sequence.repeat(num_amino_acids)

        # Expand this sequence for the entire batch
        atoms_tensor = atoms_per_amino_tensor.repeat(batch_size, 1)

        # Repeat each element in the mask tensor 4 times along the row
        expanded_mask = mask.repeat_interleave(4, dim=1)

        # Replace masked regions with 0 in the atoms tensor
        atoms_tensor[expanded_mask == False] = 0

        return atoms_tensor

    def forward(self, batch, return_vq_only=False):
        initial_x = batch['input_coords']
        x = initial_x[..., :3]
        mask = batch['masks']
        feats = torch.ones(x.shape[0], x.shape[1], dtype=torch.int).to(x.device)
        feats[mask == False] = 0
        feats = self.embedding(feats)
        se3_output = self.se3_model_encoder(feats, x, mask)
        x = torch.concatenate([se3_output.type0, torch.mean(se3_output.type1, -2), initial_x], dim=2)
        # x = x.reshape(initial_x.shape[0], self.max_length, -1)
        x = initial_x.permute(0, 2, 1)

        x = self.encoder_tail(x)
        x = self.encoder_blocks(x)
        x = self.encoder_head(x)

        x = x.permute(0, 2, 1)
        x, indices, commit_loss = self.vector_quantizer(x)
        x = x.permute(0, 2, 1)

        if return_vq_only:
            return x, indices, commit_loss

        x = self.decoder_tail(x)
        x = self.decoder_blocks(x)
        x = self.decoder_head(x)

        x = x.permute(0, 2, 1)
        return x, indices, commit_loss


class VQVAE3DTransformer(nn.Module):
    def __init__(self, codebook_size, decay, configs):
        super(VQVAE3DTransformer, self).__init__()

        self.max_length = configs.model.max_length*4
        self.encoder_dim = configs.model.vqvae.encoder.dimension
        self.decoder_dim = configs.model.vqvae.decoder.dimension

        self.embedding_encoder = nn.Embedding(num_embeddings=5, embedding_dim=64, padding_idx=0)

        self.egnn_encoder = EGNN_Network(
            num_tokens=5,
            num_positions=self.max_length,
            # unless what you are passing in is an unordered set, set this to the maximum sequence length
            dim=64,
            depth=3,
            # num_nearest_neighbors=32,
            # coor_weights_clamp_value=4.
            # absolute clamped value for the coordinate weights, needed if you increase the num neareest neighbors
        )

        # Projecting the input to the dimension expected by the Transformer
        self.input_projection = nn.Sequential(
            nn.Conv1d(64, self.encoder_dim, 1),
            nn.Conv1d(self.encoder_dim, self.encoder_dim, 3, padding=1),
            nn.BatchNorm1d(self.encoder_dim),
            nn.ReLU(),
            nn.Conv1d(self.encoder_dim, self.encoder_dim, 3, padding=1),
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

        # self.embedding_decoder = nn.Embedding(num_embeddings=5, embedding_dim=32, padding_idx=0)
        self.egnn_decoder = EGNN_Network(
            # num_tokens=5,
            num_positions=self.max_length,
            # unless what you are passing in is an unordered set, set this to the maximum sequence length
            dim=128,
            depth=3,
            # num_nearest_neighbors=8,
            # coor_weights_clamp_value=2.
            # absolute clamped value for the coordinate weights, needed if you increase the num neareest neighbors
        )

        # Projecting the output back to the original dimension
        self.output_projection = nn.Sequential(
            nn.Conv1d(12, 12, 3, padding=1),
            nn.BatchNorm1d(12),
            nn.ReLU(),
            nn.Conv1d(12, 12, 3, padding=1),
            nn.BatchNorm1d(12),
            nn.ReLU(),
            nn.Conv1d(12, 12, 1),
        )

    @staticmethod
    def create_atoms_tensor_with_mask(x, mask):
        """
        Create an atom tensor and apply the mask to replace masked regions with 0.

        Parameters:
        - x (torch.Tensor): The input tensor with shape (batch_size, num_amino_acids, 12)
        - mask (torch.Tensor): The input mask tensor with shape (batch_size, num_amino_acids)

        Returns:
        - torch.Tensor: The resulting atom tensor with shape (batch_size, num_amino_acids * 4)
        """
        batch_size, num_amino_acids, _ = x.shape

        # Create a tensor of numbers 1, 2, 3, 4 repeated for each amino acid
        atoms_sequence = torch.tensor([0, 1, 2, 3])

        # Repeat this sequence for each amino acid and then expand it for the entire batch
        atoms_per_amino_tensor = atoms_sequence.repeat(num_amino_acids)

        # Expand this sequence for the entire batch
        atoms_tensor = atoms_per_amino_tensor.repeat(batch_size, 1)

        # Repeat each element in the mask tensor 4 times along the row
        expanded_mask = mask.repeat_interleave(4, dim=1)

        # Replace masked regions with 0 in the atoms tensor
        atoms_tensor[expanded_mask == False] = 0

        return atoms_tensor

    @staticmethod
    def drop_positional_encoding(embedding, pos_embed):
        embedding = embedding + pos_embed
        return embedding

    def forward(self, batch, return_vq_only=False):
        initial_x = batch['input_coords']
        mask = batch['masks']
        reshape_x = initial_x.reshape(-1, self.max_length, 3)
        reshape_mask = mask.repeat_interleave(4, dim=1)
        feats = self.create_atoms_tensor_with_mask(initial_x, mask).to(initial_x.device)
        # feats = self.embedding_encoder(feats)
        feats_out, coors_out = self.egnn_encoder(feats, reshape_x, reshape_mask)
        # x = torch.concatenate([egnn_output.type0, torch.mean(egnn_output.type1, -2), initial_x], dim=2)

        # Apply input projection
        x = feats_out.permute(0, 2, 1)
        x = self.input_projection(x)

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

        feats_out, coors_out = self.egnn_decoder(x, torch.ones_like(reshape_x), reshape_mask)

        # coors_out = coors_out.repeat_interleave(4, dim=2)
        coors_out = coors_out.reshape(coors_out.shape[0], -1, 12)

        # Permute back to [batch, feature, sequence]
        x = coors_out.permute(0, 2, 1)

        # Apply output projection
        # x = self.output_projection(x)
        for layer in self.output_projection:
            x = layer(x)
        x = x.permute(0, 2, 1)

        return x, indices, commit_loss


def prepare_models_vqvae(configs, logger, accelerator):
    # vqvae = SE3VQVAE3DResNet(
    #     latent_dim=configs.model.vqvae.vector_quantization.dim,
    #     codebook_size=configs.model.vqvae.vector_quantization.codebook_size,
    #     decay=configs.model.vqvae.vector_quantization.decay,
    #     configs=configs
    # )

    vqvae = VQVAE3DTransformer(
        codebook_size=configs.model.vqvae.vector_quantization.codebook_size,
        decay=configs.model.vqvae.vector_quantization.decay,
        configs=configs
    )
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

    config_path = "../configs/config_egnn_vqvae.yaml"

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    test_configs = load_configs(config_file)

    test_logger = get_dummy_logger()
    accelerator = Accelerator()

    test_model = prepare_models_vqvae(test_configs, test_logger, accelerator).cuda()

    dataset = VQVAEDataset(test_configs.valid_settings.data_path, configs=test_configs)

    test_loader = DataLoader(dataset, batch_size=test_configs.train_settings.batch_size,
                             num_workers=test_configs.train_settings.num_workers, pin_memory=True)
    struct_embeddings = []
    test_model.eval()
    for batch in tqdm.tqdm(test_loader, total=len(test_loader)):
        output, _, _ = test_model(batch)
        # print(output.shape)
        # break
