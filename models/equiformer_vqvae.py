import torch.nn as nn
import torch
import numpy as np
from vector_quantize_pytorch import VectorQuantize
from utils.utils import print_trainable_parameters
from equiformer_pytorch import Equiformer
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

        self.se3_model_encoder = Equiformer(
            # num_tokens=5,
            dim=(4, 4),  # dimensions per type, ascending, length must match number of degrees (num_degrees)
            dim_head=(4, 4),  # dimension per attention head
            heads=(2, 2),
            num_linear_attn_heads=0,  # number of global linear attention heads, can see all the neighbors
            num_degrees=2,  # number of degrees
            depth=4,  # depth of equivariant transformer
            attend_self=True,  # attending to self or not
            reduce_dim_out=False,
            reversible=False,
            # whether to reduce out to dimension of 1, say for predicting new coordinates for type 1 features
            l2_dist_attention=False  # set to False to try out MLP attention
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
        mask = batch['masks']
        reshape_x = initial_x.reshape(-1, self.max_length * 4, 3)
        reshape_mask = mask.repeat_interleave(4, dim=1)
        feats = self.create_atoms_tensor_with_mask(initial_x, mask).to(initial_x.device)
        feats = self.embedding(feats)
        se3_output = self.se3_model(feats, reshape_x, reshape_mask)
        x = torch.concatenate([se3_output.type0, reshape_x], dim=2)
        x = x.reshape(initial_x.shape[0], self.max_length, -1)
        x = x.permute(0, 2, 1)

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


def prepare_models_vqvae(configs, logger, accelerator):
    vqvae = SE3VQVAE3DResNet(
        latent_dim=configs.model.vqvae.vector_quantization.dim,
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

    config_path = "../configs/config_equiformer_vqvae.yaml"

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
