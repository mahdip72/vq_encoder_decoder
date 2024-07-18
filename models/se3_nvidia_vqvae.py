import torch.nn as nn
import torch
import numpy as np
from vector_quantize_pytorch import VectorQuantize
from utils.utils import print_trainable_parameters
from SE3Transformer.se3_transformer.model import SE3Transformer, SE3TransformerPooled
from SE3Transformer.se3_transformer.model.fiber import Fiber
from SE3Transformer.se3_transformer.runtime.utils import using_tensor_cores
import torch.nn.functional as F
import flash_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim)

        # Use Flash Attention
        attn_output = flash_attn.flash_attn_func(Q, K, V, dropout_p=self.dropout.p, causal=False)

        # Merge heads
        attn_output = attn_output.contiguous().view(batch_size, seq_length, self.embed_dim)
        out = self.out(attn_output)
        return out


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, ff_dim, num_heads=4, dropout=0.0):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        attn_output = self.attention(x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        ff_output = self.ff(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        x = x.permute(0, 2, 1)
        return x


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


class SE3VQVAE3DTransformer(nn.Module):
    def __init__(self, latent_dim, codebook_size, decay, configs):
        super(SE3VQVAE3DTransformer, self).__init__()

        self.max_length = configs.model.max_length

        # Define the number of residual blocks for encoder and decoder
        self.num_encoder_blocks = configs.model.vqvae.residual_encoder.num_blocks
        self.num_decoder_blocks = configs.model.vqvae.residual_decoder.num_blocks
        self.encoder_dim = configs.model.vqvae.residual_encoder.dimension
        self.decoder_dim = configs.model.vqvae.residual_decoder.dimension

        # self.se3_model = SE3Transformer(
        #     num_tokens=257,
        #     dim=8,
        #     heads=2,
        #     depth=4,
        #     dim_head=8,
        #     num_degrees=1,
        #     valid_radius=0.5
        # )

        num_degrees = 1
        num_channels = 2
        self.se3_model = SE3Transformer(
            num_layers=1,
            num_heads=2,
            channels_div=2,
            fiber_in=Fiber({0: 3}),
            fiber_out=Fiber({0: num_degrees * num_channels}),
            fiber_hidden=Fiber({0: 2}),
            # output_dim=4,
            low_memory=True,
            # tensor_cores=using_tensor_cores(True),  # use Tensor Cores more effectively
            # **vars(args)
        )

        input_shape = 16

        # Encoder
        self.encoder_tail = nn.Sequential(
            nn.Conv1d(input_shape, self.encoder_dim, kernel_size=1),
            # TransformerBlock(start_dim, start_dim * 2),
        )

        encoder_blocks = []
        for i in range(self.num_encoder_blocks):
            encoder_blocks.append(TransformerBlock(self.encoder_dim, self.encoder_dim * 2))
        self.encoder_blocks = nn.Sequential(*encoder_blocks)

        self.pos_embed_encoder = nn.Parameter(torch.randn(1, self.max_length, latent_dim) * .02)

        self.encoder_head = nn.Sequential(
            nn.Conv1d(self.encoder_dim, latent_dim, 1),
        )

        # Vector Quantizer layer
        self.vector_quantizer = VectorQuantize(
            dim=latent_dim,
            codebook_size=codebook_size,
            decay=decay,
            commitment_weight=configs.model.vqvae.vector_quantization.commitment_weight,
            orthogonal_reg_weight=10,  # in paper, they recommended a value of 10
            orthogonal_reg_max_codes=512,
            # this would randomly sample from the codebook for the orthogonal regularization loss, for limiting memory usage
            orthogonal_reg_active_codes_only=False
            # set this to True if you have a very large codebook, and would only like to enforce the loss on the activated codes per batch
        )

        self.pos_embed_decoder = nn.Parameter(torch.randn(1, self.max_length, latent_dim) * .02)

        self.decoder_tail = nn.Sequential(
            nn.Conv1d(latent_dim, self.decoder_dim, 1),
        )

        # Decoder
        decoder_blocks = []
        for i in range(self.num_decoder_blocks):
            decoder_blocks.append(TransformerBlock(self.decoder_dim, self.decoder_dim * 2))
        self.decoder_blocks = nn.Sequential(*decoder_blocks)

        self.decoder_head = nn.Sequential(
            ResidualBlock(self.decoder_dim, self.decoder_dim),
            ResidualBlock(self.decoder_dim, self.decoder_dim),
            nn.Conv1d(self.decoder_dim, 12, 1),
        )

    def forward(self, batch, return_vq_only=False):
        initial_x = batch['input_coords'][..., 3:6]
        mask = batch['masks']
        feats = torch.tensor(range(1, self.max_length + 1)).reshape(1, self.max_length).to(initial_x.device)
        feats = feats.repeat_interleave(mask.shape[0], dim=0)
        x = self.se3_model(feats, initial_x, mask)

        x = x.type0

        x = x.permute(0, 2, 1)

        x = self.encoder_tail(x)
        x = self.encoder_blocks(x)

        # Apply positional encoding to encoder
        x = x.permute(0, 2, 1)
        x = x + self.pos_embed_encoder
        x = x.permute(0, 2, 1)

        x = self.encoder_head(x)

        x = x.permute(0, 2, 1)
        x, indices, commit_loss = self.vector_quantizer(x)
        x = x.permute(0, 2, 1)

        if return_vq_only:
            return x, indices, commit_loss

        # Apply positional encoding to decoder
        x = x.permute(0, 2, 1)
        x = x + self.pos_embed_decoder
        x = x.permute(0, 2, 1)
        x = self.decoder_tail(x)
        x = self.decoder_blocks(x)
        x = self.decoder_head(x)

        x = x.permute(0, 2, 1)
        return x, indices, commit_loss


def prepare_models_vqvae(configs, logger, accelerator):
    vqvae = SE3VQVAE3DTransformer(
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

    config_path = "../configs/config_se3_vqvae.yaml"

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    test_configs = load_configs(config_file)

    test_logger = get_dummy_logger()
    accelerator = Accelerator()

    test_model = prepare_models_vqvae(test_configs, test_logger, accelerator)

    dataset = VQVAEDataset(test_configs.valid_settings.data_path, configs=test_configs)

    test_loader = DataLoader(dataset, batch_size=test_configs.train_settings.batch_size,
                             num_workers=test_configs.train_settings.num_workers, pin_memory=True)
    struct_embeddings = []
    test_model.eval()
    for batch in tqdm.tqdm(test_loader, total=len(test_loader)):
        output, _, _ = test_model(batch)
        # print(output.shape)
        # break
