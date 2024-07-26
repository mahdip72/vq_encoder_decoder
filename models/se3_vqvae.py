import torch.nn as nn
import torch
import numpy as np
from vector_quantize_pytorch import VectorQuantize
from utils.utils import print_trainable_parameters
# from se3_transformer_pytorch import SE3Transformer
# from equiformer_pytorch import Equiformer
from egnn_pytorch import EGNN_Network
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


class ConvNeXtBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, sequence_length):
        super(ConvNeXtBlock, self).__init__()
        self.dw_conv = nn.Conv1d(input_dim, input_dim, kernel_size=7, padding=3,
                                 groups=input_dim)  # Depthwise convolution
        self.norm = nn.LayerNorm([sequence_length, input_dim])
        self.pw_conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)  # Pointwise convolution
        self.gelu = nn.GELU()
        self.pw_conv2 = nn.Conv1d(hidden_dim, input_dim, kernel_size=1)  # Pointwise convolution

    def forward(self, x):
        residual = x
        out = self.dw_conv(x)
        out = out.permute(0, 2, 1)  # Change to (batch, sequence, channels) for LayerNorm
        out = self.norm(out)
        out = out.permute(0, 2, 1)  # Change back to (batch, channels, sequence)
        out = self.pw_conv1(out)
        out = self.gelu(out)
        out = self.pw_conv2(out)
        out += residual
        return out


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

        # self.se3_model = Equiformer(
        #     num_tokens=self.max_length+1,
        #     dim=(8, 8),  # dimensions per type, ascending, length must match number of degrees (num_degrees)
        #     dim_head=(8, 8),  # dimension per attention head
        #     heads=(4, 4),
        #     num_linear_attn_heads=0,  # number of global linear attention heads, can see all the neighbors
        #     num_degrees=2,  # number of degrees
        #     depth=3,  # depth of equivariant transformer
        #     attend_self=True,  # attending to self or not
        #     reduce_dim_out=False,
        #     single_headed_kv=True,
        #     reversible=False,
        #     # whether to reduce out to dimension of 1, say for predicting new coordinates for type 1 features
        #     l2_dist_attention=False  # set to False to try out MLP attention
        # )

        # self.embedding_encoder = nn.Embedding(num_embeddings=self.max_length+1, embedding_dim=128, padding_idx=0)

        self.se3_model = EGNN_Network(
            num_tokens=self.max_length+1,
            num_positions=self.max_length,
            # unless what you are passing in is an unordered set, set this to the maximum sequence length
            dim=512,
            # m_dim=64,
            depth=2,
            # global_linear_attn_every=1,
            # global_linear_attn_heads=8,
            # global_linear_attn_dim_head=32,
            # norm_feats=True,  # whether to layernorm the features
            norm_coors=True,
            # whether to normalize the coordinates, using a strategy from the SE(3) Transformers paper
            # num_nearest_neighbors=4,
            # coor_weights_clamp_value=10.0,
            # absolute clamped value for the coordinate weights, needed if you increase the num neareest neighbors
        )
        input_shape = 512
        # Encoder
        self.encoder_tail = nn.Sequential(
            nn.Conv1d(input_shape, self.encoder_dim, kernel_size=1),
            # ResidualBlock(self.encoder_dim, self.encoder_dim),
            # TransformerBlock(start_dim, start_dim * 2),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.encoder_dim, nhead=8, dim_feedforward=self.encoder_dim * 4, activation='gelu', dropout=0.0,
            batch_first=True
        )
        self.encoder_blocks = nn.TransformerEncoder(encoder_layer, num_layers=self.num_encoder_blocks)

        self.pos_embed_encoder = nn.Parameter(torch.randn(1, self.max_length, latent_dim) * .02)

        self.encoder_head = nn.Sequential(
            nn.Conv1d(self.encoder_dim, latent_dim, 1),
        )

        # Vector Quantizer layer
        # self.vector_quantizer = VectorQuantize(
        #     dim=latent_dim,
        #     codebook_size=codebook_size,
        #     decay=decay,
        #     commitment_weight=configs.model.vqvae.vector_quantization.commitment_weight,
        #     # orthogonal_reg_weight=10,  # in paper, they recommended a value of 10
        #     # orthogonal_reg_max_codes=512,
        #     # this would randomly sample from the codebook for the orthogonal regularization loss, for limiting memory usage
        #     # orthogonal_reg_active_codes_only=False
        #     # set this to True if you have a very large codebook, and would only like to enforce the loss on the activated codes per batch
        # )

        self.pos_embed_decoder = nn.Parameter(torch.randn(1, self.max_length, latent_dim) * .02)

        self.decoder_tail = nn.Sequential(
            nn.Conv1d(latent_dim, self.decoder_dim, 1),
        )

        # Decoder
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=self.decoder_dim, nhead=8, dim_feedforward=self.decoder_dim * 4, activation='gelu', dropout=0.0,
            batch_first=True
        )
        self.decoder_blocks = nn.TransformerEncoder(decoder_layer, num_layers=self.num_decoder_blocks)

        self.decoder_head = nn.Sequential(
            # ResidualBlock(self.decoder_dim, self.decoder_dim),
            nn.Conv1d(self.decoder_dim, 9, 1),
        )

    def forward(self, batch, return_vq_only=False):
        initial_x = batch['input_coords']
        mask = batch['masks']
        feats = torch.tensor(range(1, self.max_length+1)).reshape(1, self.max_length).to(initial_x.device)
        feats = feats.repeat_interleave(mask.shape[0], dim=0)
        x = self.se3_model(feats, initial_x, mask)

        x = x[0]

        x = x.permute(0, 2, 1)
        x = self.encoder_tail(x)

        x = x.permute(0, 2, 1)
        # Apply positional encoding to encoder
        x = x + self.pos_embed_encoder
        x = self.encoder_blocks(x)
        x = x.permute(0, 2, 1)

        x = self.encoder_head(x)

        # x = x.permute(0, 2, 1)
        # x, indices, commit_loss = self.vector_quantizer(x)
        # x = x.permute(0, 2, 1)

        # if return_vq_only:
        #     return x, indices, commit_loss

        # Apply positional encoding to decoder
        x = self.decoder_tail(x)

        x = x.permute(0, 2, 1)
        x = x + self.pos_embed_decoder
        x = self.decoder_blocks(x)
        x = x.permute(0, 2, 1)

        x = self.decoder_head(x)

        x = x.permute(0, 2, 1)
        # return x, indices, commit_loss
        return x, torch.Tensor([0]).to(x.device), torch.Tensor([0]).to(x.device)


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
