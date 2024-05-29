import torch.nn as nn
import torch
from vector_quantize_pytorch import VectorQuantize
from utils.utils import print_trainable_parameters


class VQVAE3D(nn.Module):
    def __init__(self, input_dim, latent_dim, codebook_size, decay, configs):
        super(VQVAE3D, self).__init__()

        self.max_length = configs.model.max_length
        self.pos_embed = nn.Parameter(torch.randn(1, self.max_length, 12) * .02)

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

    def drop_positional_encoding(self, embedding):
        embedding = embedding + self.pos_embed
        return embedding

    def forward(self, batch, return_vq_only=False):
        # change the shape of x from (batch, num_nodes, node_dim) to (batch, node_dim, num_nodes)
        x = batch['coords'].permute(0, 2, 1)

        x = x.permute(0, 2, 1)
        x = self.drop_positional_encoding(x)
        x = x.permute(0, 2, 1)

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


def prepare_models_vqvae(configs, logger, accelerator):
    vqvae = VQVAE3D(
        input_dim=configs.model.vqvae.vector_quantization.dim*2,
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
    from data.dataset import custom_collate, GVPDataset

    config_path = "../configs/config_gvp.yaml"

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    test_configs = load_configs(config_file)

    test_logger = get_dummy_logger()
    accelerator = Accelerator()

    test_model = prepare_models_vqvae(test_configs, test_logger, accelerator)
    # print(test_model)
    print("Model loaded successfully!")

    dataset = GVPDataset(test_configs.train_settings.data_path,
                         seq_mode=test_configs.model.struct_encoder.use_seq.seq_embed_mode,
                         use_rotary_embeddings=test_configs.model.struct_encoder.use_rotary_embeddings,
                         use_foldseek=test_configs.model.struct_encoder.use_foldseek,
                         use_foldseek_vector=test_configs.model.struct_encoder.use_foldseek_vector,
                         top_k=test_configs.model.struct_encoder.top_k,
                         num_rbf=test_configs.model.struct_encoder.num_rbf,
                         num_positional_embeddings=test_configs.model.struct_encoder.num_positional_embeddings,
                         configs=test_configs)

    test_loader = DataLoader(dataset, batch_size=test_configs.train_settings.batch_size, num_workers=0, pin_memory=True,
                             collate_fn=custom_collate)
    struct_embeddings = []
    test_model.eval()
    for batch in tqdm.tqdm(test_loader, total=len(test_loader)):
        output, _, _ = test_model(batch)
        print(output.shape)
        break