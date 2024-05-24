import torch.nn as nn
from vector_quantize_pytorch import VectorQuantize
from data.data import custom_collate, ProteinGraphDataset
from gvp.models import GVPEncoder
from torch.utils.data import DataLoader


class VQVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, codebook_size, decay):
        super(VQVAE, self).__init__()

        self.encoder = nn.Conv1d(input_dim, latent_dim, 1)

        self.vector_quantizer = VectorQuantize(
            dim=latent_dim,
            codebook_size=codebook_size,
            decay=decay,
            commitment_weight=1.0
        )
        self.decoder = nn.Conv1d(latent_dim, input_dim, 1)

    def forward(self, x, return_vq_only=False):
        x = self.encoder(x)
        x, indices, commit_loss = self.vector_quantizer(x)

        if return_vq_only:
            return x, indices, commit_loss

        x = self.decoder(x)
        return x, indices, commit_loss


class GVPVQVAE(nn.Module):
    def __init__(self, gvp, vqvae, configs):
        super(GVPVQVAE, self).__init__()
        self.gvp = gvp
        self.vqvae = vqvae

        self.configs = configs

    def forward(self, x):
        _, x = self.gvp(graph=x)
        x = self.vqvae(x)
        return x


def prepare_models(configs):
    """
    Prepare the VQ-VAE model.
    :param configs: (object) configurations
    :return: (object) model
    """
    gvp = GVPEncoder(configs=configs)
    vqvae = VQVAE(
        input_dim=configs.model.struct_encoder.node_h_dim[0],
        latent_dim=configs.model.vqvae.vector_quantization.dim,
        codebook_size=configs.model.vqvae.vector_quantization.codebook_size,
        decay=configs.model.vqvae.vector_quantization.decay
    )
    gvp_vqvae = GVPVQVAE(gvp, vqvae, configs)

    return gvp_vqvae


if __name__ == '__main__':
    import yaml
    import tqdm
    from utils import load_configs

    config_path = "../configs/config_gvp.yaml"

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    main_configs = load_configs(config_file)
    test_model = prepare_models(main_configs)
    # print(test_model)
    print("Model loaded successfully!")

    dataset = ProteinGraphDataset(main_configs.train_settings.data_path,
                                  seq_mode=main_configs.model.struct_encoder.use_seq.seq_embed_mode,
                                  use_rotary_embeddings=main_configs.model.struct_encoder.use_rotary_embeddings,
                                  use_foldseek=main_configs.model.struct_encoder.use_foldseek,
                                  use_foldseek_vector=main_configs.model.struct_encoder.use_foldseek_vector,
                                  top_k=main_configs.model.struct_encoder.top_k,
                                  num_rbf=main_configs.model.struct_encoder.num_rbf,
                                  num_positional_embeddings=main_configs.model.struct_encoder.num_positional_embeddings)

    test_loader = DataLoader(dataset, batch_size=main_configs.train_settings.batch_size, num_workers=0, pin_memory=True,
                             collate_fn=custom_collate)
    struct_embeddings = []
    test_model.eval()
    for batch in tqdm.tqdm(test_loader, total=len(test_loader)):
        graph = batch["graph"]
        residue_level_feature = test_model(graph)
        print(residue_level_feature.shape)
        break
