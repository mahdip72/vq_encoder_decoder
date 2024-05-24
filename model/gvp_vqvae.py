import torch.nn as nn
from vector_quantize_pytorch import VectorQuantize
from data.data import custom_collate, ProteinGraphDataset
from gvp.models import GVPEncoder
from torch.utils.data import DataLoader


class VQVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, num_embeddings, commitment_cost):
        super(VQVAE, self).__init__()

        self.encoder = nn.Conv1d(input_dim, latent_dim, 1)

        self.vector_quantizer = VectorQuantize(
            dim=latent_dim,
            codebook_size=128,
            decay=0.8,
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
        x = self.gvp(x)
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
        input_dim=gvp.backbone,
        latent_dim=configs.model.vector_quantization.latent_dim,
        num_embeddings=configs.model.vector_quantization.codebook_size,
        commitment_cost=configs.model.vector_quantization.alpha
    )
    gvp_vqvae = GVPVQVAE(gvp, vqvae, configs)

    return gvp_vqvae


if __name__ == '__main__':
    import yaml
    from utils import load_configs

    config_path = "../configs/config_gvp.yaml"

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    main_configs = load_configs(config_file)
    test_model = prepare_models(main_configs)
    print(test_model)
    print("Model loaded successfully!")

    dataset = ProteinGraphDataset(config_file.train_settings.data_path,
                                  seq_mode=config_file.model.struct_encoder.use_seq.seq_embed_mode,
                                  use_rotary_embeddings=config_file.model.struct_encoder.use_rotary_embeddings,
                                  use_foldseek=config_file.model.struct_encoder.use_foldseek,
                                  use_foldseek_vector=config_file.model.struct_encoder.use_foldseek_vector,
                                  top_k=config_file.model.struct_encoder.top_k,
                                  num_rbf=config_file.model.struct_encoder.num_rbf,
                                  num_positional_embeddings=config_file.model.struct_encoder.num_positional_embeddings)

    val_loader = DataLoader(dataset, batch_size=config_file.train_settings.batch_size, num_workers=0, pin_memory=True,
                            collate_fn=custom_collate)
    struct_embeddings = []
    test_model.eval()
    for batch in val_loader:
        graph = batch["graph"].to('gpu:1')
        features_struct, _ = test_model(graph=graph)
        struct_embeddings.extend(features_struct.cpu().detach().numpy())
        print(struct_embeddings)
        break
