import torch.nn as nn
from vector_quantize_pytorch import VectorQuantize
from gvp.models import GVPEncoder


class LinearVQVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, num_embeddings, commitment_cost):
        super(LinearVQVAE, self).__init__()

        self.encoder = nn.Linear(input_dim, latent_dim)
        nn.init.xavier_uniform_(self.encoder.weight)

        # self.encoder = LinearEncoder(input_dim, hidden_dim, latent_dim)

        self.vector_quantizer = VectorQuantize(
            dim=latent_dim,
            codebook_size=128,
            decay=0.8,
            commitment_weight=1.0
        )
        self.decoder = nn.Linear(latent_dim, input_dim)
        nn.init.xavier_uniform_(self.decoder.weight)

        # self.decoder = LinearDecoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x, return_vq_only=False):
        x = self.encoder(x)
        x, indices, commit_loss = self.vector_quantizer(x)

        if return_vq_only:
            return x, indices, commit_loss

        x = self.decoder(x)
        return x, indices, commit_loss


def prepare_models(configs):
    """
    Prepare the VQ-VAE model.
    :param configs: (object) configurations
    :return: (object) model
    """
    model = LinearVQVAE(
        input_dim=configs.model.vector_quantization.input_dim,
        latent_dim=configs.model.vector_quantization.latent_dim,
        num_embeddings=configs.model.vector_quantization.codebook_size,
        commitment_cost=configs.model.vector_quantization.alpha
    )
    return model


if __name__ == '__main__':
    import yaml
    from utils import load_configs

    config_path = "../configs/config_gvp.yaml"

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    main_configs = load_configs(config_file)
    model = prepare_models(main_configs)
    print(model)
    print("Model loaded successfully!")
