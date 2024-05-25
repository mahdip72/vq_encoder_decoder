import torch.nn as nn
import torch
from vector_quantize_pytorch import VectorQuantize
from data.data import custom_collate, ProteinGraphDataset
from gvp.models import GVPEncoder
from utils import print_trainable_parameters


class VQVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, codebook_size, decay, configs):
        super(VQVAE, self).__init__()

        self.max_length = configs.model.max_length

        self.encoder_layers = nn.Sequential(
            nn.Conv1d(input_dim, latent_dim, 1),
            nn.ReLU(),
            nn.Conv1d(latent_dim, latent_dim, 3, padding=1),
            nn.ReLU(),
        )
        self.norm_1 = nn.LayerNorm(self.max_length)

        self.vector_quantizer = VectorQuantize(
            dim=latent_dim,
            codebook_size=codebook_size,
            decay=decay,
            commitment_weight=1.0,
            # accept_image_fmap=True,
        )
        self.decoder_layers = nn.Sequential(
            nn.Conv1d(latent_dim, input_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(input_dim, input_dim, 1),

        )

        self.norm_2 = nn.LayerNorm(self.max_length)

        self.head = nn.Sequential(
            nn.Conv1d(input_dim, 12, 1),
            nn.Tanh()
        )

    def forward(self, x, return_vq_only=False):
        for layer in self.encoder_layers:
            x = layer(x)

        x = self.norm_1(x)

        x = x.permute(0, 2, 1)
        x, indices, commit_loss = self.vector_quantizer(x)
        x = x.permute(0, 2, 1)

        if return_vq_only:
            return x, indices, commit_loss

        for layer in self.decoder_layers:
            x = layer(x)

        x = self.norm_2(x)

        for layer in self.head:
            x = layer(x)

        # make it to be (batch, num_nodes, 12)
        x = x.permute(0, 2, 1)
        return x, indices, commit_loss


class GVPVQVAE(nn.Module):
    def __init__(self, gvp, vqvae, configs):
        super(GVPVQVAE, self).__init__()
        self.gvp = gvp
        self.vqvae = vqvae

        self.configs = configs
        self.max_length = configs.model.max_length

    @staticmethod
    def separate_features(batched_features, batch):
        # Get the number of nodes in each graph
        node_counts = batch.bincount().tolist()

        # Split the features tensor into separate tensors for each graph
        features_list = torch.split(batched_features, node_counts)

        return features_list

    def merge_features(self, features_list):
        # Pad tensors
        padded_tensors = []
        for t in features_list:
            if t.size(0) < self.max_length:
                size_diff = self.max_length - t.size(0)
                pad = torch.zeros(size_diff, t.size(1))
                t_padded = torch.cat([t, pad], dim=0)
            else:
                t_padded = t
            padded_tensors.append(t_padded.unsqueeze(0))  # Add an extra dimension for concatenation

        # Concatenate tensors
        result = torch.cat(padded_tensors, dim=0)
        return result

    def forward(self, batch):
        _, x = self.gvp(graph=batch['graph'])
        x = self.separate_features(x, batch['graph'].batch)
        x = self.merge_features(x)

        # change the shape of x from (batch, num_nodes, node_dim) to (batch, node_dim, num_nodes)
        x = x.permute(0, 2, 1)

        x, indices, commit_loss = self.vqvae(x)
        return x


def prepare_models(configs, logging, accelerator):
    gvp = GVPEncoder(configs=configs)
    vqvae = VQVAE(
        input_dim=configs.model.struct_encoder.node_h_dim[0],
        latent_dim=configs.model.vqvae.vector_quantization.dim,
        codebook_size=configs.model.vqvae.vector_quantization.codebook_size,
        decay=configs.model.vqvae.vector_quantization.decay,
        configs=configs
    )
    gvp_vqvae = GVPVQVAE(gvp, vqvae, configs)

    if accelerator.is_main_process:
        print_trainable_parameters(gvp_vqvae, logging, 'VQ-VAE')

    return gvp_vqvae


if __name__ == '__main__':
    import yaml
    import tqdm
    from utils import load_configs
    from torch.utils.data import DataLoader

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
        residue_level_feature = test_model(batch)
        print(residue_level_feature.shape)
        break
