import torch.nn as nn
import torch
import torch_geometric
from vector_quantize_pytorch import VectorQuantize
from gcpnet.models import GCPNetModel
from gcpnet.utils import _normalize, batch_orientations
from utils.utils import print_trainable_parameters


class VQVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, codebook_size, decay, configs):
        super(VQVAE, self).__init__()

        self.max_length = configs.model.max_length
        self.pos_embed = nn.Parameter(torch.randn(1, self.max_length, input_dim) * .02)

        self.encoder_layers = nn.Sequential(
            nn.Conv1d(input_dim, input_dim, 1),
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
        )

    def drop_positional_encoding(self, embedding):
        embedding = embedding + self.pos_embed
        return embedding

    def forward(self, x, return_vq_only=False):
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


class VQVAE3DTransformer(nn.Module):
    def __init__(self, codebook_size, decay, configs):
        super(VQVAE3DTransformer, self).__init__()

        self.max_length = configs.model.max_length
        self.top_k = configs.model.struct_encoder.top_k
        self.encoder_dim = configs.model.vqvae.encoder.dimension
        self.decoder_dim = configs.model.vqvae.decoder.dimension

        self.chi_init_dim = configs.model.vqvae.decoder.chi_init_dimension
        self.xi_init_dim = configs.model.vqvae.decoder.xi_init_dimension

        # Projecting the input to the dimension expected by the Transformer
        self.input_projection = nn.Sequential(
            nn.Conv1d(100, self.encoder_dim, 1),
        )

        self.pos_embed_encoder = nn.Parameter(torch.randn(1, self.max_length, self.encoder_dim) * .02)

        # todo: fix batch first issue
        # Transformer Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            batch_first=True,
            d_model=self.encoder_dim,
            nhead=configs.model.vqvae.encoder.num_heads,
            dim_feedforward=configs.model.vqvae.encoder.dim_feedforward,
            activation=configs.model.vqvae.encoder.activation_function,
            dropout=0.0
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=configs.model.vqvae.encoder.num_layers)

        # Projecting the output of the Transformer to the dimension expected by the VQ layer
        self.vq_in_projection = nn.Sequential(nn.Conv1d(self.encoder_dim, configs.model.vqvae.vector_quantization.dim, 1))

        # Vector Quantizer
        self.vector_quantizer = VectorQuantize(
            dim=configs.model.vqvae.vector_quantization.dim,
            codebook_size=codebook_size,
            decay=decay,
            commitment_weight=1.0,
        )

        # Projecting the output of the VQ layer back to the decoder dimension
        self.vq_out_projection = nn.Sequential(nn.Conv1d(configs.model.vqvae.vector_quantization.dim, self.decoder_dim, 1))

        self.pos_embed_decoder = nn.Parameter(torch.randn(1, self.max_length, self.encoder_dim) * .02)

        # todo: fix batch first issue
        # Transformer Decoder
        self.decoder_layer = nn.TransformerEncoderLayer(
            batch_first=True,
            d_model=self.decoder_dim,
            nhead=configs.model.vqvae.decoder.num_heads,
            dim_feedforward=configs.model.vqvae.decoder.dim_feedforward,
            activation=configs.model.vqvae.encoder.activation_function,
            dropout=0.0
        )
        self.decoder = nn.TransformerEncoder(self.decoder_layer, num_layers=configs.model.vqvae.decoder.num_layers)

        # GCPNet output (positions) projection # 
        configs.model.struct_encoder.module_cfg.predict_backbone_positions = True
        configs.model.struct_encoder.module_cfg.predict_node_rep = False
        configs.model.struct_encoder.model_cfg.num_layers = 1

        output_projection_layers = []

        # Embedding layer
        configs.model.struct_encoder.use_rotary_embeddings = False
        configs.model.struct_encoder.use_positional_embeddings = False

        configs.model.struct_encoder.use_foldseek = False
        configs.model.struct_encoder.use_foldseek_vector = False

        configs.model.struct_encoder.model_cfg.h_input_dim = self.decoder_dim
        configs.model.struct_encoder.model_cfg.chi_input_dim = self.chi_init_dim
        configs.model.struct_encoder.model_cfg.e_input_dim = self.decoder_dim * 2 + configs.model.struct_encoder.module_cfg.num_rbf
        configs.model.struct_encoder.model_cfg.xi_input_dim = self.xi_init_dim

        output_projection_layers.append(
            GCPNetModel(
                module_cfg=configs.model.struct_encoder.module_cfg,
                model_cfg=configs.model.struct_encoder.model_cfg,
                layer_cfg=configs.model.struct_encoder.layer_cfg,
                configs=configs,
                backbone_key="x_bb",
            )
        )

        # Output projection layers
        configs.model.struct_encoder.model_cfg.h_input_dim = configs.model.struct_encoder.model_cfg.h_hidden_dim
        configs.model.struct_encoder.model_cfg.chi_input_dim = self.chi_init_dim
        configs.model.struct_encoder.model_cfg.e_input_dim = configs.model.struct_encoder.model_cfg.h_hidden_dim * 2 + configs.model.struct_encoder.module_cfg.num_rbf
        configs.model.struct_encoder.model_cfg.xi_input_dim = self.xi_init_dim

        output_projection_layers.extend(
            [
                GCPNetModel(
                    module_cfg=configs.model.struct_encoder.module_cfg,
                    model_cfg=configs.model.struct_encoder.model_cfg,
                    layer_cfg=configs.model.struct_encoder.layer_cfg,
                    configs=configs,
                    backbone_key="x_bb",
                )
                for _ in range(
                    configs.model.struct_encoder.model_cfg.num_bb_update_layers
                )
            ]
        )

        self.output_projections = nn.ModuleList(output_projection_layers)

    @staticmethod
    def drop_positional_encoding(embedding, pos_embed):
        embedding = embedding + pos_embed
        return embedding
    
    def construct_black_hole_graph_batch(self, feats, mask, batch_indices, x_slice_index):
        batch_num_nodes = mask.sum().item()
        device, dtype = feats.device, feats.dtype

        h = feats[mask]
        mask = torch.ones((batch_num_nodes,), device=device, dtype=torch.bool)

        x_bb = torch.randn((batch_num_nodes, 3, 3), device=device, dtype=dtype)
        x = x_bb[:, 1]

        chi = batch_orientations(x, x_slice_index)

        # Initialize the structural graph with scalar node feature topological information
        edge_index = torch_geometric.nn.knn_graph(
            h,
            self.top_k,
            batch=batch_indices,
            loop=False,
            flow="source_to_target",
            cosine=True,
        )

        e = torch.cat([h[edge_index[0]], h[edge_index[1]]], dim=-1)
        xi = _normalize(x[edge_index[0]] - x[edge_index[1]]).unsqueeze(-2)

        batch = torch_geometric.data.Batch(
            x=x,
            x_bb=x_bb,
            seq=None,
            h=h,
            chi=chi,
            e=e,
            xi=xi,
            edge_index=edge_index,
            mask=mask,
            batch=batch_indices,
            x_slice_index=x_slice_index,
        )

        return batch

    def construct_updated_graph_batch(self, batch):
        x = batch.x_bb[:, 1]

        chi = batch_orientations(x, batch.x_slice_index)

        edge_index = torch_geometric.nn.knn_graph(
            x,
            self.top_k,
            batch=batch.batch,
            loop=False,
            flow="source_to_target",
            cosine=False,
        )

        e = torch.cat([batch.h[edge_index[0]], batch.h[edge_index[1]]], dim=-1)
        xi = _normalize(x[edge_index[0]] - x[edge_index[1]]).unsqueeze(-2)

        new_batch = torch_geometric.data.Batch(
            x=x,
            x_bb=batch.x_bb,
            seq=None,
            h=batch.h,
            chi=chi,
            e=e,
            xi=xi,
            edge_index=edge_index,
            mask=batch.mask,
            batch=batch.batch,
            x_slice_index=batch.x_slice_index,
        )

        return new_batch

    def forward(self, x, mask, batch_indices, x_slice_index, return_vq_only=False):
        # Apply input projection
        x = x.permute(0, 2, 1)
        for layer in self.input_projection:
            x = layer(x)

        # Permute for Transformer [batch, sequence, feature]
        x = x.permute(0, 2, 1)

        # Apply positional encoding to encoder
        x = self.drop_positional_encoding(x, self.pos_embed_encoder)

        # Encoder
        x = self.encoder(x)
        x = x.permute(0, 2, 1)

        # Apply qv_in_projection
        x = self.vq_in_projection(x)

        x = x.permute(0, 2, 1)
        x, indices, commit_loss = self.vector_quantizer(x)
        x = x.permute(0, 2, 1)

        # if return_vq_only:
        #     x = x.permute(0, 2, 1)
        #     return x, indices, commit_loss

        # Apply vq_out_projection
        x = self.vq_out_projection(x)

        x = x.permute(0, 2, 1)
        # Apply positional encoding to decoder
        x = self.drop_positional_encoding(x, self.pos_embed_decoder)

        # Decoder
        x = self.decoder(x)

        # Apply output projections in a sparse graph format
        batch = self.construct_black_hole_graph_batch(x, mask, batch_indices, x_slice_index)
        _, batch.h, batch.x_bb = self.output_projections[0](batch)

        for proj in self.output_projections[1:]:
            batch = self.construct_updated_graph_batch(batch)
            _, batch.h, batch.x_bb = proj(batch)

        # Pad the output back into the original shape
        x_list = GCPNetVQVAE.separate_features(batch.x_bb.view(-1, 9), batch.batch)
        x, *_ = GCPNetVQVAE.merge_features(x_list, self.max_length)

        # return x, indices, commit_loss
        return x, torch.Tensor([0]).to(x.device), torch.Tensor([0]).to(x.device)


class GCPNetVQVAE(nn.Module):
    def __init__(self, gcpnet, vqvae, configs):
        super(GCPNetVQVAE, self).__init__()
        self.gcpnet = gcpnet
        self.vqvae = vqvae

        self.configs = configs
        self.max_length = configs.model.max_length

    @staticmethod
    def separate_features(batched_features, batch):
        # Split the features tensor into separate tensors for each graph
        features_list = torch_geometric.utils.unbatch(batched_features, batch)
        return features_list

    @staticmethod
    def merge_features(features_list, max_length):
        # Pad tensors and create masks
        device = features_list[0].device

        padded_tensors = []
        masks = []
        slice_lengths = [0]
        for t in features_list:
            # Create mask of size (original_length,)
            mask = torch.ones(t.size(0), device=t.device)

            if t.size(0) < max_length:
                size_diff = max_length - t.size(0)
                pad = torch.zeros(size_diff, t.size(1), device=t.device)
                t_padded = torch.cat([t, pad], dim=0)

                # Pad mask with zeros for the padded positions
                mask = torch.cat([mask, torch.zeros(size_diff, device=t.device)], dim=0)
            else:
                t_padded = t[:max_length, :]
                mask = mask[:max_length]  # Trim mask if necessary

            padded_tensors.append(t_padded.unsqueeze(0))  # Add an extra dimension for concatenation
            masks.append(mask.unsqueeze(0))  # Add an extra dimension to mask as well
            slice_lengths.append(min(max_length, t.size(0)))

        # Concatenate tensors and masks
        padded_features = torch.cat(padded_tensors, dim=0)
        mask = torch.cat(masks, dim=0).bool()

        # Flatten padded features and mask
        flat_mask = mask.view(-1)  # Shape: (num_batches * max_length)

        # Create batch assignment tensor
        num_batches = padded_features.size(0)
        batch_indices = torch.arange(num_batches, device=device).unsqueeze(1).repeat(1, max_length).view(-1)  # Shape: (num_batches * max_length)
        valid_batch_indices = batch_indices[flat_mask.bool()]  # Filter based on mask

        slice_indices = torch.cumsum(torch.tensor(slice_lengths), dim=0)
        slice_indices[-1] -= 1  # Decrement the last element

        return padded_features, mask, valid_batch_indices, slice_indices

    def forward(self, batch):
        _, x, _ = self.gcpnet(batch=batch['graph'])

        x = self.separate_features(x, batch['graph'].batch)
        x, mask, batch_indices, x_slice_indices = self.merge_features(x, self.max_length)

        x, indices, commit_loss = self.vqvae(x, mask, batch_indices, x_slice_indices)
        return x, indices, commit_loss


def prepare_models_gcpnet_vqvae(configs, logger, accelerator):
    gcpnet = GCPNetModel(module_cfg=configs.model.struct_encoder.module_cfg,
                         model_cfg=configs.model.struct_encoder.model_cfg,
                         layer_cfg=configs.model.struct_encoder.layer_cfg,
                         configs=configs)
    # vqvae = VQVAE(
    #     input_dim=configs.model.struct_encoder.node_h_dim[0],
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
    gcpnet_vqvae = GCPNetVQVAE(gcpnet, vqvae, configs)

    if accelerator.is_main_process:
        print_trainable_parameters(gcpnet_vqvae, logger, 'GCPNet-VQ-VAE')

    return gcpnet_vqvae


if __name__ == '__main__':
    import yaml
    import tqdm
    from utils.utils import load_configs_gcpnet, get_dummy_logger
    from torch.utils.data import DataLoader
    from accelerate import Accelerator
    from data.dataset import custom_collate, GCPNetDataset

    config_path = "../configs/config_gcpnet.yaml"

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    test_configs = load_configs_gcpnet(config_file)

    test_logger = get_dummy_logger()
    accelerator = Accelerator()

    test_model = prepare_models_gcpnet_vqvae(test_configs, test_logger, accelerator)
    # print(test_model)
    print("Model loaded successfully!")

    dataset = GCPNetDataset(test_configs.train_settings.data_path,
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
        graph = batch["graph"]
        output, _, _ = test_model(batch)
        print(output.shape)
        break
