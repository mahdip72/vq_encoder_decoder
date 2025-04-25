import torch.nn as nn
import torch
from graphein.protein.resi_atoms import PROTEIN_ATOMS, STANDARD_AMINO_ACIDS, STANDARD_AMINO_ACID_MAPPING_1_TO_3
from models.encoders import GVPTransformerEncoderWrapper
from models.vqvae import VQVAETransformer
from models.decoders import GCPNetDecoder, GeometricDecoder
from gcpnet.models.gcpnet import GCPNetModel
# from gcpnet.models.vqvae import CategoricalMixture, PairwisePredictionHead, RegressionHead
# from gcpnet.utils.structure.predicted_aligned_error import compute_predicted_aligned_error, compute_tm
from utils.utils import print_trainable_parameters
from models.utils import merge_features, separate_features


class SuperModel(nn.Module):
    def __init__(self, encoder, vqvae, decoder, configs):
        super(SuperModel, self).__init__()
        self.encoder = encoder
        self.vqvae = vqvae
        self.decoder = decoder

        self.configs = configs
        self.max_length = configs.model.max_length

    def pretrained_gcpnet_batch_preprocessing(self, one_batch):
        return self.encoder.featurise(one_batch["graph"])

    def forward(self, batch):
        dtype = batch["graph"].x_bb.dtype
        device = batch["graph"].x_bb.device
        batch_index = batch['graph'].batch

        # if self.configs.model.encoder.name == "gcpnet" and self.configs.model.encoder.pretrained.enabled:
        #     batch = self.pretrained_gcpnet_batch_preprocessing(batch)

        if self.configs.model.encoder.name == "gcpnet":
            if self.configs.model.encoder.pretrained.enabled:
                with torch.autocast(device_type=device.type, enabled=False, cache_enabled=False):
                    model_output = self.encoder(batch['graph'])
                x = model_output["node_embedding"].type(dtype)

            else:
                _, x, _ = self.encoder(batch['graph'])

            x = separate_features(x, batch_index)

        else:
            x = self.encoder(batch, output_logits=False)

        x, mask, batch_indices, x_slice_indices = merge_features(x, self.max_length)

        x, indices, commit_loss = self.vqvae(x, mask)

        x = self.decoder(x, mask, batch_indices, x_slice_indices)

        # return x, indices, commit_loss
        return x, torch.Tensor([0]).to(x[0].device), torch.Tensor([0]).to(x[0].device)


def prepare_model_vqvae(configs, logger, accelerator, **kwargs):
    if configs.model.encoder.name == "gcpnet":
        if not configs.model.encoder.pretrained.enabled:
            encoder = GCPNetModel(module_cfg=kwargs["encoder_configs"].module_cfg,
                                  model_cfg=kwargs["encoder_configs"].model_cfg,
                                  layer_cfg=kwargs["encoder_configs"].layer_cfg,
                                  configs=kwargs["encoder_configs"])

        else:
            from proteinworkshop import register_custom_omegaconf_resolvers
            from omegaconf import OmegaConf
            from proteinworkshop.models.base import BenchMarkModel

            register_custom_omegaconf_resolvers()

            pretrained_config = OmegaConf.load(configs.model.encoder.pretrained.config_path)
            pretrained_config.decoder.disable = True
            encoder = BenchMarkModel.load_from_checkpoint(configs.model.encoder.pretrained.checkpoint_path,
                                                          strict=False,
                                                          cfg=pretrained_config)

    elif configs.model.encoder.name == "gvp_transformer":
        encoder = GVPTransformerEncoderWrapper(output_logits=False, finetune=True)
    else:
        raise ValueError("Invalid encoder model specified!")

    vqvae = VQVAETransformer(
        latent_dim=configs.model.vqvae.vector_quantization.dim,
        codebook_size=configs.model.vqvae.vector_quantization.codebook_size,
        decay=configs.model.vqvae.vector_quantization.decay,
        configs=configs
    )

    if configs.model.decoder == "gcpnet":
        decoder = GCPNetDecoder(configs, decoder_configs=kwargs["decoder_configs"])
    elif configs.model.decoder == "geometric_decoder":
        decoder = GeometricDecoder(configs, decoder_configs=kwargs["decoder_configs"])
    else:
        raise ValueError("Invalid decoder model specified!")

    vqvae = SuperModel(encoder, vqvae, decoder, configs)

    vqvae = nn.SyncBatchNorm.convert_sync_batchnorm(vqvae)

    if accelerator.is_main_process:
        print_trainable_parameters(vqvae, logger, 'SuperVQVAE')

    return vqvae


if __name__ == '__main__':
    import yaml
    import tqdm
    from utils.utils import load_configs, get_dummy_logger
    from torch.utils.data import DataLoader
    from accelerate import Accelerator
    from data.dataset import custom_collate, GCPNetDataset

    config_path = "../configs/config_gcpnet.yaml"

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    test_configs = load_configs(config_file)

    test_logger = get_dummy_logger()
    test_accelerator = Accelerator()

    test_model = prepare_model_vqvae(test_configs, test_logger, test_accelerator)

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
