from gvp.data import custom_collate, ProteinGraphDataset
from utils import load_configs
import yaml
from gvp.models import GVPEncoder
from torch.utils.data import DataLoader

if __name__ == '__main__':
    
    config_path = "./gvp/config_noseq.yaml" #./config_noseq.yaml
    with open(config_path) as file:
        config_file = yaml.full_load(file)
        configs = load_configs(config_file)
    
    model_struct = GVPEncoder(configs=configs)
    model_struct = model_struct.to(configs.train_settings.device)
    dataset = ProteinGraphDataset(configs.train_settings.data_path,seq_mode = configs.model.struct_encoder.use_seq.seq_embed_mode,
                                        use_rotary_embeddings=configs.model.struct_encoder.use_rotary_embeddings,
                                        use_foldseek = configs.model.struct_encoder.use_foldseek,
                                        use_foldseek_vector = configs.model.struct_encoder.use_foldseek_vector,
                                        top_k = configs.model.struct_encoder.top_k,
                                        num_rbf = configs.model.struct_encoder.num_rbf,
                                        num_positional_embeddings = configs.model.struct_encoder.num_positional_embeddings)
    
    val_loader = DataLoader(dataset, batch_size=configs.train_settings.batch_size, num_workers=0, pin_memory=True, collate_fn=custom_collate)
    struct_embeddings = []
    # existingmodel.eval()
    for batch in val_loader:
            graph = batch["graph"].to(configs.train_settings.device)
            features_struct, _ = model_struct(graph=graph)
            struct_embeddings.extend(features_struct.cpu().detach().numpy())
            print(struct_embeddings)
            break
    
    