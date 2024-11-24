import torch
from torch import nn
import esm


class GVPTransformerEncoderWrapper(nn.Module):
    def __init__(self, last_layers_trainable=1, finetune=False, output_logits=False):
        super().__init__()
        _model, _alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        self.encoder = _model.encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        if finetune:
            for param in self.encoder.layers.parameters():
                param.requires_grad = True

            # for layer in self.encoder.layers[-last_layers_trainable:]:
            #     for param in layer.parameters():
            #         param.requires_grad = True

        # alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        # if output_logits:
        #     self.embed_dim = self.encoder.embed_tokens.embedding_dim
        #     self.out_proj = nn.Linear(self.embed_dim, len(alphabet))

    def forward(self, batch, output_logits=False, **kwargs):
        return_all_hiddens = False

        # padding_mask = torch.isnan(batch['target_coords'][:, :, 0, 0])
        padding_mask = batch['masks'] == 0
        coords = batch['input_coordinates'].reshape(batch['input_coordinates'].shape[0], -1, 4, 3)
        coords = coords[:, :, :3, :]
        confidence = torch.ones(batch['input_coordinates'].shape[0:2]).to(coords.device)
        encoder_out = self.encoder(coords, padding_mask, confidence, return_all_hiddens=return_all_hiddens)
        # encoder_out['encoder_out'][0] = torch.transpose(encoder_out['encoder_out'][0], 0, 1)
        encoder_out['feats'] = encoder_out['encoder_out'][0].transpose(0, 1)
        # if output_logits:
        #     logits = self.out_proj(encoder_out['feats'])
        #     return logits, encoder_out['feats']
        # else:
        return encoder_out['feats']


if __name__ == '__main__':
    model = GVPTransformerEncoderWrapper(output_logits=True, finetune=True, last_layers_trainable=8)

    # Print percentage of trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params: ,}")
    print(f"Trainable parameters: {trainable_params: ,}")
    print(f"Percentage of trainable parameters: {trainable_params / total_params * 100:.2f}%")

    batch = {
        'input_coordinates': torch.randn(1, 100, 12),
        'masks': torch.ones(1, 100),
    }
    output = model(batch, output_logits=True)
    print(output.shape)
    print(output.dtype)
    print(output.device)