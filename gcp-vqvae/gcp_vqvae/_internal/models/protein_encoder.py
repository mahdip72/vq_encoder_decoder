import torch
from torch import nn
from transformers import EsmModel


class ProteinEncoder(nn.Module):
    """
    ProteinEncoder wraps a pretrained protein language model (ESM) to extract per-residue embeddings.

    Args:
        logging: Logger for informational messages.
        model_name: Pretrained model identifier.
        model_type: Type of encoder to use ('esm_v2').
        quantization_4_bit: Whether to load the model in 4-bit quantized mode.

    The encoder outputs a tensor of shape (batch, seq_len, embedding_dim).
    """

    def __init__(self, logging, model_name='facebook/esm2_t12_35M_UR50D', model_type='esm_v2', quantization_4_bit=False):
        super().__init__()

        self.model_type = model_type

        if model_type == 'esm_v2':
            if quantization_4_bit:
                from transformers import BitsAndBytesConfig
                logging.info('load quantized 4-bit weights')
                # QLoRa fine-tuning:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                self.model = EsmModel.from_pretrained(model_name,
                                                      device_map="auto",
                                                      quantization_config=quantization_config)
            else:
                self.model = EsmModel.from_pretrained(model_name)

            # Freeze all layers
            for param in self.model.parameters():
                param.requires_grad = False

            for param in self.model.pooler.parameters():
                param.requires_grad = False

            for param in self.model.contact_head.parameters():
                param.requires_grad = False

            self.protein_encoder_dim = self.model.embeddings.word_embeddings.embedding_dim

        else:
            raise ValueError(f'Unknown model type: {model_type}')

    def forward(self, x):
        features = self.model(input_ids=x['input_ids'],
                              attention_mask=x['attention_mask'])
        return features.last_hidden_state
