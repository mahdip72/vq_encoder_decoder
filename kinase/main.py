from transformers import EsmTokenizer, EsmModel
import torch


def get_protein_embedding(sequence):
    # Load the tokenizer and model
    model_name = "facebook/esm2_t33_650M_UR50D"  # Example model, change as needed
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)

    # Tokenize the sequence
    inputs = tokenizer(sequence, return_tensors="pt")

    # Get the embeddings
    with torch.inference_mode():
        outputs = model(**inputs)
        embeddings = outputs.pooler_output

    return embeddings


if __name__ == '__main__':
    # Example usage
    protein_sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAYVLSMSPARGCVTRDCRVCTRVYADRTKFGINPQTFRYYTDRVRFDG"  # Replace with your sequence
    embedding = get_protein_embedding(protein_sequence)
    print(embedding.shape)
