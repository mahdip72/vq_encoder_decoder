from transformers import EsmTokenizer, EsmModel
import torch
import pandas as pd


def get_protein_embedding(sequence):
    # Load the tokenizer and model
    model_name = "facebook/esm2_t12_35M_UR50D"  # Example model, change as needed
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)

    # Tokenize the sequence
    inputs = tokenizer(sequence, return_tensors="pt")

    # Get the embeddings
    with torch.inference_mode():
        outputs = model(**inputs)
        embeddings = outputs.pooler_output

    return embeddings


def get_unique_kinases(kinase_df):
    """
    Get the unique kinase sequences from a dataset.
    :param kinase_df: DataFrame of kinase data
    :return kinase_dict: dictionary of unique name:sequence kinase pairs
    """
    # Drop duplicates to get unique kinase samples
    unique_df = kinase_df.drop_duplicates(subset='kinase')

    # Create a dictionary with kinase as keys and kinase_sequence as values
    kinase_dict = unique_df.set_index('kinase')['kinase_sequence'].to_dict()

    return kinase_dict


if __name__ == '__main__':
    # Example usage
    protein_sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAYVLSMSPARGCVTRDCRVCTRVYADRTKFGINPQTFRYYTDRVRFDG"  # Replace with your sequence
    embedding = get_protein_embedding(protein_sequence)
    print(embedding.shape)

    data_path = "../../data/valid_filtered.csv"
    kinase_df = pd.read_csv(data_path)

    kinase_dict = get_unique_kinases(kinase_df)
    print(kinase_dict.keys())