from transformers import EsmTokenizer, EsmModel
import torch
import pandas as pd
from copy import deepcopy


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


def get_many_embeddings(prot_dict):
    """
    Extract the embeddings of a dictionary of protein sequences.
    :param prot_dict: dictionary of protein sequences (name: sequence)
    :return new_prot_dict: dictionary of protein embeddings (name: embedding)
    """
    new_prot_dict = deepcopy(prot_dict)
    for name in new_prot_dict:
        prot_embedding = get_protein_embedding(new_prot_dict[name])
        new_prot_dict[name] = prot_embedding
    return new_prot_dict

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
    df = pd.read_csv(data_path)

    kinase_seq_dict = get_unique_kinases(df)
    kinase_embed_dict = get_many_embeddings(kinase_seq_dict)
    print(len(kinase_embed_dict))