from transformers import EsmTokenizer, EsmModel
import torch
import torch.nn.functional as F
from torch.nn import CosineSimilarity
import pandas as pd
from copy import deepcopy
from tqdm import tqdm


def get_protein_embedding(sequence, model, tokenizer):
    # Tokenize the sequence
    inputs = tokenizer(sequence, return_tensors="pt")
    # move inputs to the GPU
    inputs = {key: value.cuda() for key, value in inputs.items()}

    # Ensure the model is in FP16
    model.half()

    # Get the embeddings with mixed precision
    with torch.cuda.amp.autocast():
        with torch.inference_mode():
            outputs = model(**inputs)
            embeddings = outputs.pooler_output.cpu()

    return embeddings


def get_many_embeddings(prot_dict, model, tokenizer, max_length=2048, progress_bar=True):
    """
    Extract the embeddings of a dictionary of protein sequences.
    :param prot_dict: dictionary of protein sequences (name: sequence)
    :param model: pretrained model to use for embedding extraction
    :param tokenizer: tokenizer to use for embedding extraction
    :param max_length: maximum sequence length to consider
    :param progress_bar: if True, display a progress bar
    :return embedding_tensor: [N_samples, embedding_size] tensor of protein embeddings (name: embedding)
    """
    embedding_list = []
    progress_bar = tqdm(prot_dict, disable=not progress_bar)
    progress_bar.set_description("Extracting embeddings")

    for name in progress_bar:
        sequence = prot_dict[name]
        # Trim sequences that are longer than 2048 residues
        sequence = sequence[:max_length]
        prot_embedding = get_protein_embedding(sequence, model, tokenizer)
        embedding_list.append(prot_embedding.squeeze(0))

    embedding_tensor = torch.stack(embedding_list)
    return embedding_tensor


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


def calc_embedding_distance_map(embeddings, distance_type='euclidean'):
    """
    Calculate the distance map for a tensor of embeddings.
    :param embeddings: [N_samples, embedding_size] tensor of embeddings
    :param distance_type: 'euclidean' or 'cosine'
    :return distance_map: [N_samples, N_samples] distance map of embeddings
    """

    if distance_type == 'euclidean':
        # Expand dimensions to compute pairwise differences
        expanded_embeddings = embeddings.unsqueeze(1) - embeddings.unsqueeze(0)
        # Calculate Euclidean distances
        distance_map = torch.norm(expanded_embeddings, p=2, dim=2)

    elif distance_type == 'cosine':
        # Create a cosine similarity module
        cos = torch.nn.CosineSimilarity(dim=2, eps=1e-8)
        # Expand dimensions to compute pairwise cosine similarity
        embeddings1 = embeddings.unsqueeze(1).expand(-1, embeddings.size(0), -1)
        embeddings2 = embeddings.unsqueeze(0).expand(embeddings.size(0), -1, -1)
        # Calculate cosine similarities
        cosine_similarity = cos(embeddings1, embeddings2)
        # Convert cosine similarity to cosine distance
        distance_map = 1 - cosine_similarity

    else:
        raise ValueError('Distance type must be either "euclidean" or "cosine"')

    return distance_map


def get_negative_kinase_name_pairs(kinase_df, model, tokenizer, max_length=2048, distance_type='euclidean',
                                   progress_bar=True):
    """
    Get k negative kinase name pairs from a dataframe.
    :param kinase_df: DataFrame of kinase data
    :param model: pretrained model to use for embedding extraction
    :param tokenizer: tokenizer to use for embedding extraction
    :param max_length: maximum sequence length to consider
    :param distance_type: 'euclidean' or 'cosine'
    :param progress_bar: if True, display a progress bar
    :return
        kinase_name_pairs: dictionary of kinase name pairs (name:list[(nearest name, distance)])
        kinase_sequences: dictionary of kinase sequences (name:sequence)
    """
    kinase_dict = get_unique_kinases(kinase_df)
    embeddings = get_many_embeddings(kinase_dict, model, tokenizer, progress_bar=progress_bar, max_length=max_length)
    distance_map = calc_embedding_distance_map(embeddings, distance_type=distance_type)

    # Get the sorted distances and corresponding indices
    sorted_distances, indices = torch.sort(distance_map, dim=1)

    kinase_name_list = list(kinase_dict.keys())
    kinase_name_pairs = deepcopy(kinase_dict)

    for i, name in enumerate(kinase_name_pairs):
        neighbor_names_distances = []
        for j in range(1, len(kinase_name_list)):
            neighbor_idx = indices[i, j]
            distance = sorted_distances[i, j].item()
            neighbor_names_distances.append((kinase_name_list[neighbor_idx], round(distance, 4)))
        kinase_name_pairs[name] = neighbor_names_distances

    return kinase_name_pairs, kinase_dict


def prepare_negative_samples(dataset, name_pairs, kinase_seqs, k=20):
    """
    Prepare negative samples for a dataset.
    :param dataset: DataFrame of protein data
    :param name_pairs: dictionary of kinase name pairs (name:list[(nearest name, distance)])
    :param kinase_seqs: dictionary of kinase sequences (name:sequence)
    :param k: number of nearest neighbors to sample
    :return negative_samples: list of negative samples

    Returns: negative_samples: list of negative samples (substrate, kinase)
    """
    grouped_dataset = dataset.groupby('Uniprotid')

    # Initialize a list to store negative samples with additional columns
    negative_samples = []

    # Iterate through each unique protein
    for uniprot_id, group in grouped_dataset:
        # Get the unique protein sequence (substrate)
        unique_substrate = group['Sequence'].iloc[0]

        # Get all kinases and their families that interact with this protein
        kinase_labels = group['kinase'].unique()

        temp_list = []
        for kinase_label in kinase_labels:
            # Get the nearest neighbors for this kinase
            all_neighbors = name_pairs[kinase_label]

            for neighbor in all_neighbors:
                if neighbor[0] not in kinase_labels:
                    temp_list.append((neighbor[0], neighbor[1]))

            # Get the k nearest neighbors
            k_nearest_neighbors = temp_list[:k]

            # Create negative pairs with the sampled kinase sequences
            for neighbor_name, _ in k_nearest_neighbors:
                negative_samples.append([unique_substrate, kinase_seqs[neighbor_name], neighbor_name, 'disconnect'])

    return negative_samples


def compute_negative_samples(df, max_length=4096, k=20):
    """
    Compute negative kinase samples from a dataset.
    :param df: DataFrame of protein data
    :param max_length: maximum sequence length to consider for embedding extraction
    :param k: number of nearest neighbors to sample

    Returns:
    name_pairs: dictionary of kinase name pairs (name:list[k nearest names])
    kinase_seqs: dictionary of kinase sequences (name:sequence)
    """
    # Load the tokenizer and model
    model_name = "facebook/esm2_t33_650M_UR50D"  # Example model, change as needed
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name).to('cuda:0')

    name_pairs, kinase_seqs = get_negative_kinase_name_pairs(df, model, tokenizer,
                                                             distance_type='euclidean', progress_bar=False,
                                                             max_length=max_length)

    del model, tokenizer
    torch.cuda.empty_cache()
    negative_samples = prepare_negative_samples(df, name_pairs, kinase_seqs, k)
    return negative_samples, kinase_seqs


if __name__ == '__main__':
    # Load the tokenizer and model
    main_model_name = "facebook/esm2_t33_650M_UR50D"  # Example model, change as needed
    main_tokenizer = EsmTokenizer.from_pretrained(main_model_name)
    main_model = EsmModel.from_pretrained(main_model_name).cuda()

    data_path = "/mnt/hdd8/mehdi/datasets/Joint_training/kinase/train_filtered.csv"
    main_df = pd.read_csv(data_path)

    test_name_pairs, test_kinase_seqs = get_negative_kinase_name_pairs(main_df, main_model, main_tokenizer,
                                                                       distance_type='euclidean', progress_bar=True,
                                                                       max_length=4096)
    print(test_name_pairs)
    print(test_kinase_seqs)
