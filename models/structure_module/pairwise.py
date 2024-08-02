import torch
import torch.nn as nn


class RelativePosition(nn.Module):
    """
    This code was written adapted from Yining Hong.
    https://github.com/evelinehong/Transformer_Relative_Position_PyTorch/blob/master/relative_position.py
    """

    def __init__(self, max_relative_position, num_units):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, i_indices, j_indices):
        # range_vec_q = torch.arange(length_q)
        # range_vec_k = torch.arange(length_k)
        # distance_mat = range_vec_k[None, :] - range_vec_q[:, None]

        distance_mat = i_indices - j_indices
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat)
        embeddings = self.embeddings_table[final_mat]

        return embeddings


class MLP(nn.Module):
    """
    MLP module for Pairwise
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class Pairwise(nn.Module):
    """
    Module for computing a pairwise representation of the structure from the
    quantized sequence. Based on the algorithm described by Gaujac et al., 2024.
    """
    def __init__(self, input_dim, embedding_dim, hidden_dim):
        super(Pairwise, self).__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.linear_left = nn.Linear(input_dim, embedding_dim)
        self.linear_right = nn.Linear(input_dim, embedding_dim)
        self.relative_position = RelativePosition(input_dim, embedding_dim)
        self.mlp = MLP(input_dim=embedding_dim + embedding_dim, hidden_dim=hidden_dim, output_dim=embedding_dim)

    def forward(self, s):
        batch_size, N, _ = s.shape

        s_normalized = s
        # Layer normalization
        # s_normalized = self.layer_norm(s)

        # Linear transformations to get s_left and s_right
        s_left = self.linear_left(s_normalized)
        s_right = self.linear_right(s_normalized)

        # Compute the outer product between s_left and s_right
        k = torch.einsum("bnd,bkd->bnkd", s_left, s_right)

        # Prepare indices for relative positional encoding
        i_indices = torch.arange(N).view(N, 1).expand(N, N).flatten()
        j_indices = torch.arange(N).view(1, N).expand(N, N).flatten()

        # Compute positional encodings for all pairs (i, j)
        positional_encodings = self.relative_position(i_indices, j_indices).view(N, N, -1)
        # Shape: (batch_size, N, N, embedding_dim)
        positional_encodings = positional_encodings.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # Concatenate k with positional encodings
        k_concat = torch.cat((k, positional_encodings), dim=-1)

        # Apply MLP
        k = self.mlp(k_concat)

        return k


if __name__ == '__main__':

    # Example usage
    batch_size = 2
    N = 128  # Protein length
    input_dim = 64  # Input embedding dimension
    embedding_dim = 32  # Output embedding dimension
    hidden_dim = 64  # Hidden dimension for MLP

    # Create a batch of embeddings with shape (N, input_dim)
    s = torch.randn(batch_size, N, input_dim)
    print(s.shape)

    # Instantiate the model
    model = Pairwise(input_dim=input_dim, embedding_dim=embedding_dim, hidden_dim=hidden_dim)

    # Forward pass to get k
    k = model(s)
    print(k.shape)

    exit()
