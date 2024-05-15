import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from vector_quantize_pytorch import VectorQuantize
from data import *
from model import *


# class LinearEncoder(nn.Module):
#     def __init__(self, input_dim, hidden_dim, latent_dim):
#         super(LinearEncoder, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, latent_dim)
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
#
#
# class LinearDecoder(nn.Module):
#     def __init__(self, latent_dim, hidden_dim, output_dim):
#         super(LinearDecoder, self).__init__()
#         self.fc1 = nn.Linear(latent_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, output_dim)
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x


class LinearVQVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, num_embeddings, commitment_cost):
        super(LinearVQVAE, self).__init__()


        self.encoder = nn.Linear(input_dim, latent_dim)
        nn.init.xavier_uniform_(self.encoder.weight)

        # self.encoder = LinearEncoder(input_dim, hidden_dim, latent_dim)

        self.vector_quantizer = VectorQuantize(
            dim=latent_dim,
            codebook_size=128,
            decay=0.8,
            commitment_weight=1.0
        )
        self.decoder = nn.Linear(latent_dim, input_dim)
        nn.init.xavier_uniform_(self.decoder.weight)

        # self.decoder = LinearDecoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x, return_vq_only=False):
        x = self.encoder(x)
        x, indices, commit_loss = self.vector_quantizer(x)

        if return_vq_only:
            return x, indices, commit_loss

        x = self.decoder(x)
        return x, indices, commit_loss


# Example usage with your existing GVP model
embedding_dim = 100  # Example size, adjust based on your model
hidden_dim = 256
latent_dim = 64
num_embeddings = 512
commitment_cost = 0.25
vqvae = LinearVQVAE(
    input_dim=embedding_dim,
    latent_dim=latent_dim,
    num_embeddings=num_embeddings,
    commitment_cost=commitment_cost
)


optimizer = optim.Adam(vqvae.parameters(), lr=1e-3)
num_epochs = 5
dataset_path = './data/h5'  # test for piece of data
dataset = ProteinGraphDataset(dataset_path)

test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=custom_collate)

for epoch in range(num_epochs):
    for pdb_data in test_dataloader:
        model = prepare_models()
        embeddings = model(pdb_data)
        graph_embeddings = embeddings[0]
        recon_embeddings, _, commit_loss = vqvae(graph_embeddings)

        recon_loss = nn.MSELoss()(recon_embeddings, graph_embeddings)
        loss = recon_loss + commit_loss

        optimizer.zero_grad()
        recon_loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {recon_loss.item()}')

# Using the trained VQ-VAE to reconstruct PDB data
vqvae.eval()  # Set the model to evaluation mode

reconstructed_pdbs = []
with torch.no_grad():
    for pdb_data in test_dataloader:
        model = prepare_models()
        embeddings = model(pdb_data)
        recon_embeddings, _, _ = vqvae(embeddings[0])
        reconstructed_pdbs.append(recon_embeddings)


def embeddings_to_pdb(embeddings):
    # Convert embeddings back to PDB format (implementation depends on your specific case)
    pdb_data = some_conversion_function(embeddings)
    return pdb_data


# Apply the conversion function to each reconstructed embedding
# final_reconstructed_pdbs = [embeddings_to_pdb(recon_emb) for recon_emb in reconstructed_pdbs]
#
# Save or process final_reconstructed_pdbs as needed
# for pdb in final_reconstructed_pdbs:
#     save_pdb(pdb, "output_path")

print('done')