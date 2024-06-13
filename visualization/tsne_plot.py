import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.manifold import TSNE
import os
import math


def scatter_labeled_z(z_batch, colors, filename="test_plot"):
    fig = plt.gcf()
    plt.switch_backend('Agg')
    fig.set_size_inches(3.5, 3.5)
    plt.clf()
    for n in range(z_batch.shape[0]):
        result = plt.scatter(z_batch[n, 0], z_batch[n, 1], color=colors[n], s=50, marker="o", edgecolors='none')

    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    plt.xticks([], [])
    plt.yticks([], [])
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.savefig(filename)
    plt.close()
    # pylab.show()


def get_mask(seqs, max_len, pad_value=255):
    bsz = len(seqs)
    # mask_matrix = np.full((bsz,max_len), False, dtype=bool)
    mask_matrix = np.full((bsz, max_len + 2), False, dtype=bool)
    for i in range(bsz):
        leng = len(seqs[i])
        if leng >= max_len:
            mask_matrix[i, :] = True
        else:
            pad_len = max_len - leng
            mask_matrix[i, :leng] = True

        mask_matrix[i, 0] = True
        mask_matrix[i, -1] = True

    return mask_matrix


def prepare_data(fasta_file, npz_file):
    fasta_file = open(fasta_file)

    labels = {}
    for line in fasta_file:
        # labels.append(line[1:].split("|")[0].strip())
        protid = line.strip().split("|")[-1].split("/")[0]
        labels[protid] = line[1:].split("|")[0].strip()
        line = next(fasta_file)

    residue_feature = np.load(npz_file, allow_pickle=True)
    gvp_ids = [item for item in residue_feature['ids']]
    # 1469,508,1280
    labellist = []
    rep_label = {}
    seq_embeddings = []
    for index, protid in enumerate(gvp_ids):
        labellist.append(labels[protid])
        # rep_label[labels[protid]]=residue_feature['rep'][index]
        seq_embeddings.append(residue_feature['rep'][index])

    # rep_label['1.10.260.60'].shape
    # seqlen_label['1.10.260.60']

    seq_embeddings = np.array(seq_embeddings)

    return seq_embeddings, labels, labellist


def compute_plot(fasta_file, npz_file, save_path):
    seq_embeddings, labels, labellist = prepare_data(fasta_file, npz_file)

    out_figure_path = save_path

    mdel = TSNE(n_components=2, random_state=0, method='exact')
    print("Projecting to 2D by TSNE\n")
    z_tsne_seq = mdel.fit_transform(seq_embeddings)

    scores = []
    tol_archi_seq = {"3.30", "3.40", "1.10", "3.10", "2.60"}
    tol_fold_seq = {"1.10.10", "3.30.70", "2.60.40", "2.60.120", "3.40.50"}

    for digit_num in [1, 2, 3]:  # first number of digits
        color = []
        colorid = []
        keys = {}
        colorindex = 0
        if digit_num == 1:
            ct = ["blue", "red", "black", "yellow", "orange", "green", "olive", "gray", "magenta", "hotpink", "pink",
                  "cyan", "peru", "darkgray", "slategray", "gold"]
        else:
            ct = ["black", "yellow", "orange", "green", "olive", "gray", "magenta", "hotpink", "pink", "cyan", "peru",
                  "darkgray", "slategray", "gold"]

        select_label = []
        select_index = []
        color_dict = {}
        for label in labellist:
            key = ".".join([x for x in label.split(".")[0:digit_num]])
            if digit_num == 2:
                if key not in tol_archi_seq:
                    continue
            if digit_num == 3:
                if key not in tol_fold_seq:
                    continue

            if key not in keys:
                keys[key] = colorindex
                colorindex += 1

            select_label.append(keys[key])
            color.append(ct[(keys[key]) % len(ct)])
            colorid.append(keys[key])
            select_index.append(labellist.index(label))
            color_dict[keys[key]] = ct[keys[key]]

        scores.append(calinski_harabasz_score(seq_embeddings[select_index], color))
        scores.append(calinski_harabasz_score(z_tsne_seq[select_index], color))
        scatter_labeled_z(z_tsne_seq[select_index], color, filename=os.path.join(out_figure_path, f"_CATH_{digit_num}.png"))
        # add kmeans
        kmeans = KMeans(n_clusters=len(color_dict), random_state=42)
        predicted_labels = kmeans.fit_predict(z_tsne_seq[select_index])
        predicted_colors = [color_dict[label] for label in predicted_labels]
        scatter_labeled_z(z_tsne_seq[select_index], predicted_colors,
                          filename=os.path.join(out_figure_path, f"_CATH_{digit_num}_kmpred.png"))
        ari = adjusted_rand_score(colorid, predicted_labels)
        scores.append(ari)

    # [818.712930095648, 2050.5056932712114, 0.47366338028523286, 273.729149424163, 623.2356946236121, 0.32052861490323026, 100.50737136096417, 240.03891170054388, 0.33852419424771096]

    class_seq = ["1", "2", "3"]
    tol_archi_seq = {"3.30", "3.40", "1.10", "3.10", "2.60"}
    tol_fold_seq = {"1.10.10", "3.30.70", "2.60.40", "2.60.120", "3.40.50"}
    for digit_num in [1, 2, 3]:
        count = 0
        for label in labels:
            key = ".".join([x for x in label.split(".")[0:digit_num]])
            if digit_num == 1:
                if key[0] not in class_seq:
                    continue
            if digit_num == 2:
                if key not in tol_archi_seq:
                    continue
            if digit_num == 3:
                if key not in tol_fold_seq:
                    continue

            count += 1

        print(count)


if __name__ == '__main__':
    compute_plot(fasta_file="Rep_subfamily_basedon_S40pdb.fa", npz_file="gvpstruture_1553.npz",
                 save_path="./plots/gvp")
    print('test plot_GVP_CATH')
