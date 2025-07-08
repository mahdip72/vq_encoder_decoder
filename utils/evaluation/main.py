import os
from tmscore import TMscoring
import csv
import re


def extract_base_name(filename):
    """
    From any .pdb, .pdb_pred, .pdb_A_pred etc, return base name (no chain, no _pred).
    2abc_A.pdb      -> 2abc
    2xyz_pred.pdb   -> 2xyz
    1foo_B_pred.pdb -> 1foo
    """
    name = filename
    if name.endswith('.pdb'):
        name = name[:-4]
    if name.endswith('_pred'):
        name = name[:-5]
    name = re.sub(r'_[A-Z]+$', '', name)
    return name


def calculate_tm_rmsd(real_dir, pred_dir, output_csv, log_file="processing_log.txt"):
    results = []
    tm_scores = []
    rmsd_values = []
    logs = []
    real_files_map = {}

    # for real_file in os.listdir(real_dir):
    #     if not real_file.endswith('.pdb'):
    #         continue
    #
    #     # Create the predicted filename by replacing the last _<suffix> with _pred before extension
    #     name, ext = os.path.splitext(real_file)
    #     pred_file = re.sub(r'_[^_]+$', '_pred', name) + ext
    #     pred_file_path = os.path.join(pred_dir, pred_file)
    #     real_filename = os.path.join(real_dir, real_file)
    #
    #     if not os.path.isfile(pred_file_path):
    #         msg = f"SKIP: Predicted structure for {real_file} (expected: {pred_file}) not found."
    #         print(msg)
    #         logs.append(msg)
    #         continue


    for pred_file in os.listdir(pred_dir):
        # real_filename = pred_file.replace('structure_', '')
        real_filename = pred_file.replace('', '')

        real_file = os.path.join(real_dir, real_filename)
        pred_file_path = os.path.join(pred_dir, pred_file)

        if not os.path.isfile(real_file):
                msg = f"SKIP: Real structure for {pred_file} not found."
                print(msg)
                logs.append(msg)
                continue

        try:
            alignment = TMscoring(real_file, pred_file_path)
            alignment.optimise()
            tm_score = alignment.tmscore(**alignment.get_current_values())
            rmsd_value = alignment.rmsd(**alignment.get_current_values())
            msg = f"SUCCESS: {real_filename} / {pred_file} | TM-score: {tm_score:.4f} | RMSD: {rmsd_value:.4f}"
        except Exception as e:
            tm_score = 0
            rmsd_value = 0
            msg = f"ERROR: {real_filename} / {pred_file} | {str(e)}"
        print(msg)
        logs.append(msg)

        results.append({
            "id_real": real_file,
            "id_pred": pred_file_path,
            "tm-score": tm_score,
            "rmsd": rmsd_value
        })

        tm_scores.append(tm_score)
        rmsd_values.append(rmsd_value)

    # Write log file
        with open(log_file, "w") as f:
            for log_entry in logs:
                f.write(log_entry + "\n")

        avg_tm_score = sum(tm_scores) / len(tm_scores) if tm_scores else 0
        avg_rmsd = sum(rmsd_values) / len(rmsd_values) if rmsd_values else 0

        with open(output_csv, mode='w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["id_real", "id_pred", "tm-score", "rmsd"])
            writer.writeheader()
            writer.writerows(results)

    print("\nSummary:")
    print(f"Average TM-score: {avg_tm_score}")
    print(f"Average RMSD: {avg_rmsd}")

    return avg_tm_score, avg_rmsd



def replace_dot_with_underscore(directory):
    for filename in os.listdir(directory):
        # Only rename .pdb files
        if filename.endswith('.pdb'):
            new_name = filename.replace('.', '_')
            # Make sure the extension is still .pdb
            if not new_name.endswith('.pdb'):
                # Replace only dots before the extension
                name, ext = os.path.splitext(filename)
                new_name = name.replace('.', '_') + ext
            # Skip if name doesn't change
            if filename == new_name:
                continue
            src = os.path.join(directory, filename)
            dst = os.path.join(directory, new_name)
            os.rename(src, dst)
            print(f"Renamed: {filename} -> {new_name}")

def extract_base_name(filename):
    import re
    name = filename
    if name.endswith('.pdb'):
        name = name[:-4]
    if name.endswith('_pred'):
        name = name[:-5]
    name = re.sub(r'_[A-Z]+$', '', name)
    return name

# Example usage (adjust as needed)
if __name__ == "__main__":
    real_dir = "/mnt/hdd8/mehdi/projects/vq_encoder_decoder/inference_results/2025-07-06__16-09-07/original_pdb_files/"

    parent_dir = "/mnt/hdd8/mehdi/projects/vq_encoder_decoder/inference_results/2025-07-06__16-09-07/pdb_files/"

    # pred_dir = "cameo_foldtoken_4_official/cameo_rec_level8/"
    output_csv = "/mnt/hdd8/mehdi/projects/vq_encoder_decoder/inference_results/2025-07-06__16-09-07/evaluation.csv"

    # replace_dot_with_underscore(real_dir)

    calculate_tm_rmsd(real_dir, parent_dir, output_csv)
    # summary_rows = []
    #
    # for folder in os.listdir(parent_dir):
    #     full_path = os.path.join(parent_dir, folder)
    #     pred_dir = os.path.join(full_path, "pdb", "structures")
    #     if not os.path.isdir(pred_dir):
    #         continue  # skip if not a valid prediction dir
    #
    #     output_csv = os.path.join(full_path, "metrics.csv")
    #     print(f"\nRunning for {pred_dir}...")
    #
    #     Run your function (assuming calculate_tm_rmsd returns avg_tm_score, avg_rmsd)
    #     avg_tm_score, avg_rmsd = calculate_tm_rmsd(real_dir, pred_dir, output_csv)
    #
    #     summary_rows.append({
    #         "folder": folder,
    #         "pred_dir": pred_dir,
    #         "output_csv": output_csv,
    #         "avg_tm_score": avg_tm_score,
    #         "avg_rmsd": avg_rmsd
    #     })

    # Save summary
    # summary_csv = os.path.join(parent_dir, "summary_results_structure_tokenizer_v4_cameo_filtered_pdb.csv")
    # with open(summary_csv, mode='w', newline='') as f:
    #     writer = csv.DictWriter(f, fieldnames=["folder", "pred_dir", "output_csv", "avg_tm_score", "avg_rmsd"])
    #     writer.writeheader()
    #     writer.writerows(summary_rows)

    # print(f"\nSummary of all runs saved to {summary_csv}")
