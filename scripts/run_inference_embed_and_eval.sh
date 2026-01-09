#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <results_root_or_result_dir>"
  exit 1
fi

RESULTS_INPUT=$1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PST_EVAL_ROOT="/mnt/hdd8/mehdi/projects/vq_encoder_decoder/pst_evaluation"
VQVAE_ENV="/mnt/hdd8/mehdi/environments/vqvae_test/bin/activate"
STB_ENV="/mnt/hdd8/mehdi/environments/StructTokenBench/bin/activate"

DATA_ROOT_FUNCTIONAL="/home/mpngf/datasets/pst/struct_token_bench_release_data/data/functional/local"
DATA_ROOT_STRUCTURAL="/home/mpngf/datasets/pst/struct_token_bench_release_data/data/structural"
DATA_ROOT_PHYSICO="/home/mpngf/datasets/pst/struct_token_bench_release_data/data/physicochemical"
DATA_ROOT_SENSITIVITY="/home/mpngf/datasets/pst/struct_token_bench_release_data/data/sensitivity"
METRICS_DIR="${PST_EVAL_ROOT}/metrics"
SUPERVISED_CSV="${METRICS_DIR}/supervised_embedding_evaluation_metrics.csv"
UNSUPERVISED_CSV="${METRICS_DIR}/unsupervised_embedding_evaluation_metrics.csv"

TEMP_CONFIG_DIR="${REPO_ROOT}/configs/temp/inference_embed"
BASE_CONFIG="${REPO_ROOT}/configs/inference_embed_config.yaml"

DATASET_NAMES=(
  "apolo"
  "atlas"
  "biolip2_binding"
  "biolip2_catalytic"
  "interpro_binding"
  "interpro_activesite"
  "interpro_conservedsite"
  "interpro_repeat"
  "proteinglue"
  "proteinshake"
  "remote_homology"
)

DATASET_PATHS=(
  "/mnt/hdd8/mehdi/datasets/vqvae/temp/PST/apolo_fold-switching/mmcif_files_h5/"
  "/mnt/hdd8/mehdi/datasets/vqvae/temp/PST/atlas/mmcif_files_atlas_h5/"
  "/mnt/hdd8/mehdi/datasets/vqvae/temp/PST/biolip2/biolip2_binding/biolip2_binding_h5/"
  "/mnt/hdd8/mehdi/datasets/vqvae/temp/PST/biolip2/biolip2_catalytic/mmcif_files_biolip2_catalytic_h5/"
  "/mnt/hdd8/mehdi/datasets/vqvae/temp/PST/interpro/interpro_cifs/interpro_h5/"
  "/mnt/hdd8/mehdi/datasets/vqvae/temp/PST/interpro/interpro_activesite/mmcif_files_interpro_activesite_h5/"
  "/mnt/hdd8/mehdi/datasets/vqvae/temp/PST/interpro/interpro_conservedsite/mmcif_files_interpro_conservedsite_h5/"
  "/mnt/hdd8/mehdi/datasets/vqvae/temp/PST/interpro/interpro_repeat/mmcif_files_interpro_repeat_h5/"
  "/mnt/hdd8/mehdi/datasets/vqvae/temp/PST/ProteinGLUE/mmcif_files_proteinglue_epitope_h5/"
  "/mnt/hdd8/mehdi/datasets/vqvae/temp/PST/proteinshake/proteinshake_complete_h5/"
  "/mnt/hdd8/mehdi/datasets/vqvae/temp/PST/remote_homology/mmcif_files_remote_homology_separated_chains_pdb_h5/"
)

DATASET_H5=(
  "vq_embed_apolo.h5"
  "vq_embed_atlas.h5"
  "vq_embed_biolip2_binding.h5"
  "vq_embed_biolip2_catalytic.h5"
  "vq_embed_interpro_binding.h5"
  "vq_embed_interpro_activesite.h5"
  "vq_embed_interpro_conservedsite.h5"
  "vq_embed_interpro_repeat.h5"
  "vq_embed_proteinglue.h5"
  "vq_embed_proteinshake.h5"
  "vq_embed_remote_homology.h5"
)

if [[ ${#DATASET_NAMES[@]} -ne ${#DATASET_PATHS[@]} ]] || [[ ${#DATASET_NAMES[@]} -ne ${#DATASET_H5[@]} ]]; then
  echo "Dataset arrays are mismatched in length."
  exit 1
fi

activate_env() {
  local env_path=$1
  if type deactivate >/dev/null 2>&1; then
    deactivate >/dev/null 2>&1 || true
  fi
  # shellcheck source=/dev/null
  source "$env_path"
}

collect_result_dirs() {
  local input=$1
  local base
  base=$(basename "$input")
  if [[ $base =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}__[0-9]{2}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "$input"
    return
  fi
  find "$input" -type d -regextype posix-extended \
    -regex '.*/[0-9]{4}-[0-9]{2}-[0-9]{2}__[0-9]{2}-[0-9]{2}-[0-9]{2}$' | sort
}

write_config() {
  local base_config=$1
  local output_config=$2
  local trained_model_dir=$3
  local data_path=$4
  local output_base_dir=$5
  local h5_name=$6
  python - "$base_config" "$output_config" "$trained_model_dir" "$data_path" "$output_base_dir" "$h5_name" <<'PY'
import sys
import pathlib
import yaml

base_config, output_config, trained_model_dir, data_path, output_base_dir, h5_name = sys.argv[1:]
with open(base_config, "r") as f:
    cfg = yaml.safe_load(f)

cfg["trained_model_dir"] = trained_model_dir
cfg["checkpoint_path"] = "checkpoints/best_valid.pth"
cfg["data_path"] = data_path
cfg["output_base_dir"] = output_base_dir
cfg["vq_embeddings_h5_filename"] = h5_name
cfg["batch_size"] = 128
cfg["tqdm_progress_bar"] = False

pathlib.Path(output_config).parent.mkdir(parents=True, exist_ok=True)
with open(output_config, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
}

find_latest_embedding() {
  local base_dir=$1
  local h5_name=$2
  local latest
  latest=$(find "$base_dir" -type f -name "$h5_name" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n1 | cut -d' ' -f2-)
  if [[ -z $latest ]]; then
    return 1
  fi
  echo "$latest"
}

run_eval() {
  local script=$1
  local h5_path=$2
  local log_path=$3
  shift 3
  local extra_args=("$@")
  if [[ ! -f $h5_path ]]; then
    echo "Missing embedding: $h5_path (skipping $script)" >&2
    return 0
  fi
  if ! python "${PST_EVAL_ROOT}/${script}" --h5 "$h5_path" "${extra_args[@]}" >"$log_path" 2>&1; then
    echo "Eval failed for ${script}. Log output:" >&2
    sed 's/^/  /' "$log_path" >&2
    return 1
  fi
}

metrics_already_aggregated() {
  local csv_path=$1
  local run_tag=$2
  python - "$csv_path" "$run_tag" <<'PY'
import csv
import os
import sys

csv_path, run_tag = sys.argv[1:]
if not os.path.isfile(csv_path):
    sys.exit(1)
with open(csv_path, newline="") as f:
    reader = csv.reader(f)
    try:
        header = next(reader)
    except StopIteration:
        sys.exit(1)
    if run_tag not in header:
        sys.exit(1)
    idx = header.index(run_tag)
    for row in reader:
        if len(row) > idx and row[idx].strip():
            sys.exit(0)
sys.exit(1)
PY
}

if [[ ! -d $RESULTS_INPUT ]]; then
  echo "Input path is not a directory: $RESULTS_INPUT"
  exit 1
fi

mapfile -t RESULT_DIRS < <(collect_result_dirs "$RESULTS_INPUT")

if [[ ${#RESULT_DIRS[@]} -eq 0 ]]; then
  echo "No result directories found under: $RESULTS_INPUT"
  exit 1
fi

BASE_CONFIG_COPY=$(mktemp)
cp "$BASE_CONFIG" "$BASE_CONFIG_COPY"
trap 'cp "$BASE_CONFIG_COPY" "$BASE_CONFIG"; rm -f "$BASE_CONFIG_COPY"' EXIT

mkdir -p "$TEMP_CONFIG_DIR"

for result_dir in "${RESULT_DIRS[@]}"; do
  result_dir=$(realpath "$result_dir")
  result_name=$(basename "$result_dir")
  result_root="${REPO_ROOT}/inference_embed_results/temp/result_${result_name}"
  embeddings_dir="${result_root}/embeddings"
  logs_dir="${result_root}/logs"

  echo "Loading result: $result_dir"

  activate_env "$VQVAE_ENV"
  pushd "$REPO_ROOT" >/dev/null

  mkdir -p "$embeddings_dir"
  mkdir -p "$logs_dir"

  for idx in "${!DATASET_NAMES[@]}"; do
    dataset_name=${DATASET_NAMES[$idx]}
    dataset_path=${DATASET_PATHS[$idx]}
    h5_name=${DATASET_H5[$idx]}

    output_base_dir="${result_root}/${dataset_name}"
    config_path="${TEMP_CONFIG_DIR}/inference_embed_${result_name}_${dataset_name}.yaml"
    final_h5="${embeddings_dir}/${h5_name}"

    if [[ -f $final_h5 ]]; then
      echo "Embedding already present: ${h5_name}"
      continue
    fi

    existing_h5=$(find_latest_embedding "$output_base_dir" "$h5_name" || true)
    if [[ -n $existing_h5 ]]; then
      cp -f "$existing_h5" "$final_h5"
      echo "Copied embedding: ${h5_name}"
      continue
    fi

    write_config "$BASE_CONFIG_COPY" "$config_path" "$result_dir" "$dataset_path" "$output_base_dir" "$h5_name"
    cp "$config_path" "$BASE_CONFIG"

    echo "Processing dataset: ${dataset_name}"
    if ! NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 PYTHONWARNINGS=ignore \
      accelerate launch --multi_gpu --mixed_precision=bf16 --num_processes=2 inference_embed.py; then
      echo "Inference failed for ${dataset_name}." >&2
      exit 1
    fi

    latest_h5=$(find_latest_embedding "$output_base_dir" "$h5_name")
    if [[ -z ${latest_h5:-} ]]; then
      echo "Failed to locate embedding for ${dataset_name} in ${output_base_dir}" >&2
      exit 1
    fi
    cp -f "$latest_h5" "$final_h5"
    echo "Copied embedding: ${h5_name}"
  done

  popd >/dev/null

  activate_env "$STB_ENV"
  pushd "$PST_EVAL_ROOT" >/dev/null

  EMB_DIR="$embeddings_dir"
  export EMB_DIR

  supervised_done=false
  if metrics_already_aggregated "$SUPERVISED_CSV" "$result_name"; then
    supervised_done=true
  fi

  unsupervised_done=false
  if metrics_already_aggregated "$UNSUPERVISED_CSV" "$result_name"; then
    unsupervised_done=true
  fi

  if ! $supervised_done; then
    run_eval "biolip2_binding_eval.py" "${EMB_DIR}/vq_embed_biolip2_binding.h5" "${logs_dir}/eval_biolip2_binding.log" \
      --data-root "$DATA_ROOT_FUNCTIONAL"
    run_eval "biolip2_catalytic_eval.py" "${EMB_DIR}/vq_embed_biolip2_catalytic.h5" "${logs_dir}/eval_biolip2_catalytic.log" \
      --data-root "$DATA_ROOT_FUNCTIONAL"
    run_eval "proteinshake_binding_eval.py" "${EMB_DIR}/vq_embed_proteinshake.h5" "${logs_dir}/eval_proteinshake.log" \
      --data-root "$DATA_ROOT_FUNCTIONAL"
    run_eval "interpro_binding_eval.py" "${EMB_DIR}/vq_embed_interpro_binding.h5" "${logs_dir}/eval_interpro_binding.log" \
      --data-root "$DATA_ROOT_FUNCTIONAL"
    run_eval "interpro_activesite_eval.py" "${EMB_DIR}/vq_embed_interpro_activesite.h5" "${logs_dir}/eval_interpro_activesite.log" \
      --data-root "$DATA_ROOT_FUNCTIONAL"
    run_eval "interpro_conserved_eval.py" "${EMB_DIR}/vq_embed_interpro_conservedsite.h5" "${logs_dir}/eval_interpro_conserved.log" \
      --data-root "$DATA_ROOT_FUNCTIONAL"
    run_eval "interpro_repeats_eval.py" "${EMB_DIR}/vq_embed_interpro_repeat.h5" "${logs_dir}/eval_interpro_repeats.log" \
      --data-root "$DATA_ROOT_FUNCTIONAL"
    run_eval "proteinglue_epitope_region_eval.py" "${EMB_DIR}/vq_embed_proteinglue.h5" "${logs_dir}/eval_proteinglue.log" \
      --data-root "$DATA_ROOT_FUNCTIONAL"
    run_eval "atlas_rmsf_eval.py" "${EMB_DIR}/vq_embed_atlas.h5" "${logs_dir}/eval_atlas_rmsf.log" \
      --data-root "$DATA_ROOT_PHYSICO"
    run_eval "atlas_bfactor_eval.py" "${EMB_DIR}/vq_embed_atlas.h5" "${logs_dir}/eval_atlas_bfactor.log" \
      --data-root "$DATA_ROOT_PHYSICO"
    run_eval "atlas_neq_eval.py" "${EMB_DIR}/vq_embed_atlas.h5" "${logs_dir}/eval_atlas_neq.log" \
      --data-root "$DATA_ROOT_PHYSICO"
    run_eval "remote_homology_eval.py" "${EMB_DIR}/vq_embed_remote_homology.h5" "${logs_dir}/eval_remote_homology.log" \
      --data-root "$DATA_ROOT_STRUCTURAL" --progress
  else
    echo "Supervised metrics already aggregated for ${result_name}"
  fi

  if ! $unsupervised_done; then
    run_eval "apolo_unsupervised_eval.py" "${EMB_DIR}/vq_embed_apolo.h5" "${logs_dir}/eval_apolo.log" \
      --data-root "$DATA_ROOT_SENSITIVITY" --target-field tm_score
  else
    echo "Unsupervised metrics already aggregated for ${result_name}"
  fi

  if $supervised_done && $unsupervised_done; then
    echo "Metrics already aggregated for ${result_name}"
  else
    mkdir -p "$METRICS_DIR"
    skip_supervised=()
    skip_unsupervised=()
    if $supervised_done; then
      skip_supervised=(--skip-supervised)
    fi
    if $unsupervised_done; then
      skip_unsupervised=(--skip-unsupervised)
    fi
    python "${PST_EVAL_ROOT}/append_eval_metrics.py" \
      --logs-dir "$logs_dir" \
      --run-tag "$result_name" \
      --out-supervised "$SUPERVISED_CSV" \
      --out-unsupervised "$UNSUPERVISED_CSV" \
      "${skip_supervised[@]}" \
      "${skip_unsupervised[@]}"
  fi

  popd >/dev/null
done

if type deactivate >/dev/null 2>&1; then
  deactivate || true
fi
