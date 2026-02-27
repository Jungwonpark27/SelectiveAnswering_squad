#!/usr/bin/env bash
set -euo pipefail


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJ_ROOT}"

source "${PROJ_ROOT}/scripts/env.sh"


#==================================================================
USE_DEEPSPEED=1
#==================================================================

: "${CUDA_VISIBLE_DEVICES:=0,1,2,3}" 
: "${NUM_GPUS:=4}" 

DS_CONFIG="${PROJ_ROOT}/configs/ds_zero2_fp16.json" #모델 크기 따라 다르게
TRAIN_PY="${PROJ_ROOT}/src/train.py"

MODEL_TAG="qwen25_7b"
SPLITS=(g90_seed42 g80_seed42)     # split 폴더들
SEEDS=(123 428)               
VARIANTS=(baseline1 baselineB proposed)
#==================================================================

case "${MODEL_TAG}" in
  qwen25_7b)   MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" ;;
  qwen25_1p5b) MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct" ;;
  *)
    echo "[error] Unknown MODEL_TAG=${MODEL_TAG} (use qwen25_7b or qwen25_1p5b)"
    exit 1
    ;;
esac
export MODEL_NAME


MODEL_NAME_EFFECTIVE="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"

CKPT_SUBDIR="qwen25_6b"  # fallback
if [[ "${MODEL_NAME_EFFECTIVE}" =~ Qwen2\.5-([0-9]+(\.[0-9]+)?)B ]]; then
  size="${BASH_REMATCH[1]}"           # e.g. "7" or "3" or "1.5"
  size_tag="${size//./p}"            # "1.5" -> "1p5"
  CKPT_SUBDIR="qwen25_${size_tag}b"  # -> qwen25_1p5b
fi
CKPT_ROOT="${PROJ_ROOT}/ckpt/${CKPT_SUBDIR}"
DUMP_ROOT="${PROJ_ROOT}/results/config_dumps"
LOG_ROOT="${PROJ_ROOT}/results/logs"
mkdir -p "${CKPT_ROOT}" "${DUMP_ROOT}" "${LOG_ROOT}"

: "${SKIP_IF_EXISTS:=1}"

MODEL_CFG="${PROJ_ROOT}/configs/${CKPT_SUBDIR}.yaml"

if [[ ! -f "${MODEL_CFG}" ]]; then
  echo "[error] model config not found: ${MODEL_CFG}"
  echo "        expected e.g. configs/qwen25_7b.yaml or configs/qwen25_1p5b.yaml"
  exit 1
fi


# yaml dump 생성 도우미 (python 필요)
make_dump () {
  local base_cfg="$1"
  local dump_path="$2"
  local model_name="$3"
  local train_path="$4"
  local eval_path="$5"
  local output_dir="$6"
  local seed="$7"
  local variant="$8"
  python - <<PY
import os, yaml
base_cfg = "${base_cfg}"
dump_path = "${dump_path}"
os.makedirs(os.path.dirname(dump_path), exist_ok=True)

with open(base_cfg, "r") as f:
    cfg = yaml.safe_load(f) or {}

cfg["model_name"] = "${model_name}"
cfg["train_path"] = "${train_path}"
cfg["eval_path"] = "${eval_path}"
cfg["output_dir"] = "${output_dir}"
cfg["seed"] = int("${seed}")

with open(dump_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
print(dump_path)
PY
}

for split in "${SPLITS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for v in "${VARIANTS[@]}"; do

      # base cfg 선택
      base_cfg="${MODEL_CFG}"

      # 데이터 경로 규칙 (폴더만 바뀌는 구조)
      train_path="${PROJ_ROOT}/data/${split}/${v}_train.jsonl"
      eval_path="${PROJ_ROOT}/data/${split}/eval_val.jsonl"

      # ckpt/output 경로 규칙
      out_dir="${CKPT_ROOT}/${split}/${v}_seed${seed}"

      # dump/log 경로
      dump_cfg="${DUMP_ROOT}/${split}/${v}_seed${seed}.yaml"
      ts="$(date +%Y%m%d_%H%M%S)"
      log_file="${LOG_ROOT}/train_${split}_${v}_seed${seed}_${ts}.log"

      echo "=================================================="
      echo "[run] split=${split} variant=${v} seed=${seed}"
      echo "[run] train_path=${train_path}"
      echo "[run] eval_path=${eval_path}"
      echo "[run] out_dir=${out_dir}"
      echo "[run] base_cfg=${base_cfg}"
      echo "[run] dump_cfg=${dump_cfg}"
      echo "[run] log=${log_file}"
      echo "=================================================="

      if [[ "${SKIP_IF_EXISTS}" == "1" ]] && [[ -d "${out_dir}" ]] && (ls -1 "${out_dir}"/* >/dev/null 2>&1); then
        echo "[skip] output_dir already has files: ${out_dir}"
        echo
        continue
      fi

      # 호출
      make_dump "${base_cfg}" "${dump_cfg}" "${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}" \
          "${train_path}" "${eval_path}" "${out_dir}" "${seed}" "${v}" >/dev/null


      # 학습 실행: dump_cfg를 config로 사용
      if [[ "${USE_DEEPSPEED}" == "1" ]]; then
        echo "[run] launcher=deepspeed ds_config=${DS_CONFIG}"
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
          deepspeed \
            --num_gpus "${NUM_GPUS}" \
            "${TRAIN_PY}" \
            --config "${dump_cfg}" \
            --deepspeed "${DS_CONFIG}" \
            2>&1 | tee "${log_file}"
      else
        echo "[run] launcher=torchrun (no deepspeed)"
        MASTER_ADDR=127.0.0.1 MASTER_PORT=29507 \
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
        torchrun --nproc_per_node="${NUM_GPUS}" "${TRAIN_PY}" \
          --config "${dump_cfg}" \
        2>&1 | tee "${log_file}"
      fi


      echo
    done
  done
done

echo "[done] sweep complete"
