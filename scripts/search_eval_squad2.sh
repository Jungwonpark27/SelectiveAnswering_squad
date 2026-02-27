#!/usr/bin/env bash
set -euo pipefail

# 경로
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJ_ROOT}"

source "/home/qa/data2/tmp/selective_answering/scripts/env.sh"

#==========================================
CUDA_VISIBLE_DEVICES=7
#==========================================

EVAL_PY="/home/qa/data2/tmp/selective_answering/src/eval.py"

#==========================================
CKPT_ROOT="${PROJ_ROOT}/ckpt/qwen25_1p5b"
#==========================================

SQUAD2_DEV_JSONL="${PROJ_ROOT}/data/squad2_dev_1500_9to1.jsonl"

OUT_ROOT="${PROJ_ROOT}/results/eval/new2_squad2_qwen25_1p5b"
LOG_ROOT="${PROJ_ROOT}/results/logs"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

# ==========================================
# 너 스타일 그대로: 여기만 바꾸면서 sweep
SPLITS=(g90_seed42 g80_seed42 g70_seed42 g60_seed42 g50_seed42 g40_seed42 g30_seed42 g20_seed42 g10_seed42)
SEEDS=(42)

# VARIANTS를 쓰는 구조( split/variant_seed )라면 아래처럼
VARIANTS=(proposed)

# 만약 ckpt가 split 폴더 하나로 끝나면(예: .../g90_seed42) VARIANTS를 (single)로 두면 됨
# VARIANTS=(single)

THRS=(0.1 0.2 0.3 0.4)
ALPHAS=(16)
# ==========================================

# 하이퍼파라미터
MAX_CTX=1              # SQuAD는 context 1개 고정
MAX_NEW_TOKENS=64
BATCH_SIZE=4
TEMPERATURE=0
TOP_P=0.9
SAVE_SAMPLES=200

# 답 없을 때 출력 문구 (학습과 정확히 동일하게)
ABSTAIN_TEXT="모르겠습니다"

# 결과 있으면 스킵
: "${SKIP_IF_EXISTS:=1}"

if [[ ! -f "${SQUAD2_DEV_JSONL}" ]]; then
  echo "[error] SQuAD dev jsonl not found: ${SQUAD2_DEV_JSONL}"
  echo "        (convert_squad2_to_jsonl.py로 squad2_dev.jsonl 만들어야 함)"
  exit 1
fi

# echo "[env] CKPT_ROOT=${CKPT_ROOT}"
# echo "[env] SQUAD2_DEV_JSONL=${SQUAD2_DEV_JSONL}"
# echo "[env] OUT_ROOT=${OUT_ROOT}"
# echo

# for split in "${SPLITS[@]}"; do
#   for seed in "${SEEDS[@]}"; do
#     for v in "${VARIANTS[@]}"; do

#       # ------------------------------------------
#       # ✅ 체크포인트 경로 결정 로직 (2가지 구조 지원)
#       #
#       # (A) 기존 구조: CKPT_ROOT/<split>/<variant>_seed<seed>
#       # (B) 단순 구조: CKPT_ROOT/<split>   (예: .../g90_seed42)
#       # ------------------------------------------

#       CKPT_DIR_A="${CKPT_ROOT}/${split}/${v}_seed${seed}"
#       CKPT_DIR_B="${CKPT_ROOT}/${split}"

#       CKPT_DIR=""
#       CKPT_TAG=""

#       if [[ -d "${CKPT_DIR_A}" ]]; then
#         CKPT_DIR="${CKPT_DIR_A}"
#         CKPT_TAG="${split}_${v}_seed${seed}"
#       elif [[ -d "${CKPT_DIR_B}" ]]; then
#         CKPT_DIR="${CKPT_DIR_B}"
#         CKPT_TAG="${split}"
#       else
#         echo "[skip] ckpt not found: ${CKPT_DIR_A} (or ${CKPT_DIR_B})"
#         continue
#       fi

#       # 데이터는 SQuAD dev로 고정
#       DATA_PATH="${SQUAD2_DEV_JSONL}"

#       OUT_JSON="${OUT_ROOT}/squad2_dev__${CKPT_TAG}.json"
#       ts="$(date +%Y%m%d_%H%M%S)"
#       LOG_FILE="${LOG_ROOT}/eval_squad2_dev__${CKPT_TAG}__${ts}.log"

#       if [[ "${SKIP_IF_EXISTS}" == "1" ]] && [[ -f "${OUT_JSON}" ]]; then
#         echo "[skip] eval already exists: ${OUT_JSON}"
#         continue
#       fi

#       echo "=================================================="
#       echo "[eval] tag=${CKPT_TAG}"
#       echo "[eval] ckpt_dir=${CKPT_DIR}"
#       echo "[eval] data_path=${DATA_PATH}"
#       echo "[eval] out=${OUT_JSON}"
#       echo "=================================================="

#       CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
#       python "${EVAL_PY}" \
#         --model_or_ckpt "${CKPT_DIR}" \
#         --data_path "${DATA_PATH}" \
#         --out_path "${OUT_JSON}" \
#         --max_contexts "${MAX_CTX}" \
#         --max_new_tokens "${MAX_NEW_TOKENS}" \
#         --temperature "${TEMPERATURE}" \
#         --top_p "${TOP_P}" \
#         --batch_size "${BATCH_SIZE}" \
#         --save_samples "${SAVE_SAMPLES}" \
#         --abstain_text "${ABSTAIN_TEXT}" \
#         --fp16 \
#         --use_answerability_head \
#         --answerability_threshold 0.3 \
#       2>&1 | tee "${LOG_FILE}"

#       echo
#     done
#   done
# done

# echo "[done] SQuAD v2 dev eval sweep complete -> ${OUT_ROOT}"

echo "[env] CKPT_ROOT=${CKPT_ROOT}"
echo "[env] SQUAD2_DEV_JSONL=${SQUAD2_DEV_JSONL}"
echo "[env] OUT_ROOT=${OUT_ROOT}"
echo


for split in "${SPLITS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for v in "${VARIANTS[@]}"; do

      CKPT_DIR_A="${CKPT_ROOT}/${split}/${v}_seed${seed}"
      CKPT_DIR_B="${CKPT_ROOT}/${split}"

      CKPT_DIR=""
      CKPT_TAG=""

      if [[ -d "${CKPT_DIR_A}" ]]; then
        CKPT_DIR="${CKPT_DIR_A}"
        CKPT_TAG="${split}_${v}_seed${seed}"
      elif [[ -d "${CKPT_DIR_B}" ]]; then
        CKPT_DIR="${CKPT_DIR_B}"
        CKPT_TAG="${split}"
      else
        echo "[skip] ckpt not found: ${CKPT_DIR_A} (or ${CKPT_DIR_B})"
        continue
      fi

      DATA_PATH="${SQUAD2_DEV_JSONL}"

      # ✅ 추가: thr/alpha sweep 루프
      for thr in "${THRS[@]}"; do
        for alpha in "${ALPHAS[@]}"; do

          OUT_JSON="${OUT_ROOT}/squad2_dev__${CKPT_TAG}__thr${thr}__a${alpha}.json"
          ts="$(date +%Y%m%d_%H%M%S)"
          LOG_FILE="${LOG_ROOT}/eval_squad2_dev__${CKPT_TAG}__thr${thr}__a${alpha}__${ts}.log"

          if [[ "${SKIP_IF_EXISTS}" == "1" ]] && [[ -f "${OUT_JSON}" ]]; then
            echo "[skip] eval already exists: ${OUT_JSON}"
            continue
          fi

          echo "=================================================="
          echo "[eval] tag=${CKPT_TAG} thr=${thr} alpha=${alpha}"
          echo "[eval] ckpt_dir=${CKPT_DIR}"
          echo "[eval] data_path=${DATA_PATH}"
          echo "[eval] out=${OUT_JSON}"
          echo "=================================================="

          CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
          python "${EVAL_PY}" \
            --model_or_ckpt "${CKPT_DIR}" \
            --data_path "${DATA_PATH}" \
            --out_path "${OUT_JSON}" \
            --max_contexts "${MAX_CTX}" \
            --max_new_tokens "${MAX_NEW_TOKENS}" \
            --temperature "${TEMPERATURE}" \
            --top_p "${TOP_P}" \
            --batch_size "${BATCH_SIZE}" \
            --save_samples "${SAVE_SAMPLES}" \
            --abstain_text "${ABSTAIN_TEXT}" \
            --fp16 \
            --use_answerability_head \
            --answerability_threshold "${thr}" \
            --bias_alpha "${alpha}" \
          2>&1 | tee "${LOG_FILE}"

          echo
        done
      done

    done
  done
done

echo "[done] SQuAD v2 dev eval sweep complete -> ${OUT_ROOT}"

