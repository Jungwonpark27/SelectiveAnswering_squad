# eval.py
from __future__ import annotations

import os
import json
import argparse
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

from tqdm import tqdm
import math
import time


# ====== Prompt rendering (train과 동일하게 유지) ======
def render_user_prompt(query: str, contexts: List[Dict[str, Any]], max_contexts: Optional[int] = None) -> str:
    if max_contexts is not None:
        contexts = contexts[:max_contexts]

    # lines: List[str] = []
    # lines.append("다음은 질문과 참고 문서들입니다.")
    # lines.append('문서들에 답이 있으면 간결하게 답변하세요.')
    # lines.append('문서들에 답이 없으면 "모르겠습니다."라고만 답변하세요.')
    # lines.append("")
    # lines.append("[질문]")
    # lines.append((query or "").strip())
    # lines.append("")
    # lines.append("[문서들]")

    # lines: List[str] = []
    # lines.append("The following are a question and reference documents.")
    # lines.append("If the answer is found in the documents, answer concisely.")
    # lines.append('If the answer is not found in the documents, answer only "The answer is not available in the given context."')
    # lines.append("")
    # lines.append("[Question]")
    # lines.append((query or "").strip())
    # lines.append("")
    # lines.append("[Documents]")

    lines: List[str] = []
    lines.append("다음은 질문과 참고 문서들입니다.")
    #lines.append('문서들에 답이 있으면 간결하게 답변하세요.')
    #lines.append('문서들에 답이 없으면 "모르겠습니다."라고만 답변하세요.')
    lines.append("문서들을 참고해서 질문에 대한 답을 간결하게 답변하세요.")
    lines.append('문서들에 답이 없으면 "모르겠습니다"라고만 답변하세요.')
    lines.append("")
    lines.append("[질문]")
    lines.append((query or "").strip())
    lines.append("")
    lines.append("[문서들]")


    k = 0
    for c in contexts:
        txt = (c.get("text") or "").strip()
        if not txt:
            continue
        k += 1
        lines.append(f"{k}. {txt}")

    if k == 0:
        lines.append("1. (문서 없음)")

    return "\n".join(lines)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            ex["query"] = str(ex.get("query", "") or "")
            ex["target"] = str(ex.get("target", "") or "")
            ex["contexts"] = ex.get("contexts", []) or []
            try:
                ex["is_answerable"] = int(ex.get("is_answerable", 1))
            except Exception:
                ex["is_answerable"] = 1
            rows.append(ex)
    return rows


def normalize_text(s: str) -> str:
    return (s or "").strip()

def strip_trailing_punct(s: str) -> str:
    # 끝의 마침표/구두점/공백류 제거 (필요하면 더 추가)
    return (s or "").strip().rstrip(".。．!！?？…")

def is_abstain(output: str, abstain_text: str) -> bool:
    o = normalize_text(output)
    a = normalize_text(abstain_text)
    if not o:
        return True

    # ✅ 마침표/공백 같은 걸 정규화해서 비교
    def canon(s: str) -> str:
        s = s.strip()
        if s.endswith("."):
            s = s[:-1].strip()
        return s

    oc = canon(o)
    ac = canon(a)

    if oc == ac:
        return True
    if oc.startswith(ac):
        return True
    return False



#def exact_match(pred: str, gold: str) -> bool:
    #return normalize_text(pred) == normalize_text(gold) #이거 대신

def exact_match(pred: str, gold: str) -> bool:
    p = normalize_text(pred)
    g = normalize_text(gold)
    if not p or not g:
        return False
    return (p == g) or (p in g) or (g in p)




# ====== Answerability head wrapper (inference용) ======
class AnswerableWrappedCausalLM(nn.Module):
    """Train에서 쓴 wrapper와 동일한 아이디어지만,
    inference에서 answerable logit/prob을 항상 뽑을 수 있게 구성."""

    def __init__(self, base_lm):
        super().__init__()
        self.base_lm = base_lm
        hidden = getattr(base_lm.config, "hidden_size", None) or getattr(base_lm.config, "n_embd")
        self.answerability_head = nn.Linear(hidden, 1)

        # hidden_states 필요
        self.base_lm.config.output_hidden_states = True

    @property
    def device(self):
        # base_lm이 보통 device를 가지고 있음. 없으면 parameter에서 추론.
        if hasattr(self.base_lm, "device"):
            return self.base_lm.device
        return next(self.parameters()).device

    def forward(self, input_ids=None, attention_mask=None, labels=None, answerable=None, **kwargs):
        # Trainer/콜러가 answerable를 kwargs에 중복으로 넣는 경우 방지
        kwargs.pop("answerable", None)

        outputs = self.base_lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )

        hs = outputs.hidden_states[-1]  # [B, T, H]

        mask = attention_mask.unsqueeze(-1).to(hs.dtype)  # [B, T, 1]
        denom = mask.sum(dim=1).clamp(min=1.0)           # [B, 1]
        pooled = (hs * mask).sum(dim=1) / denom          # [B, H]



        #device mismatch 수정
        # pooled: [B, H] on hs.device, hs.dtype (fp16일 수 있음)
        if self.answerability_head.weight.device != pooled.device or self.answerability_head.weight.dtype != pooled.dtype:
            self.answerability_head = self.answerability_head.to(device=pooled.device, dtype=pooled.dtype)


        logits = self.answerability_head(pooled).squeeze(-1)  # [B]
        outputs.ans_logits = logits

        # 학습용으로 loss가 필요하면 계산 (eval에서는 보통 answerable=None)
        if (answerable is not None) and (outputs.loss is not None):
            ans_loss = F.binary_cross_entropy_with_logits(logits, answerable.float())
            outputs.ans_loss = ans_loss

        return outputs
    
        
    def config(self):
        return self.base_lm.config

    def generate(self, *args, **kwargs):
        return self.base_lm.generate(*args, **kwargs)



def _try_load_answerability_head(model_or_ckpt: str, wrapped: AnswerableWrappedCausalLM) -> bool:
    """체크포인트 폴더에서 answerability head 가중치를 로드.

    권장 저장 포맷(학습 코드 변경안):
      - base_lm: HuggingFace/PEFT 방식으로 output_dir에 저장
      - head:   output_dir/answerability_head.pt 에 torch.save(dict) 형태로 저장

    이 함수는 아래 순서로 로드를 시도합니다.
      1) answerability_head.pt (권장)
      2) (레거시) model.safetensors / pytorch_model.bin 안에 answerability_head.* 키가 있는 경우
    """
    ckpt_dir = model_or_ckpt
    if not os.path.isdir(ckpt_dir):
        return False

    # 1) 권장: answerability_head.pt
    head_pt = os.path.join(ckpt_dir, "answerability_head.pt")
    if os.path.exists(head_pt):
        try:
            obj = torch.load(head_pt, map_location="cpu")
            # obj는 보통 {"answerability_head": state_dict, "ans_loss_weight": float} 형태
            if isinstance(obj, dict):
                sd = obj.get("answerability_head") or obj.get("state_dict") or obj.get("head_state_dict")
                if sd is None and all(k.startswith("weight") or k.startswith("bias") for k in obj.keys()):
                    # state_dict 자체를 저장한 경우
                    sd = obj
                if sd is not None:
                    wrapped.answerability_head.load_state_dict(sd, strict=True)
                    if "ans_loss_weight" in obj:
                        try:
                            wrapped.ans_loss_weight = float(obj["ans_loss_weight"])
                        except Exception:
                            pass
                    return True
        except Exception as e:
            print(f"[warn] Failed to load answerability_head.pt: {e}")

    # 2) 레거시: 모델 파일에서 answerability_head.* 키만 추출
    state = None

    st_path = os.path.join(ckpt_dir, "model.safetensors")
    if os.path.exists(st_path):
        try:
            from safetensors.torch import load_file  # type: ignore
            state = load_file(st_path)
        except Exception:
            state = None

    if state is None:
        pt_path = os.path.join(ckpt_dir, "pytorch_model.bin")
        if os.path.exists(pt_path):
            try:
                state = torch.load(pt_path, map_location="cpu")
            except Exception:
                state = None

    if isinstance(state, dict):
        # state dict가 nested {"model": ...} 형태면 풀기
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        if "model" in state and isinstance(state["model"], dict):
            state = state["model"]

        head_keys = {}
        for k, v in state.items():
            if k.startswith("answerability_head."):
                head_keys[k.replace("answerability_head.", "")] = v
            elif k.endswith(".answerability_head.weight") or k.endswith(".answerability_head.bias"):
                # 혹시 prefix가 붙어 저장된 경우 대응
                tail = k.split("answerability_head.", 1)[-1]
                head_keys[tail] = v

        if head_keys:
            try:
                wrapped.answerability_head.load_state_dict(head_keys, strict=False)
                return True
            except Exception as e:
                print(f"[warn] Failed to load head weights from model files: {e}")

    return False

from typing import List, Optional, Tuple
import torch
from transformers.generation.logits_process import LogitsProcessor

class BatchAbstainBiasProcessor(LogitsProcessor):
    def __init__(self, abstain_token_id: int, bias_per_sample: torch.Tensor):
        self.abstain_token_id = int(abstain_token_id)
        self.bias_per_sample = bias_per_sample  # [B]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        b = self.bias_per_sample.to(device=scores.device, dtype=scores.dtype)
        scores[:, self.abstain_token_id] += b
        return scores


def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    use_answerability_head: bool = False,
    answerability_threshold: float = 0.5,
    bias_alpha: float = 8.0,
    abstain_text: str = "The answer is not available in the given context.",
) -> Tuple[List[str], Optional[List[float]]]:

    batch_messages = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": p},
        ]
        for p in prompts
    ]

    # head용 / gen용 input 분리
    input_ids_list_head = [
        tokenizer.apply_chat_template(m, add_generation_prompt=False, tokenize=True, return_tensors=None)
        for m in batch_messages
    ]
    input_ids_list_gen = [
        tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=True, return_tensors=None)
        for m in batch_messages
    ]

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def pad_left(input_ids_list_):
        max_len_ = max(len(x) for x in input_ids_list_)
        input_ids_ = torch.full((len(input_ids_list_), max_len_), pad_id, dtype=torch.long, device=model.device)
        attn_ = torch.zeros((len(input_ids_list_), max_len_), dtype=torch.long, device=model.device)
        for i, ids in enumerate(input_ids_list_):
            ids_t = torch.tensor(ids, dtype=torch.long, device=model.device)
            L = ids_t.numel()
            input_ids_[i, -L:] = ids_t
            attn_[i, -L:] = 1
        return input_ids_, attn_

    input_ids_head, attn_head = pad_left(input_ids_list_head)
    input_ids_gen, attn_gen = pad_left(input_ids_list_gen)

    ans_probs: Optional[List[float]] = None
    logits_processor = None

    if use_answerability_head:
        fw = model(input_ids=input_ids_head, attention_mask=attn_head)
        if not hasattr(fw, "ans_logits"):
            raise RuntimeError("use_answerability_head=True인데 ans_logits가 없습니다. 모델 wrap/로드 확인 필요")

        probs = torch.sigmoid(fw.ans_logits).detach()  # [B]
        ans_probs = probs.float().cpu().tolist()

        # prob 낮을수록 abstain 방향 bias↑
        #alpha = 8.0
        alpha = float(bias_alpha)
        thr = float(answerability_threshold)
        bias = (thr - probs).clamp(min=0) / max(thr, 1e-6)  # [0..1]
        bias = bias * alpha  # [0..alpha]

        abstain_ids = tokenizer.encode(abstain_text, add_special_tokens=False)
        abstain_first = abstain_ids[0] if len(abstain_ids) > 0 else tokenizer.eos_token_id

        logits_processor = [BatchAbstainBiasProcessor(abstain_first, bias)]

    gen = model.generate(
        input_ids=input_ids_gen,
        attention_mask=attn_gen,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature if temperature > 0 else None,
        top_p=top_p if temperature > 0 else None,
        pad_token_id=pad_id,
        eos_token_id=tokenizer.eos_token_id,
        logits_processor=logits_processor,
    )

    # 배치 전체 decode
    input_len = input_ids_gen.size(1)
    outputs: List[str] = []
    for j in range(gen.size(0)):
        out_ids = gen[j, input_len:]
        text = tokenizer.decode(out_ids, skip_special_tokens=True).strip()
        outputs.append(text if text else abstain_text)

    return outputs, ans_probs


def compute_metrics(rows: List[Dict[str, Any]], outputs: List[str], abstain_text: str) -> Dict[str, Any]:
    assert len(rows) == len(outputs)

    # decisions: True면 abstain, False면 answered
    abstain_flags = [is_abstain(o, abstain_text) for o in outputs]

    total = len(rows)

    # ===== mixed (전체) =====
    answered = 0
    correct_answered = 0

    un_total = 0
    un_answered = 0
    un_abstained = 0

    ans_total = 0
    ans_answered = 0
    ans_correct = 0
    ans_abstained = 0

    overall_correct = 0  # answerable correct + unanswerable abstain

    for ex, pred, ab in zip(rows, outputs, abstain_flags):
        ia = int(ex.get("is_answerable", 1))
        gold = str(ex.get("target", "") or "").strip()

        if ia == 0:
            un_total += 1
            if ab:
                un_abstained += 1
                overall_correct += 1
            else:
                un_answered += 1
        else:
            ans_total += 1
            if ab:
                ans_abstained += 1
            else:
                ans_answered += 1
                answered += 1
                if exact_match(pred, gold):
                    ans_correct += 1
                    correct_answered += 1
                    overall_correct += 1
    #수정
    answered_all = sum(1 for ab in abstain_flags if not ab)

    mixed = {
        "total": total,
        "answerable_total": ans_total,
        "unanswerable_total": un_total,
        "response_rate": answered_all / max(total, 1), #수정
        "cond_accuracy_answered": correct_answered / max(answered, 1),
        "hallucination_rate_unanswerable": un_answered / max(un_total, 1) if un_total else 0.0,
        "abstain_recall_unanswerable": un_abstained / max(un_total, 1) if un_total else 0.0,
        "overall_accuracy": overall_correct / max(total, 1),
    }

    # ===== answerable-only (정답 정확도 중심) =====
    # 여기서는 "답을 한 것 중 정확도" + "answerable에서의 응답률" 둘 다 주는게 유용
    answerable_only = {
        "answerable_total": ans_total,
        "answerable_response_rate": ans_answered / max(ans_total, 1) if ans_total else 0.0,
        "answerable_cond_accuracy_answered": ans_correct / max(ans_answered, 1) if ans_answered else 0.0,
        "answerable_abstain_rate": ans_abstained / max(ans_total, 1) if ans_total else 0.0,
    }

    # ===== unanswerable-only (안전/환각 중심) =====
    unanswerable_only = {
        "unanswerable_total": un_total,
        "unanswerable_answer_rate": un_answered / max(un_total, 1) if un_total else 0.0,
        "unanswerable_abstain_rate": un_abstained / max(un_total, 1) if un_total else 0.0,
    }

    return {
        "mixed": mixed,
        "answerable_only": answerable_only,
        "unanswerable_only": unanswerable_only,
    }



def _build_samples(
    rows: List[Dict[str, Any]],
    outputs: List[str],
    save_samples: int,
    ans_probs_all: Optional[List[float]],
    prompts: Optional[List[str]] = None,   # ✅ 추가
):
    n = min(
        save_samples,
        len(rows),
        len(outputs),
        len(prompts) if prompts is not None else 10**9,
    )
    out = []
    for i in range(n):
        ex = rows[i]
        pred = outputs[i]
        item = {
            "qid": ex.get("qid"),
            "is_answerable": ex.get("is_answerable"),
            "query": ex.get("query"),
            "gold": ex.get("target"),
            "pred": pred,
        }
        if prompts is not None:
            item["prompt"] = prompts[i]   # ✅ prompts 인자를 사용
        if ans_probs_all is not None and i < len(ans_probs_all):
            item["ans_prob"] = float(ans_probs_all[i])
        out.append(item)
    return out




def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_or_ckpt", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--out_path", type=str, required=True)

    p.add_argument("--abstain_text", type=str, default="The answer is not available in the given context.")
    p.add_argument("--use_answerability_head", action="store_true")
    p.add_argument("--answerability_threshold", type=float, default=0.5)
    p.add_argument("--max_contexts", type=int, default=5)
    p.add_argument("--max_new_tokens", type=int, default=64)

    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--batch_size", type=int, default=4)

    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--save_samples", type=int, default=200)
    p.add_argument("--base_model", type=str, default=None)
    p.add_argument("--bias_alpha", type=float, default=8.0)

    return p.parse_args()


def main():
    args = parse_args()

    base = args.base_model or args.model_or_ckpt

    tokenizer = AutoTokenizer.from_pretrained(base, use_fast=True)

    tokenizer.padding_side = "left"  # ✅ 중요: decoder-only는 left padding 권장
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        


    torch_dtype = torch.float16 if args.fp16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    model.resize_token_embeddings(len(tokenizer)) #추가함 warning문제 #위치 수정
    model.eval()

    # ====== optional: wrap with answerability head for gating ======
    ans_head_loaded = False
    if args.use_answerability_head:
        wrapped = AnswerableWrappedCausalLM(model)
        ans_head_loaded = _try_load_answerability_head(args.model_or_ckpt, wrapped)
        if not ans_head_loaded:
            print("[warn] answerability_head weights were not found/loaded from ckpt. "
                  "Gating will be based on a randomly-initialized head (not recommended).")
        model = wrapped


    rows = load_jsonl(args.data_path)

    prompts = [
        render_user_prompt(ex["query"], ex["contexts"], max_contexts=args.max_contexts)
        for ex in rows
    ]

    outputs: List[str] = []
    bs = args.batch_size
    n_batches = math.ceil(len(prompts) / bs)

    t0 = time.time()
    for bi in tqdm(range(n_batches), desc="Generating", unit="batch"):
        i = bi * bs 
    #for i in range(0, len(prompts), bs): #tqdm으로 감쌈
        batch_prompts = prompts[i:i+bs]
        batch_out, batch_probs = generate_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=batch_prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            use_answerability_head=args.use_answerability_head,
            bias_alpha=args.bias_alpha, ##
            answerability_threshold=args.answerability_threshold, 
            abstain_text=args.abstain_text,
        )
        outputs.extend(batch_out)
        if batch_probs is not None:
            # stash probs in parallel list
            if 'ans_probs_all' not in locals():
                ans_probs_all = []
            ans_probs_all.extend(batch_probs)

        done = bi + 1
        elapsed = time.time() - t0
        it_s = elapsed / done
        eta = it_s * (n_batches - done)
        tqdm.write(f"[progress] {done}/{n_batches} batches | elapsed {elapsed:.1f}s | ETA {eta:.1f}s")


    metrics = compute_metrics(rows, outputs, args.abstain_text)

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    payload = {
        "data_path": args.data_path,
        "model_or_ckpt": args.model_or_ckpt,
        "abstain_text": args.abstain_text,
        "answerability_head": {
            "use": bool(args.use_answerability_head),
            "threshold": float(args.answerability_threshold),
            "head_weights_loaded": bool(locals().get("ans_head_loaded", False)),
        },
        "decoding": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "batch_size": args.batch_size,
        },
        "metrics": metrics,
        "samples": _build_samples(rows, outputs, args.save_samples, locals().get("ans_probs_all", None), prompts),
    }
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("Saved eval to:", args.out_path)
    print("Mixed:", metrics["mixed"])
    print("Answerable-only:", metrics["answerable_only"])
    print("Unanswerable-only:", metrics["unanswerable_only"])


if __name__ == "__main__":
    main()




