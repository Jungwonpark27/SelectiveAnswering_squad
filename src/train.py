# train.py
import os
import argparse
from dataclasses import dataclass
import torch.nn.functional as F


import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

import torch.nn as nn

from peft import LoraConfig, get_peft_model, TaskType

from dataset_sft_new import JsonlSFTDataset, DataCollatorForSFT


import yaml

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
import argparse

def parse_args():
    # ---- 1) config 경로만 먼저 파싱 ----
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, _ = pre.parse_known_args()

    # ---- 2) 전체 parser 구성 ----
    p = argparse.ArgumentParser()

    p.add_argument("--config", type=str, default=None)
    p.add_argument("--deepspeed", type=str, default=None)
    p.add_argument("--local_rank", type=int, default=-1)

    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--train_path", type=str, default=None)   # required 제거
    p.add_argument("--eval_path", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)   # required 제거
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--max_seq_len", type=int, default=4096)
    p.add_argument("--max_contexts", type=int, default=5)

    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=2)

    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--fp16", action="store_true", default=True)

    #head 관련 인자 추가함
    p.add_argument("--use_answerability_head", action="store_true")
    p.add_argument("--ans_loss_weight", type=float, default=1.0)

    # ---- 3) config를 defaults로 주입 (있다면) ----
    if pre_args.config:
        cfg = load_config(pre_args.config) or {}
        p.set_defaults(**cfg)

    # ---- 4) 전체 argv를 다시 파싱 (CLI가 config를 override) ----
    args = p.parse_args()

    # ---- 5) 최종 필수값 검증 ----
    if not args.train_path:
        raise ValueError("Missing train_path. Provide --train_path or set it in --config.")
    if not args.output_dir:
        raise ValueError("Missing output_dir. Provide --output_dir or set it in --config.")

    return args


# def parse_args():
#     p = argparse.ArgumentParser()

#     # config 먼저
#     p.add_argument("--config", type=str, default=None)

#     # deepspeed / launcher 호환 (없으면 추가)
#     p.add_argument("--deepspeed", type=str, default=None)
#     p.add_argument("--local_rank", type=int, default=-1)

#     # 나머지 인자들 (required 제거!)
#     p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
#     p.add_argument("--train_path", type=str, default=None)   # <- required 제거
#     p.add_argument("--eval_path", type=str, default=None)
#     p.add_argument("--output_dir", type=str, default=None)   # <- required 제거
#     p.add_argument("--seed", type=int, default=42)

#     p.add_argument("--max_seq_len", type=int, default=4096)
#     p.add_argument("--max_contexts", type=int, default=5)

#     p.add_argument("--per_device_train_batch_size", type=int, default=1)
#     p.add_argument("--per_device_eval_batch_size", type=int, default=1)
#     p.add_argument("--gradient_accumulation_steps", type=int, default=16)
#     p.add_argument("--learning_rate", type=float, default=2e-4)
#     p.add_argument("--num_train_epochs", type=float, default=1.0)
#     p.add_argument("--warmup_ratio", type=float, default=0.03)
#     p.add_argument("--weight_decay", type=float, default=0.0)
#     p.add_argument("--logging_steps", type=int, default=10)
#     p.add_argument("--save_steps", type=int, default=500)
#     p.add_argument("--eval_steps", type=int, default=500)
#     p.add_argument("--save_total_limit", type=int, default=2)

#     p.add_argument("--lora_r", type=int, default=16)
#     p.add_argument("--lora_alpha", type=int, default=32)
#     p.add_argument("--lora_dropout", type=float, default=0.05)

#     p.add_argument("--gradient_checkpointing", action="store_true")
#     p.add_argument("--fp16", action="store_true", default=True)

#     # ---- 1) config만 먼저 파싱 ----
#     cfg_args, remaining = p.parse_known_args()
#     if cfg_args.config:
#         cfg = load_config(cfg_args.config) or {}
#         # yaml 키가 argparse 인자명과 같아야 함 
#         p.set_defaults(**cfg)

#     # ---- 2) 최종 파싱 (CLI가 config override) ----
#     args = p.parse_args(remaining, namespace=cfg_args)

#     # ---- 3) 최종 필수값 검증 ----
#     if not args.train_path:
#         raise ValueError("Missing train_path. Provide --train_path or set it in --config.")
#     if not args.output_dir:
#         raise ValueError("Missing output_dir. Provide --output_dir or set it in --config.")

#     return args


# def parse_args():
#     p = argparse.ArgumentParser()

#     p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
#     p.add_argument("--train_path", type=str, required=True)
#     p.add_argument("--eval_path", type=str, default=None)

#     p.add_argument("--output_dir", type=str, required=True)
#     p.add_argument("--seed", type=int, default=42)

#     # data / tokens
#     p.add_argument("--max_seq_len", type=int, default=4096)
#     p.add_argument("--max_contexts", type=int, default=5)

#     # config 사용시
#     p.add_argument("--config", type=str, default=None)

#     # train hyperparams
#     p.add_argument("--per_device_train_batch_size", type=int, default=1)
#     p.add_argument("--per_device_eval_batch_size", type=int, default=1)
#     p.add_argument("--gradient_accumulation_steps", type=int, default=16)
#     p.add_argument("--learning_rate", type=float, default=2e-4)
#     p.add_argument("--num_train_epochs", type=float, default=1.0)
#     p.add_argument("--warmup_ratio", type=float, default=0.03)
#     p.add_argument("--weight_decay", type=float, default=0.0)
#     p.add_argument("--logging_steps", type=int, default=10)
#     p.add_argument("--save_steps", type=int, default=500)
#     p.add_argument("--eval_steps", type=int, default=500)
#     p.add_argument("--save_total_limit", type=int, default=2)

#     # LoRA
#     p.add_argument("--lora_r", type=int, default=16)
#     p.add_argument("--lora_alpha", type=int, default=32)
#     p.add_argument("--lora_dropout", type=float, default=0.05)

#     # deepspeed
#     p.add_argument("--deepspeed", type=str, default=None,
#                    help="Path to DeepSpeed config json (e.g., ds_zero2_fp16.json)")
    
#     p.add_argument("--local_rank", type=int, default=-1,
#                   help="DeepSpeed/torch.distributed launcher argument (do not set manually)")

#     # misc
#     p.add_argument("--gradient_checkpointing", action="store_true")
#     p.add_argument("--fp16", action="store_true", default=True)  # V100이면 fp16 권장

#     return p.parse_args()


def build_lora_model(model, args):
    #target modules
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model

class AnswerableWrappedCausalLM(nn.Module):
    def __init__(self, base_lm, ans_loss_weight: float = 1.0):
        super().__init__()
        self.base_lm = base_lm
        hidden = getattr(base_lm.config, "hidden_size", None) or getattr(base_lm.config, "n_embd")
        self.answerability_head = nn.Linear(hidden, 1)

        self.ans_loss_weight = float(ans_loss_weight)

        # hidden_states 필요
        self.base_lm.config.output_hidden_states = True

    def forward(self, input_ids=None, attention_mask=None, labels=None, answerable=None, **kwargs):
        # Trainer/Collator가 answerable를 kwargs로 중복 전달하는 케이스 방지
        kwargs.pop("answerable", None)

        outputs = self.base_lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )

        lm_loss = outputs.loss

        # head 학습/로깅 안 할 때는 그대로 반환
        if answerable is None:
            return outputs

        # attention_mask 없으면 전부 토큰으로 간주
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=outputs.hidden_states[-1].device)

        hs = outputs.hidden_states[-1]  # [B, T, H]

        # ✅ mean-pool over non-pad tokens
        mask = attention_mask.unsqueeze(-1).to(dtype=hs.dtype)          # [B, T, 1]
        denom = mask.sum(dim=1).clamp(min=1.0)                          # [B, 1]
        pooled = (hs * mask).sum(dim=1) / denom                         # [B, H]

        # ✅ stop-grad: head가 LM representation을 당기지 않게
        pooled_for_head = pooled.detach()

        logits = self.answerability_head(pooled_for_head).squeeze(-1)   # [B]
        ans_loss = F.binary_cross_entropy_with_logits(logits, answerable.float())

        total_loss = lm_loss + self.ans_loss_weight * ans_loss

        outputs.loss = total_loss
        outputs.ans_loss = ans_loss
        outputs.ans_logits = logits
        return outputs


def main():
    args = parse_args()
    # config 사용 추가
    # if args.config:
    #     cfg = load_config(args.config)
    #     for k, v in cfg.items():
    #         setattr(args, k, v)

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # V100이면 bf16이 보통 안 되므로 fp16 권장
    torch_dtype = torch.float16 if args.fp16 else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        #torch_dtype=torch_dtype, #
        device_map=None,  # torchrun/DDP로 돌릴 거라 None
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False  # 체크포인팅 시 필수

    model = build_lora_model(model, args)

    if args.use_answerability_head:
        wrapped = AnswerableWrappedCausalLM(model)
        wrapped.ans_loss_weight = args.ans_loss_weight
        model = wrapped

    # if args.use_answerability_head:
    #     model = AnswerableWrappedCausalLM(model, ans_loss_weight=args.ans_loss_weight)
    # print("MODEL CLASS:", model.__class__)


    train_ds = JsonlSFTDataset(
        path=args.train_path,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        max_contexts=args.max_contexts,
        filter_only_answerable=False,  # dataset을 실험별로 이미 만들어온 전제
        add_eos=True,
    )

    import inspect
    print("train_ds type:", type(train_ds))
    print("train_ds module:", train_ds.__class__.__module__)
    print("train_ds file:", inspect.getfile(train_ds.__class__))
    print("sample keys:", train_ds[0].keys())


    eval_ds = None
    if args.eval_path:
        eval_ds = JsonlSFTDataset(
            path=args.eval_path,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            max_contexts=args.max_contexts,
            filter_only_answerable=False,
            add_eos=True,
        )

    collator = DataCollatorForSFT(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,

        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,

        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,

        save_safetensors=False,

        evaluation_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.eval_steps if eval_ds is not None else None,

        fp16=args.fp16,
        bf16=False,
        deepspeed=args.deepspeed, #추가

        report_to="none",
        dataloader_pin_memory=True,

        # 멀티GPU에서 속도/안정에 도움
        ddp_find_unused_parameters=False,

        #디버깅
        remove_unused_columns=False

    )

    #디버깅 출력
    print("eval_ds:", eval_ds is not None)
    if eval_ds is not None:
        import inspect
        print("eval_ds type:", type(eval_ds))
        print("eval_ds file:", inspect.getfile(eval_ds.__class__))
        print("eval sample keys:", eval_ds[0].keys())

    #디버깅
    print("MODEL CLASS:", model.__class__)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    # trainer.train()
    # trainer.save_model(args.output_dir)
    # tokenizer.save_pretrained(args.output_dir)


    trainer.train()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.use_answerability_head:
        # 1) base LM (LoRA 포함) 저장: 이 경로는 보통 tying 처리가 잘 됨
        model.base_lm.save_pretrained(args.output_dir)

        # 2) head 저장 (weights + 메타)
        torch.save(
            {
                "answerability_head": model.answerability_head.state_dict(),
                "ans_loss_weight": model.ans_loss_weight,
            },
            os.path.join(args.output_dir, "answerability_head.pt"),
        )
    else:
        # head 안 쓰면 기존처럼
        trainer.save_model(args.output_dir)

    tokenizer.save_pretrained(args.output_dir)



if __name__ == "__main__":
    main()
