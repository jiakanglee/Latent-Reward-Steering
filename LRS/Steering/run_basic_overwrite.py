import torch
import torch.nn as nn
import argparse
import sys
import os
import math
import traceback 
import numpy as np
import gc # 引入垃圾回收
import json
import re
import time
from glob import glob as _glob

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from datasets import load_dataset
from transformers import PreTrainedModel
from utils.utils import load_model, center_and_l2_normalize_torch
from utils.sae import load_sae 
from typing import Optional

from utils.llm_judge import evaluate_answer
from utils.rule_eval import extract_choice_letter


def _mbpp_target_function_name(test_list) -> Optional[str]:
    """从首条 assert 解析被测函数名，便于 prompt 对齐与 rule_eval 抽取。"""
    if not test_list:
        return None
    m = re.search(r"assert\s+([a-zA-Z_]\w*)\s*\(", str(test_list[0]))
    return m.group(1) if m else None


def _mcqa_judge_json_extra(
    gold_letter: str, res_base: Optional[str], res_steer: Optional[str] = None
) -> dict:
    """GPQA 等 MCQ：与 evaluate_gpqa 一致，用 extract_choice_letter 记录抽取字母（非 LLM judge）。"""
    g = (gold_letter or "").strip().upper()[:1]
    extra = {
        "gold_letter": g,
        "extracted_letter_base": extract_choice_letter(res_base or ""),
        "judge_mode": "rule_mcqa_extract_choice_letter",
    }
    if res_steer is not None:
        extra["extracted_letter_steer"] = extract_choice_letter(res_steer or "")
    return extra


def _sae_model_id_from_hf_model(model_name: str) -> str:
    """将 HF --model 的短名映射到 train-saes 文件名中的 model_id（如 open-reasoner-zero-1.5b）。"""
    tail = (model_name or "").strip().split("/")[-1].lower().replace("_", "-")
    if tail.startswith("open-reasoner-zero-"):
        return tail
    # 非 ORZ 家族时保持历史行为（与旧脚本一致）
    return "open-reasoner-zero-7b"


# =========================================================
# 1. Transformer Reward Model (保持不变)
# =========================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class LatentTransformer(nn.Module):
    # 增加 dropout=0.1 默认参数
    def __init__(self, input_dim=10, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        el = nn.TransformerEncoderLayer(d_model, nhead, d_model*4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(el, num_layers)
        
        # 🔥 关键修改：为了完美加载 train 代码的权重，这里必须加上 nn.Dropout()
        # 这样第二层 Linear 的权重名才会对应上 'head.3.weight'，避免加载报错
        self.head = nn.Sequential(
            nn.Linear(d_model, 64), 
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        ) 
        # 为了不影响你 Hook 里的 return_logits=True 逻辑，我们依然把 Sigmoid 放在外面
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, return_logits=False):
        # 1. 预处理
        x = self.pos_encoder(self.embedding(self.input_norm(x)))
        
        # 2. 核心计算：彻底删掉 mask 生成，并且不传 mask 参数
        x = self.transformer_encoder(x)
        
        logits = self.head(x) 
        return logits if return_logits else self.sigmoid(logits)

# =========================================================
# 2. Iterative Transformer Steering Hook 
# (移除 AMP 和 Sliding Window 版)
# =========================================================
def make_confidence_hook(steering_hook):
    """返回一个 hook，在 lm_head 输出后记录 max(softmax(logits)) 到 steering_hook.last_max_prob，并追加到 current_max_probs"""
    def _hook(module, input, output):
        logits = output[0] if isinstance(output, tuple) else output
        if logits.dim() == 3:
            logits = logits[:, -1, :].float()
        else:
            logits = logits.float()
        probs = torch.softmax(logits, dim=-1)
        max_prob = probs.max(dim=-1).values.mean().item()
        steering_hook.last_max_prob = max_prob
        if steering_hook.record_tokens:
            steering_hook.current_max_probs.append(max_prob)
    return _hook


class IterativeTransformerSteeringHook:
    def __init__(self, sae, reward_model, step_size=1.0, num_steps=5, sae_dim=10,
                 monitor=False, reward_threshold=0.95, confidence_threshold=0.5,
                 lm_head=None, record_tokens=False, steer_every_token=False,
                 collect_sae_latent_path=False, collect_sae_steer_trace=False):
        self.sae = sae
        self.reward_model = reward_model
        self.step_size = step_size
        self.num_steps = num_steps
        self.sae_dim = sae_dim
        self.monitor = monitor
        self.reward_threshold = reward_threshold
        self.confidence_threshold = confidence_threshold
        self.steer_every_token = bool(steer_every_token)
        self.lm_head = lm_head
        self.record_tokens = record_tokens
        self.collect_sae_latent_path = bool(collect_sae_latent_path)
        self.collect_sae_steer_trace = bool(collect_sae_steer_trace)
        # 每生成 token 一条 pre-steer 稠密 SAE latent [D]，仅当 collect_sae_latent_path 时填充
        self.sae_latent_path = []
        # 仅当本步实际执行 steer 时追加：生成步号 + steer 前后各一条长度 D 的稠密 latent（列表）
        self.sae_steer_trace = []

        self.step_count = 0
        # 上一步 token 的 max(softmax(logits))，用于条件 steering
        self.last_max_prob = None
        # 每 token 记录：用于 threshold 调参和事后分析
        self.token_records = []  # [{step, reward_prob, last_max_prob, should_steer}]
        self.current_max_probs = []  # 当前 token 的 max_prob，由 lm_head hook 追加

        # 获取 SAE 解码器权重
        if hasattr(self.sae, "W_dec"): self.decoder_weight = self.sae.W_dec
        elif hasattr(self.sae, "decoder"): self.decoder_weight = self.sae.decoder
        else: raise ValueError("Could not find decoder weights in SAE.")
        
        if isinstance(self.decoder_weight, nn.Module):
            self.decoder_weight = self.decoder_weight.weight

    def reset(self):
        """每道题开始前必须调用，彻底清空显存残留"""
        self.step_count = 0
        self.last_max_prob = None
        self.token_records = []
        self.current_max_probs = []
        self.sae_latent_path = []
        self.sae_steer_trace = []
        gc.collect()
        torch.cuda.empty_cache()

    def _get_dense_latents(self, x):
        """将 LLM 激活值通过 SAE 编码为稠密 Latent (K-sparse)"""
        flat_x = x.view(-1, x.shape[-1])
        if hasattr(self.sae, "activation_mean"):
            pre_act = center_and_l2_normalize_torch(flat_x, self.sae.activation_mean)
        else:
            pre_act = flat_x - self.sae.b_dec
        top_acts, top_indices = self.sae.encode(pre_act)
        latents = torch.zeros((top_acts.shape[0], self.sae_dim), device=x.device, dtype=torch.float32)
        latents.scatter_(1, top_indices, top_acts.float())
        return latents

    def __call__(self, module, inputs, outputs):
        original_act = outputs[0] if isinstance(outputs, tuple) else outputs
        batch_size, seq_len, _ = original_act.shape
        sae_dtype = self.sae.encoder.weight.dtype
        x = original_act.to(sae_dtype)

        # ==========================================
        # A. Prompt 阶段 (跳过)
        # ==========================================
        if seq_len > 1:
            return outputs

        # ==========================================
        # B. Generation 阶段 (逐 Token Steering)
        # ==========================================
        self.step_count += 1
        
        # 1. 提取当前 Token 的原始 SAE Latent
        with torch.no_grad():
            current_dense = self._get_dense_latents(x).view(batch_size, 1, -1)

        if self.collect_sae_latent_path:
            # [D] CPU float，与 steering / RM 门控无关，仅用于事后全长 RM 打分
            vec = current_dense.detach().float().squeeze(0).squeeze(0).cpu().clone()
            self.sae_latent_path.append(vec)

        # 2. 计算初始 Reward (Before Steering) — 条件判断和 monitor 都需要
        with torch.no_grad():
            init_logits = self.reward_model(current_dense, return_logits=True)
            init_logit_val = init_logits[:, -1, 0].sum().item() / batch_size
            init_prob_val = 1 / (1 + math.exp(-init_logit_val)) if abs(init_logit_val) < 50 else (1.0 if init_logit_val > 0 else 0.0)

        # 3. Steering 条件判断（对照实验：steer_every_token 时每步都 steer，忽略 reward/conf 门控）
        #    默认：reward_prob < threshold → steer；
        #          reward_prob >= threshold 且 last_max_prob < conf_threshold → steer；否则跳过
        if self.steer_every_token:
            should_steer = True
        elif init_prob_val < self.reward_threshold:
            should_steer = True
        elif init_prob_val >= self.reward_threshold and self.last_max_prob is not None and self.last_max_prob < self.confidence_threshold:
            should_steer = True
        else:
            should_steer = False

        # 4. 记录每 token 数据（用于 threshold 调参和事后分析）
        if self.record_tokens:
            self.token_records.append({
                "step": self.step_count,
                "reward_prob": init_prob_val,
                "last_max_prob": self.last_max_prob,
                "should_steer": should_steer,
            })

        if not should_steer:
            return outputs

        # 5. 初始化优化目标
        optimized_latent = current_dense.detach().clone()
        
        # 6. 迭代优化循环
        for step in range(self.num_steps):
            target = optimized_latent.detach().clone().requires_grad_(True)
            rm_input = target 

            with torch.enable_grad():
                logits = self.reward_model(rm_input, return_logits=True)
                target_score = logits[:, -1, 0].sum()
                grad = torch.autograd.grad(target_score, target, create_graph=False)[0]

            if grad is None: grad = torch.zeros_like(target)

            # 监控逻辑 (最后一步打印)
            if self.monitor and step == (self.num_steps - 1):
                curr_logit = target_score.item() / batch_size
                final_prob = 1 / (1 + math.exp(-curr_logit)) if abs(curr_logit) < 50 else (1.0 if curr_logit > 0 else 0.0)
                
                # 🔥 [修改] 打印格式：显示 Init -> Final
                print(f"    [Tok {self.step_count:4}] Score: {init_logit_val:7.4f} -> {curr_logit:7.4f} | Prob: {init_prob_val:6.4f} -> {final_prob:6.4f}")

            # 梯度上升更新
            norm = torch.norm(grad, dim=-1, keepdim=True) + 1e-8
            optimized_latent = target + self.step_size * (grad / norm)
            
            del rm_input, logits, target_score, grad

        # 7. 得到最终 Steered Latent
        steered_dense = optimized_latent.detach().clone()

        if self.collect_sae_steer_trace:
            pre = current_dense[0, 0].detach().float().cpu().tolist()
            post = steered_dense[0, 0].detach().float().cpu().tolist()
            self.sae_steer_trace.append({"step": self.step_count, "pre": pre, "post": post})

        # 8. 监控：计算并打印 Top-10 维度差异
        if self.monitor:
            diff = (steered_dense[0, 0] - current_dense[0, 0]).cpu().numpy()
            original = current_dense[0, 0].cpu().numpy()
            
            print(f"      Initial Latents (Base):")
            base_line = " ".join([f"{v:6.2f}" for v in original])
            print(f"      {base_line}")
            
            print(f"      Steering Diff (Add):")
            diff_line = " ".join([f"{'+' if v>0 else ''}{v:6.2f}" for v in diff])
            print(f"      {diff_line}")
            print(f"      ---------------------------------------------------------------------")

        # 9. 投影回原始空间
        with torch.no_grad():
            delta_latent = (steered_dense - current_dense).view(-1, self.sae_dim)
            W = self.decoder_weight.data
            if W.shape[0] != self.sae_dim: W = W.t()
            
            delta_act = delta_latent @ W
            final_act = x.view(-1, x.shape[-1]) + delta_act
            final_act = final_act.view_as(x)

        if isinstance(outputs, tuple):
            return (final_act.to(original_act.dtype),) + outputs[1:]
        return final_act.to(original_act.dtype)
    

# =========================================================
# 2b. Judge 目录：筛题 / 回填 Base（steer_only）
# =========================================================
def _load_judge_verdicts_by_idx(judge_dir):
    """question_idx -> {base_correct, steer_correct}（同题多 shard JSON 时后者覆盖，内容应一致）"""
    verdicts = {}
    for p in _glob(os.path.join(judge_dir, "question_*_shard_*.json")):
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        idx = d.get("question_idx")
        if idx is None:
            continue
        verdicts[idx] = {
            "base_correct": bool(d.get("base_correct")),
            "steer_correct": bool(d.get("steer_correct")),
        }
    return verdicts


def _load_prior_judge_record(prior_dir, question_idx):
    paths = sorted(_glob(os.path.join(prior_dir, f"question_{question_idx}_shard_*.json")))
    if not paths:
        return None
    with open(paths[0], "r", encoding="utf-8") as f:
        return json.load(f)


def _load_example_idx_file(path):
    """每行一个整数，或 JSON：整数组 或 {\"question_indices\": [...]}"""
    path = os.path.abspath(path)
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [int(x) for x in data]
        if isinstance(data, dict) and "question_indices" in data:
            return [int(x) for x in data["question_indices"]]
        raise ValueError(f"无法解析 JSON 题号列表: {path}")
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(int(line.split()[0]))
    return out


def _unwrap_hf_pretrained_for_generate(lm):
    """必须用 transformers.PreTrainedModel.generate，不能走 Envoy 包装后的 .generate。

    nnsight 对部分模块返回的 ``generate`` 会先进入 ``trace()``，内部 ``inspect.getsourcelines``
    在少数帧上会抛 ``OSError: could not get source code``（长批次/深层栈时更明显），整条 shard 会直接挂掉。"""
    m = getattr(lm, "_model", lm)
    for _ in range(16):
        if isinstance(m, PreTrainedModel):
            return m
        nxt = getattr(m, "_module", None)
        if nxt is None:
            break
        m = nxt
    return getattr(lm, "_model", lm)


def _greedy_generate_ids(nnsight_model, inputs_batch, max_new_tokens, do_sample=False):
    """返回 HF generate 的 LongTensor（底层走 PreTrainedModel.generate，不经 nnsight tracer）。"""
    inner = _unwrap_hf_pretrained_for_generate(nnsight_model)
    batch = inputs_batch.to(nnsight_model.device)
    return inner.generate(**batch, max_new_tokens=max_new_tokens, do_sample=do_sample)


# =========================================================
# 3. 主程序
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Open-Reasoner-Zero/Open-Reasoner-Zero-7B')
    parser.add_argument('--sae_layer', type=int, default=20)
    parser.add_argument('--n_clusters', type=int, default=10)
    parser.add_argument('--reward_model_path', type=str, default='transformer_reward_model.pt')
    
    parser.add_argument('--step_size', type=float, default=2.0) 
    parser.add_argument('--num_steps', type=int, default=5)
    
    parser.add_argument('--num_examples', type=int, default=5, help="模式1：跑前N个题目")
    parser.add_argument('--example_idx', type=int, nargs='+', default=None, help="模式2：跑指定的idx列表")
    parser.add_argument(
        '--example_idx_file',
        type=str,
        default=None,
        metavar='PATH',
        help='模式2b：从文件读题号（每行一个整数，或 JSON 数组 / {\"question_indices\":[...]}）。与 --example_idx 互斥。',
    )
    
    parser.add_argument('--max_token', type=int, default=2000)
    parser.add_argument("--print_response", action="store_true", help="Print the full decoded response.")
    parser.add_argument(
        '--monitor',
        dest='monitor',
        action='store_true',
        help='开启逐 token steering 调试打印（Score/Prob、Initial Latents、Steering Diff）',
    )
    parser.add_argument('--no-monitor', dest='monitor', action='store_false', help='关闭上述调试打印（与默认行为相同）')
    parser.set_defaults(monitor=False)
    parser.add_argument('--shard_id', type=int, default=0, help="当前分片ID")
    parser.add_argument('--num_shards', type=int, default=1, help="总分片数")
    parser.add_argument('--output_file', type=str, default="iter_steer.out", help="基础输出文件名")
    parser.add_argument('--reward_threshold', type=float, default=0.95, help="reward_prob < 此值时直接 steer")
    parser.add_argument('--confidence_threshold', type=float, default=0.5, help="reward>=reward_threshold 时，last_max_prob < 此值才 steer")
    parser.add_argument(
        '--steer_every_token',
        action='store_true',
        help="对照：生成阶段每个 token 都执行 steering 迭代，不使用 reward_threshold / confidence_threshold 门控（仍用 RM 算梯度）。",
    )
    parser.add_argument('--save_token_records', action='store_true', help="保存每 token 的 reward_prob、last_max_prob、current_max_prob、should_steer 到 JSON")
    parser.add_argument('--token_records_dir', type=str, default='logs/token_records', help="token records JSON 输出目录")
    parser.add_argument(
        '--save_rm_path_scores',
        action='store_true',
        help="Steer 阶段收集每 token 的 pre-steer 稠密 SAE latent，生成结束后用 RM 对全长 [T,D] 前向，另存 p_last（末步 sigmoid）与 p_mean（逐步 sigmoid 平均）；不改变 steering 分支逻辑",
    )
    parser.add_argument(
        '--rm_path_scores_dir',
        type=str,
        default='logs/rm_path_scores',
        help='与 --save_rm_path_scores 配合：每题一个 JSON（question_<idx>_shard_<id>.json）',
    )
    parser.add_argument(
        '--save_sae_steer_trace',
        action='store_true',
        help='仅在实际发生 steer 的生成步记录 SAE 稠密 latent（长度 n_clusters）：pre=steer 前、post=迭代优化后；每题单独 JSON',
    )
    parser.add_argument(
        '--sae_steer_trace_dir',
        type=str,
        default='logs/sae_steer_trace',
        help='与 --save_sae_steer_trace 配合：question_<idx>_shard_<id>.json',
    )
    parser.add_argument(
        '--save_judge_reason',
        action='store_true',
        help="保存判题结果到 JSON（math：math_verify；GPQA：rule MCQ 字母抽取，见 gold_letter / extracted_letter_*）",
    )
    parser.add_argument(
        '--save_judge_with_text',
        action='store_true',
        help='与 --save_judge_reason 一起用：在 JSON 写入解码全文（不写 stdout）。'
        '正常双阶段：response_base 与 response_steer。'
        '--steer_only 时默认只写 response_steer；若需把旧 judge 里的 base 全文一并写入，请加 --save_judge_with_prior_base_text。',
    )
    parser.add_argument(
        '--save_judge_with_prior_base_text',
        action='store_true',
        help='仅在与 --steer_only 及 --save_judge_with_text 同时使用时生效：把 prior judge 中的 response_base 复制进新 JSON（默认不复制）。',
    )
    parser.add_argument(
        '--save_judge_prior_base_meta',
        action='store_true',
        help='与 --steer_only 及 --save_judge_reason 同时用：在 JSON 中写入旧 judge 的 base_correct / judge_base_reason / len_base（默认不写，仅保留本轮 steer 字段）。',
    )
    parser.add_argument('--judge_reason_dir', type=str, default='logs/judge_reasons', help="判题理由 JSON 输出目录")
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['math500', 'aime24', 'aime25', 'amc23', 'gpqa_diamond', 'mbpp', 'ineqmath'],
        default='math500',
        help='Dataset: math500, aime24, aime25, amc23, gpqa_diamond, mbpp, or ineqmath (AI4Math/IneqMath，见 --ineqmath_split / --ineqmath_test_limit)',
    )
    parser.add_argument(
        '--ineqmath_split',
        type=str,
        choices=['dev', 'test'],
        default='dev',
        metavar='SPLIT',
        help='ineqmath：dev.json（100 题，有 answer，可走 math_verify）或 test.json（200 题，无公开标答）。默认 dev。',
    )
    parser.add_argument(
        '--ineqmath_test_limit',
        type=int,
        default=200,
        metavar='N',
        help='ineqmath：仅用当前 split 的前 N 条（dev 最多 100、test 最多 200；默认 200 即跑满当前文件）。判题走 math_verify。',
    )
    parser.add_argument(
        '--baseline_only',
        action='store_true',
        help='只跑 baseline 生成 + 判题（不加载 SAE / reward / steering），math 等走 rule_eval/math_verify',
    )
    parser.add_argument(
        '--filter_from_judge_dir',
        type=str,
        default=None,
        metavar='DIR',
        help='读取 DIR 下 question_*_shard_*.json 的 base_correct/steer_correct，跳过「两者都对」的题目；'
        '无记录的题号仍会加入（便于补跑）。与 --example_idx / --example_idx_file 互斥时以命令行列表或文件为准。',
    )
    parser.add_argument(
        '--steer_only',
        action='store_true',
        help='不跑 Base generate，只跑 Steer。建议与 --filter_from_judge_dir 联用；Base 侧字段从 --prior_judge_dir（默认同 filter 目录）回填。',
    )
    parser.add_argument(
        '--prior_judge_dir',
        type=str,
        default=None,
        metavar='DIR',
        help='--steer_only 时从此目录读取旧 judge（用于日志统计与可选 --save_judge_prior_base_meta / --save_judge_with_prior_base_text）；默认等于 --filter_from_judge_dir。',
    )
    parser.add_argument(
        '--attn_implementation',
        type=str,
        default=os.environ.get('ATTN_IMPLEMENTATION', 'flash_attention_2'),
        metavar='STR',
        help='HF attention: flash_attention_2（需 flash-attn，失败则自动 sdpa）| sdpa | eager。可用环境变量 ATTN_IMPLEMENTATION 覆盖。',
    )
    args = parser.parse_args()

    if args.baseline_only and args.steer_only:
        parser.error('--baseline_only 与 --steer_only 不能同时使用')
    if args.example_idx is not None and args.example_idx_file:
        parser.error('--example_idx 与 --example_idx_file 不能同时使用')

    attn = (args.attn_implementation or "").strip().lower()
    if attn in ("none", "default", ""):
        attn_impl = None
    else:
        attn_impl = args.attn_implementation

    print(f"🚀 Loading LLM: {args.model}...")
    if attn_impl:
        print(f"   attn_implementation={attn_impl}")
    model, tokenizer = load_model(model_name=args.model, device="cuda", attn_implementation=attn_impl)
    raw_model = model.model

    if args.baseline_only:
        sae = reward_model = hook = None
        layer_module = lm_head = None
        print("⏭️  baseline_only: skip SAE, reward model, steering")
    else:
        sae_model_id = _sae_model_id_from_hf_model(args.model)
        print(f"🧩 Loading SAE (checkpoint id={sae_model_id}, layer={args.sae_layer}, clusters={args.n_clusters})...")
        sae, _ = load_sae(sae_model_id, args.sae_layer, args.n_clusters)
        sae = sae.to(model.device)
        sae.k = args.n_clusters

        print(f"🧠 Loading Transformer Reward Model...")
        reward_model = LatentTransformer(
            input_dim=args.n_clusters, d_model=128
        ).to(model.device)

        reward_model.load_state_dict(torch.load(args.reward_model_path, map_location=model.device))
        reward_model.eval()

        for p in reward_model.parameters():
            p.requires_grad_(False)

        print("✅ Reward Model loaded successfully.")

    if args.dataset == "aime24":
        dataset = load_dataset("HuggingFaceH4/aime_2024")["train"]  # 30 题
    elif args.dataset == "aime25":
        dataset = load_dataset("yentinglin/aime_2025")["train"]  # 30 题
    elif args.dataset == "amc23":
        dataset = load_dataset("math-ai/amc23", split="test")  # 40 题，数值/整数标准答
    elif args.dataset == "gpqa_diamond":
        dataset = load_dataset("nichenshun/gpqa_diamond")["train"]  # 198 题
    elif args.dataset == "mbpp":
        dataset = load_dataset("google-research-datasets/mbpp", "full")["test"]  # 500 题
    elif args.dataset == "ineqmath":
        # HF 上 `load_dataset("AI4Math/IneqMath", ...)` 会因 train schema 在 builder 阶段失败；用 json 直链单文件。
        _ineq_json = (
            "dev.json"
            if args.ineqmath_split == "dev"
            else "test.json"
        )
        _ineq_url = (
            "https://huggingface.co/datasets/AI4Math/IneqMath/resolve/main/json/"
            + _ineq_json
        )
        _ineq_raw = load_dataset("json", data_files=_ineq_url, split="train")
        lim = max(1, min(int(args.ineqmath_test_limit), len(_ineq_raw)))
        dataset = _ineq_raw.select(range(lim))
        _gold_note = (
            "（含 answer，math_verify 可用）"
            if args.ineqmath_split == "dev"
            else "（无公开 answer/solution，判分多为 rule_math:no_gold）"
        )
        print(
            f"📐 IneqMath: {_ineq_json} (HF repo){_gold_note}, first {lim} / {len(_ineq_raw)} problems "
            f"(--ineqmath_split / --ineqmath_test_limit; use --num_examples >= {lim} for Mode 1 full run)."
        )
        print(
            "   判题：global question_idx < 50 → math_verify；≥ 50 → extract_choice_letter 对金标 (A)–(F)（与 GPQA 同源）。"
        )
        if args.ineqmath_split == "test" and lim > 50:
            print(
                "⚠️  test.json 在 idx≥50 处不全是 relation；按题号半分字母判分仅供试探，与 dev 含义不同。"
            )
    else:
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    
    prior_judge_dir = args.prior_judge_dir or args.filter_from_judge_dir
    if args.steer_only and not prior_judge_dir:
        print("⚠️  --steer_only 但未设置 --prior_judge_dir / --filter_from_judge_dir：Base 字段将无法从旧 judge 回填。")

    # 确定要跑的题目列表
    if args.example_idx is not None:
        all_indices = list(args.example_idx)
        mode_str = "Mode 2 (Specific List)"
    elif args.example_idx_file:
        all_indices = _load_example_idx_file(args.example_idx_file)
        mode_str = f"Mode 2b (example_idx_file={args.example_idx_file}, n={len(all_indices)})"
    elif args.filter_from_judge_dir:
        verdicts = _load_judge_verdicts_by_idx(args.filter_from_judge_dir)
        n_ds = len(dataset)
        all_indices = []
        both_ok_skip = 0
        for i in range(n_ds):
            if i not in verdicts:
                all_indices.append(i)
            else:
                b_ok = verdicts[i]["base_correct"]
                s_ok = verdicts[i]["steer_correct"]
                if b_ok and s_ok:
                    both_ok_skip += 1
                else:
                    all_indices.append(i)
        mode_str = (
            f"Mode filter_from_judge ({len(verdicts)} verdicts, both_ok skipped={both_ok_skip}, "
            f"to_run={len(all_indices)})"
        )
    else:
        n_cap = min(args.num_examples, len(dataset))
        if n_cap < args.num_examples:
            print(f"⚠️  num_examples={args.num_examples} > len(dataset)={len(dataset)}；仅用前 {n_cap} 题。")
        all_indices = list(range(n_cap))
        mode_str = "Mode 1 (First N)"

    # 执行分片过滤
    target_indices = [
        x for i, x in enumerate(all_indices) 
        if i % args.num_shards == args.shard_id
    ]

    print(f"🎯 {mode_str} | Total: {len(all_indices)} | 🚀 Shard {args.shard_id}/{args.num_shards} claimed: {len(target_indices)} tasks")

    if not args.baseline_only:
        lm_head = getattr(model, 'lm_head', None)
        if lm_head is None:
            lm_head = getattr(model.model, 'lm_head', None)
        if lm_head is None:
            raise RuntimeError("Could not find lm_head on model. Cannot use confidence-based steering.")

        hook = IterativeTransformerSteeringHook(
            sae,
            reward_model,
            step_size=args.step_size,
            num_steps=args.num_steps,
            sae_dim=args.n_clusters,
            monitor=args.monitor,
            reward_threshold=args.reward_threshold,
            confidence_threshold=args.confidence_threshold,
            lm_head=lm_head,
            record_tokens=args.save_token_records,
            steer_every_token=args.steer_every_token,
            collect_sae_latent_path=args.save_rm_path_scores,
            collect_sae_steer_trace=args.save_sae_steer_trace,
        )
        layer_module = model.model.layers[args.sae_layer]
        print(f"\n>>> Running Iterative Steering (Step={args.step_size}, Iter={args.num_steps})")
        if args.steer_every_token:
            print("    Steering condition: --steer_every_token（每生成 token 均 steer，忽略 reward/conf 门控）")
        else:
            print(f"    Steering condition: reward<{args.reward_threshold} → steer; reward>={args.reward_threshold} & last_max_prob<{args.confidence_threshold} → steer; else skip")
        if args.save_rm_path_scores:
            print(f"    RM path scores: p_last / p_mean → {args.rm_path_scores_dir}/question_<idx>_shard_<id>.json")
        if args.save_sae_steer_trace:
            print(f"    SAE steer trace (pre/post {args.n_clusters}D, steer-only steps) → {args.sae_steer_trace_dir}/question_<idx>_shard_<id>.json")
    else:
        print(f"\n>>> Running baseline only (max_new_tokens={args.max_token}, judge=rule / math_verify for math)")

    stats = {
        "base_correct": 0,
        "steer_correct": 0,
        "base_correct_lens": [],
        "steer_correct_lens": []
    }


    _COT_SUFFIX = (
          "\n\nLet's think step by step. "
       )
    _FEWSHOT_MATH = (
        "Here are 5 example problems and their solutions.\n\n"
        "Example 1:\n"
        "Problem: If $3x + 2 = 11$, what is the value of $x^2$?\n"
        "Solution: From $3x + 2 = 11$ we get $3x = 9$, so $x = 3$. Therefore $x^2 = 9$.\n"
        "Final answer: \\boxed{9}\n\n"
        "Example 2:\n"
        "Problem: What is the smallest positive integer divisible by both 6 and 8?\n"
        "Solution: Prime factorize: $6 = 2 \\cdot 3$ and $8 = 2^3$. So $\\mathrm{lcm}(6,8) = 2^3 \\cdot 3 = 24$.\n"
        "Final answer: \\boxed{24}\n\n"
        "Example 3:\n"
        "Problem: A right triangle has legs of length 5 and 12. What is the length of the hypotenuse?\n"
        "Solution: By the Pythagorean theorem, $c = \\sqrt{5^2 + 12^2} = \\sqrt{169} = 13$.\n"
        "Final answer: \\boxed{13}\n\n"
        "Example 4:\n"
        "Problem: How many ways are there to choose 3 books from a shelf of 7 distinct books?\n"
        "Solution: This is $\\binom{7}{3} = \\frac{7 \\cdot 6 \\cdot 5}{3 \\cdot 2 \\cdot 1} = 35$.\n"
        "Final answer: \\boxed{35}\n\n"
        "Example 5:\n"
        "Problem: What is the sum of the first 100 positive integers?\n"
        "Solution: Using $\\sum_{k=1}^{n} k = \\frac{n(n+1)}{2}$ with $n = 100$, the sum is $\\frac{100 \\cdot 101}{2} = 5050$.\n"
        "Final answer: \\boxed{5050}\n\n"
        "Now solve this problem.\n\nProblem: "
    )

    _FEWSHOT_GPQA = (
        "Here are 5 example multiple-choice problems and their solutions.\n\n"
        "Example 1:\n"
        "Problem: A ball is dropped from a height of 20 m. How long does it take to hit the ground? (Ignore air resistance, $g = 10\\,\\mathrm{m/s^2}$.)\n"
        "Answer Choices:\n(A) 1 s\n(B) 2 s\n(C) 4 s\n(D) 5 s\n"
        "Solution: Using $h = \\frac{1}{2} g t^2$, solve $20 = 5 t^2$, so $t^2 = 4$ and $t = 2$.\n"
        "Final answer: \\boxed{B}\n\n"
        "Example 2:\n"
        "Problem: Which of the following has the highest electronegativity?\n"
        "Answer Choices:\n(A) Carbon\n(B) Nitrogen\n(C) Oxygen\n(D) Fluorine\n"
        "Solution: Electronegativity increases across a period and decreases down a group. Fluorine has the highest value on the Pauling scale.\n"
        "Final answer: \\boxed{D}\n\n"
        "Example 3:\n"
        "Problem: Which organelle is responsible for ATP production via oxidative phosphorylation in eukaryotic cells?\n"
        "Answer Choices:\n(A) Nucleus\n(B) Ribosome\n(C) Mitochondrion\n(D) Endoplasmic reticulum\n"
        "Solution: The mitochondrion houses the electron transport chain and ATP synthase, producing ATP through oxidative phosphorylation.\n"
        "Final answer: \\boxed{C}\n\n"
        "Example 4:\n"
        "Problem: A 2 kg object moves at 3 m/s. What is its kinetic energy?\n"
        "Answer Choices:\n(A) 3 J\n(B) 6 J\n(C) 9 J\n(D) 12 J\n"
        "Solution: $KE = \\frac{1}{2} m v^2 = \\frac{1}{2}(2)(3)^2 = 9\\,\\mathrm{J}$.\n"
        "Final answer: \\boxed{C}\n\n"
        "Example 5:\n"
        "Problem: What is the pH of a $10^{-3}$ M HCl solution?\n"
        "Answer Choices:\n(A) 1\n(B) 3\n(C) 7\n(D) 11\n"
        "Solution: HCl fully dissociates, so $[\\mathrm{H}^+] = 10^{-3}$ M. Then $\\mathrm{pH} = -\\log(10^{-3}) = 3$.\n"
        "Final answer: \\boxed{B}\n\n"
        "Now solve this problem.\n\nProblem: "
    )

    _FEWSHOT_INEQMATH = (
        "Here are 5 example inequality problems and their solutions.\n\n"
        "Example 1:\n"
        "Problem: Find the largest constant $C$ such that $x^2 + y^2 \\geq C \\cdot xy$ for all real $x, y$.\n"
        "Solution: $x^2 + y^2 - 2xy = (x - y)^2 \\geq 0$, so $x^2 + y^2 \\geq 2xy$. Taking $x = y$ gives equality, so the largest valid constant is $2$.\n"
        "Final answer: \\boxed{2}\n\n"
        "Example 2:\n"
        "Problem: For all positive reals $a$ and $b$, determine the relation between $a + b$ and $2\\sqrt{ab}$.\n"
        "Solution: By AM-GM, $\\frac{a+b}{2} \\geq \\sqrt{ab}$, so $a + b \\geq 2\\sqrt{ab}$.\n"
        "Final answer: \\boxed{\\geq}\n\n"
        "Example 3:\n"
        "Problem: Find the smallest constant $C$ such that $|x + y| \\leq C \\cdot (|x| + |y|)$ for all real $x, y$.\n"
        "Solution: By the triangle inequality, $|x + y| \\leq |x| + |y|$, so $C = 1$ works. Taking $x = y = 1$ gives equality, so $C = 1$ is smallest.\n"
        "Final answer: \\boxed{1}\n\n"
        "Example 4:\n"
        "Problem: For positive reals $a, b, c$, determine the relation between $a^2 + b^2 + c^2$ and $ab + bc + ca$.\n"
        "Solution: $a^2 + b^2 + c^2 - ab - bc - ca = \\frac{1}{2}\\left((a-b)^2 + (b-c)^2 + (c-a)^2\\right) \\geq 0$, so $a^2 + b^2 + c^2 \\geq ab + bc + ca$.\n"
        "Final answer: \\boxed{\\geq}\n\n"
        "Example 5:\n"
        "Problem: Find the largest constant $C$ such that $(a+b)^2 \\geq C \\cdot ab$ for all positive reals $a, b$.\n"
        "Solution: By AM-GM, $a + b \\geq 2\\sqrt{ab}$, so $(a+b)^2 \\geq 4ab$. Taking $a = b$ gives equality, so $C = 4$.\n"
        "Final answer: \\boxed{4}\n\n"
        "Now solve this problem.\n\nProblem: "
    )
    # 根据 dataset 确定字段映射和判题类型
    CODING_TASK_PREFIX = "Task: Write a single Python function for the following problem. Do not include tests or examples in your output."
    if args.dataset == "gpqa_diamond":
        def get_item_fields(item):
            q = item['question']
            sol = item['solution']
            # solution 格式 "Answer: C"，提取字母
            ans = sol.replace("Answer:", "").strip().upper() if sol else ""
            if len(ans) > 1:
                ans = ans[0]  # 取第一个字符
            
            q = q + _FEWSHOT_GPQA

            return q, sol, ans, None
        dataset_type = "mcqa"
    elif args.dataset == "mbpp":
        # 与 MBPP 评测题无关的示意 few-shot：强调单 fence + 需要时显式 import（减少 NameError）。
        _MBPP_FEWSHOT = (
            "Below is one ILLUSTRATIVE example (not your real task). Copy the same response shape.\n\n"
            "Example (illustrative)\n"
            "Problem: Return the square root of the sum of a list of integers.\n\n"
            "Public Tests (illustrative):\n"
            "assert norm_sum([3, 4]) == 5.0\n\n"
            "```python\n"
            "import math\n"
            "from typing import List\n\n"
            "def norm_sum(values: List[int]) -> float:\n"
            "    return math.sqrt(float(sum(values)))\n"
            "```\n\n"
            "---\n"
            "Your assignment (use the SAME format as the example):\n\n"
        )
        _MBPP_FORMAT = (
            "Output format (required — follow exactly so your code can be parsed):\n"
            "1. Your entire reply must contain exactly ONE Markdown fenced code block tagged as Python.\n"
            "   Start with a line that is only: ```python\n"
            "   Then put only valid Python (the implementation). End with a line that is only: ```\n"
            "2. Inside that block: no natural language, no bullet lists, no nested ``` fences, "
            "no HTML/XML tags (e.g. <answer>, </python>), and do not paste the \"Public Tests\" lines into your answer.\n"
            "3. If you use math, re, collections, itertools, or typing (e.g. List, Dict, Optional), "
            "add the necessary import/from lines at the top of the code block.\n"
            "4. Outside that single block you may write at most a few words if needed; prefer code only.\n"
        )

        def get_item_fields(item):
            text = item["text"]
            test_list = item.get("test_list", [])
            tests_section = "\n\nPublic Tests:\n" + "\n".join(test_list) if test_list else ""
            tgt = _mbpp_target_function_name(test_list)
            name_rule = (
                f"5. Define the solution as `def {tgt}(...):` using this exact function name "
                f"(the tests call `{tgt}`).\n"
                if tgt
                else "5. Define the function using the exact name appearing in the Public Tests `assert ...` lines.\n"
            )
            question = (
                f"{_MBPP_FEWSHOT}"
                f"{CODING_TASK_PREFIX}\n\nProblem: {text}{tests_section}\n\n"
                f"{_MBPP_FORMAT}{name_rule}"
            )
            code = item["code"]
            return question, code, code, test_list
        dataset_type = "coding"
    elif args.dataset == "amc23":
        def get_item_fields(item):
            q = item["question"]
            a = str(item["answer"]).strip()
            q = _FEWSHOT_INEQMATH + q
            return q, a, a, None
        dataset_type = "math"
    else:
        # MATH-500、AIME24/25、IneqMath：仅 user=题干文本（与其它数学集一致的 zero-shot）
        def get_item_fields(item):
            return _FEWSHOT_MATH + item['problem'], item['solution'], item['answer'], None
        dataset_type = "math"

    for i in target_indices:
        try:
            item = dataset[i]
            question, solution, answer, test_list = get_item_fields(item)
            _ineq_qidx = i if args.dataset == "ineqmath" else None

            print(f"\n======== Question {i} ========")
            messages = [{"role": "user", "content": question}]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            input_len = inputs.input_ids.shape[1]

            correct_ref = answer if dataset_type == "mcqa" else solution
            out_base = None
            res_base = None
            len_base = 0
            is_base_correct = False
            judge_base = "N/A"
            seconds_base = None
            seconds_steer = None

            # 1. Baseline（steer_only 时跳过生成，从 prior judge 回填）
            if args.steer_only:
                print("1️⃣ Baseline: skipped (--steer_only)", flush=True)
                prior_record = (
                    _load_prior_judge_record(prior_judge_dir, i) if prior_judge_dir else None
                )
                if prior_record is not None:
                    is_base_correct = bool(prior_record.get("base_correct"))
                    judge_base = prior_record.get("judge_base_reason") or "N/A"
                    len_base = int(prior_record.get("len_base") or 0)
                    res_base = prior_record.get("response_base")
                else:
                    is_base_correct = False
                    judge_base = "N/A (no prior judge JSON)"
                    len_base = 0
                    res_base = None
                if is_base_correct:
                    stats["base_correct"] += 1
                    stats["base_correct_lens"].append(len_base)
            else:
                print("1️⃣ Generating Baseline...", end="", flush=True)
                t0_base = time.perf_counter()
                with torch.no_grad():
                    out_base = _greedy_generate_ids(model, inputs, args.max_token, do_sample=False)
                seconds_base = time.perf_counter() - t0_base

                len_base = out_base.shape[1] - input_len
                res_base = tokenizer.decode(out_base[0][input_len:], skip_special_tokens=False)
                print(f" Done. (Len: {len_base})")

                if args.print_response:
                    print("\n" + "="*20 + " [Baseline Output] " + "="*20)
                    print(res_base)
                    print("="*60 + "\n")
                is_base_correct, judge_base = evaluate_answer(
                    res_base, correct_ref, answer, question, "Baseline",
                    dataset_type=dataset_type, test_list=test_list,
                    ineqmath_question_idx=_ineq_qidx,
                )
                if is_base_correct:
                    stats["base_correct"] += 1
                    stats["base_correct_lens"].append(len_base)

            if args.baseline_only:
                if args.save_judge_reason:
                    os.makedirs(args.judge_reason_dir, exist_ok=True)
                    out_path = os.path.join(args.judge_reason_dir, f"question_{i}_shard_{args.shard_id}.json")
                    payload = {
                        "question_idx": i,
                        "baseline_only": True,
                        "base_correct": is_base_correct,
                        "judge_base_reason": judge_base,
                        "len_base": len_base,
                        "seconds_base": seconds_base,
                    }
                    if dataset_type == "mcqa":
                        payload.update(_mcqa_judge_json_extra(answer, res_base, None))
                    if args.save_judge_with_text:
                        payload["response_base"] = res_base
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(payload, f, indent=2, ensure_ascii=False)
                print("-" * 50)
                print(f"Base [{'✅' if is_base_correct else '❌'}] Len: {len_base} | judge: {judge_base[:120]}...")
                print("-" * 50)
                continue

            # 2. Steered
            print("2️⃣ Generating Steered...", end="", flush=True)
            hook.reset() 
            handle = layer_module.register_forward_hook(hook)
            conf_handle = lm_head.register_forward_hook(make_confidence_hook(hook))
            
            res_steer = "[FAILED]" 
            len_steer = 0
            is_steer_correct = False
            judge_steer = "N/A"
            
            try:
                t0_steer = time.perf_counter()
                out_steer = _greedy_generate_ids(model, inputs, args.max_token, do_sample=False)
                seconds_steer = time.perf_counter() - t0_steer

                len_steer = out_steer.shape[1] - input_len
                res_steer = tokenizer.decode(out_steer[0][input_len:], skip_special_tokens=False)
                print(f" Done. (Len: {len_steer})")
                
                if args.print_response:
                    print("\n" + "="*20 + " [Steered Output] " + "="*20)
                    print(res_steer)
                    print("="*60 + "\n")
                is_steer_correct, judge_steer = evaluate_answer(
                    res_steer,
                    correct_ref,
                    answer,
                    question,
                    "Steered",
                    dataset_type=dataset_type,
                    test_list=test_list,
                    ineqmath_question_idx=_ineq_qidx,
                )
                if is_steer_correct: 
                    stats["steer_correct"] += 1
                    stats["steer_correct_lens"].append(len_steer)

                # 保存判题理由（便于分析做错原因）
                if args.save_judge_reason:
                    os.makedirs(args.judge_reason_dir, exist_ok=True)
                    out_path = os.path.join(args.judge_reason_dir, f"question_{i}_shard_{args.shard_id}.json")
                    if args.steer_only:
                        payload = {
                            "question_idx": i,
                            "steer_correct": is_steer_correct,
                            "judge_steer_reason": judge_steer,
                            "len_steer": len_steer,
                            "steer_only_rerun": True,
                        }
                        if args.save_judge_prior_base_meta:
                            payload["base_correct"] = is_base_correct
                            payload["judge_base_reason"] = judge_base
                            payload["len_base"] = len_base
                        if dataset_type == "mcqa":
                            ex = _mcqa_judge_json_extra(answer, res_base, res_steer)
                            payload.update(ex)
                    else:
                        payload = {
                            "question_idx": i,
                            "base_correct": is_base_correct,
                            "steer_correct": is_steer_correct,
                            "judge_base_reason": judge_base,
                            "judge_steer_reason": judge_steer,
                            "len_base": len_base,
                            "len_steer": len_steer,
                        }
                        if dataset_type == "mcqa":
                            payload.update(_mcqa_judge_json_extra(answer, res_base, res_steer))
                    payload["steer_event_count"] = len(hook.sae_steer_trace)
                    payload["seconds_steer"] = seconds_steer
                    if seconds_base is not None:
                        payload["seconds_base"] = seconds_base
                    if args.save_judge_with_text:
                        payload["response_steer"] = res_steer
                        if (not args.steer_only or args.save_judge_with_prior_base_text) and res_base is not None:
                            payload["response_base"] = res_base
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(payload, f, indent=2, ensure_ascii=False)

                # 合并 token_records 与 current_max_probs，保存到 JSON
                if args.save_token_records and hook.token_records:
                    os.makedirs(args.token_records_dir, exist_ok=True)
                    records = []
                    for j, rec in enumerate(hook.token_records):
                        r = dict(rec)
                        r["current_max_prob"] = hook.current_max_probs[j] if j < len(hook.current_max_probs) else None
                        records.append(r)
                    out_path = os.path.join(args.token_records_dir, f"question_{i}_shard_{args.shard_id}.json")
                    tr_payload = {
                        "question_idx": i,
                        "steer_correct": is_steer_correct,
                        "len_steer": len_steer,
                        "records": records,
                    }
                    if args.steer_only:
                        tr_payload["steer_only_rerun"] = True
                        if args.save_judge_prior_base_meta:
                            tr_payload["base_correct"] = is_base_correct
                            tr_payload["len_base"] = len_base
                    else:
                        tr_payload["base_correct"] = is_base_correct
                        tr_payload["len_base"] = len_base
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(tr_payload, f, indent=2, ensure_ascii=False)

                if args.save_sae_steer_trace:
                    os.makedirs(args.sae_steer_trace_dir, exist_ok=True)
                    trace_path = os.path.join(
                        args.sae_steer_trace_dir,
                        f"question_{i}_shard_{args.shard_id}.json",
                    )
                    trace_payload = {
                        "question_idx": i,
                        "sae_dim": args.n_clusters,
                        "steer_event_count": len(hook.sae_steer_trace),
                        "events": hook.sae_steer_trace,
                    }
                    if args.steer_only:
                        trace_payload["steer_only_rerun"] = True
                    with open(trace_path, "w", encoding="utf-8") as f:
                        json.dump(trace_payload, f, indent=2, ensure_ascii=False)

                if args.save_rm_path_scores and hook.sae_latent_path:
                    os.makedirs(args.rm_path_scores_dir, exist_ok=True)
                    path_tensor = torch.stack(hook.sae_latent_path, dim=0)
                    rm_dev = next(reward_model.parameters()).device
                    rm_dtype = next(reward_model.parameters()).dtype
                    with torch.no_grad():
                        x_rm = path_tensor.unsqueeze(0).to(device=rm_dev, dtype=rm_dtype)
                        logits_full = reward_model(x_rm, return_logits=True)
                        probs_full = torch.sigmoid(logits_full.float())
                    p_last = probs_full[0, -1, 0].item()
                    p_mean = probs_full[0, :, 0].mean().item()
                    rm_out = os.path.join(
                        args.rm_path_scores_dir,
                        f"question_{i}_shard_{args.shard_id}.json",
                    )
                    rm_payload = {
                        "question_idx": i,
                        "T": int(path_tensor.shape[0]),
                        "p_last": p_last,
                        "p_mean": p_mean,
                        "steer_correct": is_steer_correct,
                        "len_steer": len_steer,
                        "base_correct": is_base_correct,
                    }
                    if args.steer_only:
                        rm_payload["steer_only_rerun"] = True
                    with open(rm_out, "w", encoding="utf-8") as f:
                        json.dump(rm_payload, f, indent=2, ensure_ascii=False)
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n❌ OOM Error on idx {i}. Skipping this item.")
                else:
                    print(f"\n❌ Runtime Error: {e}")
                    traceback.print_exc()
            except Exception as e:
                print(f"\n❌ Error: {e}")
                traceback.print_exc()
            finally:
                handle.remove()
                conf_handle.remove()

            # Report
            print("-" * 50)
            status_base = "✅" if is_base_correct else "❌"
            status_steer = "✅" if is_steer_correct else "❌"
            if args.steer_only:
                print(
                    f"Base  (prior, 未本轮生成)[{status_base}] Len: {len_base} | judge: {judge_base[:240]}"
                )
            else:
                print(f"Base  [{status_base}] Len: {len_base} | judge: {judge_base[:240]}")
            print(f"Steer [{status_steer}] Len: {len_steer} | judge: {judge_steer[:240]}")
            if not is_base_correct and is_steer_correct: print("🚀 SUCCESS! Steering fixed the error!")
            print("-" * 50)

        finally:
            del inputs
            if 'out_base' in locals(): del out_base
            if 'out_steer' in locals(): del out_steer
            torch.cuda.empty_cache()
            gc.collect()

    # === Final Statistics ===
    print("\n" + "="*50)
    if args.baseline_only:
        print(f"📊 Final Report (baseline_only, max_token={args.max_token})")
        print(f"  Base : {stats['base_correct']}/{len(target_indices)}")
    else:
        print(f"📊 Final Report (Step={args.step_size}, Iter={args.num_steps})")
        print(f"Accuracy:")
        if args.steer_only:
            print(f"  Base : {stats['base_correct']}/{len(target_indices)} (来自 prior judge，未重新生成)")
        else:
            print(f"  Base : {stats['base_correct']}/{len(target_indices)}")
        print(f"  Steer: {stats['steer_correct']}/{len(target_indices)}")

        print(f"\nAverage Token Length (Correct Answers Only):")
        avg_base_len = np.mean(stats["base_correct_lens"]) if stats["base_correct_lens"] else 0
        avg_steer_len = np.mean(stats["steer_correct_lens"]) if stats["steer_correct_lens"] else 0

        print(f"  Base : {avg_base_len:.1f} tokens")
        print(f"  Steer: {avg_steer_len:.1f} tokens")

        delta = avg_steer_len - avg_base_len
        if delta > 0: print(f"📈 On average, steering increased thinking by +{delta:.1f} tokens.")
        else: print(f"📉 On average, steering decreased thinking by {delta:.1f} tokens.")

    if args.baseline_only:
        avg_base_len = np.mean(stats["base_correct_lens"]) if stats["base_correct_lens"] else 0
        print(f"\nAverage Token Length (Base correct only): {avg_base_len:.1f} tokens")
    print("="*50)

if __name__ == "__main__":
    main()