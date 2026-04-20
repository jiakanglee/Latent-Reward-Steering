import math
import random
import gc
from contextlib import contextmanager
from typing import List, Tuple, Callable, Optional, Union

import torch
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

torch.autograd.set_detect_anomaly(True)

# =============================================================
# 1.  Context‑manager helpers
# =============================================================

@contextmanager
def hf_hooks_contextmanager(model, hook_infos: List[Tuple[int, Callable]]):
    """Attach *forward‑pre* hooks to a HF decoder‑only model inside a 'with' block."""
    hooks = [model.model.layers[layer].register_forward_pre_hook(hook) for layer, hook in hook_infos]
    try:
        yield
    finally:
        for h in hooks:
            h.remove()

# =============================================================
# 2.  Batch hook factories
# =============================================================

def make_batch_linear_hook(
    vector: torch.Tensor,
    slices: List[slice],
    static_vectors: list,
    *,
    projection_clamp: bool,
):
    """Create a forward_pre batch hook for linear steering over row-wise `slices`.

    Assumptions:
    - `vector` is (d_model,)
    - Every static vector in `static_vectors` is (d_model,)
    - Each slice in `slices` indexes tokens dimension
    """

    def hook_fn(_module, args):
        (x,) = args
        assert x.dim() == 3, "Expected hidden states of shape (batch, seq, d_model)"
        d_model = x.shape[-1]

        v_local = vector.to(x)
        assert v_local.dim() == 1 and v_local.shape[0] == d_model, "vector must be 1-D of length d_model"

        stat_vecs_on_device = [sv.to(x.device) for sv in static_vectors]
        for sv in stat_vecs_on_device:
            assert sv.dim() == 1 and sv.shape[0] == d_model, "static vector must be 1-D of length d_model"

        x_new = x.clone()
        for row, sl in enumerate(slices):
            seg = x[row, sl]
            if projection_clamp:
                coef = (seg @ v_local) / (v_local.norm() ** 2)
                y = seg - coef.unsqueeze(-1) * v_local + v_local
            else:
                y = seg + v_local
            if stat_vecs_on_device:
                for sv in stat_vecs_on_device:
                    y = y + sv
            x_new[row, sl] = y

        return (x_new,)

    return hook_fn


def make_batch_adaptive_linear_hook(
    vector: torch.Tensor,
    W1: torch.Tensor,
    b1: torch.Tensor,
    W2: torch.Tensor,
    b2: torch.Tensor,
    slices: List[slice],
    static_vectors: list,
    *,
    projection_clamp: bool,
):
    """Create a forward_pre batch hook for *adaptive linear* steering.

    Assumptions:
    - `vector` is (d_model,)
    - MLP parameters have shapes: W1: (d_model, hidden), b1: (hidden,), W2: (hidden, 1), b2: (1,)
    - Each slice in `slices` indexes tokens dimension
    - Output gate is per-token scalar in [0,1] via sigmoid
    """

    def hook_fn(_module, args):
        (x,) = args
        assert x.dim() == 3, "Expected hidden states of shape (batch, seq, d_model)"
        d_model = x.shape[-1]

        v_local = vector.to(x)
        W1l = W1.to(x)
        b1l = b1.to(x)
        W2l = W2.to(x)
        b2l = b2.to(x)

        assert v_local.dim() == 1 and v_local.shape[0] == d_model, "vector must be 1-D of length d_model"
        assert W1l.shape[0] == d_model and W2l.shape[1] == 1, "W1 must map d_model->hidden; W2 must map hidden->1"
        assert b1l.dim() == 1 and b1l.shape[0] == W1l.shape[1], "b1 must be (hidden,)"
        assert b2l.numel() == 1, "b2 must be scalar"

        stat_vecs_on_device = [sv.to(x.device) for sv in static_vectors]
        for sv in stat_vecs_on_device:
            assert sv.dim() == 1 and sv.shape[0] == d_model, "static vector must be 1-D of length d_model"

        x_new = x.clone()
        for row, sl in enumerate(slices):
            seg = x[row, sl]
            # Shape assertions
            assert seg.dim() == 2 and seg.shape[-1] == d_model, "segment must be (tokens, d_model)"
            # Compute per-token gate g in [0,1]
            h = torch.matmul(seg, W1l) + b1l  # (tokens, hidden)
            h = torch.nn.functional.gelu(h)
            g = torch.matmul(h, W2l) + b2l  # (tokens, 1)
            g = torch.sigmoid(g).squeeze(-1)  # (tokens,)
            assert g.dim() == 1 and g.shape[0] == seg.shape[0], "gate must be (tokens,)"

            if projection_clamp:
                coef = (seg @ v_local) / (v_local.norm() ** 2)
                y = seg - coef.unsqueeze(-1) * v_local + g.unsqueeze(-1) * v_local
            else:
                y = seg + g.unsqueeze(-1) * v_local

            if stat_vecs_on_device:
                for sv in stat_vecs_on_device:
                    y = y + sv
            x_new[row, sl] = y

        return (x_new,)

    return hook_fn


def make_batch_resid_lora_hook(
    A: torch.Tensor,
    B: torch.Tensor,
    alpha: torch.Tensor,
    slices: List[slice],
    static_loras: list,
):
    """Create a forward_pre batch hook for residual LoRA steering over row-wise `slices`.

    Assumptions:
    - A: (d_model, r), B: (r, d_model), alpha: scalar tensor
    - static_loras: list of dicts with keys 'A','B','alpha'
    """

    def hook_fn(_module, args):
        (x,) = args
        assert x.dim() == 3, "Expected hidden states of shape (batch, seq, d_model)"
        d_model = x.shape[-1]

        Al = A.to(x)
        Bl = B.to(x)
        al = alpha.to(x)
        assert Al.shape[0] == d_model and Bl.shape[1] == d_model, "A and B must map d_model -> r -> d_model"

        x_new = x.clone()
        for row, sl in enumerate(slices):
            seg = x[row, sl]
            seg_det = seg.detach()
            y = seg_det + al * ((seg_det @ Al) @ Bl)
            if static_loras:
                for l in static_loras:
                    assert isinstance(l, dict) and all(k in l for k in ("A","B","alpha")), "LoRA static expects dicts with A,B,alpha"
                    Al_s = l['A'].to(x)
                    Bl_s = l['B'].to(x)
                    al_s = l['alpha'].to(x)
                    assert Al_s.shape[0] == d_model and Bl_s.shape[1] == d_model, "Static A,B must map d_model -> r -> d_model"
                    y = y + al_s * ((seg_det @ Al_s) @ Bl_s)
            x_new[row, sl] = y

        return (x_new,)

    return hook_fn

# =============================================================
# 3.  Generic cosine‑anneal scheduler (optionally disabled)
# =============================================================

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=0):
    base_lr = optimizer.param_groups[0]["lr"]

    def lr_lambda(step):
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return cosine_decay * (1 - min_lr / base_lr) + min_lr / base_lr

    return LambdaLR(optimizer, lr_lambda)


# =============================================================
# 4.  Refactored helpers (pure logic; no behavior changes)
# =============================================================

def _init_steering_parameters(model, steering_type: str, starting_norm: float, rank: int, adaptive_hidden: Optional[int] = None):
    """Initialize trainable steering parameters for the chosen parameterization."""
    d_model = model.config.hidden_size
    if steering_type == "linear":
        vector = torch.randn(d_model, device=model.device)
        vector = starting_norm * vector / vector.norm()
        vector.requires_grad_(True)
        params_to_opt = [vector]
        return {"type": "linear", "vector": vector, "params": params_to_opt}
    elif steering_type == "adaptive_linear":
        hidden = adaptive_hidden if adaptive_hidden is not None else max(64, d_model // 16)
        vector = torch.randn(d_model, device=model.device)
        vector = starting_norm * vector / vector.norm()
        W1 = torch.randn(d_model, hidden, device=model.device) / math.sqrt(d_model)
        b1 = torch.zeros(hidden, device=model.device)
        W2 = torch.randn(hidden, 1, device=model.device) / math.sqrt(hidden)
        b2 = torch.zeros(1, device=model.device)
        for p in (vector, W1, b1, W2, b2):
            p.requires_grad_(True)
        params_to_opt = [vector, W1, b1, W2, b2]
        return {
            "type": "adaptive_linear",
            "vector": vector,
            "W1": W1,
            "b1": b1,
            "W2": W2,
            "b2": b2,
            "hidden": hidden,
            "params": params_to_opt,
        }
    elif steering_type == "resid_lora":
        assert rank >= 1 and rank <= d_model, "Invalid LoRA rank"
        A = torch.randn(d_model, rank, device=model.device) / math.sqrt(d_model)
        B = torch.randn(rank, d_model, device=model.device) / math.sqrt(d_model)
        alpha_param = torch.tensor(1.0, device=model.device)
        A.requires_grad_(True)
        B.requires_grad_(True)
        alpha_param.requires_grad_(True)
        params_to_opt = [A, B, alpha_param]
        return {"type": "resid_lora", "A": A, "B": B, "alpha": alpha_param, "params": params_to_opt}
    else:
        raise ValueError("steering_type must be 'linear' or 'resid_lora'")


def _tok_batch(tokenizer, strs: List[str]):
    toks, lengths = [], []
    for s in strs:
        t = tokenizer(s, return_tensors="pt").input_ids[0]
        toks.append(t)
        lengths.append(t.size(0))
    return toks, lengths


def _precompute_base_completions(
    model,
    tokenizer,
    prompts: List[str],
    target_lens: List[Union[int, torch.Tensor]],
    layer: int,
    base_gen_minibatch_size: int,
    static_vectors: Optional[List[torch.Tensor]],
):
    base_completions: list[str] = []
    descr = "Precomputing base completions" if static_vectors is None else "Precomputing base completions with static vectors"
    for i in tqdm(range(0, len(prompts), base_gen_minibatch_size), desc=descr):
        batch_prompts = prompts[i:i + base_gen_minibatch_size]
        batch_target_lens = target_lens[i:i + base_gen_minibatch_size]
        max_new = max(1, max(tl.item() if isinstance(tl, torch.Tensor) else int(tl) for tl in batch_target_lens))
        inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=False).to(model.device)
        hooks = []
        if static_vectors is not None:
            clean_static_vectors = [sv for sv in static_vectors if sv is not None]
            if clean_static_vectors:
                prompt_lens_tensor = inputs['attention_mask'].sum(dim=1)
                def static_vec_batch_hook(_m, args, p_lens=prompt_lens_tensor):
                    (x,) = args
                    for row in range(x.shape[0]):
                        p_len = p_lens[row].item()
                        sl = slice(p_len, x.shape[1])
                        for sv in clean_static_vectors:
                            x[row, sl] += sv.to(x)
                    return (x,)
                hooks.append((layer, static_vec_batch_hook))
        with torch.no_grad():
            generation_kwargs = dict(
                max_new_tokens=max_new,
                pad_token_id=tokenizer.eos_token_id,
                suppress_tokens=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None,
            )
            if hooks:
                with hf_hooks_contextmanager(model, hooks):
                    gen = model.generate(**inputs, **generation_kwargs)
            else:
                gen = model.generate(**inputs, **generation_kwargs)
        decoded_texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for j, txt in enumerate(decoded_texts):
            original_prompt = batch_prompts[j]
            base_completions.append(txt[len(original_prompt):])
    return base_completions


def _build_right_padded_batch(
    tokenizer,
    prompt_tokens: List[torch.Tensor],
    target_tokens: List[torch.Tensor],
    batch_indices: List[int],
    prompt_lens: List[int],
    target_lens: List[int],
    steering_token_window: Optional[int],
    device,
):
    seqs = [torch.cat([prompt_tokens[i], target_tokens[i]]) for i in batch_indices]
    max_len = max(s.size(0) for s in seqs)
    pad_id = tokenizer.pad_token_id
    input_ids = torch.full((len(batch_indices), max_len), pad_id, device=device)
    attn_mask = torch.zeros_like(input_ids)
    steering_slices = []
    for row, (i, seq) in enumerate(zip(batch_indices, seqs)):
        L = seq.size(0)
        input_ids[row, :L] = seq
        attn_mask[row, :L] = 1
        start = prompt_lens[i] if steering_token_window is None else (
            prompt_lens[i] + max(0, target_lens[i] - (steering_token_window if steering_token_window is not None else 0))
        )
        steering_slices.append(slice(start, L))
    return input_ids, attn_mask, steering_slices


def _compute_target_cross_entropy(logits: torch.Tensor, input_ids: torch.Tensor, steering_slices: List[slice]):
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    target_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
    for row, sl in enumerate(steering_slices):
        tgt_start = sl.start - 1
        tgt_end = sl.stop - 1 if sl.stop is not None else shift_labels.size(1)
        target_mask[row, tgt_start:tgt_end] = True
    active_logits = shift_logits[target_mask]
    active_labels = shift_labels[target_mask]
    ce_target = torch.nn.functional.cross_entropy(active_logits, active_labels, reduction="mean") / math.log(10)
    return ce_target, (shift_logits, shift_labels, active_logits, active_labels)


def _compute_base_objective_for_batch(
    model,
    tokenizer,
    layer: int,
    steering_type: str,
    vector_or_lora,
    static_vectors_local: list,
    prompt_tokens: List[torch.Tensor],
    base_tokens: List[torch.Tensor],
    prompt_lens: List[int],
    base_lens: List[int],
    batch_indices: List[int],
    steering_token_window: Optional[int],
    pad_id: int,
    coldness: float,
    eps: float,
    projection_clamp: bool,
):
    base_seqs = [torch.cat([prompt_tokens[i], base_tokens[i]]) for i in batch_indices]
    max_len_b = max(s.size(0) for s in base_seqs)
    input_ids_b = torch.full((len(base_seqs), max_len_b), pad_id, device=model.device)
    attn_mask_b = torch.zeros_like(input_ids_b)
    steering_slices_b = []
    for row, i in enumerate(batch_indices):
        seq = base_seqs[row]
        Lb = seq.size(0)
        input_ids_b[row, :Lb] = seq
        attn_mask_b[row, :Lb] = 1
        start_b = prompt_lens[i] if steering_token_window is None else (
            prompt_lens[i] + max(0, (base_lens[i] if isinstance(base_lens[i], int) else int(base_lens[i])) - steering_token_window)
        )
        steering_slices_b.append(slice(start_b, Lb))
    if steering_type == "linear":
        batch_hook_base = make_batch_linear_hook(
            vector_or_lora, steering_slices_b, static_vectors_local, projection_clamp=projection_clamp
        )
    elif steering_type == "adaptive_linear":
        batch_hook_base = make_batch_adaptive_linear_hook(
            vector_or_lora['vector'], vector_or_lora['W1'], vector_or_lora['b1'], vector_or_lora['W2'], vector_or_lora['b2'], steering_slices_b, static_vectors_local, projection_clamp=projection_clamp
        )
    elif steering_type == "resid_lora":
        batch_hook_base = make_batch_resid_lora_hook(
            vector_or_lora['A'], vector_or_lora['B'], vector_or_lora['alpha'], steering_slices_b, static_vectors_local
        )
    else:
        raise ValueError(f"Unknown steering_type: {steering_type}")
    with hf_hooks_contextmanager(model, [(layer, batch_hook_base)]):
        out_b = model(input_ids=input_ids_b, attention_mask=attn_mask_b)
        logits_b = out_b.logits * coldness
    shift_logits_b = logits_b[:, :-1].contiguous()
    shift_labels_b = input_ids_b[:, 1:].contiguous()
    target_mask_b = torch.zeros_like(shift_labels_b, dtype=torch.bool)
    for row, sl in enumerate(steering_slices_b):
        tgt_start_b = sl.start - 1
        tgt_end_b = sl.stop - 1 if sl.stop is not None else shift_labels_b.size(1)
        if tgt_start_b < tgt_end_b:
            target_mask_b[row, tgt_start_b:tgt_end_b] = True
    active_logits_b = shift_logits_b[target_mask_b]
    active_labels_b = shift_labels_b[target_mask_b]
    if active_logits_b.numel() > 0:
        log_probs_b = torch.log_softmax(active_logits_b, dim=-1)
        idx = torch.arange(active_labels_b.size(0), device=active_labels_b.device)
        p_b = torch.exp(log_probs_b[idx, active_labels_b])
        base_loss = (-torch.log(torch.clamp(1.0 - p_b, min=eps))).mean() / math.log(10)
    else:
        base_loss = torch.tensor(0.0, device=model.device)
    del input_ids_b, attn_mask_b, shift_logits_b, shift_labels_b, active_logits_b, active_labels_b, out_b, logits_b
    torch.cuda.empty_cache()
    return base_loss


def _compute_eval_loss(
    model,
    tokenizer,
    eval_prompt_tokens: List[torch.Tensor],
    eval_target_tokens: List[torch.Tensor],
    eval_prompt_lens: List[int],
    eval_target_lens: List[int],
    layer: int,
    steering_type: str,
    vector_or_lora,
    static_vectors_local: list,
    coldness: float,
    steering_token_window: Optional[int],
    include_base_objective: bool,
    base_tokens_eval: Optional[List[torch.Tensor]],
    base_lens_eval: Optional[List[int]],
    eps: float,
    optim_minibatch_size: int,
    projection_clamp: bool,
    base_loss_weight: float,
):
    running_eval_loss = 0.0
    eval_batches = 0
    for i in range(0, len(eval_prompt_tokens), optim_minibatch_size):
        batch_indices = list(range(i, min(i + optim_minibatch_size, len(eval_prompt_tokens))))
        seqs_eval = [torch.cat([eval_prompt_tokens[j], eval_target_tokens[j]]) for j in batch_indices]
        if not seqs_eval:
            continue
        max_len_e = max(s.size(0) for s in seqs_eval)
        pad_id = tokenizer.pad_token_id
        input_ids_e = torch.full((len(seqs_eval), max_len_e), pad_id, device=model.device)
        attn_mask_e = torch.zeros_like(input_ids_e)
        steering_slices_e = []
        for row, j in enumerate(batch_indices):
            L = seqs_eval[row].size(0)
            input_ids_e[row, :L] = seqs_eval[row]
            attn_mask_e[row, :L] = 1
            start_e = eval_prompt_lens[j] if steering_token_window is None else (
                eval_prompt_lens[j] + max(0, eval_target_lens[j] - steering_token_window)
            )
            steering_slices_e.append(slice(start_e, L))
        if steering_type == "linear":
            batch_hook_eval = make_batch_linear_hook(vector_or_lora, steering_slices_e, static_vectors_local, projection_clamp=projection_clamp)
        elif steering_type == "adaptive_linear":
            batch_hook_eval = make_batch_adaptive_linear_hook(vector_or_lora['vector'], vector_or_lora['W1'], vector_or_lora['b1'], vector_or_lora['W2'], vector_or_lora['b2'], steering_slices_e, static_vectors_local, projection_clamp=projection_clamp)
        elif steering_type == "resid_lora":
            batch_hook_eval = make_batch_resid_lora_hook(vector_or_lora['A'], vector_or_lora['B'], vector_or_lora['alpha'], steering_slices_e, static_vectors_local)
        else:
            raise ValueError(f"Unknown steering_type: {steering_type}")
        with hf_hooks_contextmanager(model, [(layer, batch_hook_eval)]):
            out_e = model(input_ids=input_ids_e, attention_mask=attn_mask_e)
            logits_e = out_e.logits * coldness
        shift_logits_e = logits_e[:, :-1].contiguous()
        shift_labels_e = input_ids_e[:, 1:].contiguous()
        target_mask_e = torch.zeros_like(shift_labels_e, dtype=torch.bool)
        for row, sl in enumerate(steering_slices_e):
            tgt_start_e = sl.start - 1
            tgt_end_e = sl.stop - 1 if sl.stop is not None else shift_labels_e.size(1)
            if tgt_start_e < tgt_end_e:
                target_mask_e[row, tgt_start_e:tgt_end_e] = True
        active_logits_e = shift_logits_e[target_mask_e]
        active_labels_e = shift_labels_e[target_mask_e]
        ce_target_e = torch.nn.functional.cross_entropy(active_logits_e, active_labels_e, reduction="mean") / math.log(10)
        base_loss_e = 0.0
        if include_base_objective and base_tokens_eval is not None and base_lens_eval is not None:
            base_seqs_e = [torch.cat([eval_prompt_tokens[j], base_tokens_eval[j]]) for j in batch_indices]
            max_len_be = max(s.size(0) for s in base_seqs_e)
            input_ids_be = torch.full((len(base_seqs_e), max_len_be), pad_id, device=model.device)
            attn_mask_be = torch.zeros_like(input_ids_be)
            steering_slices_be = []
            for row, j in enumerate(batch_indices):
                seq = base_seqs_e[row]
                Lb = seq.size(0)
                input_ids_be[row, :Lb] = seq
                attn_mask_be[row, :Lb] = 1
                start_be = eval_prompt_lens[j] if steering_token_window is None else (
                    eval_prompt_lens[j] + max(0, (base_lens_eval[j] if isinstance(base_lens_eval[j], int) else int(base_lens_eval[j])) - steering_token_window)
                )
                steering_slices_be.append(slice(start_be, Lb))
            if steering_type == "linear":
                batch_hook_base_eval = make_batch_linear_hook(vector_or_lora, steering_slices_be, static_vectors_local, projection_clamp=projection_clamp)
            elif steering_type == "resid_lora":
                batch_hook_base_eval = make_batch_resid_lora_hook(vector_or_lora['A'], vector_or_lora['B'], vector_or_lora['alpha'], steering_slices_be, static_vectors_local)
            else:
                raise ValueError(f"Unknown steering_type: {steering_type}")
            with hf_hooks_contextmanager(model, [(layer, batch_hook_base_eval)]):
                out_be = model(input_ids=input_ids_be, attention_mask=attn_mask_be)
                logits_be = out_be.logits * coldness
            shift_logits_be = logits_be[:, :-1].contiguous()
            shift_labels_be = input_ids_be[:, 1:].contiguous()
            target_mask_be = torch.zeros_like(shift_labels_be, dtype=torch.bool)
            for row, sl in enumerate(steering_slices_be):
                tgt_start_be = sl.start - 1
                tgt_end_be = sl.stop - 1 if sl.stop is not None else shift_labels_be.size(1)
                if tgt_start_be < tgt_end_be:
                    target_mask_be[row, tgt_start_be:tgt_end_be] = True
            active_logits_be = shift_logits_be[target_mask_be]
            active_labels_be = shift_labels_be[target_mask_be]
            if active_logits_be.numel() > 0:
                log_probs_be = torch.log_softmax(active_logits_be, dim=-1)
                idx_be = torch.arange(active_labels_be.size(0), device=active_labels_be.device)
                p_be = torch.exp(log_probs_be[idx_be, active_labels_be])
                base_loss_e = (-torch.log(torch.clamp(1.0 - p_be, min=eps))).mean() / math.log(10)
            else:
                base_loss_e = torch.tensor(0.0, device=model.device)
            del input_ids_be, attn_mask_be, shift_logits_be, shift_labels_be, active_logits_be, active_labels_be, out_be, logits_be
            torch.cuda.empty_cache()
        batch_eval_loss = (ce_target_e + (base_loss_weight * base_loss_e if include_base_objective else 0.0)).item()
        running_eval_loss += batch_eval_loss
        eval_batches += 1
        del input_ids_e, attn_mask_e, shift_logits_e, shift_labels_e, active_logits_e, active_labels_e, out_e, logits_e
        torch.cuda.empty_cache()
    if eval_batches > 0:
        return running_eval_loss / eval_batches
    return None


# =============================================================
# 4.  Optimisation loop
# =============================================================

def optimize_vector_simple(
    model,
    tokenizer,
    prompts,
    target_completions,
    layer: int,
    *,
    lr: float = 0.01,
    max_iters: int = 50,
    optim_minibatch_size: int = 32,
    base_gen_minibatch_size: int = 32,
    warmup_steps: int = 0,
    min_lr: float = 0.0,
    coldness: float = 0.7,
    grad_clip: Optional[float] = None,
    early_stopping_patience: int = 5,
    early_stopping_min_delta: float = 1e-4,
    max_norm: Optional[float] = None,
    starting_norm: float = 1.0,
    debug: bool = False,
    return_info: bool = True,
    return_loss_history: bool = False,
    steering_token_window: Optional[int] = None,
    projection_clamp: bool = False,
    wandb_run=None,
    static_vectors: Optional[list[torch.Tensor]] = None,
    include_base_objective: bool = False,
    base_loss_weight: float = 1.0,
    eps: float = 1e-6,
    eval_prompts: Optional[list[str]] = None,
    eval_target_completions: Optional[list[str]] = None,
    # Unified steering controls
    steering_type: str = "linear",  # "linear", "adaptive_linear" or "resid_lora"
    rank: int = 1,
    adaptive_hidden: int = 128,
):
    """One‑stop promotion‑steering optimiser matching the paper’s details.

    Supports two parameterisations under one API:
      - steering_type == "linear": train a 1-D vector added to hidden states
      - steering_type == "resid_lora":   train a rank-r residual LoRA: x += alpha * ((x @ A) @ B)
    """

    # ---------------- Pre‑flight checks ---------------- #
    if len(prompts) != len(target_completions):
        raise ValueError("Prompts and completions length mismatch.")
    # Evaluation logic removed; optimisation now relies only on training loss.

    # ---------------- Parameter initialisation ----------- #
    init_payload = _init_steering_parameters(model, steering_type, starting_norm, rank, adaptive_hidden)
    if steering_type == "linear":
        vector = init_payload["vector"]
        params_to_opt = init_payload["params"]
    elif steering_type == "adaptive_linear":
        vector = init_payload["vector"]
        W1 = init_payload["W1"]
        b1 = init_payload["b1"]
        W2 = init_payload["W2"]
        b2 = init_payload["b2"]
        adaptive_hidden_local = init_payload["hidden"]
        params_to_opt = init_payload["params"]
    else:
        A = init_payload["A"]
        B = init_payload["B"]
        alpha_param = init_payload["alpha"]
        params_to_opt = init_payload["params"]

    # ---------------- Tokenisation helpers -------------- #
    prompt_tokens, prompt_lens = _tok_batch(tokenizer, prompts)
    target_tokens, target_lens = _tok_batch(tokenizer, target_completions)

    # ---- Optional evaluation tokenisation ----
    have_eval = eval_prompts is not None and eval_target_completions is not None and len(eval_prompts) > 0
    if have_eval:
        eval_prompt_tokens, eval_prompt_lens = _tok_batch(tokenizer, eval_prompts)
        eval_target_tokens, eval_target_lens = _tok_batch(tokenizer, eval_target_completions)

    # ---- Precompute unsteered base completions (used as "src" completions in the objective) ----
    if include_base_objective:
        base_completions = _precompute_base_completions(
            model,
            tokenizer,
            prompts,
            target_lens,
            layer,
            base_gen_minibatch_size,
            static_vectors,
        )
    else:
        base_completions = [""] * len(prompts)

    base_tokens, base_lens = _tok_batch(tokenizer, base_completions)

    # Precompute base completions for eval set if provided and base objective enabled
    if have_eval:
        if include_base_objective:
            descr_eval = "Precomputing base completions (eval)" if static_vectors is None else "Precomputing base completions with static vectors (eval)"
            base_completions_eval: list[str] = []
            for i in tqdm(range(0, len(eval_prompts), base_gen_minibatch_size), desc=descr_eval):
                batch_prompts = eval_prompts[i:i + base_gen_minibatch_size]
                batch_target_lens = eval_target_lens[i:i + base_gen_minibatch_size]
                
                max_new = max(1, max(tl.item() if isinstance(tl, torch.Tensor) else int(tl) for tl in batch_target_lens))

                inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=False).to(model.device)
                
                hooks = []
                if static_vectors is not None:
                    clean_static_vectors = [sv for sv in static_vectors if sv is not None]
                    if clean_static_vectors:
                        prompt_lens_tensor = inputs['attention_mask'].sum(dim=1)
                        def static_vec_batch_hook_eval(_m, args, p_lens=prompt_lens_tensor):
                            (x,) = args
                            for row in range(x.shape[0]):
                                p_len = p_lens[row].item()
                                sl = slice(p_len, x.shape[1])
                                for sv in clean_static_vectors:
                                    x[row, sl] += sv.to(x)
                            return (x,)
                        hooks.append((layer, static_vec_batch_hook_eval))

                with torch.no_grad():
                    generation_kwargs = dict(
                        max_new_tokens=max_new,
                        pad_token_id=tokenizer.eos_token_id,
                        suppress_tokens=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None,
                    )
                    if hooks:
                        with hf_hooks_contextmanager(model, hooks):
                            gen = model.generate(**inputs, **generation_kwargs)
                    else:
                        gen = model.generate(**inputs, **generation_kwargs)
                
                decoded_texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
                for j, txt in enumerate(decoded_texts):
                    original_prompt = batch_prompts[j]
                    base_completions_eval.append(txt[len(original_prompt):])
            
            base_tokens_eval, base_lens_eval = _tok_batch(tokenizer, base_completions_eval)
        else:
            base_tokens_eval = [torch.tensor([], device=model.device)] * len(eval_prompts)  # type: ignore
            base_lens_eval = [0] * len(eval_prompts)  # type: ignore
    else:
        base_tokens_eval, base_lens_eval = None, None

    # Evaluation dataset and metrics have been removed.
    eval_hist = []

    # ---------------- Optimiser & LR schedule ----------- #
    optim = torch.optim.Adam(params_to_opt, lr=lr)

    # Ensure static_vectors is a list of tensors on correct device
    if static_vectors is None:
        static_vectors_local: list = []
    else:
        static_vectors_local = [sv.to(model.device).detach() if isinstance(sv, torch.Tensor) else sv for sv in static_vectors]

    if steering_type == "resid_lora" and static_vectors_local:
        # Expect dicts with A,B,alpha for LoRA static adapters
        for sv in static_vectors_local:
            assert isinstance(sv, dict) and all(k in sv for k in ("A","B","alpha")), "LoRA static_vectors must be dicts with A,B,alpha"

    # ---- learning-rate schedule ----
    # We want the scheduler to advance *once per optimiser step* (i.e. per minibatch).
    # Hence, total training steps = max_iters * num_batches_per_epoch.
    num_batches_per_epoch = (len(prompts) + optim_minibatch_size - 1) // optim_minibatch_size
    total_training_steps = max_iters * num_batches_per_epoch

    sched = (
        get_cosine_schedule_with_warmup(optim, warmup_steps, total_training_steps, min_lr)
        if total_training_steps
        else None
    )

    # ---------------- Training loop --------------------- #
    best_vec, best_lora, best_adapt, best_loss = None, None, None, float("inf")
    loss_hist = []
    patience = 0

    total_pbar = tqdm(total=total_training_steps, desc="total", position=0, leave=True)
    epoch_pbar = tqdm(range(max_iters), desc="", position=1, leave=False)

    for step in epoch_pbar:
        # ----- minibatch shuffling -----
        idxs = list(range(len(prompts)))
        random.shuffle(idxs)

        running_loss, batches = 0.0, 0
        batch_pbar = tqdm(range(0, len(idxs), optim_minibatch_size), desc=f"batches {step+1}/{max_iters}", position=2, leave=True)
        for bs in batch_pbar:
            batch = idxs[bs : bs + optim_minibatch_size]

            input_ids, attn_mask, steering_slices = _build_right_padded_batch(
                tokenizer,
                prompt_tokens,
                target_tokens,
                batch,
                prompt_lens,
                target_lens,
                steering_token_window,
                model.device,
            )

            # Hook for this batch
            if steering_type == "linear":
                batch_hook = make_batch_linear_hook(vector, steering_slices, static_vectors_local, projection_clamp=projection_clamp)
            elif steering_type == "adaptive_linear":
                batch_hook = make_batch_adaptive_linear_hook(vector, W1, b1, W2, b2, steering_slices, static_vectors_local, projection_clamp=projection_clamp)
            elif steering_type == "resid_lora":
                batch_hook = make_batch_resid_lora_hook(A, B, alpha_param, steering_slices, static_vectors_local)
            else:
                raise ValueError(f"Unknown steering_type: {steering_type}")

            with hf_hooks_contextmanager(model, [(layer, batch_hook)]):
                out = model(input_ids=input_ids, attention_mask=attn_mask)
                logits = out.logits * coldness

            ce_target, (shift_logits, shift_labels, active_logits, active_labels) = _compute_target_cross_entropy(
                logits, input_ids, steering_slices
            )

            # ---- Base (unsteered) completion objective: -log(1 - p_base) on base tokens ----
            base_loss = 0.0
            if include_base_objective:
                if steering_type == "linear":
                    vector_or_lora = vector
                elif steering_type == "adaptive_linear":
                    vector_or_lora = {"vector": vector, "W1": W1, "b1": b1, "W2": W2, "b2": b2}
                else:
                    vector_or_lora = {"A": A, "B": B, "alpha": alpha_param}
                base_loss = _compute_base_objective_for_batch(
                    model,
                    tokenizer,
                    layer,
                    steering_type,
                    vector_or_lora,
                    static_vectors_local,
                    prompt_tokens,
                    base_tokens,
                    prompt_lens,
                    base_lens,
                    batch,
                    steering_token_window,
                    tokenizer.pad_token_id,
                    coldness,
                    eps,
                    projection_clamp,
                )

            loss = ce_target + (base_loss_weight * base_loss if include_base_objective else 0.0)

            optim.zero_grad()
            loss.backward()

            if grad_clip:
                torch.nn.utils.clip_grad_norm_(params_to_opt, grad_clip)

            optim.step()

            # Advance the LR scheduler once per optimiser update
            if sched:
                sched.step()

            if max_norm is not None:
                if steering_type == "linear":
                    if vector.norm() > max_norm:
                        with torch.no_grad():
                            vector.mul_(max_norm / vector.norm())
                elif steering_type == "adaptive_linear":
                    with torch.no_grad():
                        v_norm = vector.norm()
                        W1_norm = W1.norm()
                        b1_norm = b1.norm()
                        W2_norm = W2.norm()
                        b2_norm = b2.norm()
                        combined = torch.sqrt(v_norm*v_norm + W1_norm*W1_norm + b1_norm*b1_norm + W2_norm*W2_norm + b2_norm*b2_norm)
                        if combined > max_norm:
                            scale = max_norm / combined
                            vector.mul_(scale)
                            W1.mul_(scale)
                            b1.mul_(scale)
                            W2.mul_(scale)
                            b2.mul_(scale)
                elif steering_type == "resid_lora":
                    # Enforce a joint norm constraint over (A, B, alpha)
                    with torch.no_grad():
                        A_norm = A.norm()
                        B_norm = B.norm()
                        a_norm = alpha_param.abs()
                        combined = torch.sqrt(A_norm * A_norm + B_norm * B_norm + a_norm * a_norm)
                        if combined > max_norm:
                            scale = max_norm / combined
                            A.mul_(scale)
                            B.mul_(scale)
                            alpha_param.mul_(scale)
                else:
                    raise ValueError(f"Unknown steering_type: {steering_type}")

            running_loss += loss.item()
            batches += 1
            loss_hist.append(loss.item())

            # update total training progress (ETA across all batches)
            total_pbar.update(1)

            # ---- lightweight cleanup to mitigate CUDA OOM ----
            del input_ids, attn_mask, shift_logits, shift_labels, active_logits, active_labels, out, logits
            torch.cuda.empty_cache()

        batch_pbar.close()
        train_loss = running_loss / max(1, batches)

        # ---- Evaluation loss on eval set (if provided) ----
        eval_loss = None
        if have_eval:
            with torch.no_grad():
                vector_or_lora = vector if steering_type == "linear" else {"A": A, "B": B, "alpha": alpha_param}
                eval_loss = _compute_eval_loss(
                    model,
                    tokenizer,
                    eval_prompt_tokens,
                    eval_target_tokens,
                    eval_prompt_lens,
                    eval_target_lens,
                    layer,
                    steering_type,
                    vector_or_lora,
                    static_vectors_local,
                    coldness,
                    steering_token_window,
                    include_base_objective,
                    base_tokens_eval,
                    base_lens_eval,
                    eps,
                    optim_minibatch_size,
                    projection_clamp,
                    base_loss_weight,
                )
                if eval_loss is not None:
                    eval_hist.append(eval_loss)

        # --- wandb logging ---
        if wandb_run is not None:
            log_payload = {
                'train_loss': train_loss,
                'step': step,
            }
            if steering_type == 'linear':
                log_payload['vector_norm'] = vector.norm().item()
            elif steering_type == 'adaptive_linear':
                log_payload['vector_norm'] = vector.norm().item()
                log_payload['W1_norm'] = W1.norm().item()
                log_payload['W2_norm'] = W2.norm().item()
            elif steering_type == 'resid_lora':
                log_payload['A_norm'] = A.norm().item()
                log_payload['B_norm'] = B.norm().item()
                log_payload['alpha'] = float(alpha_param.item())
            if eval_loss is not None:
                log_payload['eval_loss'] = eval_loss
            wandb_run.log(log_payload, step=step)

        metric = train_loss
        if metric + early_stopping_min_delta < best_loss:
            best_loss = metric
            if steering_type == "linear":
                best_vec = vector.detach().clone()
            elif steering_type == "adaptive_linear":
                best_adapt = (
                    vector.detach().clone(),
                    W1.detach().clone(),
                    b1.detach().clone(),
                    W2.detach().clone(),
                    b2.detach().clone(),
                )
            else:
                best_lora = (A.detach().clone(), B.detach().clone(), alpha_param.detach().clone())
            patience = 0
        else:
            patience += 1
            if patience >= early_stopping_patience:
                if debug:
                    print("Early stopping.")
                break

        # Scheduler is already stepped inside the minibatch loop

        # --- final cleanup for this step ---
        gc.collect()
        torch.cuda.empty_cache()

        current_lr = optim.param_groups[0]["lr"]
        epoch_pbar.set_description(
            f"step {step} lr {current_lr:.2e} train {train_loss:.4f}"
        )

    epoch_pbar.close()
    total_pbar.close()

    if steering_type == "linear":
        if best_vec is not None:
            vector.data.copy_(best_vec)
    elif steering_type == "adaptive_linear":
        if best_adapt is not None:
            vector.data.copy_(best_adapt[0])
            W1.data.copy_(best_adapt[1])
            b1.data.copy_(best_adapt[2])
            W2.data.copy_(best_adapt[3])
            b2.data.copy_(best_adapt[4])
    else:
        if best_lora is not None:
            A.data.copy_(best_lora[0])
            B.data.copy_(best_lora[1])
            alpha_param.data.copy_(best_lora[2])

    if wandb_run is not None:
        if steering_type == "linear":
            wandb_run.log({'final_vector_norm': vector.norm().item()})
        elif steering_type == "adaptive_linear":
            wandb_run.log({'final_vector_norm': vector.norm().item(), 'final_W1_norm': W1.norm().item(), 'final_W2_norm': W2.norm().item()})
        else:
            wandb_run.log({'final_A_norm': A.norm().item(), 'final_B_norm': B.norm().item(), 'final_alpha': float(alpha_param.item())})

    if return_info:
        if steering_type == "linear":
            info = dict(
                iters=len(loss_hist),
                final_loss=best_loss,
                norm=vector.norm().item(),
                loss_history=loss_hist if return_loss_history else None,
                eval_loss_history=eval_hist if (return_loss_history and have_eval) else None,
            )
            return vector, info
        elif steering_type == "adaptive_linear":
            combined_norm = (vector.norm()**2 + W1.norm()**2 + b1.norm()**2 + W2.norm()**2 + b2.norm()**2).sqrt().item()
            info = dict(
                iters=len(loss_hist),
                final_loss=best_loss,
                vector_norm=vector.norm().item(),
                W1_norm=W1.norm().item(),
                W2_norm=W2.norm().item(),
                combined_norm=combined_norm,
                hidden_dim=adaptive_hidden_local,
                loss_history=loss_hist if return_loss_history else None,
                eval_loss_history=eval_hist if (return_loss_history and have_eval) else None,
            )
            return {"vector": vector, "W1": W1, "b1": b1, "W2": W2, "b2": b2}, info
        else:
            info = dict(
                iters=len(loss_hist),
                final_loss=best_loss,
                A_norm=A.norm().item(),
                B_norm=B.norm().item(),
                alpha=float(alpha_param.item()),
                loss_history=loss_hist if return_loss_history else None,
                eval_loss_history=eval_hist if (return_loss_history and have_eval) else None,
            )
            return {'A': A, 'B': B, 'alpha': alpha_param}, info
    if steering_type == "linear":
        return vector
    elif steering_type == "adaptive_linear":
        return {"vector": vector, "W1": W1, "b1": b1, "W2": W2, "b2": b2}
    else:
        return {'A': A, 'B': B, 'alpha': alpha_param}


# Evaluation helper removed.
