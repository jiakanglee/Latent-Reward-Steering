# %%
import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import argparse
from matplotlib.patches import Rectangle
import html as html_mod


def _escape_token(t: str) -> str:
    return t.replace('$', '\\$')


def load_record(jsonl_path: str, task_index: int):
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"File not found: {jsonl_path}")
    records = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    if not records:
        raise RuntimeError("No records found in JSONL.")
    if task_index < 0 or task_index >= len(records):
        raise IndexError(f"task_index {task_index} out of range [0, {len(records)-1}]")
    return records[task_index], len(records)


def build_arrays(per_token, steering_selection):
    # If per_token has content, derive tokens, categories, coefficients, windows
    if per_token and len(per_token) == len(steering_selection):
        tokens = [pt.get("token", "") for pt in per_token]
        categories = [pt.get("latent_title", "No Steering") for pt in per_token]
        coefs = [pt.get("coefficient", None) for pt in per_token]
        windows = [pt.get("window", None) for pt in per_token]
    else:
        # Fallback: we only know which steps were steered
        n = len(steering_selection)
        tokens = [str(i) for i in range(n)]
        categories = ["No Steering" for _ in range(n)]
        coefs = [None for _ in range(n)]
        windows = [None for _ in range(n)]
    steered_mask = [1 if s == "steered" else 0 for s in steering_selection]
    return tokens, categories, coefs, windows, steered_mask


def colorize_categories(categories, steered_mask):
    # Map categories (only on steered positions) to integer IDs; 0 is reserved for unsteered (grey)
    steered_cats = [c for c, m in zip(categories, steered_mask) if m == 1]
    unique_cats = sorted(list(set(steered_cats)))
    cat_to_id = {c: i + 1 for i, c in enumerate(unique_cats)}  # start at 1
    # Build index row
    idx_row = []
    for c, m in zip(categories, steered_mask):
        if m == 0:
            idx_row.append(0)
        else:
            idx_row.append(cat_to_id.get(c, 0))
    # Colormap: first color grey for unsteered, then tab20 colors
    tab20 = plt.get_cmap("tab20")
    colors = ["#AAAAAA"]
    for i in range(len(unique_cats)):
        colors.append(tab20(i % tab20.N))
    cmap = ListedColormap(colors)
    return np.array([idx_row], dtype=np.int32), cmap, unique_cats


def _assign_category_colors(categories):
    # Only assign colors to categories that appear on steered tokens
    unique = [c for c in sorted(set(categories)) if c and c != "No Steering"]
    n = max(1, len(unique))
    colors = {}
    for i, cat in enumerate(unique):
        hue = (i / n) * 360.0
        # soft pastel backgrounds, readable with black text
        colors[cat] = {
            "bg": f"hsl({hue:.1f}, 85%, 88%)",
            "border": f"hsl({hue:.1f}, 70%, 55%)",
        }
    return colors


def render_html(record, task_index: int, save_html: bool = False, output_file: str = None):
    answers = record.get("answers", {})
    hybrid_text = answers.get("hybrid", "")
    base_text = answers.get("base", "")
    details = record.get("hybrid_details", {})
    per_token = details.get("per_token", []) or []
    steering_selection = details.get("steering_selection", []) or []

    if not per_token:
        # Fallback: show plain text with a note
        # Compute steering percentage from selection list if available
        steered_count = sum(1 for s in steering_selection if s == "steered")
        total_tokens = len(steering_selection)
        steered_pct = (steered_count / total_tokens * 100.0) if total_tokens else 0.0
        try:
            print(f"Steered tokens: {steered_count}/{total_tokens} ({steered_pct:.1f}%)")
        except Exception:
            pass
        doc = f"""
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>Hybrid steering — task {task_index}</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, \"Apple Color Emoji\", \"Segoe UI Emoji\"; margin: 16px; }}
    .comparison-container {{ display: flex; gap: 20px; margin-top: 20px; }}
    .response-panel {{ flex: 1; }}
    .response-panel h3 {{ margin: 0 0 12px 0; font-size: 18px; color: #374151; }}
    .text {{ white-space: pre-wrap; line-height: 1.6; font-size: 16px; }}
    .note {{ background: #fff3cd; border: 1px solid #ffe69c; padding: 10px 12px; border-radius: 6px; margin-bottom: 12px; }}
    .meta {{ color: #555; margin-bottom: 10px; }}
    @media (max-width: 768px) {{ .comparison-container {{ flex-direction: column; }} }}
  </style>
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  </head>
<body>
  <div class=\"meta\">Steered: {steered_pct:.1f}% ({steered_count}/{total_tokens})</div>
  <div class=\"note\">Per-token details missing in record. Re-run evaluation with <code>--store_per_token_details</code> to see colored steering spans.</div>
  <div class=\"comparison-container\">
    <div class=\"response-panel\">
      <h3>Base Response (Unsteered)</h3>
      <div class=\"text\">{html_mod.escape(base_text)}</div>
    </div>
    <div class=\"response-panel\">
      <h3>Hybrid Response (Steered)</h3>
      <div class=\"text\">{html_mod.escape(hybrid_text)}</div>
    </div>
  </div>
</body>
</html>
"""
        # Save to file if requested
        if save_html:
            if output_file is None:
                output_file = f"hybrid_steering_task_{task_index}.html"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(doc)
            print(f"HTML saved to: {output_file}")
        
        # Display in interactive window if available
        try:
            from IPython.display import HTML as IPyHTML, display as ipy_display
            ipy_display(IPyHTML(doc))
            return
        except Exception:
            pass
        # Fallback: show minimal figure with plain text
        plt.figure(figsize=(12, 6))
        plt.axis('off')
        plt.text(0, 1, "Per-token details missing.\n\n" + hybrid_text, va='top', ha='left', fontsize=12, family='monospace')
        plt.tight_layout()
        plt.show()
        return

    # Build arrays from per_token (prefer fields embedded in each entry)
    tokens = [pt.get("token", "") for pt in per_token]
    selections = [pt.get("selection", None) for pt in per_token]
    categories = [pt.get("latent_title", "No Steering") for pt in per_token]
    coefs = [pt.get("coefficient", None) for pt in per_token]
    windows = [pt.get("window", None) for pt in per_token]
    perpls = [pt.get("perplexity", None) for pt in per_token]

    cat_colors = _assign_category_colors(categories)
    # Steering stats
    steered_mask = [1 if sel == "steered" else 0 for sel in selections]
    steered_count = int(sum(steered_mask))
    total_tokens = int(len(selections))
    steered_pct = (steered_count / total_tokens * 100.0) if total_tokens else 0.0
    try:
        print(f"Steered tokens: {steered_count}/{total_tokens} ({steered_pct:.1f}%)")
    except Exception:
        pass

    # Legend entries (unique in order of appearance)
    legend_seen = set()
    legend_items = []
    for cat in categories:
        if cat in (None, "No Steering"):
            continue
        if cat in legend_seen:
            continue
        legend_seen.add(cat)
        col = cat_colors.get(cat)
        if not col:
            continue
        legend_items.append((cat, col["bg"], col["border"]))

    # Build token HTML
    tok_html_parts = []
    for tok, sel, cat, coef, win, ppl in zip(tokens, selections, categories, coefs, windows, perpls):
        safe_tok = html_mod.escape(tok)
        if sel == "steered" and cat and cat != "No Steering":
            col = cat_colors.get(cat) or {"bg": "#fff1b8", "border": "#f59e0b"}
            title = f"{cat} | coef={coef} | win={win}" if coef is not None else f"{cat}"
            if ppl is not None:
                title += f" | ppl={ppl:.2f}" if isinstance(ppl, (int, float)) else f" | ppl={ppl}"
            style = f"background:{col['bg']};border-bottom:2px solid {col['border']};border-radius:2px;padding:0 1px;color:#000;"
            span = f"<span title=\"{html_mod.escape(title)}\" style=\"{style}\">{safe_tok}</span>"
            tok_html_parts.append(span)
        else:
            tok_html_parts.append(safe_tok)

    # HTML document
    legend_html = "".join(
        [
            f"<div class=\"legend-item\"><span class=\"color\" style=\"background:{bg};border:1px solid {bd}\"></span><span class=\"name\" style=\"color:#000\">{html_mod.escape(cat)}</span></div>"
            for cat, bg, bd in legend_items
        ]
    )
    doc = f"""
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>Hybrid steering — task {task_index}</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, \"Apple Color Emoji\", \"Segoe UI Emoji\"; margin: 16px; }}
    h1 {{ font-size: 20px; margin: 0 0 12px 0; }}
    .meta {{ color: #555; margin-bottom: 14px; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 8px; row-gap: 6px; margin-bottom: 10px; }}
    .legend-item {{ display: inline-flex; align-items: center; gap: 6px; padding: 2px 6px; border: 1px solid #e5e7eb; border-radius: 6px; background: #fafafa; }}
    .legend-item .color {{ display: inline-block; width: 14px; height: 14px; border-radius: 3px; }}
    .legend-item .name {{ color: #000; }}
    .comparison-container {{ display: flex; gap: 20px; margin-top: 20px; }}
    .response-panel {{ flex: 1; }}
    .response-panel h3 {{ margin: 0 0 12px 0; font-size: 18px; color: #374151; }}
    .text {{ white-space: pre-wrap; line-height: 1.7; font-size: 16px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Liberation Mono\", \"Courier New\", monospace; }}
    .note {{ background: #eef6ff; border: 1px solid #b6d6ff; padding: 8px 10px; border-radius: 6px; margin-bottom: 12px; }}
    @media (max-width: 768px) {{ .comparison-container {{ flex-direction: column; }} }}
  </style>
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  </head>
<body>
  <h1>Hybrid steering — task {task_index}</h1>
  <div class=\"meta\">Dataset: {html_mod.escape(str(record.get('dataset', '')))} | Correct: {html_mod.escape(str(record.get('judges', {}).get('hybrid', {}).get('correct')))} | Steered: {steered_pct:.1f}% ({steered_count}/{total_tokens})</div>
  <div class=\"legend\">{legend_html}</div>
  <div class=\"comparison-container\">
    <div class=\"response-panel\">
      <h3>Base Response (Unsteered)</h3>
      <div class=\"text\">{html_mod.escape(base_text)}</div>
    </div>
    <div class=\"response-panel\">
      <h3>Hybrid Response (Steered)</h3>
      <div class=\"text\">{''.join(tok_html_parts)}</div>
    </div>
  </div>
</body>
</html>
"""

    # Save to file if requested
    if save_html:
        if output_file is None:
            output_file = f"hybrid_steering_task_{task_index}.html"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(doc)
        print(f"HTML saved to: {output_file}")
    
    # Display in interactive window if available
    try:
        from IPython.display import HTML as IPyHTML, display as ipy_display
        ipy_display(IPyHTML(doc))
        return
    except Exception:
        pass
    # Fallback: approximate with a matplotlib text view
    plt.figure(figsize=(12, 6))
    plt.axis('off')
    plt.text(0, 1, ''.join([pt.get('token', '') for pt in per_token]), va='top', ha='left', fontsize=12, family='monospace')
    plt.tight_layout()
    plt.show()
    return


def plot_example(record, task_index: int, max_xticks: int = 60):
    question = record.get("question") or record.get("Question") or ""
    gold = record.get("gold_answer") or record.get("Correct") or ""
    answers = record.get("answers", {})
    judges = record.get("judges", {})
    details = record.get("hybrid_details", {})
    per_token = details.get("per_token", []) or []
    steering_selection = details.get("steering_selection", []) or []

    tokens, categories, coefs, windows, steered_mask = build_arrays(per_token, steering_selection)

    # Category heat row
    cat_row, cat_cmap, cat_labels = colorize_categories(categories, steered_mask)

    # Selection row (0/1)
    sel_row = np.array([[1 if s == "steered" else 0 for s in steering_selection]], dtype=np.int32)

    n_tokens = len(tokens)
    xtick_step = max(1, n_tokens // max_xticks)
    xticks = list(range(0, n_tokens, xtick_step))
    xticklabels = [_escape_token(tokens[i]) for i in xticks]

    fig = plt.figure(figsize=(min(16, max(10, n_tokens * 0.15)), 6))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.4])

    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(sel_row, aspect="auto", cmap=ListedColormap(["#dddddd", "#2ecc71"]), vmin=0, vmax=1)
    ax0.set_yticks([])
    ax0.set_xticks(xticks)
    ax0.set_xticklabels(xticklabels, rotation=90, fontsize=8)
    ax0.set_title("Steering usage (0 = unsteered, 1 = steered)")

    ax1 = fig.add_subplot(gs[1, 0])
    im1 = ax1.imshow(cat_row, aspect="auto", cmap=cat_cmap, vmin=0, vmax=len(cat_cmap.colors) - 1)
    ax1.set_yticks([])
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticklabels, rotation=90, fontsize=8)
    ax1.set_title("Latent category when steered (grey = unsteered)")
    # Legend for categories
    if cat_labels:
        handles = [Rectangle((0, 0), 1, 1, fc=cat_cmap(i + 1)) for i in range(len(cat_labels))]
        ax1.legend(handles, cat_labels, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=min(5, len(cat_labels)), frameon=False, fontsize=8)

    ax2 = fig.add_subplot(gs[2, 0])
    x_steered = [i for i, m in enumerate(steered_mask) if m == 1]
    y_coefs = [float(coefs[i]) if coefs[i] is not None else np.nan for i in x_steered]
    if x_steered:
        ax2.stem(x_steered, y_coefs, linefmt="#888888", markerfmt="o", basefmt=" ")
    ax2.set_xlim(-1, max(1, n_tokens))
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticklabels, rotation=90, fontsize=8)
    ax2.set_ylabel("Coefficient")
    ax2.set_title("Steering coefficients by token (only for steered tokens)")

    fig.suptitle(f"Hybrid steering — task {task_index}\nCorrect: {str(judges.get('hybrid', {}).get('correct'))} | Dataset: {record.get('dataset', '')}", fontsize=12)
    fig.tight_layout()

    plt.show()

    # Console summary
    steered_count = sum(steered_mask)
    total = len(steered_mask)
    print(f"Steered tokens: {steered_count}/{total} ({(steered_count/total*100 if total else 0):.1f}%)")
    if per_token:
        cat_counter = Counter([c for c, m in zip(categories, steered_mask) if m == 1])
        print("Top categories:")
        for cat, cnt in cat_counter.most_common(10):
            print(f"  {cat}: {cnt}")
        # Show first few steered instances with details
        print("\nFirst 10 steered tokens:")
        shown = 0
        for i, m in enumerate(steered_mask):
            if m == 1:
                coef = coefs[i]
                win = windows[i]
                tok = tokens[i]
                cat = categories[i]
                print(f"  idx={i:>4}  token={tok!r}  cat={cat}  coef={coef}  win={win}")
                shown += 1
                if shown >= 10:
                    break
    else:
        print("Per-token details missing in record (likely collected with collect_details=False). Showing selection only.")


def main():
    parser = argparse.ArgumentParser(description="Inspect a single hybrid-rolling example with steering visualization")
    parser.add_argument("--rolling-file", default="/workspace/cot-interp/hybrid/results/rolling/rolling_llama-3.1-8b_gsm8k.jsonl", help="Path to rolling_*.jsonl produced during evaluation")
    parser.add_argument("--task-index", type=int, default=50, help="0-based index of the record to inspect in the JSONL")
    parser.add_argument("--mode", choices=["html", "plot"], default="html", help="Visualization mode: html (colored text) or plot (heatmaps)")
    parser.add_argument("--save-html", action="store_true", default=False, help="Save HTML output to file instead of just displaying")
    args, _ = parser.parse_known_args()

    record, total = load_record(args.rolling_file, args.task_index)
    print(f"Loaded record {args.task_index} of {total} from {args.rolling_file}")
    if args.mode == "html":
        render_html(record, args.task_index, save_html=args.save_html, output_file=f"./hybrid_steering_task_{args.task_index}_{args.rolling_file.split('/')[-1].split('.')[0]}.html")
    else:
        plot_example(record, args.task_index)


if __name__ == "__main__":
    main()



# %%
