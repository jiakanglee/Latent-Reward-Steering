import os
import re
import json
import argparse
from typing import Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze rolling hybrid run logs to find steering usage stats per file."
    )
    parser.add_argument(
        "--rolling_dir",
        type=str,
        default=None,
        help="Directory containing rolling *.jsonl files. Defaults to hybrid/results/rolling next to this script.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="How many top latent titles/keys to display. None = all.",
    )
    parser.add_argument(
        "--save_json",
        action="store_true",
        default=False,
        help="If set, writes a *_vector_stats.json next to each rolling file.",
    )
    return parser.parse_known_args()[0]


def _default_rolling_dir() -> str:
    here = os.path.dirname(__file__)
    d = os.path.join(here, "results", "rolling")
    return d


def _iter_jsonl(path: str):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _analyze_file(path: str, *, top_k: Optional[int]) -> Dict[str, object]:
    return _analyze_group([path], display_name=os.path.basename(path), top_k=top_k)


def _analyze_group(paths: List[str], *, display_name: str, top_k: Optional[int]) -> Dict[str, object]:
    num_records: int = 0

    steered_tokens_total: int = 0
    tokens_total: int = 0

    # Latent tallies across all steered tokens
    latent_title_to_count: Dict[str, int] = {}
    latent_key_to_count: Dict[str, int] = {}

    # Coeff and window usage (only for steered tokens)
    coef_to_count: Dict[float, int] = {}
    window_to_count: Dict[int, int] = {}

    # Per-problem aggregates
    per_problem_steered_counts: List[int] = []
    per_problem_token_counts: List[int] = []
    missing_details_records: int = 0

    for path in paths:
        for rec in _iter_jsonl(path):
            num_records += 1
            assert "hybrid_details" in rec, "Missing 'hybrid_details' in record"
            hd = rec["hybrid_details"]
            assert isinstance(hd, dict)
            assert "steering_selection" in hd, "Missing 'steering_selection'"
            assert "per_token" in hd, "Missing 'per_token'"

            steering_selection = hd["steering_selection"]
            per_token = hd["per_token"]
            assert isinstance(steering_selection, list)
            assert isinstance(per_token, list)
            if len(per_token) == 0:
                missing_details_records += 1
            else:
                assert len(steering_selection) == len(per_token), "Length mismatch between steering_selection and per_token"

            n_tok = len(steering_selection)
            n_steered = sum(1 for s in steering_selection if s == "steered")

            tokens_total += n_tok
            steered_tokens_total += n_steered
            per_problem_token_counts.append(n_tok)
            per_problem_steered_counts.append(n_steered)

            # Tally only for steered positions
            for s, info in zip(steering_selection, per_token):
                assert isinstance(info, dict)
                if s != "steered":
                    continue
                title = info.get("latent_title")
                key = info.get("latent_key")
                coef = info.get("coefficient")
                window = info.get("window")

                assert title is not None, "Expected latent_title for steered token"
                latent_title_to_count[title] = latent_title_to_count.get(title, 0) + 1

                if key is not None:
                    latent_key_to_count[key] = latent_key_to_count.get(key, 0) + 1

                if coef is not None:
                    try:
                        c = float(coef)
                    except Exception:
                        raise AssertionError("Coefficient must be numeric")
                    coef_to_count[c] = coef_to_count.get(c, 0) + 1

                if window is not None:
                    try:
                        w = int(window)
                    except Exception:
                        raise AssertionError("Window must be int-like")
                    window_to_count[w] = window_to_count.get(w, 0) + 1

    assert num_records > 0, f"No records found in {display_name}"
    assert tokens_total > 0, "No generated tokens recorded"

    avg_steered_tokens_per_problem = sum(per_problem_steered_counts) / len(per_problem_steered_counts)
    avg_tokens_per_problem = sum(per_problem_token_counts) / len(per_problem_token_counts)
    avg_steered_fraction = steered_tokens_total / tokens_total
    steered_fractions = []
    for steered, total in zip(per_problem_steered_counts, per_problem_token_counts):
        assert total > 0, "Encountered problem with zero tokens"
        steered_fractions.append(steered / total)
    avg_steered_fraction_per_problem = sum(steered_fractions) / len(steered_fractions)

    def _top_n(d: Dict, k: Optional[int]):
        items = sorted(d.items(), key=lambda kv: kv[1], reverse=True)
        if k is None or k <= 0:
            return items
        return items[:k]

    top_titles = _top_n(latent_title_to_count, top_k)
    top_keys = _top_n(latent_key_to_count, top_k)
    top_coefs = _top_n(coef_to_count, top_k)
    top_windows = _top_n(window_to_count, top_k)

    most_used_title: Optional[str] = top_titles[0][0] if top_titles else None
    most_used_key: Optional[str] = top_keys[0][0] if top_keys else None

    return {
        "file": display_name,
        "source_files": [os.path.basename(p) for p in paths],
        "num_problems": num_records,
        "total_tokens": tokens_total,
        "total_steered_tokens": steered_tokens_total,
        "avg_steered_tokens_per_problem": avg_steered_tokens_per_problem,
        "avg_tokens_per_problem": avg_tokens_per_problem,
        "avg_steered_fraction": avg_steered_fraction,
        "avg_steered_fraction_per_problem": avg_steered_fraction_per_problem,
        "records_missing_per_token": missing_details_records,
        "most_used_latent_title": most_used_title,
        "most_used_latent_key": most_used_key,
        "top_latent_titles": top_titles,
        "top_latent_keys": top_keys,
        "top_coefficients": top_coefs,
        "top_windows": top_windows,
    }


def _split_prefix_parts(path: str) -> Tuple[str, Optional[int]]:
    base = os.path.basename(path)
    parent = os.path.dirname(path)
    if base.endswith(".jsonl"):
        base = base[:-6]
    m = re.match(r"^(.*)_(\d+)$", base)
    if not m:
        return os.path.join(parent, base + ".jsonl"), None
    prefix = os.path.join(parent, m.group(1) + ".jsonl")
    return prefix, int(m.group(2))


def _group_rollings(rolling_dir: str) -> Dict[str, List[str]]:
    files = {}
    for name in os.listdir(rolling_dir):
        if not name.endswith(".jsonl"):
            continue
        if not name.startswith("rolling_"):
            continue
        full = os.path.join(rolling_dir, name)
        prefix, part = _split_prefix_parts(full)
        if prefix not in files:
            files[prefix] = []
        if part is None:
            # Legacy single file; treat as part index -1 so we process first
            files[prefix].append(full)
        else:
            files[prefix].append((part, full))

    grouped: Dict[str, List[str]] = {}
    for prefix, entries in files.items():
        sorted_paths: List[str] = []
        parts = [e for e in entries if isinstance(e, tuple)]
        legacy = [e for e in entries if isinstance(e, str)]
        if legacy:
            sorted_paths.extend(sorted(legacy))
        for _, path in sorted(parts, key=lambda x: x[0]):
            sorted_paths.append(path)
        grouped[prefix] = sorted_paths
    return grouped


def _find_rolling_files(rolling_dir: str) -> List[Tuple[str, List[str]]]:
    grouped = _group_rollings(rolling_dir)
    all_groups: List[Tuple[str, List[str]]] = sorted(grouped.items(), key=lambda kv: kv[0])
    return all_groups


def main() -> None:
    args = parse_args()
    rolling_dir = args.rolling_dir or _default_rolling_dir()
    assert os.path.isdir(rolling_dir), f"Rolling dir not found: {rolling_dir}"
    groups = _find_rolling_files(rolling_dir)
    assert len(groups) > 0, f"No rolling files in {rolling_dir}"

    total_groups = len(groups)
    total_files = sum(len(paths) for _, paths in groups)
    print(f"Scanning {total_files} rolling files across {total_groups} groups in {rolling_dir}\n")
    for prefix, paths in groups:
        display_name = os.path.basename(prefix)
        stats = _analyze_group(paths, display_name=display_name, top_k=args.top_k)
        print(f"== {stats['file']} ==")
        if len(stats["source_files"]) > 1:
            joined = ", ".join(stats["source_files"])
            print(f"source parts: {joined}")
        print(f"problems: {stats['num_problems']}")
        print(f"avg steered tokens/problem: {stats['avg_steered_tokens_per_problem']:.2f}")
        print(f"avg total tokens/problem: {stats['avg_tokens_per_problem']:.2f}")
        print(f"avg steered fraction: {stats['avg_steered_fraction']:.3f}")
        print(f"avg steered fraction (per-problem): {stats['avg_steered_fraction_per_problem']:.3f}")
        print(f"most used latent title: {stats['most_used_latent_title']}")
        if int(stats.get("records_missing_per_token", 0)) > 0:
            print(f"records missing per_token details: {int(stats['records_missing_per_token'])}")
        total_titles = sum(cnt for _, cnt in stats["top_latent_titles"]) if stats["top_latent_titles"] else 0  # type: ignore[index]
        print(f"top latent titles: total={total_titles}")
        for title, cnt in stats["top_latent_titles"]:  # type: ignore[index]
            pct = (cnt / total_titles * 100.0) if total_titles > 0 else 0.0
            print(f"  {title}: {pct:.1f}% ({cnt})")
        total_coefs = sum(cnt for _, cnt in stats["top_coefficients"]) if stats["top_coefficients"] else 0  # type: ignore[index]
        print(f"top coefficients: total={total_coefs}")
        for coef, cnt in stats["top_coefficients"]:  # type: ignore[index]
            pct = (cnt / total_coefs * 100.0) if total_coefs > 0 else 0.0
            print(f"  {coef}: {pct:.1f}% ({cnt})")
        print()

        if args.save_json:
            out_path = prefix.replace(".jsonl", "_vector_stats.json")
            with open(out_path, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"Saved {out_path}\n")


if __name__ == "__main__":
    main()


