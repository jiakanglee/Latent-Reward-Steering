#!/usr/bin/env python3
"""
合并 generate_data_7B.py 多卡分片输出为单个 .pt 文件。
用法: python merge_collect_shards.py --base collected_sae_latents_10dim_4000_mbpp
"""
import argparse
import glob
import os
import re
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=str, required=True,
        help='输出文件基础名，不含 .pt（如 collected_sae_latents_10dim_4000_mbpp）')
    parser.add_argument('--output', type=str, default=None,
        help='最终输出路径（默认 {base}.pt）')
    args = parser.parse_args()

    pattern = f"{args.base}_shard_*.pt"
    files = glob.glob(pattern)

    def _shard_num(p):
        m = re.search(r"_shard_(\d+)\.pt", p)
        return int(m.group(1)) if m else 999

    files = sorted(files, key=_shard_num)

    if not files:
        print(f"❌ No shard files found: {pattern}")
        return

    all_data = []
    for f in files:
        data = torch.load(f)
        all_data.extend(data)
        print(f"  + {os.path.basename(f)}: {len(data)} items")

    out_path = args.output or f"{args.base}.pt"
    torch.save(all_data, out_path)
    print(f"\n✅ Merged {len(all_data)} items to {out_path}")


if __name__ == "__main__":
    main()
