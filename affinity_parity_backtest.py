#!/usr/bin/env python3
"""
Quick backtest: does the majority parity of the top-k% by affinity (winner-independent)
predict the actual winner's parity?

- Uses your existing winners CSV.
- Builds "today" context from the previous draw (seed) for each day.
- Generates a synthetic pool for that day that matches your constraints:
  * pool_mode="1digit": each combo shares >=1 distinct digit with the seed (default)
  * pool_mode="2digit": each combo shares >=2 distinct digits with the seed
- Scores every combo with the SAME combo_affinity heuristic we added in recommender.
- Checks the majority parity among the top-k% (k ∈ {10, 20, 30}% by default) of that day's pool.
- Compares against the true winner's parity (both sum-parity and parity-major labels).
- Also reports a baseline: majority parity over the whole pool (no affinity).

Outputs two CSVs:
  - parity_affinity_backtest.csv   (one row per day with predictions)
  - parity_affinity_summary.csv    (aggregate accuracies & lift over baseline)

Run (examples):
  python affinity_parity_backtest.py --winners DC5_Midday_Full_Cleaned_Expanded.csv
  python affinity_parity_backtest.py --winners DC5_Midday_Full_Cleaned_Expanded.csv \
         --n_days 240 --pool_mode 1digit --pool_size 1800 --top_pcts 0.2,0.3
"""

import argparse
import csv
import math
import random
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd

# Import utilities from your recommender
from recommender import (
    load_winners,
    build_env_for_draw,
    digits_of,
    classify_structure,
)

# ---------------------------
# Helper labels
# ---------------------------

def parity_major_label(digs):
    return "even>=3" if sum(1 for d in digs if d % 2 == 0) >= 3 else "even<=2"


def sum_parity_label(digs):
    return "even" if (sum(digs) % 2 == 0) else "odd"


# ---------------------------
# Affinity (same heuristic used in recommender)
# ---------------------------

def combo_affinity(env_now: dict, combo: str) -> float:
    cd = digits_of(combo)
    seedd = env_now['seed_digits_list']

    combo_sum = sum(cd)
    seed_sum = env_now['seed_sum']
    sum_prox = math.exp(-abs(combo_sum - seed_sum) / 2.0)

    spread_c = max(cd) - min(cd)
    spread_s = env_now['spread_seed']
    spread_prox = math.exp(-abs(spread_c - spread_s) / 2.0)

    struct_match = 1.0 if classify_structure(cd) == classify_structure(seedd) else 0.0
    parity_match = 1.0 if parity_major_label(cd) == parity_major_label(seedd) else 0.0

    # High-digit (>=8) pattern handling
    hi8_seed = sum(1 for d in seedd if d >= 8)
    hi8_combo = sum(1 for d in cd if d >= 8)
    if hi8_seed >= 3:
        # If seed is unusually heavy in high digits, prefer combos with fewer highs (<=1)
        hi8_score = math.exp(-max(0, hi8_combo - 1))
    else:
        # Otherwise prefer closeness in count of high digits
        hi8_score = math.exp(-abs(hi8_combo - hi8_seed) / 1.5)

    overlap = len(set(seedd) & set(cd))
    overlap_eq1 = 1.0 if overlap == 1 else 0.0

    # Weights (tunable; mirror recommender)
    W_SUM = 0.8
    W_SPD = 0.3
    W_STR = 0.4
    W_PAR = 0.15
    W_HI8 = 0.35
    W_OV1 = 0.25

    return (
        W_SUM * sum_prox +
        W_SPD * spread_prox +
        W_STR * struct_match +
        W_PAR * parity_match +
        W_HI8 * hi8_score +
        W_OV1 * overlap_eq1
    )


# ---------------------------
# Pool generation for a day
# ---------------------------

def gen_pool_for_seed(seed_digits: list[int], pool_size: int, mode: str, rng: random.Random) -> list[str]:
    """Generate a pool of 5-digit strings under a seed-anchored constraint.
       mode="1digit": require >=1 distinct shared digit with seed
       mode="2digit": require >=2 distinct shared digits with seed
    """
    seed_set = set(seed_digits)
    required = 1 if mode == "1digit" else 2
    out = []
    seen = set()
    while len(out) < pool_size:
        s = ''.join(str(rng.randrange(10)) for _ in range(5))
        if s in seen:
            continue
        seen.add(s)
        digs = set(digits_of(s))
        if len(digs & seed_set) >= required:
            out.append(s)
    return out


# ---------------------------
# Main backtest
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--winners', required=True, help='Path to winners CSV')
    ap.add_argument('--n_days', type=int, default=240, help='How many most-recent days to test')
    ap.add_argument('--pool_mode', choices=['1digit','2digit'], default='1digit', help='Pool constraint mode')
    ap.add_argument('--pool_size', type=int, default=1800, help='Pool size to generate per day')
    ap.add_argument('--top_pcts', type=str, default='0.2,0.3', help='Top percent thresholds, comma sep (e.g. 0.1,0.2)')
    ap.add_argument('--out_prefix', type=str, default='parity_affinity', help='Prefix for output CSVs')
    ap.add_argument('--seed', type=int, default=12345, help='Random seed for reproducibility')
    args = ap.parse_args()
ap.add_argument('--use_archived_pools', action='store_true',
                help='Load real pools via recommender.get_pool_for_seed')

    winners = load_winners(args.winners)
    N = len(winners)
# Also keep a DataFrame view so get_pool_for_seed can read the date column
winners_df_full = pd.read_csv(args.winners)

    if N < 3:
        raise SystemExit('Not enough winners to run a backtest')

    # Indices to test: last n_days (each index t uses seed winners[t-1], winner winners[t])
    end = N - 1
    start = max(2, end - args.n_days)

    rng = random.Random(args.seed)

    top_ps = [float(x) for x in args.top_pcts.split(',') if x.strip()]

    rows = []
    # For aggregated accuracy
    agg = defaultdict(lambda: {'sum_hits':0,'sum_total':0,'maj_hits':0,'maj_total':0,'base_sum_hits':0,'base_maj_hits':0})

    for t in range(start, end):
        env = build_env_for_draw(t, winners)
        seed_digits = env['seed_digits_list']
        true_win = winners[t]
        true_digs = digits_of(true_win)
        true_sum_par = sum_parity_label(true_digs)
        true_maj_par = parity_major_label(true_digs)

       # Generate/Load pool for the day
if args.use_archived_pools:
    import recommender as rec
    # Use the actual seed row so get_pool_for_seed can grab the correct date
    seed_row = winners_df_full.iloc[t-1]
    pool_df = rec.get_pool_for_seed(seed_row)  # returns DataFrame with 'combo'
    pool = pool_df["combo"].astype(str).tolist()
else:
    day_rng = random.Random((args.seed * 1000003) ^ t)
    pool = gen_pool_for_seed(seed_digits, args.pool_size, args.pool_mode, day_rng)


        # Affinity per combo
        aff = {c: combo_affinity(env, c) for c in pool}
        sorted_pool = sorted(pool, key=lambda c: aff[c], reverse=True)

        # Baseline predictions from full pool (no affinity)
        base_sum_counts = Counter(sum_parity_label(digits_of(c)) for c in pool)
        base_maj_counts = Counter(parity_major_label(digits_of(c)) for c in pool)
        base_sum_pred, base_sum_cnt = base_sum_counts.most_common(1)[0]
        base_maj_pred, base_maj_cnt = base_maj_counts.most_common(1)[0]

        for p in top_ps:
            k = max(1, int(len(sorted_pool) * p))
            topk = sorted_pool[:k]

            # Majority predictions within top-k
            sum_counts = Counter(sum_parity_label(digits_of(c)) for c in topk)
            maj_counts = Counter(parity_major_label(digits_of(c)) for c in topk)
            # Handle possible ties deterministically by label sort
            sum_pred = sorted(sum_counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
            maj_pred = sorted(maj_counts.items(), key=lambda x: (-x[1], x[0]))[0][0]

            sum_hit = int(sum_pred == true_sum_par)
            maj_hit = int(maj_pred == true_maj_par)
            base_sum_hit = int(base_sum_pred == true_sum_par)
            base_maj_hit = int(base_maj_pred == true_maj_par)

            rows.append({
                't': t,
                'seed': ''.join(str(d) for d in seed_digits),
                'winner': true_win,
                'top_pct': p,
                'sum_pred': sum_pred,
                'sum_true': true_sum_par,
                'sum_hit': sum_hit,
                'maj_pred': maj_pred,
                'maj_true': true_maj_par,
                'maj_hit': maj_hit,
                'base_sum_pred': base_sum_pred,
                'base_sum_hit': base_sum_hit,
                'base_maj_pred': base_maj_pred,
                'base_maj_hit': base_maj_hit,
                'pool_mode': args.pool_mode,
                'pool_size': args.pool_size,
            })

            key = f'p={p}'
            agg[key]['sum_hits'] += sum_hit
            agg[key]['sum_total'] += 1
            agg[key]['maj_hits'] += maj_hit
            agg[key]['maj_total'] += 1
            agg[key]['base_sum_hits'] += base_sum_hit
            agg[key]['base_maj_hits'] += base_maj_hit

    # Write per-day CSV
    out1 = f'{args.out_prefix}_backtest.csv'
    pd.DataFrame(rows).to_csv(out1, index=False)

    # Summary CSV
    summary_rows = []
    for key, d in agg.items():
        sum_acc = d['sum_hits'] / max(1, d['sum_total'])
        maj_acc = d['maj_hits'] / max(1, d['maj_total'])
        base_sum_acc = d['base_sum_hits'] / max(1, d['sum_total'])
        base_maj_acc = d['base_maj_hits'] / max(1, d['maj_total'])
        summary_rows.append({
            'top_pct': key,
            'sum_acc': round(sum_acc, 4),
            'sum_base_acc': round(base_sum_acc, 4),
            'sum_lift': round(sum_acc - base_sum_acc, 4),
            'maj_acc': round(maj_acc, 4),
            'maj_base_acc': round(base_maj_acc, 4),
            'maj_lift': round(maj_acc - base_maj_acc, 4),
            'n_days': d['sum_total'],
            'pool_mode': args.pool_mode,
            'pool_size': args.pool_size,
        })

    out2 = f'{args.out_prefix}_summary.csv'
    pd.DataFrame(summary_rows).to_csv(out2, index=False)

    print(f"Wrote {out1} and {out2}.")


if __name__ == '__main__':
    main()
