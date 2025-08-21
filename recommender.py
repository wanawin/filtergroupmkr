# ---- Helper shims (ensure env builder exists) ----
from collections import Counter
from typing import List, Dict, Tuple, Optional
import itertools as it

# Try to reuse from profiler if present
try:
    from profiler import build_env_for_draw, classify_structure, hot_cold_due, sum_category
    from profiler import MIRROR, VTRAC, digits_of
except Exception:
    # Minimal built-ins used by recommender
    MIRROR = {0:5,1:6,2:7,3:8,4:9,5:0,6:1,7:2,8:3,9:4}
    VTRAC  = {0:1,5:1,1:2,6:2,2:3,7:3,3:4,8:4,4:5,9:5}

    def digits_of(s: str) -> List[int]:
        return [int(ch) for ch in str(s) if ch.isdigit()]

    def sum_category(total: int) -> str:
        if 0 <= total <= 15: return "Very Low"
        if 16 <= total <= 24: return "Low"
        if 25 <= total <= 33: return "Mid"
        return "High"

    def classify_structure(digs: List[int]) -> str:
        c = Counter(digs)
        counts = sorted(c.values(), reverse=True)
        if counts == [5]:       return "quint"
        if counts == [4,1]:     return "quad"
        if counts == [3,2]:     return "triple_double"
        if counts == [3,1,1]:   return "triple"
        if counts == [2,2,1]:   return "double_double"
        if counts == [2,1,1,1]: return "double"
        return "single"

    def hot_cold_due(history: List[List[int]], k_hotcold: int = 10):
        flat = [d for row in history[-k_hotcold:] for d in row]
        cnt = Counter(flat)
        hot, cold = set(), set()
        if cnt:
            most = cnt.most_common()
            topk = 6
            thresh = most[topk-1][1] if len(most) >= topk else most[-1][1]
            hot = {d for d,c in most if c >= thresh}
            least = sorted(cnt.items(), key=lambda x: (x[1], x[0]))
            coldk = 4
            if least:
                cth = least[coldk-1][1] if len(least) >= coldk else least[0][1]
                cold = {d for d,c in least if c <= cth}
        last2 = set(d for row in history[-2:] for d in row)
        due = set(range(10)) - last2
        return hot, cold, due

    def build_env_for_draw(idx: int, winners: List[str]) -> Dict[str, object]:
        seed   = winners[idx-1]
        combo  = winners[idx]
        seed_list  = digits_of(seed)
        combo_list = sorted(digits_of(combo))

        seed_sum = sum(seed_list)
        combo_sum = sum(combo_list)

        prev_seed = winners[idx-1]
        prev_prev_seed = winners[idx-2] if idx >= 2 else None
        prev_seed_digits = digits_of(prev_seed)
        prev_prev_seed_digits = digits_of(prev_prev_seed) if prev_prev_seed is not None else []

        history_digits = [digits_of(s) for s in winners[:idx]]
        hot, cold, due = hot_cold_due(history_digits, k_hotcold=10)

        def parity_label(digs): return "Even" if sum(digs) % 2 == 0 else "Odd"
        prev_pattern = []
        for digs in (prev_prev_seed_digits, prev_seed_digits, seed_list):
            prev_pattern.extend([sum_category(sum(digs)), parity_label(digs)])
        prev_pattern = tuple(prev_pattern)

        env = {
            "combo": combo,
            "combo_digits": set(combo_list),
            "combo_digits_list": combo_list,
            "combo_sum": combo_sum,
            "combo_sum_cat": sum_category(combo_sum),
            "combo_sum_category": sum_category(combo_sum),

            "seed": seed,
            "seed_digits": set(seed_list),
            "seed_digits_list": seed_list,
            "seed_sum": seed_sum,
            "prev_sum_cat": sum_category(seed_sum),
            "seed_sum_category": sum_category(seed_sum),

            "spread_seed": max(seed_list) - min(seed_list) if seed_list else 0,
            "spread_combo": max(combo_list) - min(combo_list) if combo_list else 0,

            "seed_vtracs": set(VTRAC[d] for d in seed_list),
            "combo_vtracs": set(VTRAC[d] for d in combo_list),

            "last2": set(seed_list) | set(prev_seed_digits),

            "prev_seed": prev_seed,
            "prev_seed_digits": prev_seed_digits,
            "prev_seed_digits_list": prev_seed_digits,
            "prev_prev_seed": prev_prev_seed,
            "prev_prev_seed_digits": prev_prev_seed_digits,
            "prev_prev_seed_digits_list": prev_prev_seed_digits,
            "prev_pattern": prev_pattern,

            "new_seed_digits": set(seed_list) - set(prev_seed_digits),
            "seed_counts": Counter(seed_list),
            "common_to_both": set(seed_list) & set(prev_seed_digits),

            "hot_digits_10": hot,
            "cold_digits_10": cold,
            "hot_digits_20": hot,
            "cold_digits_20": cold,
            "hot_digits": sorted(hot),
            "cold_digits": sorted(cold),
            "due_digits": sorted(due),
            "due_digits_2": due,

            "mirror": MIRROR,
            "vtrac": VTRAC,

            # safe builtins
            "any": any, "all": all, "len": len, "sum": sum,
            "max": max, "min": min, "set": set, "sorted": sorted, "Counter": Counter,
        }
        return env
# ---- end helper shims ----
