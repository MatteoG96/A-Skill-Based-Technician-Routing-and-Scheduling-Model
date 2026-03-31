#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional
import json
import math
import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


def _euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _rand_point(rng: random.Random, lo: float, hi: float) -> Tuple[float, float]:
    return (rng.uniform(lo, hi), rng.uniform(lo, hi))


def _sample_without_replacement(rng: random.Random, items: List[str], k: int) -> List[str]:
    k = min(k, len(items))
    items2 = items[:]
    rng.shuffle(items2)
    return items2[:k]


def save_instance_json(data: Dict[str, Any], path: str) -> None:
    """Save instance to JSON. Converts tuple keys to strings automatically."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def encode_keys(obj):
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if isinstance(k, tuple):
                    kk = "|".join(map(str, k))
                else:
                    kk = str(k)
                out[kk] = encode_keys(v)
            return out
        if isinstance(obj, list):
            return [encode_keys(x) for x in obj]
        return obj

    with open(path, "w", encoding="utf-8") as f:
        json.dump(encode_keys(data), f, indent=2, ensure_ascii=False)

# -------------------------
# Styles / config
# -------------------------

@dataclass(frozen=True)
class GenConfig:
    style: str = "clustered"        # "clustered" | "scattered" | "multi_region"
    seed: int = 1
    H: int = 30

    n_bases: int = 2
    n_techs: int = 6
    n_plants: int = 20
    n_tickets: int = 40

    modes: Tuple[str, ...] = ("car", "train", "air")

    # Geometry / travel
    coord_lo: float = 0.0
    coord_hi: float = 500.0
    km_per_day_by_mode: Dict[str, float] = None  # if None, defaults below

    # Durations
    dur_min: int = 1
    dur_max: int = 4
    skill_spread: float = 0.35      # variability per technician (higher -> more heterogeneity)
    bigM_dur: int = 999             # if technician cannot do ticket, set dur[t,k]=bigM_dur

    # Tickets timing
    release_window: Tuple[int, int] = (1, 20)  # a_t in [..]
    sla_slack_window: Tuple[int, int] = (2, 10)  # b_t = a_t + slack, clipped to H

    # Hotel rule
    hotel_distance_threshold: float = 40.0      # km from base to plant => hotel required
    hotel_prob_extra: float = 0.10             # additional random chance (noise) for hotel flag

    # Feasibility / capability
    capable_k_per_ticket: Tuple[int, int] = (2, 4)  # each ticket doable by this many techs (approx)

    # Objective weights
    alpha: float = 1.0
    beta: float = 1.0
    eta: float = 1.0
    gamma: float = 10.0

    # Costs
    c_hotel: float = 80.0
    EF_by_mode: Dict[str, float] = None         # kgCO2/km

    # Away policy
    X: int = 5


def _default_km_per_day():
    return {"car": 500.0, "train": 800.0, "air": 1500.0}


def _default_EF():
    # toy numbers; replace with your own assumptions
    return {"car": 0.200, "train": 0.035, "air": 0.285}


# # GENERATE DATA

# In[ ]:


from dataclasses import dataclass
from typing import Dict, Any, Tuple, List
import math
import random


# -------------------------
# Helpers
# -------------------------
def _euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _rand_point(rng: random.Random, lo: float, hi: float) -> Tuple[float, float]:
    return (rng.uniform(lo, hi), rng.uniform(lo, hi))


def _sample_without_replacement(rng: random.Random, items: List[str], k: int) -> List[str]:
    items2 = list(items)
    rng.shuffle(items2)
    return items2[:k]


def _default_km_per_day() -> Dict[str, float]:
    return {"car": 320.0, "train": 700.0, "air": 1400.0}


def _default_EF() -> Dict[str, float]:
    return {"car": 0.18, "train": 0.06, "air": 0.25}


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _jitter(
    rng: random.Random,
    center: Tuple[float, float],
    rx: float,
    ry: float,
) -> Tuple[float, float]:
    return (
        center[0] + rng.uniform(-rx, rx),
        center[1] + rng.uniform(-ry, ry),
    )


def _point_on_segment(
    rng: random.Random,
    a: Tuple[float, float],
    b: Tuple[float, float],
    lateral_noise: float,
) -> Tuple[float, float]:
    u = rng.uniform(0.0, 1.0)
    x = a[0] + u * (b[0] - a[0])
    y = a[1] + u * (b[1] - a[1])

    dx = b[0] - a[0]
    dy = b[1] - a[1]
    norm = math.hypot(dx, dy)
    if norm > 1e-9:
        nx, ny = -dy / norm, dx / norm
        off = rng.uniform(-lateral_noise, lateral_noise)
        x += off * nx
        y += off * ny
    return (x, y)


def _sample_separated_points(
    rng: random.Random,
    n: int,
    lo: float,
    hi: float,
    min_sep: float,
    max_tries: int = 5000,
) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    for _ in range(max_tries):
        if len(pts) >= n:
            break
        cand = _rand_point(rng, lo, hi)
        if all(_euclid(cand, p) >= min_sep for p in pts):
            pts.append(cand)

    while len(pts) < n:
        pts.append(_rand_point(rng, lo, hi))

    return pts


# -------------------------
# Generator config
# -------------------------
@dataclass(frozen=True)
class GenConfig:
    style: str
    seed: int
    H: int
    n_bases: int
    n_techs: int
    n_plants: int
    n_tickets: int
    modes: Tuple[str, ...]

    coord_lo: float = 0.0
    coord_hi: float = 500.0

    release_window: Tuple[int, int] = (1, 20)
    sla_slack_window: Tuple[int, int] = (2, 10)

    dur_min: int = 1
    dur_max: int = 5
    capable_k_per_ticket: Tuple[int, int] = (2, 4)
    bigM_dur: int = 999

    skill_spread: float = 0.15

    hotel_distance_threshold: float = 180.0
    hotel_prob_extra: float = 0.08
    c_hotel: float = 120.0

    X: int = 5

    alpha: float = 1.0
    beta: float = 1.0
    eta: float = 1.0
    gamma: float = 1.0

    c_idle_away: float = 300.0
    delta: float = 1.0

    km_per_day_by_mode: Dict[str, float] = None
    EF_by_mode: Dict[str, float] = None


# -------------------------
# Seed-driven geography
# -------------------------
def _generate_hubs_clustered(
    rng: random.Random,
    n_hubs: int,
    lo: float,
    hi: float,
) -> List[Tuple[float, float]]:
    span = hi - lo
    min_sep = 0.22 * span
    return _sample_separated_points(
        rng=rng,
        n=n_hubs,
        lo=lo + 0.12 * span,
        hi=hi - 0.12 * span,
        min_sep=min_sep,
    )


def _generate_hubs_multi_region(
    rng: random.Random,
    n_hubs: int,
    lo: float,
    hi: float,
) -> List[Tuple[float, float]]:
    span = hi - lo
    n_regions = min(4, max(2, n_hubs))

    region_centers = _sample_separated_points(
        rng=rng,
        n=n_regions,
        lo=lo + 0.15 * span,
        hi=hi - 0.15 * span,
        min_sep=0.35 * span,
    )

    hubs: List[Tuple[float, float]] = []
    for i in range(n_hubs):
        rc = region_centers[i % n_regions]
        pt = _jitter(rng, rc, 0.05 * span, 0.05 * span)
        hubs.append((_clip(pt[0], lo, hi), _clip(pt[1], lo, hi)))
    return hubs


def _generate_hubs_scattered(
    rng: random.Random,
    n_hubs: int,
    lo: float,
    hi: float,
) -> List[Tuple[float, float]]:
    span = hi - lo
    return _sample_separated_points(
        rng=rng,
        n=n_hubs,
        lo=lo + 0.08 * span,
        hi=hi - 0.08 * span,
        min_sep=0.12 * span,
    )


def _build_corridors(
    hubs: List[Tuple[float, float]]
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    if len(hubs) < 2:
        return []

    remaining = list(range(len(hubs)))
    remaining.sort(key=lambda i: (hubs[i][0], hubs[i][1]))

    used = {remaining[0]}
    corridors: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []

    while len(used) < len(hubs):
        best_pair = None
        best_d = float("inf")
        for i in used:
            for j in remaining:
                if j in used:
                    continue
                d = _euclid(hubs[i], hubs[j])
                if d < best_d:
                    best_d = d
                    best_pair = (i, j)
        i, j = best_pair
        corridors.append((hubs[i], hubs[j]))
        used.add(j)

    # add one or two extra links for realism
    if len(hubs) >= 3:
        hub_ids = list(range(len(hubs)))
        extra_candidates = []
        for i in hub_ids:
            for j in hub_ids:
                if i < j:
                    seg = (hubs[i], hubs[j])
                    if seg not in corridors and (hubs[j], hubs[i]) not in corridors:
                        extra_candidates.append((i, j, _euclid(hubs[i], hubs[j])))
        extra_candidates.sort(key=lambda x: x[2])
        for i, j, _ in extra_candidates[:2]:
            corridors.append((hubs[i], hubs[j]))

    return corridors


def _generate_coords(
    rng: random.Random,
    B: List[str],
    P: List[str],
    cfg: GenConfig,
) -> Dict[str, Tuple[float, float]]:
    lo, hi = cfg.coord_lo, cfg.coord_hi
    span = hi - lo
    coords: Dict[str, Tuple[float, float]] = {}

    n_hubs = max(2, min(6, max(cfg.n_bases, 3)))

    if cfg.style == "clustered":
        hubs = _generate_hubs_clustered(rng, n_hubs, lo, hi)
    elif cfg.style == "multi_region":
        hubs = _generate_hubs_multi_region(rng, n_hubs, lo, hi)
    elif cfg.style == "scattered":
        hubs = _generate_hubs_scattered(rng, n_hubs, lo, hi)
    else:
        raise ValueError(f"Unknown style='{cfg.style}'. Use: clustered | scattered | multi_region")

    corridors = _build_corridors(hubs)

    # bases near hubs
    assigned_hubs = _sample_without_replacement(rng, list(range(len(hubs))), min(len(B), len(hubs)))
    while len(assigned_hubs) < len(B):
        assigned_hubs.append(rng.randrange(len(hubs)))

    for i, b in enumerate(B):
        h = hubs[assigned_hubs[i]]
        pt = _jitter(rng, h, 0.03 * span, 0.03 * span)
        coords[b] = (_clip(pt[0], lo, hi), _clip(pt[1], lo, hi))

    # plants generated from style-specific mixture
    for p in P:
        r = rng.random()

        if cfg.style == "clustered":
            if r < 0.70:
                h = hubs[rng.randrange(len(hubs))]
                pt = _jitter(rng, h, 0.08 * span, 0.07 * span)
            elif r < 0.90 and corridors:
                a, b = corridors[rng.randrange(len(corridors))]
                pt = _point_on_segment(rng, a, b, lateral_noise=0.03 * span)
            else:
                h = hubs[rng.randrange(len(hubs))]
                angle = rng.uniform(0, 2 * math.pi)
                rad = rng.uniform(0.12 * span, 0.22 * span)
                pt = (h[0] + rad * math.cos(angle), h[1] + rad * math.sin(angle))

        elif cfg.style == "multi_region":
            if r < 0.58:
                h = hubs[rng.randrange(len(hubs))]
                pt = _jitter(rng, h, 0.10 * span, 0.08 * span)
            elif r < 0.82 and corridors:
                a, b = corridors[rng.randrange(len(corridors))]
                pt = _point_on_segment(rng, a, b, lateral_noise=0.035 * span)
            else:
                h = hubs[rng.randrange(len(hubs))]
                angle = rng.uniform(0, 2 * math.pi)
                rad = rng.uniform(0.16 * span, 0.28 * span)
                pt = (h[0] + rad * math.cos(angle), h[1] + rad * math.sin(angle))

        else:  # scattered
            if r < 0.40:
                h = hubs[rng.randrange(len(hubs))]
                pt = _jitter(rng, h, 0.12 * span, 0.12 * span)
            elif r < 0.70 and corridors:
                a, b = corridors[rng.randrange(len(corridors))]
                pt = _point_on_segment(rng, a, b, lateral_noise=0.045 * span)
            else:
                pt = _rand_point(rng, lo, hi)

        coords[p] = (_clip(pt[0], lo, hi), _clip(pt[1], lo, hi))

    return coords


def _travel_days(
    dij: float,
    mode: str,
    cfg: GenConfig,
    is_remote: bool = False,
) -> int:
    if dij <= 1e-9:
        return 0

    km_per_day = float(cfg.km_per_day_by_mode[mode])

    if mode == "car":
        overhead = 0.05 if dij < 80 else 0.15
    elif mode == "train":
        overhead = 0.20 if dij < 180 else 0.35
    elif mode == "air":
        overhead = 0.60 if dij < 350 else 0.85
    else:
        overhead = 0.20

    if is_remote:
        overhead += 0.20

    effective_days = dij / km_per_day + overhead
    return max(1, int(math.ceil(effective_days)))


# -------------------------
# Main generator
# -------------------------
def generate_instance(
    *,
    style: str = "clustered",
    seed: int = 1,
    n_bases: int = 2,
    n_techs: int = 6,
    n_plants: int = 20,
    n_tickets: int = 40,
    H: int = 30,
    X: int = 5,
    modes: Tuple[str, ...] = ("car", "train", "air"),
    c_idle_away: float = 300.0,
    delta: float = 1.0,
) -> Dict[str, Any]:
    cfg = GenConfig(
        style=style,
        seed=seed,
        H=H,
        n_bases=n_bases,
        n_techs=n_techs,
        n_plants=n_plants,
        n_tickets=n_tickets,
        modes=modes,
        X=X,
        km_per_day_by_mode=_default_km_per_day(),
        EF_by_mode=_default_EF(),
        c_idle_away=float(c_idle_away),
        delta=float(delta),
    )

    rng = random.Random(cfg.seed)

    B = [f"b{b+1}" for b in range(cfg.n_bases)]
    K = [f"k{k+1}" for k in range(cfg.n_techs)]
    P = [f"p{p+1}" for p in range(cfg.n_plants)]
    T = [f"t{t+1}" for t in range(cfg.n_tickets)]
    D = list(range(1, cfg.H + 1))
    M = list(cfg.modes)
    N = B + P

    coords = _generate_coords(rng, B, P, cfg)

    base: Dict[str, str] = {}
    for idx, k in enumerate(K):
        base[k] = B[idx % len(B)]

    nearest_base_dist: Dict[str, float] = {}
    for p in P:
        nearest_base_dist[p] = min(_euclid(coords[p], coords[b]) for b in B)

    plant_scores: List[float] = []
    for p in P:
        score = 1.0 / (1.0 + 0.01 * nearest_base_dist[p])
        score *= rng.uniform(0.85, 1.15)
        plant_scores.append(score)

    total_score = sum(plant_scores)
    cum_scores: List[float] = []
    s = 0.0
    for sc in plant_scores:
        s += sc / total_score
        cum_scores.append(s)

    def sample_plant() -> str:
        u = rng.random()
        for i, cs in enumerate(cum_scores):
            if u <= cs:
                return P[i]
        return P[-1]

    node: Dict[str, str] = {t: sample_plant() for t in T}

    a: Dict[str, int] = {}
    b: Dict[str, int] = {}
    w: Dict[str, float] = {}

    rel_lo, rel_hi = cfg.release_window
    slack_lo, slack_hi = cfg.sla_slack_window
    rel_hi = min(rel_hi, cfg.H)

    for t in T:
        a_t = rng.randint(rel_lo, rel_hi)
        severity = rng.random()

        if severity < 0.20:
            weight = rng.choice([4, 5])
            slack = rng.randint(max(1, slack_lo - 1), max(2, min(4, slack_hi)))
        elif severity < 0.60:
            weight = rng.choice([2, 2, 3])
            slack = rng.randint(max(2, slack_lo), min(7, slack_hi))
        else:
            weight = rng.choice([1, 1, 2])
            slack = rng.randint(max(4, slack_lo), slack_hi)

        a[t] = int(a_t)
        b[t] = int(min(cfg.H, a_t + slack))
        w[t] = float(weight)

    dist: Dict[Tuple[str, str], float] = {}
    tt: Dict[Tuple[str, str, str], int] = {}

    remote_flag: Dict[str, bool] = {
        p: nearest_base_dist[p] > cfg.hotel_distance_threshold * 0.95 for p in P
    }

    for i in N:
        for j in N:
            dij = 0.0 if i == j else _euclid(coords[i], coords[j])
            dist[(i, j)] = float(round(dij, 3))

            is_remote_leg = False
            if i in P and remote_flag[i]:
                is_remote_leg = True
            if j in P and remote_flag[j]:
                is_remote_leg = True

            for md in M:
                tt[(i, j, md)] = _travel_days(dij, md, cfg, is_remote=is_remote_leg)

    EF = dict(cfg.EF_by_mode)

    Hkp: Dict[Tuple[str, str], int] = {}
    for k in K:
        bk = base[k]
        for p in P:
            d = dist[(bk, p)]
            need = 0

            if d > cfg.hotel_distance_threshold:
                need = 1

            car_days = tt[(bk, p, "car")] if "car" in M else 99
            train_days = tt[(bk, p, "train")] if "train" in M else 99
            best_ground = min(car_days, train_days)

            if best_ground >= 2:
                need = 1

            if remote_flag[p] and d > 0.7 * cfg.hotel_distance_threshold and rng.random() < 0.35:
                need = 1

            if rng.random() < cfg.hotel_prob_extra:
                need = 1

            Hkp[(k, p)] = int(need)

    dur: Dict[Tuple[str, str], int] = {}
    cap_lo, cap_hi = cfg.capable_k_per_ticket

    tech_factor: Dict[str, float] = {}
    for k in K:
        tech_factor[k] = max(0.65, rng.gauss(1.0, cfg.skill_spread))

    base_strength: Dict[str, float] = {b: rng.uniform(0.85, 1.15) for b in B}

    for t in T:
        p = node[t]
        base_duration = rng.randint(cfg.dur_min, cfg.dur_max)

        if nearest_base_dist[p] > cfg.hotel_distance_threshold and rng.random() < 0.5:
            base_duration = min(cfg.dur_max, base_duration + 1)

        cap_k = rng.randint(cap_lo, min(cap_hi, len(K)))

        tech_rank = sorted(
            K,
            key=lambda k: _euclid(coords[base[k]], coords[p]) + rng.uniform(0.0, 20.0)
        )
        capable_pool = tech_rank[:max(cap_k + 1, min(len(K), cap_k + 2))]
        capable_techs = set(_sample_without_replacement(rng, capable_pool, min(cap_k, len(capable_pool))))

        if len(capable_techs) < cap_k:
            missing = [k for k in K if k not in capable_techs]
            extra = _sample_without_replacement(rng, missing, cap_k - len(capable_techs))
            capable_techs.update(extra)

        nearest_b = min(B, key=lambda b_: _euclid(coords[b_], coords[p]))

        for k in K:
            if k not in capable_techs:
                dur[(t, k)] = int(cfg.bigM_dur)
                continue

            familiarity = 0.92 if base[k] == nearest_b else 1.05
            specialization = base_strength[base[k]]
            raw = base_duration * tech_factor[k] * familiarity / specialization
            dur[(t, k)] = int(max(1, round(raw)))

    data: Dict[str, Any] = {
        "B": B,
        "K": K,
        "P": P,
        "T": T,
        "D": D,
        "M": M,
        "N": N,
        "a": a,
        "b": b,
        "w": w,
        "dur": dur,
        "node": node,
        "base": base,
        "dist": dist,
        "tt": tt,
        "EF": EF,
        "Hkp": Hkp,
        "c_hotel": float(cfg.c_hotel),
        "X": int(cfg.X),
        "alpha": float(cfg.alpha),
        "beta": float(cfg.beta),
        "eta": float(cfg.eta),
        "gamma": float(cfg.gamma),
        "c_idle_away": float(cfg.c_idle_away),
        "delta": float(cfg.delta),
        "_meta": {
            "style": cfg.style,
            "seed": cfg.seed,
            "coords": coords,
        },
    }
    return data


# In[ ]:


#inst = generate_instance(style="clustered", seed=4, n_bases=2, n_techs=4, n_plants=6, n_tickets=10, H=30, X=5)
inst = generate_instance(style="clustered", seed=4, n_bases=3, n_techs=12, n_plants=25, n_tickets=50, H=30)

save_instance_json(inst, "instances/demo_clustered_seed4.json")
print("Saved:", "instances/demo_clustered_seed4.json")


# # VISUALIZATION

# In[ ]:


PATH = r"G:\Il mio Drive\LAVORO\1.RICERCA\0.CONFERENCE PAPER\4.KES\2026\1.CODE\instances\demo_clustered_seed4.json"

with open(PATH, "r") as f:
    data = json.load(f)

print(data.keys())


# In[ ]:


tickets = list(data["T"])

df_tw = pd.DataFrame({
    "ticket": tickets,
    "a": [data["a"][t] for t in tickets],
    "b": [data["b"][t] for t in tickets],
    "weight": [data["w"][t] for t in tickets],
    "plant": [data["node"][t] for t in tickets]   # ← ADDED
})

df_tw["duration"] = df_tw["b"] - df_tw["a"]

# Sort by plant then by start day (optional but cleaner visually)
df_tw = df_tw.sort_values(["plant", "a"]).reset_index(drop=True)

plt.figure(figsize=(10, 12))

# Use seaborn color palette
plants = df_tw["plant"].unique()
palette = dict(zip(plants, sns.color_palette("tab20", len(plants))))

for i, row in df_tw.iterrows():
    plt.barh(
        row["ticket"],
        row["duration"],
        left=row["a"],
        color=palette[row["plant"]],
        edgecolor="black"
    )

plt.xlabel("Day")
plt.ylabel("Ticket")
plt.title("Ticket Time Windows (Colored by Plant)")
plt.xticks(range(1, 31))
plt.grid(axis="x", linestyle="--", alpha=0.5)

# Legend
handles = [
    plt.Rectangle((0,0),1,1, color=palette[p])
    for p in plants
]
plt.legend(handles, plants, title="Plant", bbox_to_anchor=(1.02,1), loc="upper left")

plt.tight_layout()
plt.show()


# In[ ]:


dur = data["dur"]

records = []
for key, value in dur.items():
    t, k = key.split("|")
    if value < 999:   # only feasible
        records.append((t, k, value))

df_dur = pd.DataFrame(records, columns=["ticket", "technician", "duration"])

pivot = df_dur.pivot(index="ticket", columns="technician", values="duration")

plt.figure(figsize=(8, 10))
sns.heatmap(pivot, cmap="coolwarm", annot=False)
plt.title("Skill Matrix (Duration if Assigned)")
plt.show()


# In[ ]:


node = data["node"]

df_node = pd.DataFrame({
    "ticket": list(node.keys()),
    "plant": list(node.values())
})

counts = df_node["plant"].value_counts()

plt.figure(figsize=(10, 4))
counts.sort_index().plot(kind="bar")
plt.title("Number of Tickets per Plant")
plt.ylabel("Count")
plt.show()


# In[ ]:


weights = list(data["w"].values())

plt.figure(figsize=(6,4))
plt.hist(weights, bins=5)
plt.title("Ticket Priority Distribution")
plt.xlabel("Weight")
plt.ylabel("Frequency")
plt.show()


# In[ ]:


ef = data["EF"]

plt.figure(figsize=(6,4))
plt.bar(ef.keys(), ef.values())
plt.title("Emission Factors by Transport Mode")
plt.ylabel("Emission Factor")
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
from collections import Counter


def plot_instance_map(data, figsize=(10, 8), show_labels=True):
    """
    Plot bases and plants using synthetic coordinates stored in data["_meta"]["coords"].
    """
    coords = data["_meta"]["coords"]
    B = data["B"]
    P = data["P"]

    bx = [coords[b][0] for b in B]
    by = [coords[b][1] for b in B]

    px = [coords[p][0] for p in P]
    py = [coords[p][1] for p in P]

    plt.figure(figsize=figsize)
    plt.scatter(px, py, s=70, marker="o", label="Plants")
    plt.scatter(bx, by, s=180, marker="^", label="Bases")

    if show_labels:
        for b in B:
            x, y = coords[b]
            plt.text(x + 0.8, y + 0.8, b, fontsize=9)
        for p in P:
            x, y = coords[p]
            plt.text(x + 0.5, y + 0.5, p, fontsize=8)

    plt.title("Synthetic Service Network Map")
    plt.xlabel("X (km)")
    plt.ylabel("Y (km)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def plot_instance_map_with_ticket_load(data, figsize=(10, 8), show_labels=True):
    """
    Plot plants with marker size proportional to number of assigned tickets.
    """
    coords = data["_meta"]["coords"]
    B = data["B"]
    P = data["P"]
    node = data["node"]  # ticket -> plant

    plant_load = Counter(node[t] for t in data["T"])

    bx = [coords[b][0] for b in B]
    by = [coords[b][1] for b in B]

    px = [coords[p][0] for p in P]
    py = [coords[p][1] for p in P]
    ps = [80 + 60 * plant_load.get(p, 0) for p in P]

    plt.figure(figsize=figsize)
    plt.scatter(px, py, s=ps, marker="o", label="Plants (size = ticket load)")
    plt.scatter(bx, by, s=180, marker="^", label="Bases")

    if show_labels:
        for b in B:
            x, y = coords[b]
            plt.text(x + 0.8, y + 0.8, b, fontsize=9)
        for p in P:
            x, y = coords[p]
            plt.text(x + 0.5, y + 0.5, f"{p} ({plant_load.get(p, 0)})", fontsize=8)

    plt.title("Synthetic Map with Plant Ticket Load")
    plt.xlabel("X (km)")
    plt.ylabel("Y (km)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


# In[ ]:


plot_instance_map(data)
plot_instance_map_with_ticket_load(data)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




