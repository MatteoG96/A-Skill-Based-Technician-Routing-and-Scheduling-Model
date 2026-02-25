# model_ticket_routing.py
# Pyomo MILP model: technician scheduling with travel, hotel, SLA + "return to base if not redirected"
#
# Usage from a notebook (mani.ipynb):
#   from model_ticket_routing import build_model
#   data = {...}  # see template at bottom
#   m = build_model(data)
#   # solve with your solver of choice (gurobi/cbc/glpk)

from __future__ import annotations

from typing import Dict, Any, Iterable, Tuple
import pyomo.environ as pyo


def build_model(data: Dict[str, Any]) -> pyo.ConcreteModel:
    """
    Build Pyomo model from a data dictionary.

    Required keys in `data` (suggested):
      Sets:
        B, K, P, T, D, M, N
      Params:
        a, b, w, dur, node, base, dist, tt, EF, Hkp, c_hotel, X, alpha, beta, eta, gamma

    Indexing conventions:
      - a[t], b[t], w[t]            with t in T
      - dur[t,k]                    with (t,k) in T x K
      - node[t]                     with t in T (node in N)
      - base[k]                     with k in K (base in B)
      - dist[i,j]                   with (i,j) in N x N
      - tt[i,j,m]                   with (i,j,m) in N x N x M
      - EF[m]                       with m in M
      - Hkp[k,p]                    with (k,p) in K x P
      - c_hotel (scalar), X (int), alpha/beta/eta/gamma (scalar)
    """

    m = pyo.ConcreteModel(name="TechScheduling_Routing_ReturnToBase")

    # -------------------------
    # Sets
    # -------------------------
    m.B = pyo.Set(initialize=data["B"], ordered=False)
    m.K = pyo.Set(initialize=data["K"], ordered=False)
    m.P = pyo.Set(initialize=data["P"], ordered=False)
    m.T = pyo.Set(initialize=data["T"], ordered=False)
    m.D = pyo.Set(initialize=data["D"], ordered=True)  # 1..H recommended
    m.M = pyo.Set(initialize=data["M"], ordered=False)
    m.N = pyo.Set(initialize=data["N"], ordered=False)

    # Helper: H (last day)
    m.H = pyo.Param(initialize=max(list(m.D)), within=pyo.PositiveIntegers)

    # -------------------------
    # Params
    # -------------------------
    # Ticket timing / weights
    m.a = pyo.Param(m.T, within=pyo.PositiveIntegers, initialize=data["a"])
    m.b = pyo.Param(m.T, within=pyo.PositiveIntegers, initialize=data["b"])
    m.w = pyo.Param(m.T, within=pyo.NonNegativeReals, initialize=data["w"])

    # Exact durations per (t,k)
    m.dur = pyo.Param(m.T, m.K, within=pyo.PositiveIntegers, initialize=data["dur"])

    # Ticket node, technician base
    m.node = pyo.Param(m.T, within=m.N, initialize=data["node"])
    m.base = pyo.Param(m.K, within=m.B, initialize=data["base"])

    # Travel / CO2
    m.dist = pyo.Param(m.N, m.N, within=pyo.NonNegativeReals, initialize=data["dist"])
    m.tt = pyo.Param(m.N, m.N, m.M, within=pyo.NonNegativeIntegers, initialize=data["tt"])
    m.EF = pyo.Param(m.M, within=pyo.NonNegativeReals, initialize=data["EF"])

    # Hotel
    m.Hkp = pyo.Param(m.K, m.P, within=pyo.Binary, initialize=data["Hkp"])
    m.c_hotel = pyo.Param(within=pyo.NonNegativeReals, initialize=float(data["c_hotel"]))

    # Policy
    m.X = pyo.Param(within=pyo.PositiveIntegers, initialize=int(data["X"]))

    # Objective weights
    m.alpha = pyo.Param(within=pyo.NonNegativeReals, initialize=float(data["alpha"]))
    m.beta = pyo.Param(within=pyo.NonNegativeReals, initialize=float(data["beta"]))
    m.eta = pyo.Param(within=pyo.NonNegativeReals, initialize=float(data["eta"]))
    m.gamma = pyo.Param(within=pyo.NonNegativeReals, initialize=float(data["gamma"]))

    # Derived param: co2[i,j,m] = EF[m] * dist[i,j]
    def _co2_init(mm, i, j, md):
        return pyo.value(mm.EF[md]) * pyo.value(mm.dist[i, j])

    m.co2 = pyo.Param(m.N, m.N, m.M, within=pyo.NonNegativeReals, initialize=_co2_init)

    # -------------------------
    # Variables
    # -------------------------
    m.start = pyo.Var(m.T, m.K, m.D, domain=pyo.Binary)
    m.z = pyo.Var(m.T, m.K, m.D, domain=pyo.Binary)

    m.loc = pyo.Var(m.N, m.K, m.D, domain=pyo.Binary)
    m.y = pyo.Var(m.N, m.N, m.K, m.D, m.M, domain=pyo.Binary)

    m.away = pyo.Var(m.K, m.D, domain=pyo.Binary)
    m.hotel = pyo.Var(m.K, m.D, domain=pyo.Binary)

    m.C = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.L = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.busy = pyo.Var(m.K, m.D, domain=pyo.Binary)

    # -------------------------
    # Objective
    # -------------------------
    def obj_rule(mm):
        work = mm.alpha * sum(mm.z[t, k, d] for t in mm.T for k in mm.K for d in mm.D)
        travel = mm.beta * sum(
            mm.co2[i, j, md] * mm.y[i, j, k, d, md]
            for k in mm.K for d in mm.D for i in mm.N for j in mm.N for md in mm.M
        )
        hotel = mm.eta * sum(mm.c_hotel * mm.hotel[k, d] for k in mm.K for d in mm.D)
        sla = mm.gamma * sum(mm.w[t] * mm.L[t] for t in mm.T)
        return work + travel + hotel + sla

    m.OBJ = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # -------------------------
    # Constraints
    # -------------------------

    # (1) Each ticket started exactly once (k, d>=a_t)
    def c1_rule(mm, t):
        return sum(mm.start[t, k, d] for k in mm.K for d in mm.D if d >= mm.a[t]) == 1

    m.C1_start_once = pyo.Constraint(m.T, rule=c1_rule)

    # (2) Start -> consecutive work: z[t,k,tau] = sum_{d covers tau} start[t,k,d]
    def c2_rule(mm, t, k, tau):
        return mm.z[t, k, tau] == sum(
            mm.start[t, k, d]
            for d in mm.D
            if (d <= tau) and (tau <= d + mm.dur[t, k] - 1) and (d + mm.dur[t, k] - 1 <= mm.H)
        )

    m.C2_consecutive_work = pyo.Constraint(m.T, m.K, m.D, rule=c2_rule)

    # (3) Forbid start that would exceed horizon: start[t,k,d]=0 if d+dur-1>H
    def c3_rule(mm, t, k, d):
        if d + mm.dur[t, k] - 1 > mm.H:
            return mm.start[t, k, d] == 0
        return pyo.Constraint.Skip

    m.C3_forbid_overflow_start = pyo.Constraint(m.T, m.K, m.D, rule=c3_rule)

    # (4) At most one ticket per technician per day
    def c4_rule(mm, k, d):
        return sum(mm.z[t, k, d] for t in mm.T) <= 1

    m.C4_one_ticket_per_day = pyo.Constraint(m.K, m.D, rule=c4_rule)

    # (5) Work requires being at ticket node
    def c5_rule(mm, t, k, d):
        return mm.z[t, k, d] <= mm.loc[mm.node[t], k, d]

    m.C5_work_location = pyo.Constraint(m.T, m.K, m.D, rule=c5_rule)

    # (6) Unique location per technician per day
    def c6_rule(mm, k, d):
        return sum(mm.loc[i, k, d] for i in mm.N) == 1

    m.C6_unique_location = pyo.Constraint(m.K, m.D, rule=c6_rule)

    # (7) Initial condition: start at base on day 1
    def c7_rule(mm, k):
        return mm.loc[mm.base[k], k, 1] == 1

    m.C7_initial_at_base = pyo.Constraint(m.K, rule=c7_rule)

    # (8) Can depart from i only if located at i
    def c8_rule(mm, i, k, d):
        return sum(mm.y[i, j, k, d, md] for j in mm.N for md in mm.M) <= mm.loc[i, k, d]

    m.C8_depart_if_there = pyo.Constraint(m.N, m.K, m.D, rule=c8_rule)

    # (9) Atomic day: either work or travel (departure), not both
    def c9_rule(mm, k, d):
        return (
            sum(mm.y[i, j, k, d, md] for i in mm.N for j in mm.N for md in mm.M)
            + sum(mm.z[t, k, d] for t in mm.T)
            <= 1
        )

    m.C9_atomic_day = pyo.Constraint(m.K, m.D, rule=c9_rule)

    # (10) Arrival after travel time: loc[j,k,d+tt] >= y[i,j,k,d,m]
    def c10_rule(mm, i, j, k, d, md):
        arr = d + mm.tt[i, j, md]
        if arr <= mm.H:
            return mm.loc[j, k, arr] >= mm.y[i, j, k, d, md]
        return pyo.Constraint.Skip

    m.C10_arrival = pyo.Constraint(m.N, m.N, m.K, m.D, m.M, rule=c10_rule)

    # (11) Persistence if no departure from i on day d: loc[i,k,d+1] >= loc[i,k,d] - sum_{j,m} y[i,j,k,d,m]
    def c11_rule(mm, i, k, d):
        if d < mm.H:
            return mm.loc[i, k, d + 1] >= mm.loc[i, k, d] - sum(
                mm.y[i, j, k, d, md] for j in mm.N for md in mm.M
            )
        return pyo.Constraint.Skip

    m.C11_persistence = pyo.Constraint(m.N, m.K, m.D, rule=c11_rule)

    # (12) away definition
    def c12_rule(mm, k, d):
        return mm.away[k, d] == 1 - mm.loc[mm.base[k], k, d]

    m.C12_away_def = pyo.Constraint(m.K, m.D, rule=c12_rule)

    # (13) Max X consecutive days away (window constraint)
    def c13_rule(mm, k, d):
        if d <= mm.H - mm.X:
            return sum(mm.away[k, tau] for tau in mm.D if d <= tau <= d + mm.X) <= mm.X
        return pyo.Constraint.Skip

    m.C13_max_consecutive_away = pyo.Constraint(m.K, m.D, rule=c13_rule)

    # (14) Hotel activation: hotel[k,d] >= sum_{p} Hkp[k,p]*loc[p,k,d]
    def c14_rule(mm, k, d):
        return mm.hotel[k, d] >= sum(mm.Hkp[k, p] * mm.loc[p, k, d] for p in mm.P)

    m.C14_hotel = pyo.Constraint(m.K, m.D, rule=c14_rule)

    # (15) Completion day: C[t] = sum_{k,d} (d + dur[t,k] - 1) * start[t,k,d]
    def c15_rule(mm, t):
        return mm.C[t] == sum(
            (d + mm.dur[t, k] - 1) * mm.start[t, k, d]
            for k in mm.K for d in mm.D
            if d + mm.dur[t, k] - 1 <= mm.H
        )

    m.C15_completion = pyo.Constraint(m.T, rule=c15_rule)

    # (16) Lateness: L[t] >= C[t] - b[t]
    def c16_rule(mm, t):
        return mm.L[t] >= mm.C[t] - mm.b[t]

    m.C16_lateness = pyo.Constraint(m.T, rule=c16_rule)

    # (17) L[t] >= 0 already by domain; keep if you want explicit:
    def c17_rule(mm, t):
        return mm.L[t] >= 0

    m.C17_lateness_nonneg = pyo.Constraint(m.T, rule=c17_rule)

    # (18) Return-to-base departure if not redirected (coherent with atomic day & tt)
    # If at plant p on day d, and day d+1 would be idle, then on day d+1 must depart to base
    # (or already be at base at start of day d+1).
    def busy_lb_rule(mm, k, d):
        return mm.busy[k, d] >= sum(mm.z[t, k, d] for t in mm.T)
    m.BUSY_LB1 = pyo.Constraint(m.K, m.D, rule=busy_lb_rule)

    def busy_lb2_rule(mm, k, d):
        return mm.busy[k, d] >= sum(mm.y[i, j, k, d, md] for i in mm.N for j in mm.N for md in mm.M)
    m.BUSY_LB2 = pyo.Constraint(m.K, m.D, rule=busy_lb2_rule)

    def busy_ub_rule(mm, k, d):
        return mm.busy[k, d] <= (
            sum(mm.z[t, k, d] for t in mm.T)
            + sum(mm.y[i, j, k, d, md] for i in mm.N for j in mm.N for md in mm.M)
        )
    m.BUSY_UB = pyo.Constraint(m.K, m.D, rule=busy_ub_rule)
    
    def c18_rule_fixed(mm, k, d, p):
        if d < mm.H:
            base = mm.base[k]
            depart_to_base = sum(mm.y[p, base, k, d + 1, md] for md in mm.M)
            return mm.loc[p, k, d] <= mm.busy[k, d + 1] + mm.loc[base, k, d + 1] + depart_to_base
        return pyo.Constraint.Skip

    m.C18_return_if_idle_next = pyo.Constraint(m.K, m.D, m.P, rule=c18_rule_fixed)

    return m