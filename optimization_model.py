# model_ticket_routing.py
# Pyomo MILP model: technician scheduling with travel, hotel, SLA
# + ECONOMIC PENALTY for idle days away from base (no explicit "return-if-idle" constraint)
#
# Key change vs your version:
# - Remove C18_return_if_idle_next
# - Add idle_away[k,d] and penalize it in the objective.
#   This makes “wait idle at a plant” expensive, so the model will *naturally*
#   prefer returning to base (or otherwise staying productive) without forcing it.
#
# Usage:
#   from model_ticket_routing import build_model
#   m = build_model(data)
#   # solve with gurobi/cbc/glpk etc.
#
# Data dictionary keys (required/suggested):
#   Sets:    B, K, P, T, D, M, N
#   Params:  a, b, w, dur, node, base, dist, tt, EF, Hkp, c_hotel,
#            X, alpha, beta, eta, gamma,
#            (NEW) c_idle_away, delta

from __future__ import annotations

from typing import Dict, Any
import pyomo.environ as pyo


def build_model(data: Dict[str, Any]) -> pyo.ConcreteModel:
    m = pyo.ConcreteModel(name="TechScheduling_Routing_EconIdlePenalty")

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

    # NEW: cost and weight for idle-away days
    # Interpret c_idle_away as "daily wage / opportunity cost" for being away and idle.
    # If you prefer using only a weight, set c_idle_away=1 and tune delta.
    m.c_idle_away = pyo.Param(within=pyo.NonNegativeReals, initialize=float(data.get("c_idle_away", 1.0)))
    m.delta = pyo.Param(within=pyo.NonNegativeReals, initialize=float(data.get("delta", 0.0)))

    # Derived param: co2[i,j,m] = EF[m] * dist[i,j]
    def _co2_init(mm, i, j, md):
        return pyo.value(mm.EF[md]) * pyo.value(mm.dist[i, j])

    m.co2 = pyo.Param(m.N, m.N, m.M, within=pyo.NonNegativeReals, initialize=_co2_init)

    # -------------------------
    # Decision Variables
    # -------------------------
    m.start = pyo.Var(m.T, m.K, m.D, domain=pyo.Binary)              # start[t,k,d]
    m.z = pyo.Var(m.T, m.K, m.D, domain=pyo.Binary)                  # work day indicator
    m.loc = pyo.Var(m.N, m.K, m.D, domain=pyo.Binary)                # location
    m.y = pyo.Var(m.N, m.N, m.K, m.D, m.M, domain=pyo.Binary)        # travel arcs
    m.away = pyo.Var(m.K, m.D, domain=pyo.Binary)                    # away from base
    m.hotel = pyo.Var(m.K, m.D, domain=pyo.Binary)                   # hotel
    m.C = pyo.Var(m.T, domain=pyo.NonNegativeReals)                  # completion day
    m.L = pyo.Var(m.T, domain=pyo.NonNegativeReals)                  # lateness
    m.busy = pyo.Var(m.K, m.D, domain=pyo.Binary)                    # any work or travel on day

    # NEW: idle-away indicator (away AND not busy)
    m.idle_away = pyo.Var(m.K, m.D, domain=pyo.Binary)

    # -------------------------
    # Objective
    # -------------------------
    def obj_rule(mm):
        # "work" term here is a proxy cost for consuming technician days;
        # minimizing it tends to pick assignments with shorter durations.
        work = mm.alpha * sum(mm.z[t, k, d] for t in mm.T for k in mm.K for d in mm.D)

        travel = mm.beta * sum(
            mm.co2[i, j, md] * mm.y[i, j, k, d, md]
            for k in mm.K for d in mm.D for i in mm.N for j in mm.N for md in mm.M
        )

        hotel = mm.eta * sum(mm.c_hotel * mm.hotel[k, d] for k in mm.K for d in mm.D)

        sla = mm.gamma * sum(mm.w[t] * mm.L[t] for t in mm.T)

        # NEW: penalize idle days while away from home base
        idle_away_cost = mm.delta * sum(mm.c_idle_away * mm.idle_away[k, d] for k in mm.K for d in mm.D)

        return work + travel + hotel + sla + idle_away_cost

    m.OBJ = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # -------------------------
    # Constraints
    # -------------------------

    # (1) Each ticket started exactly once (k, d>=a_t)
    def c1_rule(mm, t):
        return sum(mm.start[t, k, d] for k in mm.K for d in mm.D if d >= mm.a[t]) == 1

    m.C1_start_once = pyo.Constraint(m.T, rule=c1_rule)

    # (2) Start -> consecutive work days
    def c2_rule(mm, t, k, tau):
        return mm.z[t, k, tau] == sum(
            mm.start[t, k, d]
            for d in mm.D
            if (d <= tau) and (tau <= d + mm.dur[t, k] - 1) and (d + mm.dur[t, k] - 1 <= mm.H)
        )

    m.C2_consecutive_work = pyo.Constraint(m.T, m.K, m.D, rule=c2_rule)

    # (3) Forbid start that would exceed horizon
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

    # (9) Atomic day: either work OR travel (departure), not both
    def c9_rule(mm, k, d):
        return (
            sum(mm.y[i, j, k, d, md] for i in mm.N for j in mm.N for md in mm.M)
            + sum(mm.z[t, k, d] for t in mm.T)
            <= 1
        )

    m.C9_atomic_day = pyo.Constraint(m.K, m.D, rule=c9_rule)

    # (10) Arrival after travel time
    def c10_rule(mm, i, j, k, d, md):
        arr = d + mm.tt[i, j, md]
        if arr <= mm.H:
            return mm.loc[j, k, arr] >= mm.y[i, j, k, d, md]
        return pyo.Constraint.Skip

    m.C10_arrival = pyo.Constraint(m.N, m.N, m.K, m.D, m.M, rule=c10_rule)

    # (11) Persistence lower bound (stay unless depart)
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

    # (14) Hotel activation
    def c14_rule(mm, k, d):
        return mm.hotel[k, d] >= sum(mm.Hkp[k, p] * mm.loc[p, k, d] for p in mm.P)

    m.C14_hotel = pyo.Constraint(m.K, m.D, rule=c14_rule)

    # (15) Completion day
    def c15_rule(mm, t):
        return mm.C[t] == sum(
            (d + mm.dur[t, k] - 1) * mm.start[t, k, d]
            for k in mm.K for d in mm.D
            if d + mm.dur[t, k] - 1 <= mm.H
        )

    m.C15_completion = pyo.Constraint(m.T, rule=c15_rule)

    # (16) Lateness
    def c16_rule(mm, t):
        return mm.L[t] >= mm.C[t] - mm.b[t]

    m.C16_lateness = pyo.Constraint(m.T, rule=c16_rule)

    # (17) L[t] >= 0 (explicit)
    def c17_rule(mm, t):
        return mm.L[t] >= 0

    m.C17_lateness_nonneg = pyo.Constraint(m.T, rule=c17_rule)

    # -------------------------
    # Busy definition (kept)
    # -------------------------
    def busy_lb_work(mm, k, d):
        return mm.busy[k, d] >= sum(mm.z[t, k, d] for t in mm.T)

    m.BUSY_LB_WORK = pyo.Constraint(m.K, m.D, rule=busy_lb_work)

    def busy_lb_travel(mm, k, d):
        return mm.busy[k, d] >= sum(mm.y[i, j, k, d, md] for i in mm.N for j in mm.N for md in mm.M)

    m.BUSY_LB_TRAVEL = pyo.Constraint(m.K, m.D, rule=busy_lb_travel)

    def busy_ub(mm, k, d):
        return mm.busy[k, d] <= (
            sum(mm.z[t, k, d] for t in mm.T)
            + sum(mm.y[i, j, k, d, md] for i in mm.N for j in mm.N for md in mm.M)
        )

    m.BUSY_UB = pyo.Constraint(m.K, m.D, rule=busy_ub)

    # -------------------------
    # NEW: idle-away linking constraints
    # idle_away = 1 iff (away == 1 and busy == 0)
    # -------------------------
    def idle_away_lb(mm, k, d):
        # if away=1 and busy=0 => idle_away must be 1
        return mm.idle_away[k, d] >= mm.away[k, d] - mm.busy[k, d]

    m.IDLE_AWAY_LB = pyo.Constraint(m.K, m.D, rule=idle_away_lb)

    def idle_away_ub1(mm, k, d):
        return mm.idle_away[k, d] <= mm.away[k, d]

    m.IDLE_AWAY_UB1 = pyo.Constraint(m.K, m.D, rule=idle_away_ub1)

    def idle_away_ub2(mm, k, d):
        return mm.idle_away[k, d] <= 1 - mm.busy[k, d]

    m.IDLE_AWAY_UB2 = pyo.Constraint(m.K, m.D, rule=idle_away_ub2)

    # -------------------------
    # Fixes: prevent teleportation + forbid moving to other bases
    # -------------------------

    # (A) Forbid self-loop travel
    def cA_no_self_loop(mm, i, k, d, md):
        return mm.y[i, i, k, d, md] == 0

    m.CA_no_self_loop = pyo.Constraint(m.N, m.K, m.D, m.M, rule=cA_no_self_loop)

    # (B) Forbid being located at bases other than technician's own base
    def cB_no_other_bases(mm, b, k, d):
        if b != mm.base[k]:
            return mm.loc[b, k, d] == 0
        return pyo.Constraint.Skip

    m.CB_no_other_bases = pyo.Constraint(m.B, m.K, m.D, rule=cB_no_other_bases)

    # (C) Forbid traveling to/from other bases
    def cC_no_travel_other_bases(mm, i, j, k, d, md):
        if (i in mm.B) and (i != mm.base[k]):
            return mm.y[i, j, k, d, md] == 0
        if (j in mm.B) and (j != mm.base[k]):
            return mm.y[i, j, k, d, md] == 0
        return pyo.Constraint.Skip

    m.CC_no_travel_other_bases = pyo.Constraint(m.N, m.N, m.K, m.D, m.M, rule=cC_no_travel_other_bases)

    # Helper expressions: departures and arrivals-on-day
    def dep_from(mm, j, k, d):
        return sum(mm.y[j, h, k, d, md] for h in mm.N for md in mm.M)

    def arr_to_on_day(mm, j, k, d):
        return sum(
            mm.y[i, j, k, d0, md]
            for i in mm.N
            for md in mm.M
            for d0 in mm.D
            if d0 + mm.tt[i, j, md] == d
        )

    # (D) No-teleport upper bound (flow UB)
    def cD_flow_ub(mm, j, k, d):
        if d == 1:
            return pyo.Constraint.Skip
        return mm.loc[j, k, d] <= mm.loc[j, k, d - 1] - dep_from(mm, j, k, d - 1) + arr_to_on_day(mm, j, k, d)

    m.CD_location_flow_ub = pyo.Constraint(m.N, m.K, m.D, rule=cD_flow_ub)

    return m