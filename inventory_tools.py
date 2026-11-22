"""
Inventory Tools Module for Supply Chain Analytics.

Implements pure deterministic numerical calculations for:
1. calculate_eoq()
2. calculate_quantity_discount()
3. calculate_rop()
4. calculate_periodic_review()
5. calculate_newsvendor()

EVERY calculation tool strictly returns:
1. Inputs Extracted ('inputs')
2. Variable Definitions ('variable_definitions')
3. Formula ('formula')
4. Formula with Substituted Values ('formula_with_substituted_values')
5. Intermediate Calculations ('intermediate_calculations')
6. Final Answer ('final_answer')
7. Verification Checks ('verification_checks')
8. Assumptions ('assumptions')

All numerical calculations come directly from deterministic Python logic.
"""

import math
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import norm


# =============================================================================
# 1. ECONOMIC ORDER QUANTITY (EOQ)
# =============================================================================
def calculate_eoq(
    demand: float,
    order_cost: float,
    holding_cost: float,
    unit_cost: Optional[float] = None,
    work_days_per_year: int = 250,
    lead_time_days: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calculate Economic Order Quantity (EOQ): Q* = sqrt(2DS / H).

    Validates inputs strictly without inventing missing numerical inputs.
    Returns all 8 pedagogical components for student verification.
    """
    # Strict Input Validation
    if demand is None:
        raise ValueError("Missing required input: 'demand' (D) must be provided.")
    if order_cost is None:
        raise ValueError("Missing required input: 'order_cost' (S) must be provided.")
    if holding_cost is None:
        raise ValueError("Missing required input: 'holding_cost' (H) must be provided.")

    if not isinstance(demand, (int, float)) or demand <= 0:
        raise ValueError(f"Annual demand must be a strictly positive number, got {demand}.")
    if not isinstance(order_cost, (int, float)) or order_cost <= 0:
        raise ValueError(f"Order cost must be a strictly positive number, got {order_cost}.")
    if not isinstance(holding_cost, (int, float)) or holding_cost <= 0:
        raise ValueError(f"Holding cost must be a strictly positive number, got {holding_cost}.")

    if unit_cost is not None and (not isinstance(unit_cost, (int, float)) or unit_cost < 0):
        raise ValueError(f"Unit cost cannot be negative, got {unit_cost}.")
    if work_days_per_year <= 0:
        raise ValueError(f"Work days per year must be strictly positive, got {work_days_per_year}.")
    if lead_time_days is not None and (not isinstance(lead_time_days, (int, float)) or lead_time_days < 0):
        raise ValueError(f"Lead time days cannot be negative, got {lead_time_days}.")

    # Deterministic Core Numerical Calculation
    numerator_2ds = 2.0 * demand * order_cost
    eoq_squared = numerator_2ds / holding_cost
    eoq = math.sqrt(eoq_squared)
    optimal_q_int = int(round(eoq))

    # Intermediate Calculations
    orders_per_year = demand / eoq
    cycle_time_years = eoq / demand
    cycle_time_days = cycle_time_years * work_days_per_year

    annual_ordering_cost = orders_per_year * order_cost
    annual_holding_cost = (eoq / 2.0) * holding_cost
    total_inventory_cost = annual_ordering_cost + annual_holding_cost

    purchase_cost = (demand * unit_cost) if unit_cost is not None else None
    total_annual_cost = (total_inventory_cost + purchase_cost) if purchase_cost is not None else total_inventory_cost

    reorder_point = None
    if lead_time_days is not None:
        daily_demand = demand / work_days_per_year
        reorder_point = daily_demand * lead_time_days

    # 1. Inputs Extracted
    inputs_extracted = {
        "demand (D)": float(demand),
        "order_cost (S)": float(order_cost),
        "holding_cost (H)": float(holding_cost),
        "unit_cost (C)": float(unit_cost) if unit_cost is not None else None,
        "work_days_per_year": int(work_days_per_year),
        "lead_time_days (L)": float(lead_time_days) if lead_time_days is not None else None,
    }

    # 2. Variable Definitions
    variable_definitions = {
        "D": "Annual demand rate in units per year",
        "S": "Fixed ordering / setup cost per replenishment order in dollars ($/order)",
        "H": "Inventory holding / carrying cost per unit per year in dollars ($/unit/year)",
        "Q*": "Optimal Economic Order Quantity batch size in units",
        "N": "Order frequency (number of replenishment orders placed per year): N = D / Q*",
        "T": "Cycle time interval between successive orders in working days: T = (Q* / D) * work_days",
        "Annual Ordering Cost": "Cost incurred placing orders: (D / Q*) * S",
        "Annual Holding Cost": "Cost incurred storing cycle stock: (Q* / 2) * H",
        "Total Inventory Cost (TC)": "Sum of annual ordering and holding costs: Annual Ordering Cost + Annual Holding Cost",
        "Reorder Point (ROP)": "Inventory level that triggers replenishment: (D / work_days) * L",
    }

    # 3. Formula
    formula = "Q* = sqrt((2 * D * S) / H)"

    # 4. Formula with Substituted Values
    formula_with_substituted_values = (
        f"Q* = sqrt((2 * {demand:,.2f} * {order_cost:,.2f}) / {holding_cost:,.2f})\n"
        f"   = sqrt({numerator_2ds:,.2f} / {holding_cost:,.2f})\n"
        f"   = sqrt({eoq_squared:,.4f})\n"
        f"   = {eoq:,.2f} units (approx {optimal_q_int:,} units)"
    )

    # 5. Intermediate Calculations
    intermediate_calculations = {
        "numerator_2DS": round(numerator_2ds, 2),
        "eoq_squared_variance_ratio": round(eoq_squared, 4),
        "raw_eoq": round(eoq, 4),
        "orders_per_year": round(orders_per_year, 4),
        "cycle_time_years": round(cycle_time_years, 4),
        "cycle_time_days": round(cycle_time_days, 2),
        "annual_ordering_cost": round(annual_ordering_cost, 2),
        "annual_holding_cost": round(annual_holding_cost, 2),
        "cost_difference_holding_vs_ordering": round(abs(annual_holding_cost - annual_ordering_cost), 4),
        "total_inventory_cost": round(total_inventory_cost, 2),
        "annual_purchase_cost": round(purchase_cost, 2) if purchase_cost is not None else None,
        "total_annual_cost": round(total_annual_cost, 2),
        "reorder_point": round(reorder_point, 2) if reorder_point is not None else None,
    }

    # 6. Final Answer
    final_answer = (
        f"The optimal Economic Order Quantity is {round(eoq, 2):,} units ({optimal_q_int:,} rounded). "
        f"Place {round(orders_per_year, 2)} orders per year (every {round(cycle_time_days, 1)} operating days) "
        f"to achieve the minimum total annual inventory cost of ${round(total_inventory_cost, 2):,.2f}."
    )

    # 7. Verification Checks
    cost_diff = abs(annual_holding_cost - annual_ordering_cost)
    cost_balanced = cost_diff < 0.50 or (cost_diff / (total_inventory_cost + 1e-6) < 0.01)
    second_derivative = (2.0 * demand * order_cost) / (eoq**3)

    verification_checks = [
        {
            "check": "Positivity of Order Quantity",
            "condition": "Q* > 0",
            "calculated_value": round(eoq, 2),
            "status": "PASSED" if eoq > 0 else "FAILED",
            "student_note": "Confirms physically feasible, non-negative procurement quantity.",
        },
        {
            "check": "First-Order Trade-Off Optimality",
            "condition": "Annual Holding Cost ≈ Annual Ordering Cost",
            "calculated_value": f"Holding: ${annual_holding_cost:,.2f} vs Ordering: ${annual_ordering_cost:,.2f} (diff = ${cost_diff:.4f})",
            "status": "PASSED" if cost_balanced else "FAILED",
            "student_note": "At EOQ, marginal holding cost equals marginal ordering cost (dTC/dQ = 0).",
        },
        {
            "check": "Second-Order Convexity Check",
            "condition": "d^2(TC)/dQ^2 = (2 * D * S) / Q^3 > 0",
            "calculated_value": round(second_derivative, 6),
            "status": "PASSED" if second_derivative > 0 else "FAILED",
            "student_note": "Positive second derivative mathematically confirms total cost curve is strictly convex, guaranteeing a global minimum.",
        },
    ]

    # 8. Assumptions
    assumptions = [
        "Annual demand (D) is deterministic, constant, and continuous over an infinite horizon.",
        "Replenishment is instantaneous (entire batch arrives in a single delivery).",
        "Lead time (L) is constant and deterministic.",
        "Ordering cost (S) is fixed and independent of order batch size.",
        "Holding cost (H) is constant per unit per year and linear with average inventory (Q/2).",
        "Stockouts, shortages, and backorders are not permitted.",
        "Unit purchasing price (C) is constant without volume price breaks.",
    ]

    return {
        "model": "Economic Order Quantity (EOQ)",
        "inputs": inputs_extracted,
        "variable_definitions": variable_definitions,
        "formula": formula,
        "formula_with_substituted_values": formula_with_substituted_values,
        "intermediate_calculations": intermediate_calculations,
        "final_answer": final_answer,
        "verification_checks": verification_checks,
        "assumptions": assumptions,
        # Structured results for programmatic inspection
        "results": {
            "eoq": round(eoq, 2),
            "optimal_order_quantity_int": optimal_q_int,
            "orders_per_year": round(orders_per_year, 2),
            "cycle_time_days": round(cycle_time_days, 1),
            "annual_ordering_cost": round(annual_ordering_cost, 2),
            "annual_holding_cost": round(annual_holding_cost, 2),
            "total_inventory_cost": round(total_inventory_cost, 2),
            "annual_purchase_cost": round(purchase_cost, 2) if purchase_cost is not None else None,
            "total_annual_cost": round(total_annual_cost, 2),
            "reorder_point": round(reorder_point, 2) if reorder_point is not None else None,
        },
    }


# =============================================================================
# 2. QUANTITY DISCOUNT MODEL
# =============================================================================
def calculate_quantity_discount(
    demand: float,
    order_cost: float,
    discount_tiers: List[Dict[str, Any]],
    holding_cost_rate: Optional[float] = None,
    fixed_holding_cost: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calculate optimal order quantity under all-units quantity discount.
    Evaluates feasible quantities and compares total annual cost.
    Supports fixed holding cost ($/unit/year) and percentage holding cost rate (i * Price).
    Returns all 8 pedagogical components.
    """
    if demand is None or order_cost is None:
        raise ValueError("Missing required inputs: 'demand' and 'order_cost' must be provided.")
    if not isinstance(demand, (int, float)) or demand <= 0:
        raise ValueError("Annual demand must be strictly positive.")
    if not isinstance(order_cost, (int, float)) or order_cost <= 0:
        raise ValueError("Order cost must be strictly positive.")

    if not discount_tiers or not isinstance(discount_tiers, list):
        raise ValueError("Must provide at least one discount tier in 'discount_tiers'.")

    if holding_cost_rate is None and fixed_holding_cost is None:
        raise ValueError("Must provide either 'holding_cost_rate' (percentage) or 'fixed_holding_cost' ($/unit).")
    if holding_cost_rate is not None and holding_cost_rate <= 0:
        raise ValueError("Holding cost rate must be strictly positive.")
    if fixed_holding_cost is not None and fixed_holding_cost <= 0:
        raise ValueError("Fixed holding cost must be strictly positive.")

    evaluated_tiers = []
    min_total_cost = float("inf")
    best_tier = None
    step_descriptions = []

    for idx, tier in enumerate(discount_tiers):
        min_qty = float(tier.get("min_qty", 0.0))
        max_qty = float(tier["max_qty"]) if tier.get("max_qty") is not None else None
        price = float(tier["unit_price"])

        if price <= 0:
            raise ValueError(f"Tier {idx+1} unit price must be positive, got {price}.")
        if min_qty < 0:
            raise ValueError(f"Tier {idx+1} min_qty cannot be negative.")
        if max_qty is not None and max_qty < min_qty:
            raise ValueError(f"Tier {idx+1} max_qty ({max_qty}) cannot be less than min_qty ({min_qty}).")

        # Holding cost for tier
        if holding_cost_rate is not None:
            tier_h = holding_cost_rate * price
        else:
            tier_h = fixed_holding_cost

        # Unconstrained EOQ
        raw_eoq = math.sqrt((2.0 * demand * order_cost) / tier_h)

        # Boundary adjustments
        if raw_eoq < min_qty:
            candidate_q = min_qty
            status = "Adjusted upward to minimum threshold to qualify for tier discount"
        elif max_qty is not None and raw_eoq > max_qty:
            candidate_q = max_qty
            status = "Adjusted downward to tier ceiling"
        else:
            candidate_q = raw_eoq
            status = "Unconstrained EOQ is feasible within tier bracket"

        annual_purchase = demand * price
        annual_ordering = (demand / candidate_q) * order_cost
        annual_holding = (candidate_q / 2.0) * tier_h
        total_cost = annual_purchase + annual_ordering + annual_holding

        tier_summary = {
            "tier_index": idx + 1,
            "min_qty": min_qty,
            "max_qty": max_qty,
            "unit_price": price,
            "holding_cost_unit": round(tier_h, 4),
            "raw_eoq": round(raw_eoq, 2),
            "candidate_q": round(candidate_q, 2),
            "status": status,
            "annual_purchase_cost": round(annual_purchase, 2),
            "annual_ordering_cost": round(annual_ordering, 2),
            "annual_holding_cost": round(annual_holding, 2),
            "total_annual_cost": round(total_cost, 2),
        }
        evaluated_tiers.append(tier_summary)

        max_str = f"{max_qty:,.0f}" if max_qty is not None else "∞"
        step_descriptions.append(
            f"Tier {idx+1} [{min_qty:,.0f} to {max_str} @ ${price:,.2f}]:\n"
            f"  - Holding Cost H = {f'{holding_cost_rate*100:.1f}% * ${price:.2f} = ${tier_h:.2f}' if holding_cost_rate else f'${tier_h:.2f}'}\n"
            f"  - Unconstrained EOQ = sqrt((2 * {demand:,.0f} * {order_cost:,.2f}) / {tier_h:.2f}) = {raw_eoq:,.2f}\n"
            f"  - Feasible Candidate Q = {candidate_q:,.2f} ({status})\n"
            f"  - Total Annual Cost = Purchase (${annual_purchase:,.2f}) + Order (${annual_ordering:,.2f}) + Hold (${annual_holding:,.2f}) = ${total_cost:,.2f}"
        )

        if total_cost < min_total_cost:
            min_total_cost = total_cost
            best_tier = tier_summary

    # 1. Inputs Extracted
    inputs_extracted = {
        "demand (D)": float(demand),
        "order_cost (S)": float(order_cost),
        "holding_cost_rate (i)": float(holding_cost_rate) if holding_cost_rate is not None else None,
        "fixed_holding_cost (H)": float(fixed_holding_cost) if fixed_holding_cost is not None else None,
        "discount_tiers": discount_tiers,
    }

    # 2. Variable Definitions
    variable_definitions = {
        "D": "Annual demand in units",
        "S": "Fixed ordering cost per order in dollars",
        "P_j": "Unit purchase price for tier bracket j in dollars ($/unit)",
        "H_j": "Unit holding cost per year for tier j ($/unit/year)",
        "Q_raw": "Unconstrained EOQ calculated at price P_j: sqrt(2 * D * S / H_j)",
        "Q_candidate": "Feasible quantity bounded to valid tier bracket [min_qty_j, max_qty_j]",
        "Annual Purchase Cost": "Total acquisition expense: D * P_j",
        "Annual Ordering Cost": "Replenishment expense: (D / Q_candidate) * S",
        "Annual Holding Cost": "Carrying expense: (Q_candidate / 2) * H_j",
        "Total Annual Cost (TC)": "Sum of purchase, ordering, and holding costs",
    }

    # 3. Formula
    formula = (
        "For each price tier j:\n"
        "  1. H_j = i * P_j (or fixed H)\n"
        "  2. Q_raw = sqrt((2 * D * S) / H_j)\n"
        "  3. Q_candidate = min_qty if Q_raw < min_qty else (max_qty if Q_raw > max_qty else Q_raw)\n"
        "  4. TC(Q) = (D * P_j) + (D / Q_candidate * S) + (Q_candidate / 2 * H_j)\n"
        "Select Tier j with global minimum TC(Q)."
    )

    # 4. Formula with Substituted Values
    formula_with_substituted_values = "\n\n".join(step_descriptions)

    # 5. Intermediate Calculations
    intermediate_calculations = {
        "evaluated_tiers": evaluated_tiers,
        "optimal_tier_index": best_tier["tier_index"],
        "optimal_order_quantity": best_tier["candidate_q"],
        "optimal_unit_price": best_tier["unit_price"],
        "optimal_total_cost": best_tier["total_annual_cost"],
    }

    # 6. Final Answer
    final_answer = (
        f"Select Tier {best_tier['tier_index']} at an optimal order quantity of {best_tier['candidate_q']:,} units "
        f"at ${best_tier['unit_price']:.2f}/unit. This minimizes total annual acquisition and inventory carrying costs "
        f"at ${best_tier['total_annual_cost']:,.2f}."
    )

    # 7. Verification Checks
    all_costs = [t["total_annual_cost"] for t in evaluated_tiers]
    is_min = best_tier["total_annual_cost"] == min(all_costs)
    opt_q = best_tier["candidate_q"]
    in_bounds = opt_q >= best_tier["min_qty"] and (best_tier["max_qty"] is None or opt_q <= best_tier["max_qty"])

    verification_checks = [
        {
            "check": "Bracket Feasibility",
            "condition": "min_qty <= Q* <= max_qty",
            "calculated_value": f"{best_tier['min_qty']:,} <= {opt_q:,} <= {best_tier['max_qty'] or '∞'}",
            "status": "PASSED" if in_bounds else "FAILED",
            "student_note": "Ensures the selected quantity qualifies for the supplier's price break.",
        },
        {
            "check": "Global Total Cost Minimization",
            "condition": "TC(selected tier) <= TC(all other tiers)",
            "calculated_value": f"Selected ${best_tier['total_annual_cost']:,.2f} vs other tiers {[t['total_annual_cost'] for t in evaluated_tiers]}",
            "status": "PASSED" if is_min else "FAILED",
            "student_note": "Evaluates trade-off: purchase savings exceed extra inventory holding costs.",
        },
    ]

    # 8. Assumptions
    assumptions = [
        "All-units quantity discount applies to every unit purchased in the order.",
        "Supplier price breaks and capacity remain constant across the planning year.",
        "Demand is completely independent of the quantity discount price.",
        "Sufficient storage capacity and working capital exist to hold larger volume batches.",
    ]

    return {
        "model": "Quantity Discount Model",
        "inputs": inputs_extracted,
        "variable_definitions": variable_definitions,
        "formula": formula,
        "formula_with_substituted_values": formula_with_substituted_values,
        "intermediate_calculations": intermediate_calculations,
        "final_answer": final_answer,
        "verification_checks": verification_checks,
        "assumptions": assumptions,
        "results": {
            "optimal_order_quantity": best_tier["candidate_q"],
            "optimal_unit_price": best_tier["unit_price"],
            "optimal_tier": best_tier["tier_index"],
            "minimum_total_annual_cost": best_tier["total_annual_cost"],
            "tier_breakdown": evaluated_tiers,
        },
    }


# =============================================================================
# 3. SAFETY STOCK AND REORDER POINT (ROP)
# =============================================================================
def calculate_rop(
    avg_demand: float,
    lead_time: float,
    service_level: Optional[float] = None,
    z_score: Optional[float] = None,
    demand_std_dev: float = 0.0,
    lead_time_std_dev: float = 0.0,
    holding_cost: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calculate Safety Stock (SS) and Reorder Point (ROP) for continuous review systems.

    Supports all 4 uncertainty conditions:
    1. Constant demand and constant lead time (sigma_d = 0, sigma_L = 0)
    2. Variable demand and constant lead time (sigma_d > 0, sigma_L = 0)
    3. Constant demand and variable lead time (sigma_d = 0, sigma_L > 0)
    4. Variable demand and variable lead time (sigma_d > 0, sigma_L > 0)

    Returns all 8 pedagogical components.
    """
    # Validation
    if avg_demand is None or lead_time is None:
        raise ValueError("Missing required inputs: 'avg_demand' and 'lead_time' must be provided.")
    if avg_demand < 0:
        raise ValueError(f"Average demand cannot be negative, got {avg_demand}.")
    if lead_time < 0:
        raise ValueError(f"Lead time cannot be negative, got {lead_time}.")
    if demand_std_dev < 0 or lead_time_std_dev < 0:
        raise ValueError("Standard deviations cannot be negative.")

    # Determine z-score and service level
    if z_score is not None:
        if z_score < 0:
            raise ValueError(f"z_score cannot be negative, got {z_score}.")
        z = float(z_score)
        csl = float(norm.cdf(z))
    elif service_level is not None:
        if service_level <= 0:
            raise ValueError(f"Service level must be strictly positive, got {service_level}.")
        if service_level >= 100.0:
            raise ValueError("Service level percentage must be strictly less than 100%.")
        csl = float(service_level / 100.0 if service_level > 1.0 else service_level)
        if not (0.0 < csl < 1.0):
            raise ValueError(f"Service level must be strictly between 0 and 1, got {csl}.")
        z = float(norm.ppf(csl))
    else:
        raise ValueError("Must provide either 'service_level' (e.g. 0.95) or 'z_score' (e.g. 1.645).")

    # Case Identification & Formula Evaluation
    is_demand_variable = demand_std_dev > 0.0
    is_lead_time_variable = lead_time_std_dev > 0.0

    if not is_demand_variable and not is_lead_time_variable:
        case_name = "constant demand and constant lead time"
        var_demand_comp = 0.0
        var_lead_time_comp = 0.0
        total_variance_lt = 0.0
        sigma_lead_time = 0.0
        safety_stock = 0.0
        formula_expr = "sigma_LT = 0; SS = 0; ROP = d * L"
    elif is_demand_variable and not is_lead_time_variable:
        case_name = "variable demand and constant lead time"
        var_demand_comp = lead_time * (demand_std_dev**2)
        var_lead_time_comp = 0.0
        total_variance_lt = var_demand_comp
        sigma_lead_time = math.sqrt(total_variance_lt)
        safety_stock = z * sigma_lead_time
        formula_expr = "sigma_LT = sqrt(L * sigma_d^2); SS = z * sigma_LT; ROP = d * L + SS"
    elif not is_demand_variable and is_lead_time_variable:
        case_name = "constant demand and variable lead time"
        var_demand_comp = 0.0
        var_lead_time_comp = (avg_demand**2) * (lead_time_std_dev**2)
        total_variance_lt = var_lead_time_comp
        sigma_lead_time = math.sqrt(total_variance_lt)
        safety_stock = z * sigma_lead_time
        formula_expr = "sigma_LT = sqrt(d^2 * sigma_L^2) = d * sigma_L; SS = z * sigma_LT; ROP = d * L + SS"
    else:
        case_name = "variable demand and variable lead time"
        var_demand_comp = lead_time * (demand_std_dev**2)
        var_lead_time_comp = (avg_demand**2) * (lead_time_std_dev**2)
        total_variance_lt = var_demand_comp + var_lead_time_comp
        sigma_lead_time = math.sqrt(total_variance_lt)
        safety_stock = z * sigma_lead_time
        formula_expr = "sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2); SS = z * sigma_LT; ROP = d * L + SS"

    lead_time_demand = avg_demand * lead_time
    reorder_point = lead_time_demand + safety_stock
    annual_ss_cost = (safety_stock * holding_cost) if holding_cost is not None else None

    # 1. Inputs Extracted
    inputs_extracted = {
        "average_demand (d)": float(avg_demand),
        "lead_time (L)": float(lead_time),
        "cycle_service_level (CSL)": round(csl, 4),
        "z_score (z)": round(z, 4),
        "demand_std_dev (sigma_d)": float(demand_std_dev),
        "lead_time_std_dev (sigma_L)": float(lead_time_std_dev),
        "holding_cost (H)": float(holding_cost) if holding_cost is not None else None,
    }

    # 2. Variable Definitions
    variable_definitions = {
        "d": "Average customer demand rate per period",
        "L": "Average replenishment lead time duration",
        "sigma_d": "Standard deviation of periodic customer demand",
        "sigma_L": "Standard deviation of supplier lead time",
        "CSL": "Cycle Service Level: probability of zero stockouts during lead time",
        "z": "Standard normal critical factor corresponding to CSL: z = norm.ppf(CSL)",
        "DL": "Expected demand during lead time: DL = d * L",
        "sigma_LT": "Total standard deviation of demand during lead time",
        "SS": "Safety Stock required to absorb lead-time volatility: SS = z * sigma_LT",
        "ROP": "Reorder Point inventory trigger: ROP = DL + SS",
    }

    # 3. Formula
    formula = (
        f"Case: {case_name.title()}\n"
        f"General Formula:\n"
        f"  Total Variance Var(DL) = (L * sigma_d^2) + (d^2 * sigma_L^2)\n"
        f"  sigma_LT = sqrt(Var(DL))\n"
        f"  Safety Stock (SS) = z * sigma_LT\n"
        f"  Reorder Point (ROP) = (d * L) + SS"
    )

    # 4. Formula with Substituted Values
    formula_with_substituted_values = (
        f"1. Expected Lead Time Demand (DL) = {avg_demand} * {lead_time} = {lead_time_demand:,.2f} units\n"
        f"2. Demand Variance Component = {lead_time} * ({demand_std_dev}^2) = {var_demand_comp:,.2f}\n"
        f"3. Lead Time Variance Component = ({avg_demand}^2) * ({lead_time_std_dev}^2) = {var_lead_time_comp:,.2f}\n"
        f"4. Total Lead Time Variance = {var_demand_comp:,.2f} + {var_lead_time_comp:,.2f} = {total_variance_lt:,.2f}\n"
        f"5. sigma_LT = sqrt({total_variance_lt:,.2f}) = {sigma_lead_time:,.4f} units\n"
        f"6. Safety Stock (SS) = {z:.4f} * {sigma_lead_time:,.4f} = {safety_stock:,.2f} units ({int(math.ceil(safety_stock)):,} rounded)\n"
        f"7. Reorder Point (ROP) = {lead_time_demand:,.2f} + {safety_stock:,.2f} = {reorder_point:,.2f} units ({int(math.ceil(reorder_point)):,} rounded)"
    )

    # 5. Intermediate Calculations
    intermediate_calculations = {
        "case_identified": case_name,
        "lead_time_demand_DL": round(lead_time_demand, 4),
        "demand_variance_component": round(var_demand_comp, 4),
        "lead_time_variance_component": round(var_lead_time_comp, 4),
        "total_lead_time_variance": round(total_variance_lt, 4),
        "sigma_lead_time_demand": round(sigma_lead_time, 4),
        "z_critical_factor": round(z, 4),
        "safety_stock_exact": round(safety_stock, 4),
        "reorder_point_exact": round(reorder_point, 4),
        "annual_safety_stock_holding_cost": round(annual_ss_cost, 2) if annual_ss_cost is not None else None,
    }

    # 6. Final Answer
    final_answer = (
        f"Under {case_name}, maintain a Safety Stock of {round(safety_stock, 2):,} units ({int(math.ceil(safety_stock)):,} rounded) "
        f"to achieve a {round(csl * 100, 1)}% Cycle Service Level. Place a replenishment order whenever the "
        f"inventory position reaches {round(reorder_point, 2):,} units ({int(math.ceil(reorder_point)):,} rounded)."
    )

    # 7. Verification Checks
    verification_checks = [
        {
            "check": "Conservation of ROP Components",
            "condition": "ROP == DL + SS",
            "calculated_value": f"{reorder_point:,.2f} == {lead_time_demand:,.2f} + {safety_stock:,.2f}",
            "status": "PASSED" if abs(reorder_point - (lead_time_demand + safety_stock)) < 1e-4 else "FAILED",
            "student_note": "Confirms reorder point strictly covers expected lead time demand plus safety buffer.",
        },
        {
            "check": "Safety Stock Non-Negativity",
            "condition": "SS >= 0 for CSL >= 50%",
            "calculated_value": round(safety_stock, 2),
            "status": "PASSED" if safety_stock >= 0 else "FAILED",
            "student_note": "A service level >= 50% guarantees positive buffer stock under normal distribution.",
        },
        {
            "check": "Cumulative Service Level Consistency",
            "condition": "norm.cdf(z) == CSL",
            "calculated_value": f"Phi({z:.4f}) = {norm.cdf(z):.4f} (target: {csl:.4f})",
            "status": "PASSED" if abs(norm.cdf(z) - csl) < 1e-3 else "FAILED",
            "student_note": "Verifies that the z-factor exactly matches standard normal cumulative distribution probability.",
        },
    ]

    # 8. Assumptions
    assumptions = [
        f"Inventory uncertainty condition: {case_name}.",
        "Demand across successive time periods is independent and identically distributed (i.i.d.).",
        "Lead time duration is statistically independent of customer demand volume.",
        "Cycle service level targets non-stockout probability during replenishment cycle.",
        "Unfulfilled demand during lead time is backordered, not lost.",
    ]

    return {
        "model": "Safety Stock & Reorder Point",
        "case": case_name,
        "inputs": inputs_extracted,
        "variable_definitions": variable_definitions,
        "formula": formula,
        "formula_with_substituted_values": formula_with_substituted_values,
        "intermediate_calculations": intermediate_calculations,
        "final_answer": final_answer,
        "verification_checks": verification_checks,
        "assumptions": assumptions,
        "results": {
            "lead_time_demand": round(lead_time_demand, 2),
            "sigma_lead_time_demand": round(sigma_lead_time, 2),
            "z_score": round(z, 4),
            "safety_stock": round(safety_stock, 2),
            "safety_stock_int": int(math.ceil(safety_stock)),
            "reorder_point": round(reorder_point, 2),
            "reorder_point_int": int(math.ceil(reorder_point)),
            "annual_safety_stock_cost": round(annual_ss_cost, 2) if annual_ss_cost is not None else None,
        },
    }


calculate_safety_stock_and_rop = calculate_rop


# =============================================================================
# 4. FIXED-PERIOD REVIEW MODEL (PERIODIC REVIEW / (R, S))
# =============================================================================
def calculate_periodic_review(
    avg_demand: float,
    review_period: float,
    lead_time: float,
    service_level: Optional[float] = None,
    z_score: Optional[float] = None,
    demand_std_dev: float = 0.0,
    current_inventory: Optional[float] = None,
    on_order_inventory: float = 0.0,
    backorders: float = 0.0,
) -> Dict[str, Any]:
    """
    Calculate Order-Up-To Target Level and Order Quantity in a Fixed-Period Review System.
    Protection period = review period + lead time (R + L).
    Target inventory = demand during protection period + safety stock.
    Order quantity = max(0, target inventory - inventory position).
    Returns all 8 pedagogical components.
    """
    if avg_demand is None or review_period is None or lead_time is None:
        raise ValueError("Missing required inputs: 'avg_demand', 'review_period', and 'lead_time' must be provided.")
    if avg_demand < 0:
        raise ValueError(f"Average demand cannot be negative, got {avg_demand}.")
    if review_period <= 0:
        raise ValueError(f"Review period must be strictly positive, got {review_period}.")
    if lead_time < 0:
        raise ValueError(f"Lead time cannot be negative, got {lead_time}.")
    if demand_std_dev < 0:
        raise ValueError(f"Standard deviation of demand cannot be negative, got {demand_std_dev}.")
    if backorders < 0:
        raise ValueError(f"Backorders cannot be negative, got {backorders}.")

    # Service level / z-score
    if z_score is not None:
        if z_score < 0:
            raise ValueError(f"z_score cannot be negative, got {z_score}.")
        z = float(z_score)
        csl = float(norm.cdf(z))
    elif service_level is not None:
        if service_level <= 0:
            raise ValueError(f"Service level must be strictly positive, got {service_level}.")
        if service_level >= 100.0:
            raise ValueError("Service level percentage must be strictly less than 100%.")
        csl = float(service_level / 100.0 if service_level > 1.0 else service_level)
        if not (0.0 < csl < 1.0):
            raise ValueError("Service level must be strictly between 0 and 1.")
        z = float(norm.ppf(csl))
    else:
        raise ValueError("Must provide either 'service_level' or 'z_score'.")

    # Protection Period Calculation
    protection_period = review_period + lead_time
    expected_protection_demand = avg_demand * protection_period
    sigma_protection = demand_std_dev * math.sqrt(protection_period)
    safety_stock = z * sigma_protection
    target_inventory = expected_protection_demand + safety_stock

    order_quantity = None
    inventory_position = None
    if current_inventory is not None:
        inventory_position = float(current_inventory) + float(on_order_inventory) - float(backorders)
        order_quantity = max(0.0, target_inventory - inventory_position)

    # 1. Inputs Extracted
    inputs_extracted = {
        "average_demand (d)": float(avg_demand),
        "review_period (R)": float(review_period),
        "lead_time (L)": float(lead_time),
        "cycle_service_level (CSL)": round(csl, 4),
        "z_score (z)": round(z, 4),
        "demand_std_dev (sigma_d)": float(demand_std_dev),
        "on_hand_inventory (I)": float(current_inventory) if current_inventory is not None else None,
        "on_order_inventory": float(on_order_inventory),
        "backorders": float(backorders),
    }

    # 2. Variable Definitions
    variable_definitions = {
        "d": "Average demand rate per period",
        "R": "Review period duration between successive inventory audits",
        "L": "Replenishment delivery lead time",
        "R + L": "Total protection interval duration: order must cover demand over R + L periods",
        "sigma_d": "Demand standard deviation per period",
        "sigma_protection": "Demand volatility over entire protection window: sigma_d * sqrt(R + L)",
        "SS": "Safety stock required for periodic review: z * sigma_protection",
        "S": "Order-Up-To Target Inventory Level: d * (R + L) + SS",
        "IP": "Net Inventory Position: On-Hand + On-Order - Backorders",
        "Q": "Recommended order quantity to place: max(0, S - IP)",
    }

    # 3. Formula
    formula = (
        "Protection Period (T_p) = R + L\n"
        "Expected Protection Demand = d * (R + L)\n"
        "sigma_protection = sigma_d * sqrt(R + L)\n"
        "Safety Stock (SS) = z * sigma_protection\n"
        "Target Inventory Level (S) = d * (R + L) + SS\n"
        "Inventory Position (IP) = On-Hand + On-Order - Backorders\n"
        "Order Quantity (Q) = max(0, S - IP)"
    )

    # 4. Formula with Substituted Values
    target_int = int(math.ceil(target_inventory))
    formula_with_substituted_values = (
        f"1. Protection Period = {review_period} + {lead_time} = {protection_period:.2f} periods\n"
        f"2. Expected Demand over Protection Period = {avg_demand} * {protection_period:.2f} = {expected_protection_demand:,.2f} units\n"
        f"3. sigma_protection = {demand_std_dev} * sqrt({protection_period:.2f}) = {sigma_protection:,.4f} units\n"
        f"4. Safety Stock (SS) = {z:.4f} * {sigma_protection:,.4f} = {safety_stock:,.2f} units ({int(math.ceil(safety_stock)):,} rounded)\n"
        f"5. Target Inventory Level (S) = {expected_protection_demand:,.2f} + {safety_stock:,.2f} = {target_inventory:,.2f} units ({target_int:,} rounded)\n"
    )
    if inventory_position is not None:
        order_q_int = int(math.ceil(order_quantity))
        formula_with_substituted_values += (
            f"6. Inventory Position (IP) = {current_inventory:,.2f} (on-hand) + {on_order_inventory:,.2f} (on-order) - {backorders:,.2f} (backorders) = {inventory_position:,.2f} units\n"
            f"7. Order Quantity (Q) = max(0, {target_inventory:,.2f} - {inventory_position:,.2f}) = {order_quantity:,.2f} units ({order_q_int:,} rounded)"
        )

    # 5. Intermediate Calculations
    intermediate_calculations = {
        "protection_period": round(protection_period, 4),
        "expected_protection_demand": round(expected_protection_demand, 4),
        "sigma_protection": round(sigma_protection, 4),
        "safety_stock": round(safety_stock, 4),
        "target_inventory_exact": round(target_inventory, 4),
        "inventory_position": round(inventory_position, 4) if inventory_position is not None else None,
        "order_quantity_exact": round(order_quantity, 4) if order_quantity is not None else None,
    }

    # 6. Final Answer
    if order_quantity is not None:
        final_answer = (
            f"Set Target Order-Up-To Level (S) to {round(target_inventory, 2):,} units ({target_int:,} rounded) "
            f"to protect across the {protection_period} period window. With current inventory position of "
            f"{inventory_position:,.2f} units, place an order for {round(order_quantity, 2):,} units ({int(math.ceil(order_quantity)):,} rounded)."
        )
    else:
        final_answer = (
            f"Set Target Order-Up-To Level (S) to {round(target_inventory, 2):,} units ({target_int:,} rounded) "
            f"across the {protection_period} period protection window (Review Period {review_period} + Lead Time {lead_time})."
        )

    # 7. Verification Checks
    verification_checks = [
        {
            "check": "Protection Period Formulation",
            "condition": "Protection Period == R + L",
            "calculated_value": f"{protection_period} == {review_period} + {lead_time}",
            "status": "PASSED" if abs(protection_period - (review_period + lead_time)) < 1e-4 else "FAILED",
            "student_note": "In periodic review, stock is only evaluated every R periods, so protection must extend across R + L.",
        },
        {
            "check": "Target Inventory Consistency",
            "condition": "S == Expected Demand + Safety Stock",
            "calculated_value": f"{target_inventory:,.2f} == {expected_protection_demand:,.2f} + {safety_stock:,.2f}",
            "status": "PASSED" if abs(target_inventory - (expected_protection_demand + safety_stock)) < 1e-4 else "FAILED",
            "student_note": "Ensures target stock level rigorously equals expected protection demand plus safety buffer.",
        },
        {
            "check": "Non-Negative Order Quantity",
            "condition": "Q >= 0",
            "calculated_value": round(order_quantity, 2) if order_quantity is not None else "N/A",
            "status": "PASSED" if (order_quantity is None or order_quantity >= 0) else "FAILED",
            "student_note": "If inventory position exceeds target S, order quantity is bounded at zero (no negative order).",
        },
    ]

    # 8. Assumptions
    assumptions = [
        "Inventory levels are reviewed and replenished strictly at fixed review intervals of R periods.",
        "Protection interval must span both review duration R and supplier lead time L.",
        "Demand across the protection period is normally distributed.",
        "Unfulfilled demand during the cycle is backordered.",
        "Inventory position equals on-hand physical stock plus open pipeline orders minus backorders.",
    ]

    return {
        "model": "Fixed-Period Review (Periodic Review / (R, S))",
        "inputs": inputs_extracted,
        "variable_definitions": variable_definitions,
        "formula": formula,
        "formula_with_substituted_values": formula_with_substituted_values,
        "intermediate_calculations": intermediate_calculations,
        "final_answer": final_answer,
        "verification_checks": verification_checks,
        "assumptions": assumptions,
        "results": {
            "protection_interval": round(protection_period, 2),
            "expected_protection_demand": round(expected_protection_demand, 2),
            "sigma_protection_interval": round(sigma_protection, 2),
            "safety_stock": round(safety_stock, 2),
            "safety_stock_int": int(math.ceil(safety_stock)),
            "target_inventory_level": round(target_inventory, 2),
            "target_inventory_level_int": target_int,
            "order_quantity": round(order_quantity, 2) if order_quantity is not None else None,
            "order_quantity_int": int(math.ceil(order_quantity)) if order_quantity is not None else None,
            "inventory_position": round(inventory_position, 2) if inventory_position is not None else None,
        },
    }


calculate_fixed_period_review = calculate_periodic_review


# =============================================================================
# 5. NEWSVENDOR MODEL (SINGLE-PERIOD STOCHASTIC)
# =============================================================================
def calculate_newsvendor(
    selling_price: Optional[float] = None,
    cost_price: Optional[float] = None,
    mean_demand: Optional[float] = None,
    std_demand: Optional[float] = None,
    salvage_value: float = 0.0,
    penalty_cost: float = 0.0,
    underage_cost: Optional[float] = None,
    overage_cost: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calculate optimal order quantity for single-period stochastic inventory (Newsvendor Model).
    Supports price/cost inputs OR direct underage/overage cost inputs.
    Returns all 8 pedagogical components.
    """
    if mean_demand is None or std_demand is None:
        raise ValueError("Missing required inputs: 'mean_demand' and 'std_demand' must be provided.")

    if underage_cost is not None and overage_cost is not None:
        if underage_cost <= 0 or overage_cost <= 0:
            raise ValueError("Underage cost and overage cost must be strictly positive.")
        cost_underage = float(underage_cost)
        cost_overage = float(overage_cost)
        cost_source = "Direct Underage/Overage Input"
    elif selling_price is not None and cost_price is not None:
        if selling_price <= cost_price:
            raise ValueError(f"Selling price (${selling_price}) must exceed cost price (${cost_price}) for positive unit margin.")
        if cost_price <= salvage_value:
            raise ValueError(f"Cost price (${cost_price}) must exceed salvage value (${salvage_value}) to define positive overage cost.")
        cost_underage = float(selling_price - cost_price + penalty_cost)
        cost_overage = float(cost_price - salvage_value)
        cost_source = "Margin Economics from Selling Price, Cost, and Salvage"
    else:
        raise ValueError("Must provide either ('selling_price' and 'cost_price') or ('underage_cost' and 'overage_cost').")

    if mean_demand < 0 or std_demand < 0:
        raise ValueError("Demand mean and standard deviation cannot be negative.")
    if penalty_cost < 0:
        raise ValueError("Penalty cost cannot be negative.")

    # Critical Ratio Calculation
    critical_ratio = cost_underage / (cost_underage + cost_overage)

    if std_demand == 0.0:
        z_star = 0.0
        optimal_q = float(mean_demand)
        expected_lost_sales = 0.0
        expected_sales = float(mean_demand)
        expected_leftover = 0.0
    else:
        z_star = float(norm.ppf(critical_ratio))
        optimal_q = mean_demand + z_star * std_demand

        # Standard normal loss function L(z) = phi(z) - z * (1 - Phi(z))
        loss_z = norm.pdf(z_star) - z_star * (1.0 - norm.cdf(z_star))
        expected_lost_sales = std_demand * loss_z
        expected_sales = mean_demand - expected_lost_sales
        expected_leftover = optimal_q - expected_sales

    # Expected Profit
    expected_profit = None
    if selling_price is not None and cost_price is not None:
        expected_profit = (
            (selling_price * expected_sales)
            + (salvage_value * expected_leftover)
            - (cost_price * optimal_q)
            - (penalty_cost * expected_lost_sales)
        )

    # 1. Inputs Extracted
    inputs_extracted = {
        "selling_price (p)": float(selling_price) if selling_price is not None else None,
        "cost_price (c)": float(cost_price) if cost_price is not None else None,
        "salvage_value (s)": float(salvage_value),
        "penalty_cost (g)": float(penalty_cost),
        "underage_cost (Cu)": float(cost_underage),
        "overage_cost (Co)": float(cost_overage),
        "mean_demand (mu)": float(mean_demand),
        "std_demand (sigma)": float(std_demand),
    }

    # 2. Variable Definitions
    variable_definitions = {
        "p": "Selling price per unit during sales period ($/unit)",
        "c": "Unit procurement or production cost ($/unit)",
        "s": "Salvage value per unsold unit at season end ($/unit)",
        "g": "Goodwill / penalty cost per unit of unsatisfied customer demand ($/unit)",
        "Cu": "Underage cost (marginal profit lost per unit of unmet demand): Cu = p - c + g",
        "Co": "Overage cost (marginal loss suffered per unsold unit): Co = c - s",
        "CR": "Critical Ratio / Fractile: optimal in-stock probability = Cu / (Cu + Co)",
        "mu": "Mean expected seasonal demand",
        "sigma": "Standard deviation of demand",
        "z*": "Optimal standard normal factor: norm.ppf(CR)",
        "Q*": "Optimal order quantity: mu + z* * sigma",
        "Expected Sales": "Forecasted units sold at full retail price",
        "Expected Leftover": "Forecasted leftover units sold at markdown/salvage value",
        "Expected Profit": "Expected revenue + salvage revenue - procurement cost - penalty cost",
    }

    # 3. Formula
    formula = (
        "Underage Cost (Cu) = Selling Price - Cost Price + Penalty Cost\n"
        "Overage Cost (Co) = Cost Price - Salvage Value\n"
        "Critical Ratio (CR) = Cu / (Cu + Co)\n"
        "z* = norm.ppf(CR)\n"
        "Optimal Order Quantity (Q*) = mu + (z* * sigma)"
    )

    # 4. Formula with Substituted Values
    opt_q_int = int(round(optimal_q))
    formula_with_substituted_values = (
        f"1. Underage Cost (Cu) = {cost_underage:.2f}\n"
        f"2. Overage Cost (Co) = {cost_overage:.2f}\n"
        f"3. Critical Ratio (CR) = {cost_underage:.2f} / ({cost_underage:.2f} + {cost_overage:.2f}) = {critical_ratio:.4f} ({critical_ratio*100:.2f}%)\n"
        f"4. z* = norm.ppf({critical_ratio:.4f}) = {z_star:.4f}\n"
        f"5. Q* = {mean_demand} + ({z_star:.4f} * {std_demand}) = {optimal_q:.2f} units ({opt_q_int:,} rounded)"
    )

    # 5. Intermediate Calculations
    intermediate_calculations = {
        "underage_cost_Cu": round(cost_underage, 4),
        "overage_cost_Co": round(cost_overage, 4),
        "critical_ratio": round(critical_ratio, 6),
        "critical_ratio_pct": round(critical_ratio * 100.0, 2),
        "z_star": round(z_star, 4),
        "expected_lost_sales": round(expected_lost_sales, 4),
        "expected_sales": round(expected_sales, 4),
        "expected_leftover_units": round(expected_leftover, 4),
        "expected_profit": round(expected_profit, 2) if expected_profit is not None else None,
    }

    # 6. Final Answer
    profit_str = f" with an expected net profit of ${round(expected_profit, 2):,.2f}" if expected_profit is not None else ""
    final_answer = (
        f"The optimal order quantity is {round(optimal_q, 2):,} units ({opt_q_int:,} rounded). "
        f"This balances marginal underage cost ($C_u = ${cost_underage:.2f}) and overage cost ($C_o = ${cost_overage:.2f}) "
        f"at a {round(critical_ratio * 100, 2)}% critical service level{profit_str}."
    )

    # 7. Verification Checks
    verification_checks = [
        {
            "check": "Critical Ratio Probability Boundary",
            "condition": "0 < CR < 1",
            "calculated_value": round(critical_ratio, 4),
            "status": "PASSED" if (0.0 < critical_ratio < 1.0) else "FAILED",
            "student_note": "Confirms the critical ratio is a valid cumulative probability fractile.",
        },
        {
            "check": "Economic Viability of Product",
            "condition": "Cu > 0 and Co > 0",
            "calculated_value": f"Cu=${cost_underage:.2f}, Co=${cost_overage:.2f}",
            "status": "PASSED" if (cost_underage > 0 and cost_overage > 0) else "FAILED",
            "student_note": "Ensures selling margin exceeds procurement cost, and procurement cost exceeds salvage value.",
        },
        {
            "check": "Service Level In-Stock Alignment",
            "condition": "P(Demand <= Q*) == Critical Ratio",
            "calculated_value": f"Phi({z_star:.4f}) = {norm.cdf(z_star):.4f} (CR: {critical_ratio:.4f})",
            "status": "PASSED" if abs(norm.cdf(z_star) - critical_ratio) < 1e-3 else "FAILED",
            "student_note": "Mathematically proves that ordering Q* targets the exact critical fractile balance point.",
        },
    ]

    # 8. Assumptions
    assumptions = [
        "Single sales period with strictly no mid-season replenishment possible.",
        "Demand is a random variable following a normal distribution N(mu, sigma).",
        "Unsold stock is salvaged at salvage value s < c (or incurs overage cost Co).",
        "Unsatisfied customer demand results in lost sales margin Cu.",
        "Selling price, cost price, and salvage value remain constant throughout the period.",
    ]

    return {
        "model": "Newsvendor Model (Single-Period Stochastic)",
        "inputs": inputs_extracted,
        "variable_definitions": variable_definitions,
        "formula": formula,
        "formula_with_substituted_values": formula_with_substituted_values,
        "intermediate_calculations": intermediate_calculations,
        "final_answer": final_answer,
        "verification_checks": verification_checks,
        "assumptions": assumptions,
        "results": {
            "underage_cost_Cu": round(cost_underage, 2),
            "overage_cost_Co": round(cost_overage, 2),
            "critical_ratio": round(critical_ratio, 4),
            "critical_ratio_pct": round(critical_ratio * 100.0, 2),
            "z_star": round(z_star, 4),
            "optimal_order_quantity": round(optimal_q, 2),
            "optimal_order_quantity_int": opt_q_int,
            "in_stock_probability": round(critical_ratio, 4),
            "expected_sales": round(expected_sales, 2),
            "expected_lost_sales": round(expected_lost_sales, 2),
            "expected_leftover_units": round(expected_leftover, 2),
            "expected_profit": round(expected_profit, 2) if expected_profit is not None else None,
        },
    }


# =============================================================================
# 6. MULTICRITERIA INVENTORY CLASSIFICATION (A, B, C, D)
# =============================================================================
def classify_inventory_multicriteria(
    df: pd.DataFrame,
    sales_col: Optional[str] = None,
    hits_col: Optional[str] = None,
    item_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Performs deterministic multi-criteria inventory classification (A, B, C, D)
    using the exact Inventory Analytics module procedure:

    STEP 1: Classify Sales ($) and Hits separately
    - Clean text/numeric fields.
    - Sort descending by criterion, break ties by Item Number ascending.
    - Cumulative % = running sum / total (unrounded comparison).
    - Thresholds: A <= 60%, B <= 80%, C <= 90%, D > 90%.
    - Scores: A = 4, B = 3, C = 2, D = 1.
    - Merge results back to original DataFrame by Item Number without dropping rows.

    STEP 2: Combined Weighted Score
    - Combined Weighted Score = 0.65 * (Sales Score) + 0.35 * (Hits Score).

    STEP 3: Final Inventory Class
    - A: score >= 3.25
    - B: score >= 2.50 and < 3.25
    - C: score >= 1.75 and < 2.50
    - D: score < 1.75 (guarded against floating-point error).

    Returns:
        Dict with 'classified_df', 'summary_df', 'validation_messages', 'warnings', and 'metrics'.
    """
    if df is None or df.empty:
        raise ValueError("The provided dataframe is empty or None.")

    res_df = df.copy()
    warnings = []

    # 1. Column Detection & Validation
    # Required conceptual columns: Item Number, Item Name, UoM, Sales ($), Hits
    col_map = {c.strip().lower(): c for c in res_df.columns}

    # Item Number detection
    if item_col is None:
        for candidate in ["item number", "item_number", "item no", "item #", "sku", "product"]:
            if candidate in col_map:
                item_col = col_map[candidate]
                break
        if item_col is None:
            # Fall back to first column
            item_col = res_df.columns[0]
            warnings.append(f"Using '{item_col}' as Item Number identifier.")

    # Sales column detection
    if sales_col is None:
        for candidate in ["sales ($)", "sales", "total sales", "revenue", "sales_usd"]:
            if candidate in col_map:
                sales_col = col_map[candidate]
                break
        if sales_col is None:
            for c in res_df.columns:
                if "sale" in c.lower():
                    sales_col = c
                    break
    if sales_col is None or sales_col not in res_df.columns:
        raise ValueError(f"Sales column not found. Available columns: {list(res_df.columns)}")

    # Hits column detection
    if hits_col is None:
        for candidate in ["hits", "pick hits", "order hits", "picks", "transactions"]:
            if candidate in col_map:
                hits_col = col_map[candidate]
                break
        if hits_col is None:
            for c in res_df.columns:
                if "hit" in c.lower() or "pick" in c.lower():
                    hits_col = c
                    break
    if hits_col is None or hits_col not in res_df.columns:
        raise ValueError(f"Hits column not found. Available columns: {list(res_df.columns)}")

    # 2. Data Cleaning
    # Strip whitespace from text columns
    for col in res_df.columns:
        if not pd.api.types.is_numeric_dtype(res_df[col]):
            res_df[col] = res_df[col].astype(str).str.strip()

    # Check for blank / empty Item Numbers
    blank_items = res_df[res_df[item_col].isna() | (res_df[item_col] == "") | (res_df[item_col] == "nan")]
    if not blank_items.empty:
        warnings.append(f"Found {len(blank_items)} row(s) with blank or empty {item_col}.")

    # Check for duplicate Item Numbers
    duplicate_items = res_df[res_df.duplicated(subset=[item_col], keep=False)]
    if not duplicate_items.empty:
        dup_keys = duplicate_items[item_col].unique().tolist()
        warnings.append(
            f"Found {len(duplicate_items)} rows with duplicate {item_col} ({len(dup_keys)} unique keys: {dup_keys[:5]}{'...' if len(dup_keys)>5 else ''})."
        )

    # Clean numeric Sales ($) column
    raw_sales = res_df[sales_col].copy()
    if not pd.api.types.is_numeric_dtype(res_df[sales_col]):
        cleaned_sales = (
            res_df[sales_col]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        res_df[sales_col] = pd.to_numeric(cleaned_sales, errors="coerce")
    else:
        res_df[sales_col] = pd.to_numeric(res_df[sales_col], errors="coerce")

    # Clean numeric Hits column
    raw_hits = res_df[hits_col].copy()
    if not pd.api.types.is_numeric_dtype(res_df[hits_col]):
        cleaned_hits = (
            res_df[hits_col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        res_df[hits_col] = pd.to_numeric(cleaned_hits, errors="coerce")
    else:
        res_df[hits_col] = pd.to_numeric(res_df[hits_col], errors="coerce")

    # Check for non-numeric or NaN values
    nan_sales = res_df[res_df[sales_col].isna()]
    if not nan_sales.empty:
        warnings.append(f"Found {len(nan_sales)} row(s) with non-numeric or blank {sales_col}; treated as 0.0.")
        res_df[sales_col] = res_df[sales_col].fillna(0.0)

    nan_hits = res_df[res_df[hits_col].isna()]
    if not nan_hits.empty:
        warnings.append(f"Found {len(nan_hits)} row(s) with non-numeric or blank {hits_col}; treated as 0.0.")
        res_df[hits_col] = res_df[hits_col].fillna(0.0)

    # Assign internal stable row index for safe merging even if duplicate item numbers exist
    res_df["_row_id"] = np.arange(len(res_df))

    total_sales = float(res_df[sales_col].sum())
    total_hits = float(res_df[hits_col].sum())

    if total_sales <= 0:
        warnings.append("Total Sales ($) across all items is <= 0; all items assigned Class D for sales.")
    if total_hits <= 0:
        warnings.append("Total Hits across all items is <= 0; all items assigned Class D for hits.")

    # =========================================================================
    # STEP 1: Classify Sales and Hits Separately
    # =========================================================================

    # --- Sales Ranking ---
    # Sort descending by sales, tie-break by Item Number ascending
    df_sales = res_df.sort_values(by=[sales_col, item_col], ascending=[False, True]).copy()
    if total_sales > 0:
        df_sales["_sales_running_sum"] = df_sales[sales_col].cumsum()
        df_sales["Sales Cumulative %"] = (df_sales["_sales_running_sum"] / total_sales) * 100.0

        def get_sales_class(cum_pct):
            val = round(cum_pct, 9)
            if val <= 60.0:
                return "A"
            elif val <= 80.0:
                return "B"
            elif val <= 90.0:
                return "C"
            else:
                return "D"

        df_sales["Sales Class"] = df_sales["Sales Cumulative %"].apply(get_sales_class)
    else:
        df_sales["Sales Cumulative %"] = 100.0
        df_sales["Sales Class"] = "D"

    score_map = {"A": 4, "B": 3, "C": 2, "D": 1}
    df_sales["Sales Score"] = df_sales["Sales Class"].map(score_map)

    # --- Hits Ranking ---
    # Sort descending by hits, tie-break by Item Number ascending
    df_hits = res_df.sort_values(by=[hits_col, item_col], ascending=[False, True]).copy()
    if total_hits > 0:
        df_hits["_hits_running_sum"] = df_hits[hits_col].cumsum()
        df_hits["Hits Cumulative %"] = (df_hits["_hits_running_sum"] / total_hits) * 100.0

        def get_hits_class(cum_pct):
            val = round(cum_pct, 9)
            if val <= 60.0:
                return "A"
            elif val <= 80.0:
                return "B"
            elif val <= 90.0:
                return "C"
            else:
                return "D"

        df_hits["Hits Class"] = df_hits["Hits Cumulative %"].apply(get_hits_class)
    else:
        df_hits["Hits Cumulative %"] = 100.0
        df_hits["Hits Class"] = "D"

    df_hits["Hits Score"] = df_hits["Hits Class"].map(score_map)

    # Merge results back to original DataFrame by _row_id (guarantees Item Number alignment and zero row loss)
    sales_merge_cols = ["_row_id", "Sales Cumulative %", "Sales Class", "Sales Score"]
    hits_merge_cols = ["_row_id", "Hits Cumulative %", "Hits Class", "Hits Score"]

    merged_df = res_df.merge(df_sales[sales_merge_cols], on="_row_id", how="left")
    merged_df = merged_df.merge(df_hits[hits_merge_cols], on="_row_id", how="left")
    merged_df = merged_df.drop(columns=["_row_id"])

    # =========================================================================
    # STEP 2: Combined Weighted Score
    # Combined Weighted Score = 0.65 * (Sales Score) + 0.35 * (Hits Score)
    # Check: Sales Class A (4) & Hits Class B (3) => 0.65*4 + 0.35*3 = 3.65
    # =========================================================================
    merged_df["Combined Weighted Score"] = (0.65 * merged_df["Sales Score"]) + (0.35 * merged_df["Hits Score"])

    # =========================================================================
    # STEP 3: Final Inventory Class
    # - A: score >= 3.25
    # - B: score >= 2.50 and < 3.25
    # - C: score >= 1.75 and < 2.50
    # - D: score < 1.75
    # Use rounded values (round(score, 6)) to guard against floating-point error
    # =========================================================================
    def assign_final_inventory_class(score):
        s_guard = round(float(score), 6)
        if s_guard >= 3.25:
            return "A"
        elif s_guard >= 2.50:
            return "B"
        elif s_guard >= 1.75:
            return "C"
        else:
            return "D"

    merged_df["Final Inventory Class"] = merged_df["Combined Weighted Score"].apply(assign_final_inventory_class)

    # Order rows by Item Number or descending Sales for clear presentation
    classified_df = merged_df.copy()

    # =========================================================================
    # REQUIRED CLASSIFICATION SUMMARY TABLE (One row per class + Total row)
    # =========================================================================
    total_items = len(classified_df)
    summary_rows = []
    for cls in ["A", "B", "C", "D"]:
        cls_subset = classified_df[classified_df["Final Inventory Class"] == cls]
        cls_count = int(len(cls_subset))
        cls_sales = float(cls_subset[sales_col].sum())
        summary_rows.append({
            "Final Class": cls,
            "Number of Items": cls_count,
            "Total Sales ($)": cls_sales,
        })

    sum_count = sum(r["Number of Items"] for r in summary_rows)
    sum_sales = sum(r["Total Sales ($)"] for r in summary_rows)

    summary_rows.append({
        "Final Class": "Total",
        "Number of Items": sum_count,
        "Total Sales ($)": sum_sales,
    })

    summary_df = pd.DataFrame(summary_rows)

    # Validation confirmations
    count_reconciled = (sum_count == total_items)
    sales_reconciled = abs(sum_sales - total_sales) < 1e-4

    validation_messages = [
        f"Validation Check 1 (Item Counts): Sum of class items ({sum_count:,}) matches total dataset rows ({total_items:,}) -> {'PASSED' if count_reconciled else 'FAILED'}",
        f"Validation Check 2 (Sales Reconciliation): Sum of class sales (${sum_sales:,.2f}) matches total sales (${total_sales:,.2f}) -> {'PASSED' if sales_reconciled else 'FAILED'}",
    ]

    metrics = {
        "total_items": total_items,
        "total_sales": total_sales,
        "total_hits": total_hits,
        "count_reconciled": count_reconciled,
        "sales_reconciled": sales_reconciled,
        "class_a_items": int(summary_df.loc[summary_df["Final Class"] == "A", "Number of Items"].iloc[0]),
        "class_b_items": int(summary_df.loc[summary_df["Final Class"] == "B", "Number of Items"].iloc[0]),
        "class_c_items": int(summary_df.loc[summary_df["Final Class"] == "C", "Number of Items"].iloc[0]),
        "class_d_items": int(summary_df.loc[summary_df["Final Class"] == "D", "Number of Items"].iloc[0]),
        "class_a_sales": float(summary_df.loc[summary_df["Final Class"] == "A", "Total Sales ($)"].iloc[0]),
        "class_b_sales": float(summary_df.loc[summary_df["Final Class"] == "B", "Total Sales ($)"].iloc[0]),
        "class_c_sales": float(summary_df.loc[summary_df["Final Class"] == "C", "Total Sales ($)"].iloc[0]),
        "class_d_sales": float(summary_df.loc[summary_df["Final Class"] == "D", "Total Sales ($)"].iloc[0]),
    }

    return {
        "classified_df": classified_df,
        "summary_df": summary_df,
        "validation_messages": validation_messages,
        "warnings": warnings,
        "metrics": metrics,
        "sales_col": sales_col,
        "hits_col": hits_col,
        "item_col": item_col,
    }


