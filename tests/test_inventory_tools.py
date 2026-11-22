"""
Unit and Integration Tests for Deterministic Inventory Tools.

Covers:
1. calculate_eoq()
2. calculate_quantity_discount()
3. calculate_rop() (all 4 cases: constant/variable demand & constant/variable lead time)
4. calculate_periodic_review()
5. calculate_newsvendor()
6. Input validation, intermediate calculations, final answer, assumptions, and agent integration.
"""

import pytest
import math
from scipy.stats import norm
import inventory_tools as it
from inventory_agent import InventoryDecisionAgent, InventoryModelType


# =============================================================================
# 1. calculate_eoq() TESTS
# =============================================================================
def test_calculate_eoq_standard():
    res = it.calculate_eoq(demand=10000, order_cost=50, holding_cost=2)

    # Validate structured return
    assert "inputs" in res
    assert "intermediate_calculations" in res
    assert "results" in res
    assert "final_answer" in res
    assert "assumptions" in res
    assert len(res["assumptions"]) > 0

    # Validate math: Q = sqrt(2 * 10000 * 50 / 2) = sqrt(500000) = 707.11
    assert pytest.approx(res["results"]["eoq"], 0.01) == 707.11
    assert res["results"]["optimal_order_quantity_int"] == 707

    # Check intermediate calculations
    inter = res["intermediate_calculations"]
    assert pytest.approx(inter["annual_ordering_cost"], 0.01) == 707.11
    assert pytest.approx(inter["annual_holding_cost"], 0.01) == 707.11
    assert pytest.approx(inter["total_inventory_cost"], 0.01) == 1414.21
    assert pytest.approx(inter["orders_per_year"], 0.01) == 14.14


def test_calculate_eoq_with_unit_cost_and_lead_time():
    res = it.calculate_eoq(
        demand=12000,
        order_cost=60,
        holding_cost=3,
        unit_cost=15,
        work_days_per_year=240,
        lead_time_days=4,
    )
    # daily demand = 12000 / 240 = 50 -> ROP = 50 * 4 = 200
    assert res["results"]["reorder_point"] == 200.0
    assert res["results"]["annual_purchase_cost"] == 180000.0
    assert res["results"]["total_annual_cost"] == 180000.0 + res["results"]["total_inventory_cost"]


def test_calculate_eoq_input_validation():
    # Missing required inputs (never invent inputs)
    with pytest.raises(ValueError, match="Missing required input"):
        it.calculate_eoq(demand=None, order_cost=50, holding_cost=2)
    with pytest.raises(ValueError, match="Annual demand must be a strictly positive number"):
        it.calculate_eoq(demand=-100, order_cost=50, holding_cost=2)
    with pytest.raises(ValueError, match="Order cost must be a strictly positive number"):
        it.calculate_eoq(demand=1000, order_cost=0, holding_cost=2)
    with pytest.raises(ValueError, match="Holding cost must be a strictly positive number"):
        it.calculate_eoq(demand=1000, order_cost=50, holding_cost=-1)


# =============================================================================
# 2. calculate_quantity_discount() TESTS
# =============================================================================
def test_calculate_quantity_discount_percentage_holding_cost():
    demand = 15000.0
    order_cost = 100.0
    holding_cost_rate = 0.20
    tiers = [
        {"min_qty": 0, "max_qty": 999, "unit_price": 10.0},
        {"min_qty": 1000, "max_qty": 2499, "unit_price": 9.0},
        {"min_qty": 2500, "max_qty": None, "unit_price": 8.0},
    ]

    res = it.calculate_quantity_discount(
        demand=demand,
        order_cost=order_cost,
        discount_tiers=tiers,
        holding_cost_rate=holding_cost_rate,
    )

    assert res["results"]["optimal_tier"] == 3
    assert res["results"]["optimal_order_quantity"] == 2500.0
    assert res["results"]["optimal_unit_price"] == 8.0
    assert res["results"]["minimum_total_annual_cost"] == 122600.0

    # Check intermediate calculations
    assert "intermediate_calculations" in res
    assert len(res["intermediate_calculations"]["evaluated_tiers"]) == 3
    assert "final_answer" in res
    assert "assumptions" in res


def test_calculate_quantity_discount_fixed_holding_cost():
    demand = 5000.0
    order_cost = 40.0
    fixed_holding_cost = 2.0
    tiers = [
        {"min_qty": 0, "max_qty": 499, "unit_price": 20.0},
        {"min_qty": 500, "max_qty": None, "unit_price": 18.0},
    ]

    res = it.calculate_quantity_discount(
        demand=demand,
        order_cost=order_cost,
        discount_tiers=tiers,
        fixed_holding_cost=fixed_holding_cost,
    )

    assert "minimum_total_annual_cost" in res["results"]
    assert len(res["results"]["tier_breakdown"]) == 2


def test_calculate_quantity_discount_validation():
    with pytest.raises(ValueError, match="Must provide at least one discount tier"):
        it.calculate_quantity_discount(demand=1000, order_cost=50, discount_tiers=[])
    with pytest.raises(ValueError, match="Must provide either 'holding_cost_rate'"):
        it.calculate_quantity_discount(
            demand=1000, order_cost=50, discount_tiers=[{"min_qty": 0, "max_qty": 100, "unit_price": 5}]
        )


# =============================================================================
# 3. calculate_rop() TESTS (All 4 Cases)
# =============================================================================
def test_calculate_rop_case1_constant_demand_constant_lead_time():
    # Case 1: sigma_d = 0, sigma_L = 0 -> SS = 0, ROP = d * L
    res = it.calculate_rop(avg_demand=50, lead_time=5, service_level=0.95, demand_std_dev=0.0, lead_time_std_dev=0.0)

    assert res["case"] == "constant demand and constant lead time"
    assert res["results"]["safety_stock"] == 0.0
    assert res["results"]["reorder_point"] == 250.0  # 50 * 5


def test_calculate_rop_case2_variable_demand_constant_lead_time():
    # Case 2: sigma_d > 0, sigma_L = 0 -> sigma_LT = sqrt(L) * sigma_d
    avg_d = 50.0
    lt = 5.0
    sig_d = 8.0
    res = it.calculate_rop(avg_demand=avg_d, lead_time=lt, service_level=0.95, demand_std_dev=sig_d, lead_time_std_dev=0.0)

    assert res["case"] == "variable demand and constant lead time"
    expected_sigma_lt = math.sqrt(5.0) * 8.0  # 17.8885
    assert pytest.approx(res["results"]["sigma_lead_time_demand"], 0.01) == expected_sigma_lt
    assert pytest.approx(res["results"]["safety_stock"], 0.01) == 1.64485 * expected_sigma_lt
    assert pytest.approx(res["results"]["reorder_point"], 0.01) == 250.0 + (1.64485 * expected_sigma_lt)


def test_calculate_rop_case3_constant_demand_variable_lead_time():
    # Case 3: sigma_d = 0, sigma_L > 0 -> sigma_LT = d * sigma_L
    avg_d = 40.0
    lt = 4.0
    sig_lt = 1.5
    res = it.calculate_rop(avg_demand=avg_d, lead_time=lt, service_level=0.95, demand_std_dev=0.0, lead_time_std_dev=sig_lt)

    assert res["case"] == "constant demand and variable lead time"
    expected_sigma_lt = 40.0 * 1.5  # 60.0
    assert pytest.approx(res["results"]["sigma_lead_time_demand"], 0.01) == expected_sigma_lt
    assert pytest.approx(res["results"]["safety_stock"], 0.01) == 1.64485 * 60.0


def test_calculate_rop_case4_variable_demand_variable_lead_time():
    # Case 4: sigma_d > 0, sigma_L > 0 -> sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)
    avg_d = 100.0
    lt = 4.0
    sig_d = 10.0
    sig_lt = 1.0
    z = 1.96

    res = it.calculate_rop(
        avg_demand=avg_d,
        lead_time=lt,
        z_score=z,
        demand_std_dev=sig_d,
        lead_time_std_dev=sig_lt,
    )

    assert res["case"] == "variable demand and variable lead time"
    # Var = 4 * 10^2 + 100^2 * 1^2 = 400 + 10000 = 10400
    # sigma_LT = sqrt(10400) = 101.9804
    expected_sigma_lt = math.sqrt(10400.0)
    assert pytest.approx(res["results"]["sigma_lead_time_demand"], 0.01) == expected_sigma_lt
    assert pytest.approx(res["results"]["safety_stock"], 0.05) == z * expected_sigma_lt
    assert pytest.approx(res["results"]["reorder_point"], 0.05) == 400.0 + (z * expected_sigma_lt)


def test_calculate_rop_validation():
    with pytest.raises(ValueError, match="Average demand cannot be negative"):
        it.calculate_rop(avg_demand=-10, lead_time=5, service_level=0.95)
    with pytest.raises(ValueError, match="Service level percentage must be strictly less than 100%"):
        it.calculate_rop(avg_demand=10, lead_time=5, service_level=105.0)
    with pytest.raises(ValueError, match="Service level must be strictly positive"):
        it.calculate_rop(avg_demand=10, lead_time=5, service_level=0.0)


# =============================================================================
# 4. calculate_periodic_review() TESTS
# =============================================================================
def test_calculate_periodic_review_standard():
    avg_demand = 100.0
    review_period = 7.0
    lead_time = 3.0
    demand_std_dev = 15.0
    service_level = 0.98
    current_inventory = 400.0
    on_order = 100.0

    res = it.calculate_periodic_review(
        avg_demand=avg_demand,
        review_period=review_period,
        lead_time=lead_time,
        demand_std_dev=demand_std_dev,
        service_level=service_level,
        current_inventory=current_inventory,
        on_order_inventory=on_order,
    )

    # Protection period = 7 + 3 = 10
    # Expected demand during protection = 100 * 10 = 1000
    # sigma_protection = 15 * sqrt(10) = 47.434
    # z = 2.0537
    # SS = 2.0537 * 47.434 = 97.42
    # Target inventory S = 1000 + 97.42 = 1097.42 -> 1098
    # Order Q = max(0, 1097.42 - 500) = 597.42 -> 598
    assert res["results"]["protection_interval"] == 10.0
    assert res["results"]["expected_protection_demand"] == 1000.0
    assert pytest.approx(res["results"]["safety_stock"], 0.01) == 97.42
    assert res["results"]["target_inventory_level_int"] == 1098
    assert res["results"]["order_quantity_int"] == 598

    # Validate structure
    assert "intermediate_calculations" in res
    assert "final_answer" in res
    assert "assumptions" in res


def test_calculate_periodic_review_inventory_surplus():
    res = it.calculate_periodic_review(
        avg_demand=20,
        review_period=5,
        lead_time=2,
        service_level=0.90,
        current_inventory=500,  # Exceeds target
    )
    assert res["results"]["order_quantity"] == 0.0


# =============================================================================
# 5. calculate_newsvendor() TESTS
# =============================================================================
def test_calculate_newsvendor_standard():
    selling_price = 120.0
    cost_price = 70.0
    salvage_value = 30.0
    penalty_cost = 0.0
    mean_demand = 500.0
    std_demand = 80.0

    res = it.calculate_newsvendor(
        selling_price=selling_price,
        cost_price=cost_price,
        salvage_value=salvage_value,
        penalty_cost=penalty_cost,
        mean_demand=mean_demand,
        std_demand=std_demand,
    )

    # Cu = 120 - 70 = 50
    # Co = 70 - 30 = 40
    # CR = 50 / (50 + 40) = 50/90 = 0.5556
    # z* = norm.ppf(0.5556) = 0.1397
    # Q* = 500 + 0.1397 * 80 = 511.18 -> 511
    assert res["results"]["underage_cost_Cu"] == 50.0
    assert res["results"]["overage_cost_Co"] == 40.0
    assert pytest.approx(res["results"]["critical_ratio"], 0.001) == 0.5556
    assert res["results"]["optimal_order_quantity_int"] == 511
    assert res["results"]["expected_profit"] > 0

    # Intermediate calculations
    assert "intermediate_calculations" in res
    assert "expected_lost_sales" in res["intermediate_calculations"]
    assert "final_answer" in res
    assert "assumptions" in res


def test_calculate_newsvendor_validation():
    with pytest.raises(ValueError, match="Selling price .* must exceed cost price"):
        it.calculate_newsvendor(selling_price=50, cost_price=60, mean_demand=100, std_demand=10)
    with pytest.raises(ValueError, match="Cost price .* must exceed salvage value"):
        it.calculate_newsvendor(selling_price=100, cost_price=50, salvage_value=60, mean_demand=100, std_demand=10)


# =============================================================================
# 6. AGENT PIPELINE INTEGRATION TESTS
# =============================================================================
def test_agent_identify_all_models():
    agent = InventoryDecisionAgent()

    queries = {
        InventoryModelType.EOQ: "What is the economic order quantity for an annual demand of 8000 units with order cost $40 and holding cost $1.50?",
        InventoryModelType.QUANTITY_DISCOUNT: "We get a quantity discount: Tier 1: 0-499 units at $20, Tier 2: 500+ at $18. Demand is 5000, setup cost is $50.",
        InventoryModelType.SAFETY_STOCK_ROP: "Daily demand is 40 units with std dev 5 units. Lead time is 4 days. Calculate safety stock and reorder point at 95% service level.",
        InventoryModelType.FIXED_PERIOD_REVIEW: "We review inventory every 14 days with lead time of 5 days. Average demand is 50 units, target inventory level.",
        InventoryModelType.NEWSVENDOR: "Calculate optimal order for a seasonal perishable product: selling price $80, cost price $40, salvage $15, mean demand 300 with std dev 60.",
    }

    for expected_model, q in queries.items():
        model_name, confidence = agent.identify_model(q)
        assert model_name == expected_model
        assert confidence >= 0.70


def test_agent_end_to_end_pipeline():
    agent = InventoryDecisionAgent()
    query = "Our annual demand is 10,000 units, each order costs $50, and holding cost is $2 per unit per year. What is the optimal EOQ?"
    resp = agent.process_query(query)

    assert resp.model_name == InventoryModelType.EOQ
    assert resp.tool_name == "calculate_eoq"
    assert resp.calculation_results is not None
    assert resp.verification is not None
    assert resp.verification.is_valid()
    assert "707" in resp.explanation


# =============================================================================
# 8. EIGHT PEDAGOGICAL VERIFICATION COMPONENTS TESTS (STUDENT VERIFICATION)
# =============================================================================
REQUIRED_VERIFICATION_KEYS = [
    "inputs",
    "variable_definitions",
    "formula",
    "formula_with_substituted_values",
    "intermediate_calculations",
    "final_answer",
    "verification_checks",
    "assumptions",
]


def test_eight_verification_components_eoq():
    res = it.calculate_eoq(demand=12000, order_cost=75, holding_cost=3)
    for key in REQUIRED_VERIFICATION_KEYS:
        assert key in res, f"Missing required verification key '{key}' in calculate_eoq()"

    assert isinstance(res["inputs"], dict)
    assert res["inputs"]["demand (D)"] == 12000.0
    assert res["inputs"]["order_cost (S)"] == 75.0
    assert res["inputs"]["holding_cost (H)"] == 3.0

    assert "Q*" in res["variable_definitions"]
    assert "D" in res["variable_definitions"]
    assert "S" in res["variable_definitions"]
    assert "H" in res["variable_definitions"]

    assert "Q* = sqrt((2 * D * S) / H)" in res["formula"]
    assert "12,000.00" in res["formula_with_substituted_values"]
    assert "75.00" in res["formula_with_substituted_values"]
    assert "3.00" in res["formula_with_substituted_values"]

    assert "annual_ordering_cost" in res["intermediate_calculations"]
    assert "annual_holding_cost" in res["intermediate_calculations"]
    assert "775" in res["final_answer"]

    assert len(res["verification_checks"]) >= 3
    for chk in res["verification_checks"]:
        assert chk["status"] == "PASSED", f"Verification check failed: {chk}"
        assert "check" in chk
        assert "condition" in chk
        assert "calculated_value" in chk
        assert "student_note" in chk

    assert len(res["assumptions"]) >= 4


def test_eight_verification_components_quantity_discount():
    tiers = [
        {"min_qty": 0, "max_qty": 999, "unit_price": 10.0},
        {"min_qty": 1000, "max_qty": 2499, "unit_price": 9.0},
        {"min_qty": 2500, "max_qty": None, "unit_price": 8.0},
    ]
    res = it.calculate_quantity_discount(
        demand=15000, order_cost=100, discount_tiers=tiers, holding_cost_rate=0.20
    )
    for key in REQUIRED_VERIFICATION_KEYS:
        assert key in res, f"Missing required verification key '{key}' in calculate_quantity_discount()"

    assert "discount_tiers" in res["inputs"]
    assert "Q_candidate" in res["variable_definitions"]
    assert "TC(Q)" in res["formula"]
    assert "15,000" in res["formula_with_substituted_values"]
    assert "evaluated_tiers" in res["intermediate_calculations"]
    assert "Tier 3" in res["final_answer"]

    for chk in res["verification_checks"]:
        assert chk["status"] == "PASSED", f"Verification check failed: {chk}"

    assert len(res["assumptions"]) >= 3


def test_eight_verification_components_rop_all_four_cases():
    cases = [
        {"avg_demand": 50, "lead_time": 5, "service_level": 0.95, "demand_std_dev": 0.0, "lead_time_std_dev": 0.0, "exp_case": "constant demand and constant lead time"},
        {"avg_demand": 50, "lead_time": 5, "service_level": 0.95, "demand_std_dev": 8.0, "lead_time_std_dev": 0.0, "exp_case": "variable demand and constant lead time"},
        {"avg_demand": 40, "lead_time": 4, "service_level": 0.95, "demand_std_dev": 0.0, "lead_time_std_dev": 1.5, "exp_case": "constant demand and variable lead time"},
        {"avg_demand": 100, "lead_time": 4, "service_level": 0.95, "demand_std_dev": 10.0, "lead_time_std_dev": 1.0, "exp_case": "variable demand and variable lead time"},
    ]

    for case_data in cases:
        exp_case = case_data.pop("exp_case")
        res = it.calculate_rop(**case_data)
        for key in REQUIRED_VERIFICATION_KEYS:
            assert key in res, f"Missing required key '{key}' for ROP case '{exp_case}'"
        assert res["case"] == exp_case
        assert "ROP" in res["variable_definitions"]
        assert "SS" in res["variable_definitions"]
        assert "Reorder Point (ROP) = " in res["formula"]
        assert len(res["verification_checks"]) >= 3
        for chk in res["verification_checks"]:
            assert chk["status"] == "PASSED", f"Verification check failed in '{exp_case}': {chk}"
        assert len(res["assumptions"]) >= 4


def test_eight_verification_components_periodic_review():
    res = it.calculate_periodic_review(
        avg_demand=100,
        review_period=7,
        lead_time=3,
        demand_std_dev=15,
        service_level=0.98,
        current_inventory=400,
        on_order_inventory=100,
        backorders=10,
    )
    for key in REQUIRED_VERIFICATION_KEYS:
        assert key in res, f"Missing required verification key '{key}' in calculate_periodic_review()"

    assert "review_period (R)" in res["inputs"]
    assert "backorders" in res["inputs"]
    assert "IP" in res["variable_definitions"]
    assert "S" in res["variable_definitions"]
    assert "Target Inventory Level (S)" in res["formula"]
    assert "1,000.00 units" in res["formula_with_substituted_values"]
    assert "protection_period" in res["intermediate_calculations"]
    assert "target_inventory_exact" in res["intermediate_calculations"]
    assert "inventory_position" in res["intermediate_calculations"]
    assert "order_quantity_exact" in res["intermediate_calculations"]

    for chk in res["verification_checks"]:
        assert chk["status"] == "PASSED", f"Verification check failed: {chk}"

    assert len(res["assumptions"]) >= 4


def test_eight_verification_components_newsvendor():
    res = it.calculate_newsvendor(
        selling_price=150,
        cost_price=80,
        salvage_value=30,
        penalty_cost=10,
        mean_demand=400,
        std_demand=60,
    )
    for key in REQUIRED_VERIFICATION_KEYS:
        assert key in res, f"Missing required verification key '{key}' in calculate_newsvendor()"

    assert "selling_price (p)" in res["inputs"]
    assert "cost_price (c)" in res["inputs"]
    assert "Cu" in res["variable_definitions"]
    assert "Co" in res["variable_definitions"]
    assert "Critical Ratio (CR) = Cu / (Cu + Co)" in res["formula"]
    assert "Underage Cost (Cu) = 80.00" in res["formula_with_substituted_values"]
    assert "Critical Ratio (CR) = 80.00 / (80.00 + 50.00)" in res["formula_with_substituted_values"]
    assert "critical_ratio" in res["intermediate_calculations"]
    assert "z_star" in res["intermediate_calculations"]
    assert "expected_sales" in res["intermediate_calculations"]
    assert "expected_leftover_units" in res["intermediate_calculations"]
    assert "expected_lost_sales" in res["intermediate_calculations"]

    for chk in res["verification_checks"]:
        assert chk["status"] == "PASSED", f"Verification check failed: {chk}"

    assert len(res["assumptions"]) >= 4

def test_eoq_ordering_cost_synonyms():
    agent = InventoryDecisionAgent()
    synonym_phrasings = [
        "Annual demand is 5000 units, ordering cost is $60, and holding cost is $2.50.",
        "Annual demand is 5000 units, order cost is $60, and holding cost is $2.50.",
        "Annual demand is 5000 units, setup cost is $60, and holding cost is $2.50.",
        "Annual demand is 5000 units, setup expense is $60, and holding cost is $2.50.",
        "Annual demand is 5000 units, cost per order is $60, and holding cost is $2.50.",
        "Annual demand is 5000 units, cost of placing an order is $60, and holding cost is $2.50.",
        "Annual demand is 5000 units, ordering/setup cost is $60, and holding cost is $2.50.",
    ]

    for phrase in synonym_phrasings:
        resp = agent.process_query(phrase)
        assert resp.model_name == InventoryModelType.EOQ
        assert resp.missing_inputs == [], f"Failed to extract order_cost from: '{phrase}'"
        assert resp.extracted_inputs["order_cost"] == 60.0
        assert resp.extracted_inputs["demand"] == 5000.0
        assert resp.extracted_inputs["holding_cost"] == 2.50


def test_safety_stock_distinguish_parameters():
    agent = InventoryDecisionAgent()
    query = (
        "Average demand is 120 units per day with a demand standard deviation of 15 units. "
        "Average lead time is 6 days with a lead-time standard deviation of 1.2 days. "
        "We require a 95% cycle service level."
    )
    resp = agent.process_query(query)

    assert resp.model_name == InventoryModelType.SAFETY_STOCK_ROP
    assert resp.missing_inputs == []
    assert resp.extracted_inputs["avg_demand"] == 120.0
    assert resp.extracted_inputs["demand_std_dev"] == 15.0
    assert resp.extracted_inputs["lead_time"] == 6.0
    assert resp.extracted_inputs["lead_time_std_dev"] == 1.2
    assert resp.extracted_inputs["service_level"] == 95.0
    assert resp.calculation_results["case"] == "variable demand and variable lead time"


def test_quantity_discount_distinguish_parameters():
    agent = InventoryDecisionAgent()
    query = (
        "Annual demand is 10000 units. Cost of placing an order is $80. "
        "Percentage holding cost is 18%. Price breaks: 0-999 units at $25, 1000+ at $22."
    )
    resp = agent.process_query(query)

    assert resp.model_name == InventoryModelType.QUANTITY_DISCOUNT
    assert resp.missing_inputs == []
    assert resp.extracted_inputs["demand"] == 10000.0
    assert resp.extracted_inputs["order_cost"] == 80.0
    assert resp.extracted_inputs["holding_cost_rate"] == 0.18
    assert len(resp.extracted_inputs["discount_tiers"]) == 2


def test_fixed_period_review_distinguish_parameters():
    agent = InventoryDecisionAgent()
    query = (
        "Daily demand is 80 units with demand standard deviation of 10. "
        "Review period is 14 days and replenishment lead time is 4 days. "
        "Target service level is 95%. On-hand inventory is 350 units, on-order inventory is 100 units, "
        "and backorders are 20 units."
    )
    resp = agent.process_query(query)

    assert resp.model_name == InventoryModelType.FIXED_PERIOD_REVIEW
    assert resp.missing_inputs == []
    assert resp.extracted_inputs["avg_demand"] == 80.0
    assert resp.extracted_inputs["demand_std_dev"] == 10.0
    assert resp.extracted_inputs["review_period"] == 14.0
    assert resp.extracted_inputs["lead_time"] == 4.0
    assert resp.extracted_inputs["current_inventory"] == 350.0
    assert resp.extracted_inputs["on_order_inventory"] == 100.0
    assert resp.extracted_inputs["backorders"] == 20.0


def test_newsvendor_distinguish_parameters():
    agent = InventoryDecisionAgent()
    query = (
        "Selling price is $150, purchase cost is $80, salvage value is $30, "
        "penalty cost is $10. Mean demand is 400 units with demand standard deviation of 60 units."
    )
    resp = agent.process_query(query)

    assert resp.model_name == InventoryModelType.NEWSVENDOR
    assert resp.missing_inputs == []
    assert resp.extracted_inputs["selling_price"] == 150.0
    assert resp.extracted_inputs["cost_price"] == 80.0
    assert resp.extracted_inputs["salvage_value"] == 30.0
    assert resp.extracted_inputs["penalty_cost"] == 10.0
    assert resp.extracted_inputs["mean_demand"] == 400.0
    assert resp.extracted_inputs["std_demand"] == 60.0


# =============================================================================
# 9. CSV DATA PROCESSING & ANALYSIS TESTS
# =============================================================================
def test_csv_inventory_data_analysis():
    import io
    import pandas as pd

    csv_data = """Product,Category,Units,Unit Cost
Widget A,Electronics,500,$12.50
Widget B,Electronics,200,$25.00
Tool X,Hardware,1500,$4.00
Tool Y,Hardware,300,$15.00
"""
    df = pd.read_csv(io.StringIO(csv_data))

    # Schema validation
    required_cols = ["Product", "Category", "Units", "Unit Cost"]
    for col in required_cols:
        assert col in df.columns

    # Clean numeric
    df["Units"] = pd.to_numeric(df["Units"], errors="coerce")
    df["Unit Cost"] = pd.to_numeric(
        df["Unit Cost"].astype(str).str.replace("$", "").str.replace(",", "").str.strip(),
        errors="coerce",
    )

    # 1. Total units
    total_units = df["Units"].sum()
    assert total_units == 2500

    # 2. Average unit cost: (12.50 + 25.00 + 4.00 + 15.00) / 4 = 56.50 / 4 = 14.125
    assert pytest.approx(df["Unit Cost"].mean(), 0.01) == 14.125

    # 3. Product with highest units: Tool X (1500)
    max_idx = df["Units"].idxmax()
    assert df.loc[max_idx, "Product"] == "Tool X"
    assert df.loc[max_idx, "Units"] == 1500

    # 4. Group by Category
    cat_summary = df.groupby("Category")["Units"].sum().to_dict()
    assert cat_summary["Electronics"] == 700
    assert cat_summary["Hardware"] == 1800

    # 5. Inventory Value = Units * Unit Cost
    df["Inventory Value"] = df["Units"] * df["Unit Cost"]
    assert df.loc[0, "Inventory Value"] == 500 * 12.50  # 6250
    assert df.loc[1, "Inventory Value"] == 200 * 25.00  # 5000
    assert df.loc[2, "Inventory Value"] == 1500 * 4.00  # 6000
    assert df.loc[3, "Inventory Value"] == 300 * 15.00  # 4500

    # 6. Total Inventory Value: 6250 + 5000 + 6000 + 4500 = 21750
    assert df["Inventory Value"].sum() == 21750.0


def test_csv_inventory_missing_columns():
    import io
    import pandas as pd

    invalid_csv = """Item,Category,Quantity
Widget A,Electronics,500
"""
    df = pd.read_csv(io.StringIO(invalid_csv))
    required_cols = ["Product", "Category", "Units", "Unit Cost"]
    missing = [c for c in required_cols if c not in df.columns]
    assert "Product" in missing
    assert "Units" in missing
    assert "Unit Cost" in missing


# =============================================================================
# 10. MULTICRITERIA INVENTORY CLASSIFICATION TESTS
# =============================================================================
def test_classify_inventory_multicriteria_hand_calculated():
    import io
    import pandas as pd

    # Small hand-calculated dataset designed to test:
    # 1. Exact cumulative percentage boundaries: 60%, 80%, 90%
    # 2. Example check score: Sales A (4) & Hits B (3) => 0.65*4 + 0.35*3 = 3.65
    # 3. Deterministic tie-breaking on Item Number ascending
    # 4. Final inventory classes A, B, C, D
    # 5. Item counts and sales reconciliation
    csv_data = """Item Number,Item Name,UoM,Sales ($),Hits
SKU-A,Precision Spindle,EA,$550.00,20
SKU-B,Servo Drive,EA,$150.00,55
SKU-C,Sensor Cable,EA,$100.00,5
SKU-D,Terminal Block,EA,$100.00,10
SKU-E,Coupling Ring,EA,$100.00,10
"""
    # Total Sales = 550 + 150 + 100 + 100 + 100 = 1000.00
    # Total Hits = 20 + 55 + 5 + 10 + 10 = 100
    df = pd.read_csv(io.StringIO(csv_data))
    res = it.classify_inventory_multicriteria(df)

    clf_df = res["classified_df"]
    summary_df = res["summary_df"]

    # Verify column existence
    required_cols = [
        "Sales Cumulative %", "Sales Class", "Sales Score",
        "Hits Cumulative %", "Hits Class", "Hits Score",
        "Combined Weighted Score", "Final Inventory Class"
    ]
    for col in required_cols:
        assert col in clf_df.columns, f"Missing column {col}"

    # 1. Sales Ranking & Boundaries:
    # Ranked order: SKU-A (550), SKU-B (150), SKU-C (100), SKU-D (100), SKU-E (100)
    # Tie-break on Sales=100: SKU-C < SKU-D < SKU-E
    # Cumulative Sales %:
    # SKU-A: 550/1000 = 55.0% -> <= 60% -> Class A (Score 4)
    # SKU-B: 700/1000 = 70.0% -> > 60% & <= 80% -> Class B (Score 3)
    # SKU-C: 800/1000 = 80.0% -> exactly 80% boundary -> Class B (Score 3)
    # SKU-D: 900/1000 = 90.0% -> exactly 90% boundary -> Class C (Score 2)
    # SKU-E: 1000/1000 = 100.0% -> > 90% -> Class D (Score 1)
    sku_a = clf_df[clf_df["Item Number"] == "SKU-A"].iloc[0]
    sku_b = clf_df[clf_df["Item Number"] == "SKU-B"].iloc[0]
    sku_c = clf_df[clf_df["Item Number"] == "SKU-C"].iloc[0]
    sku_d = clf_df[clf_df["Item Number"] == "SKU-D"].iloc[0]
    sku_e = clf_df[clf_df["Item Number"] == "SKU-E"].iloc[0]

    assert sku_a["Sales Class"] == "A"
    assert sku_a["Sales Score"] == 4
    assert pytest.approx(sku_a["Sales Cumulative %"], 0.01) == 55.0

    assert sku_b["Sales Class"] == "B"
    assert sku_b["Sales Score"] == 3
    assert pytest.approx(sku_b["Sales Cumulative %"], 0.01) == 70.0

    assert sku_c["Sales Class"] == "B"
    assert sku_c["Sales Score"] == 3
    assert pytest.approx(sku_c["Sales Cumulative %"], 0.01) == 80.0

    assert sku_d["Sales Class"] == "C"
    assert sku_d["Sales Score"] == 2
    assert pytest.approx(sku_d["Sales Cumulative %"], 0.01) == 90.0

    assert sku_e["Sales Class"] == "D"
    assert sku_e["Sales Score"] == 1
    assert pytest.approx(sku_e["Sales Cumulative %"], 0.01) == 100.0

    # 2. Hits Ranking & Boundaries:
    # Ranked order: SKU-B (55), SKU-A (20), SKU-D (10), SKU-E (10), SKU-C (5)
    # Tie-break on Hits=10: SKU-D < SKU-E
    # Cumulative Hits %:
    # SKU-B: 55/100 = 55.0% -> Class A (Score 4)
    # SKU-A: 75/100 = 75.0% -> Class B (Score 3)
    # SKU-D: 85/100 = 85.0% -> Class C (Score 2)
    # SKU-E: 95/100 = 95.0% -> Class D (Score 1)
    # SKU-C: 100/100 = 100.0% -> Class D (Score 1)
    assert sku_b["Hits Class"] == "A"
    assert sku_b["Hits Score"] == 4

    assert sku_a["Hits Class"] == "B"
    assert sku_a["Hits Score"] == 3

    assert sku_d["Hits Class"] == "C"
    assert sku_d["Hits Score"] == 2

    assert sku_e["Hits Class"] == "D"
    assert sku_e["Hits Score"] == 1

    assert sku_c["Hits Class"] == "D"
    assert sku_c["Hits Score"] == 1

    # 3. Example Check Score: Sales Class A (4) & Hits Class B (3) => 0.65*4 + 0.35*3 = 3.65
    assert pytest.approx(sku_a["Combined Weighted Score"], 0.001) == 3.65
    assert sku_a["Final Inventory Class"] == "A"  # 3.65 >= 3.25

    # 4. Final Classification Scores:
    # SKU-B: 0.65(3) + 0.35(4) = 1.95 + 1.40 = 3.35 -> Class A
    assert pytest.approx(sku_b["Combined Weighted Score"], 0.001) == 3.35
    assert sku_b["Final Inventory Class"] == "A"

    # SKU-C: 0.65(3) + 0.35(1) = 1.95 + 0.35 = 2.30 -> Class C (>= 1.75 and < 2.50)
    assert pytest.approx(sku_c["Combined Weighted Score"], 0.001) == 2.30
    assert sku_c["Final Inventory Class"] == "C"

    # SKU-D: 0.65(2) + 0.35(2) = 1.30 + 0.70 = 2.00 -> Class C (>= 1.75 and < 2.50)
    assert pytest.approx(sku_d["Combined Weighted Score"], 0.001) == 2.00
    assert sku_d["Final Inventory Class"] == "C"

    # SKU-E: 0.65(1) + 0.35(1) = 0.65 + 0.35 = 1.00 -> Class D (< 1.75)
    assert pytest.approx(sku_e["Combined Weighted Score"], 0.001) == 1.00
    assert sku_e["Final Inventory Class"] == "D"

    # 5. Summary Table Validation & Reconciliation
    assert len(summary_df) == 5  # A, B, C, D, Total
    assert set(summary_df["Final Class"].tolist()) == {"A", "B", "C", "D", "Total"}

    total_row = summary_df[summary_df["Final Class"] == "Total"].iloc[0]
    assert total_row["Number of Items"] == 5
    assert pytest.approx(total_row["Total Sales ($)"], 0.01) == 1000.00

    # Class A has 2 items (SKU-A: $550 + SKU-B: $150 = $700)
    class_a_row = summary_df[summary_df["Final Class"] == "A"].iloc[0]
    assert class_a_row["Number of Items"] == 2
    assert pytest.approx(class_a_row["Total Sales ($)"], 0.01) == 700.00

    # Reconciliation validation checks in result
    assert res["metrics"]["count_reconciled"] is True
    assert res["metrics"]["sales_reconciled"] is True


def test_classify_inventory_multicriteria_validation():
    import pandas as pd

    # Empty df raises ValueError
    with pytest.raises(ValueError, match="empty or None"):
        it.classify_inventory_multicriteria(pd.DataFrame())

    # Missing sales column raises ValueError
    bad_df = pd.DataFrame({"Item Number": ["A", "B"], "Hits": [10, 20]})
    with pytest.raises(ValueError, match="Sales column not found"):
        it.classify_inventory_multicriteria(bad_df)

    # Missing hits column raises ValueError
    bad_df2 = pd.DataFrame({"Item Number": ["A", "B"], "Sales ($)": [100, 200]})
    with pytest.raises(ValueError, match="Hits column not found"):
        it.classify_inventory_multicriteria(bad_df2)


def test_classify_inventory_multicriteria_non_numeric_and_duplicates():
    import io
    import pandas as pd

    # Test handling of duplicate Item Numbers, blanks, and non-numeric values without crashing
    csv_data = """Item Number,Item Name,UoM,Sales ($),Hits
SKU-01,Part 1,EA,$300.00,40
SKU-01,Part 1 Dup,EA,$200.00,30
SKU-02,Part 2,EA,N/A,15
SKU-03,Part 3,EA,$100.00,INVALID
,Blank Part,EA,$50.00,5
"""
    df = pd.read_csv(io.StringIO(csv_data))
    res = it.classify_inventory_multicriteria(df)

    assert "classified_df" in res
    assert len(res["classified_df"]) == 5  # No rows dropped
    assert len(res["warnings"]) >= 3  # Warnings generated for duplicates, non-numeric, blanks
    assert res["metrics"]["count_reconciled"] is True




