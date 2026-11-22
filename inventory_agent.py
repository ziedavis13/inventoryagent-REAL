"""
Supply Chain Analytics Inventory Decision Agent.

Architecture Pipeline:
1. Customer Question  -> Receives natural-language query
2. Identify Model     -> Classifies query into 1 of 5 inventory models
3. Extract Inputs     -> Extracts numeric parameters via NLP pattern matching
4. Select Python Tool -> Maps to deterministic calculation function in inventory_tools
5. Calculate          -> Computes exact figures using Python (SciPy/NumPy)
6. Verify             -> Performs mathematical & domain sanity checks
7. Explain            -> Generates executive summary, formula breakdown & managerial insights
"""

import re
import math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Callable
import inventory_tools as it


class InventoryModelType:
    EOQ = "Economic Order Quantity (EOQ)"
    QUANTITY_DISCOUNT = "Quantity Discount Model"
    SAFETY_STOCK_ROP = "Safety Stock & Reorder Point"
    FIXED_PERIOD_REVIEW = "Fixed-Period Review (Periodic Review / (R, S))"
    NEWSVENDOR = "Newsvendor Model (Single-Period Stochastic)"


@dataclass
class VerificationResult:
    status: str  # "PASSED", "WARNING", "FAILED"
    checks_passed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    audit_notes: List[str] = field(default_factory=list)

    def is_valid(self) -> bool:
        return self.status in ["PASSED", "WARNING"]


@dataclass
class AgentResponse:
    question: str
    model_name: str
    confidence: float
    extracted_inputs: Dict[str, Any]
    missing_inputs: List[str]
    tool_name: str
    calculation_results: Optional[Dict[str, Any]]
    verification: Optional[VerificationResult]
    explanation: str
    pipeline_trace: List[Dict[str, Any]] = field(default_factory=list)


class InventoryDecisionAgent:
    """Autonomous agent orchestrating natural language inventory analysis."""

    def __init__(self):
        # Keywords mapping for model identification
        self.model_keywords = {
            InventoryModelType.NEWSVENDOR: [
                "newsvendor", "newsboy", "single period", "perishable", "seasonal",
                "salvage", "markdown", "underage", "overage", "critical ratio",
                "spoilage", "fashion", "short life cycle", "leftover", "excess inventory"
            ],
            InventoryModelType.QUANTITY_DISCOUNT: [
                "quantity discount", "discount tier", "discount tiers", "price break", "price breaks",
                "bracket", "brackets", "tier 1", "tier 2", "tier 3", "tiers",
                "bulk discount", "volume discount", "unit price tier", "price schedule",
                "all-units discount", "cheaper in bulk"
            ],
            InventoryModelType.FIXED_PERIOD_REVIEW: [
                "fixed period", "fixed-period", "periodic review", "review period",
                "order up to", "order-up-to", "target inventory", "(r, s)", "(r,s)",
                "replenishment cycle", "review interval", "protection interval",
                "periodic inventory", "review every", "review inventory every"
            ],
            InventoryModelType.SAFETY_STOCK_ROP: [
                "safety stock", "reorder point", "rop", "buffer stock",
                "service level", "cycle service level", "lead time demand",
                "stockout risk", "standard deviation of demand", "demand uncertainty",
                "lead time uncertainty", "z-score", "csl"
            ],
            InventoryModelType.EOQ: [
                "eoq", "economic order quantity", "order cost", "setup cost",
                "holding cost", "carrying cost", "annual demand", "batch size",
                "optimal order size", "how many units should we order", "orders per year",
                "cycle time"
            ]
        }

    # =========================================================================
    # STAGE 1: IDENTIFY MODEL
    # =========================================================================
    def identify_model(self, query: str) -> Tuple[str, float]:
        """
        Classify customer question into one of the 5 supported inventory models.
        Returns: (model_name, confidence_score [0.0 - 1.0])
        """
        q_lower = query.lower()
        scores: Dict[str, float] = {
            InventoryModelType.NEWSVENDOR: 0.0,
            InventoryModelType.QUANTITY_DISCOUNT: 0.0,
            InventoryModelType.FIXED_PERIOD_REVIEW: 0.0,
            InventoryModelType.SAFETY_STOCK_ROP: 0.0,
            InventoryModelType.EOQ: 0.0,
        }

        # Strong keyword / pattern boosts
        if re.search(r"\b(newsvendor|newsboy|single[- ]period|salvage value|salvage|critical ratio|perishable|seasonal demand)\b", q_lower):
            scores[InventoryModelType.NEWSVENDOR] += 6.0

        if re.search(r"\b(quantity discount|price breaks?|discount tiers?|brackets?|tier\s*\d+|price schedule)\b", q_lower):
            scores[InventoryModelType.QUANTITY_DISCOUNT] += 6.0

        if re.search(r"\b(fixed[- ]period|periodic review|review period|review interval|order[- ]up[- ]to|target inventory|review (?:inventory )?every)\b", q_lower):
            scores[InventoryModelType.FIXED_PERIOD_REVIEW] += 6.0

        if re.search(r"\b(safety stock|reorder point|rop|cycle service level)\b", q_lower):
            scores[InventoryModelType.SAFETY_STOCK_ROP] += 5.0

        if re.search(r"\b(eoq|economic order quantity)\b", q_lower):
            scores[InventoryModelType.EOQ] += 4.5

        # Weighted keyword frequency scoring
        for model, keywords in self.model_keywords.items():
            for kw in keywords:
                if kw in q_lower:
                    scores[model] += 1.0

        # Disambiguation rules
        # If Quantity Discount keywords exist, it supersedes plain EOQ
        if scores[InventoryModelType.QUANTITY_DISCOUNT] > 2.0:
            scores[InventoryModelType.EOQ] = max(0.0, scores[InventoryModelType.EOQ] - 3.0)

        # If Fixed Period Review keywords exist (e.g. review every, target inventory), it supersedes Safety Stock ROP
        if scores[InventoryModelType.FIXED_PERIOD_REVIEW] > 3.0:
            scores[InventoryModelType.SAFETY_STOCK_ROP] = max(0.0, scores[InventoryModelType.SAFETY_STOCK_ROP] - 3.0)

        # If Safety Stock / ROP is specifically asked, it supersedes plain EOQ
        if re.search(r"\b(safety stock|reorder point|rop)\b", q_lower):
            scores[InventoryModelType.SAFETY_STOCK_ROP] += 3.0

        best_model = max(scores, key=scores.get)
        max_score = scores[best_model]

        if max_score == 0:
            return InventoryModelType.EOQ, 0.40

        confidence = min(0.99, max(0.55, 0.50 + (max_score * 0.10)))
        return best_model, round(confidence, 2)

    # =========================================================================
    # STAGE 2: EXTRACT INPUTS
    # =========================================================================
    def _extract_first(self, patterns: List[str], text: str) -> Optional[float]:
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                clean_str = match.group(1).replace(",", "").replace("$", "").strip()
                try:
                    return float(clean_str)
                except ValueError:
                    continue
        return None

    def extract_inputs(self, query: str, model: str) -> Tuple[Dict[str, Any], List[str]]:
        """
        Extract numeric inputs specific to the identified model, recognizing
        common inventory terminology and synonyms across the entire question.
        Returns: (inputs_dict, missing_required_keys)
        """
        q = query
        inputs: Dict[str, Any] = {}
        missing: List[str] = []

        if model == InventoryModelType.EOQ:
            # 1. Demand (D): annual / yearly demand, sales, usage, requirement
            demand = self._extract_first([
                r"(?:annual|yearly|total)\s*(?:demand|sales|usage|requirements?|volume)(?:\s*(?:is|of|=|:|estimated at|averages?))\s*\$?([0-9,]+(?:\.[0-9]+)?)",
                r"demand\s*(?:of|is|=|:)\s*\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:units|items|cases|boxes|pieces)?\s*(?:per|a|\/)\s*(?:year|annum|yr)",
                r"([0-9,]+(?:\.[0-9]+)?)\s*(?:units|items|cases|boxes|pieces)\s*(?:per|a|\/)\s*(?:year|annum|yr)",
                r"demand\s*(?:is|of|=|:)\s*\$?([0-9,]+(?:\.[0-9]+)?)",
                r"(?:sales|usage|requirements?)\s*(?:is|of|=|:)\s*\$?([0-9,]+(?:\.[0-9]+)?)",
                r"demand\s*of\s*([0-9,]+(?:\.[0-9]+)?)",
            ], q)

            # 2. Ordering Cost (S): ordering cost, order cost, setup cost, setup expense,
            #    cost per order, cost of placing an order, ordering/setup cost, each order costs
            order_cost = self._extract_first([
                r"(?:ordering\/setup|ordering\s*and\s*setup|ordering\s*or\s*setup|ordering|order|setup)\s*(?:costs?|expenses?|fees?)(?:\s*(?:is|are|of|=|:))?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
                r"(?:cost|expense|fee)\s*(?:per|for\s*each|of\s*placing\s*(?:a|an)?|to\s*place\s*(?:a|an)?)\s*(?:order|setup)(?:\s*(?:is|are|of|=|:))?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
                r"(?:cost\s*of\s*placing\s*(?:a|an)?\s*order|cost\s*to\s*place\s*(?:a|an)?\s*order)(?:\s*(?:is|are|of|=|:))?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
                r"(?:each|every)\s*(?:order|setup)\s*(?:costs?|expenses?|fees?)(?:\s*(?:is|are|of|=|:))?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
                r"setup\s*expense(?:\s*(?:is|are|of|=|:))?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
                r"\$([0-9,]+(?:\.[0-9]+)?)\s*(?:per|each|\/)\s*(?:order|setup|placement)",
                r"(?:order|ordering|setup)\s*(?:is|are|=|:)\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            ], q)

            # 3. Holding Cost (H): holding cost, carrying cost, cost of holding, cost of carrying,
            #    storage cost, holding expense, inventory carrying cost
            holding_cost = self._extract_first([
                r"(?:holding\/carrying|holding\s*and\s*carrying|holding\s*or\s*carrying|holding|carrying|storage)\s*(?:costs?|expenses?)(?:\s*(?:is|are|of|=|:))?\s*\$?([0-9,]+(?:\.[0-9]+)?)(?!\s*%)",
                r"(?:cost\s*of\s*(?:holding|carrying|storing))(?:\s*(?:is|are|of|=|:))?\s*\$?([0-9,]+(?:\.[0-9]+)?)(?!\s*%)",
                r"inventory\s*(?:holding|carrying)\s*(?:cost|expense)(?:\s*(?:is|are|of|=|:))?\s*\$?([0-9,]+(?:\.[0-9]+)?)(?!\s*%)",
                r"\$([0-9,]+(?:\.[0-9]+)?)\s*(?:per|each|\/)\s*(?:unit|item|piece|case)\s*(?:per|a|\/)\s*(?:year|annum|yr)",
                r"(?:holding|carrying)\s*(?:is|are|of|=|:)\s*\$?([0-9,]+(?:\.[0-9]+)?)(?!\s*%)",
            ], q)

            # 4. Optional Unit Price / Purchase Cost
            unit_cost = self._extract_first([
                r"(?:unit\s*purchase\s*price|unit\s*cost|unit\s*price|purchase\s*price|price\s*per\s*unit|cost\s*per\s*unit)(?:\s*(?:is|of|=|:))\s*\$?([0-9,]+(?:\.[0-9]+)?)",
                r"purchased?\s*(?:at|for)\s*\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:per|each|\/)\s*unit",
            ], q)

            # 5. Optional Lead Time
            lead_time_days = self._extract_first([
                r"(?:lead\s*time|replenishment\s*time|delivery\s*time)(?:\s*(?:is|of|=|:))?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:days?|d)\b",
                r"takes\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:days?|d)\s*(?:to\s*arrive|for\s*delivery|lead\s*time)?",
            ], q)

            if demand is not None:
                inputs["demand"] = demand
            else:
                missing.append("demand (D)")

            if order_cost is not None:
                inputs["order_cost"] = order_cost
            else:
                missing.append("order_cost (S)")

            if holding_cost is not None:
                inputs["holding_cost"] = holding_cost
            else:
                missing.append("holding_cost (H)")

            if unit_cost is not None:
                inputs["unit_cost"] = unit_cost
            if lead_time_days is not None:
                inputs["lead_time_days"] = lead_time_days

        elif model == InventoryModelType.QUANTITY_DISCOUNT:
            # 1. Demand
            demand = self._extract_first([
                r"(?:annual|yearly|total)?\s*(?:demand|sales|usage|requirements?)(?:\s*(?:is|of|=|:))?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
                r"demand\s*(?:is|of|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
                r"([0-9,]+(?:\.[0-9]+)?)\s*(?:units|items|cases)\s*(?:per|a|\/)\s*(?:year|yr)",
            ], q)

            # 2. Ordering cost synonyms
            order_cost = self._extract_first([
                r"(?:ordering\/setup|ordering\s*and\s*setup|ordering|order|setup)\s*(?:costs?|expenses?|fees?)(?:\s*(?:is|of|=|:|\b))\s*\$?([0-9,]+(?:\.[0-9]+)?)",
                r"(?:cost|expense|fee)\s*(?:per|for\s*each|of\s*placing\s*(?:an?)?|to\s*place\s*(?:an?)?)\s*(?:order|setup)(?:\s*(?:is|of|=|:|\b))\s*\$?([0-9,]+(?:\.[0-9]+)?)",
                r"setup\s*expense(?:\s*(?:is|of|=|:|\b))\s*\$?([0-9,]+(?:\.[0-9]+)?)",
                r"\$([0-9,]+(?:\.[0-9]+)?)\s*(?:per|each|\/)\s*(?:order|setup)",
            ], q)

            # 3. Holding cost: distinguish percentage vs fixed
            holding_cost_rate = self._extract_first([
                r"(?:percentage\s*holding\s*cost|holding\s*cost\s*rate|carrying\s*cost\s*rate|carrying\s*rate|annual\s*carrying\s*charge|inventory\s*carrying\s*rate)(?:\s*(?:is|of|=|:))?\s*([0-9,]+(?:\.[0-9]+)?)\s*%",
                r"(?:holding|carrying)\s*costs?\s*(?:rate|percentage)?(?:\s*(?:is|of|=|:))?\s*([0-9,]+(?:\.[0-9]+)?)\s*%",
                r"([0-9,]+(?:\.[0-9]+)?)\s*%\s*(?:of\s*(?:the\s*)?(?:purchase\s*price|unit\s*price|price)|holding|carrying)",
            ], q)

            fixed_holding_cost = None
            if holding_cost_rate is not None:
                holding_cost_rate = holding_cost_rate / 100.0 if holding_cost_rate > 1.0 else holding_cost_rate
            else:
                fixed_holding_cost = self._extract_first([
                    r"(?:fixed\s*holding\s*cost|fixed\s*carrying\s*cost)(?:\s*(?:is|of|=|:))?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
                    r"(?:holding|carrying)\s*costs?(?:\s*(?:is|of|=|:))?\s*\$?([0-9,]+(?:\.[0-9]+)?)(?!\s*%)",
                ], q)

            # 4. Quantity break / discount tiers / brackets
            tiers = []
            # Match formats: "0-999: $10", "Tier 1: 0 to 999 units at $10", "1-499: $20"
            tier_matches = re.findall(
                r"(?:tier\s*\d+\s*:?|\b)\s*([0-9,]+)\s*(?:-|to)\s*([0-9,]+|\+)?\s*(?:units?|items?|cases?)?\s*(?:at|@|for|costs?|price)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
                q,
                re.IGNORECASE,
            )
            for tm in tier_matches:
                min_q = float(tm[0].replace(",", ""))
                max_str = tm[1].replace(",", "").replace("+", "").strip() if tm[1] else None
                max_q = float(max_str) if (max_str and max_str.isdigit()) else None
                price = float(tm[2].replace(",", ""))
                tiers.append({"min_qty": min_q, "max_qty": max_q, "unit_price": price})

            # Match "2500+ at $8", "1000 or more: $8", "2500 and above: $8"
            plus_matches = re.findall(
                r"(?:tier\s*\d+\s*:?|\b)\s*([0-9,]+)(?:\+|\s*or more|\s*and above|\s*and up)\s*(?:units?|items?|cases?)?\s*(?:at|@|for|costs?|price)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
                q,
                re.IGNORECASE,
            )
            for pm in plus_matches:
                min_q = float(pm[0].replace(",", ""))
                price = float(pm[1].replace(",", ""))
                if not any(t["min_qty"] == min_q and t["max_qty"] is None for t in tiers):
                    tiers.append({"min_qty": min_q, "max_qty": None, "unit_price": price})

            # Match "under 500 units at $15", "less than 500: $15"
            under_matches = re.findall(
                r"(?:under|less than|fewer than)\s*([0-9,]+)\s*(?:units?|items?)?\s*(?:at|@|for|costs?|price)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
                q,
                re.IGNORECASE,
            )
            for um in under_matches:
                max_q = float(um[0].replace(",", "")) - 1.0
                price = float(um[1].replace(",", ""))
                if not any(t["min_qty"] == 0 and t["unit_price"] == price for t in tiers):
                    tiers.append({"min_qty": 0.0, "max_qty": max_q, "unit_price": price})

            # Sort tiers by min_qty
            tiers = sorted(tiers, key=lambda x: x["min_qty"])

            if demand is not None:
                inputs["demand"] = demand
            else:
                missing.append("demand (D)")

            if order_cost is not None:
                inputs["order_cost"] = order_cost
            else:
                missing.append("order_cost (S)")

            if holding_cost_rate is not None:
                inputs["holding_cost_rate"] = holding_cost_rate
            elif fixed_holding_cost is not None:
                inputs["fixed_holding_cost"] = fixed_holding_cost
            else:
                missing.append("holding_cost_rate (i) or fixed_holding_cost (H)")

            if tiers:
                inputs["discount_tiers"] = tiers
            else:
                missing.append("discount_tiers (e.g. 0-999: $10, 1000+: $8)")

        elif model == InventoryModelType.SAFETY_STOCK_ROP:
            # Distinguish:
            # 1. Average Demand (d)
            avg_demand = self._extract_first([
                r"(?:average\s*daily\s*demand|mean\s*daily\s*demand|daily\s*demand|average\s*weekly\s*demand|mean\s*weekly\s*demand|weekly\s*demand)(?:\s*(?:is|are|of|=|:))?\s*([0-9,]+(?:\.[0-9]+)?)",
                r"(?:average\s*demand|mean\s*demand|expected\s*demand|demand\s*rate|usage\s*rate)(?:\s*(?:is|are|of|=|:))?\s*([0-9,]+(?:\.[0-9]+)?)",
                r"demand\s*(?:averages|is|are|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
                r"([0-9,]+(?:\.[0-9]+)?)\s*(?:units|items)\s*(?:per|a|\/)\s*(?:day|week|month)",
            ], q)

            # 2. Average Lead Time (L)
            lead_time = self._extract_first([
                r"(?:average\s*lead[- ]time|mean\s*lead[- ]time|expected\s*lead[- ]time|average\s*lead\s*time)(?:\s*(?:is|are|of|=|:))?\s*([0-9,]+(?:\.[0-9]+)?)",
                r"(?:replenishment\s*lead\s*time|delivery\s*lead\s*time|lead[- ]time|lead\s*time)(?:\s*(?:is|are|of|=|:))?\s*([0-9,]+(?:\.[0-9]+)?)",
                r"takes\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:days?|weeks?)\s*(?:for\s*delivery|to\s*arrive|lead\s*time)?",
            ], q)

            # 3. Lead-time Standard Deviation (sigma_L) - Extract before demand std dev to avoid collision
            lead_time_std_dev = self._extract_first([
                r"(?:lead[- ]time\s*standard\s*deviation|lead\s*time\s*std\s*dev|sigma_l)(?:\s*(?:is|are|of|=|:))?\s*([0-9,]+(?:\.[0-9]+)?)",
                r"(?:standard\s*deviation|std\s*dev)\s*of\s*(?:the\s*)?lead[- ]time(?:\s*(?:is|are|of|=|:))?\s*([0-9,]+(?:\.[0-9]+)?)",
                r"lead\s*time\s*variability(?:\s*(?:is|are|of|=|:))?\s*([0-9,]+(?:\.[0-9]+)?)",
                r"(?:lead\s*time|delivery\s*time)\s*fluctuates?\s*by\s*\+\/-\s*([0-9,]+(?:\.[0-9]+)?)",
            ], q)

            # 4. Demand Standard Deviation (sigma_d)
            demand_std_dev = self._extract_first([
                r"(?:demand\s*standard\s*deviation|demand\s*std\s*dev|sigma_d)(?:\s*(?:is|are|of|=|:))?\s*([0-9,]+(?:\.[0-9]+)?)",
                r"(?:standard\s*deviation|std\s*dev)\s*of\s*(?:daily\s*|weekly\s*|customer\s*)?demand(?:\s*(?:is|are|of|=|:))?\s*([0-9,]+(?:\.[0-9]+)?)",
                r"(?:standard\s*deviation|std\s*dev|sigma)(?:\s*(?:is|are|of|=|:))?\s*([0-9,]+(?:\.[0-9]+)?)(?!\s*(?:days|weeks|months))",
                r"demand\s*fluctuates?\s*by\s*\+\/-\s*([0-9,]+(?:\.[0-9]+)?)",
            ], q)

            # 5. Service Level (CSL) or Z-score
            service_level = self._extract_first([
                r"(?:cycle\s*service\s*level|target\s*service\s*level|desired\s*service\s*level|service\s*level|csl|in-stock\s*probability)(?:\s*(?:is|are|of|=|:))?\s*([0-9,]+(?:\.[0-9]+)?)\s*%",
                r"([0-9,]+(?:\.[0-9]+)?)\s*%\s*(?:cycle\s*service\s*level|service\s*level|csl|in-stock\s*probability)",
                r"(?:service\s*level|csl)(?:\s*(?:is|are|of|=|:))?\s*([0-9,]+(?:\.[0-9]+)?)",
            ], q)

            z_score = self._extract_first([
                r"(?:z[- ]score|critical\s*value|z\s*=)(?:\s*(?:is|are|of|=|:))?\s*([0-9,]+(?:\.[0-9]+)?)",
            ], q)

            holding_cost = self._extract_first([
                r"(?:holding|carrying)\s*costs?(?:\s*(?:is|are|of|=|:))?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            ], q)

            if avg_demand is not None:
                inputs["avg_demand"] = avg_demand
            else:
                missing.append("avg_demand (d)")

            if lead_time is not None:
                inputs["lead_time"] = lead_time
            else:
                missing.append("lead_time (L)")

            if service_level is not None:
                inputs["service_level"] = service_level
            elif z_score is not None:
                inputs["z_score"] = z_score
            else:
                inputs["service_level"] = 0.95

            inputs["demand_std_dev"] = demand_std_dev if demand_std_dev is not None else 0.0
            inputs["lead_time_std_dev"] = lead_time_std_dev if lead_time_std_dev is not None else 0.0
            if holding_cost is not None:
                inputs["holding_cost"] = holding_cost

        elif model == InventoryModelType.FIXED_PERIOD_REVIEW:
            # Distinguish:
            # 1. Demand (d)
            avg_demand = self._extract_first([
                r"(?:average\s*daily\s*demand|daily\s*demand|average\s*demand|mean\s*demand|demand\s*rate)(?:\s*(?:is|of|=|:))\s*([0-9,]+(?:\.[0-9]+)?)",
                r"demand\s*(?:is|of|=|:)\s*([0-9,]+(?:\.[0-9]+)?)",
            ], q)

            # 2. Review Period (R)
            review_period = self._extract_first([
                r"(?:review\s*period|review\s*interval|review\s*cycle|replenishment\s*cycle)(?:\s*(?:is|of|=|:|\b))\s*([0-9,]+(?:\.[0-9]+)?)",
                r"review\s*(?:inventory\s*|stock\s*)?every\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:days?|weeks?|d|w)?",
                r"orders?\s*placed?\s*every\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:days?|weeks?|d|w)?",
            ], q)

            # 3. Lead Time (L)
            lead_time = self._extract_first([
                r"(?:replenishment\s*lead\s*time|delivery\s*lead\s*time|lead[- ]time|lead\s*time)(?:\s*(?:is|of|=|:))?\s*([0-9,]+(?:\.[0-9]+)?)",
                r"takes\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:days?|weeks?)\s*(?:to\s*arrive|for\s*delivery)?",
            ], q)

            # 4. Service Level or Z-score
            service_level = self._extract_first([
                r"(?:target\s*service\s*level|service\s*level|csl)(?:\s*(?:is|of|=|:|\b))\s*([0-9,]+(?:\.[0-9]+)?)\s*%",
                r"([0-9,]+(?:\.[0-9]+)?)\s*%\s*service\s*level",
            ], q)
            z_score = self._extract_first([
                r"(?:z[- ]score|z\s*=)(?:\s*(?:is|of|=|:|\b))\s*([0-9,]+(?:\.[0-9]+)?)",
            ], q)

            # 5. Demand Standard Deviation
            demand_std_dev = self._extract_first([
                r"(?:demand\s*standard\s*deviation|standard\s*deviation\s*of\s*demand|demand\s*std\s*dev|sigma_d)(?:\s*(?:is|of|=|:))\s*([0-9,]+(?:\.[0-9]+)?)",
                r"(?:standard\s*deviation|std\s*dev|sigma)(?:\s*(?:is|of|=|:))\s*([0-9,]+(?:\.[0-9]+)?)",
            ], q)

            # 6. On-hand Inventory (I)
            on_hand = self._extract_first([
                r"(?:on[- ]hand\s*inventory|stock\s*on\s*hand|on[- ]hand\s*stock|on[- ]hand|physical\s*(?:stock|inventory))(?:\s*(?:is|are|of|=|:|\b))\s*([0-9,]+(?:\.[0-9]+)?)",
                r"(?:current\s*inventory|current\s*stock|existing\s*inventory)(?:\s*(?:is|are|of|=|:|\b))\s*([0-9,]+(?:\.[0-9]+)?)",
            ], q)

            # 7. On-order Inventory
            on_order = self._extract_first([
                r"(?:on[- ]order\s*inventory|on[- ]order\s*stock|on[- ]order|in[- ]transit|pipeline\s*inventory|open\s*purchase\s*orders?)(?:\s*(?:is|are|of|=|:|\b))\s*([0-9,]+(?:\.[0-9]+)?)",
            ], q)

            # 8. Backorders
            backorders = self._extract_first([
                r"(?:backorders?|backlogged\s*orders?|unfulfilled\s*demand|pending\s*backorders?)(?:\s*(?:is|are|of|=|:|\b))\s*([0-9,]+(?:\.[0-9]+)?)",
            ], q)

            # 9. Direct Inventory Position (if specified directly)
            inventory_position = self._extract_first([
                r"(?:inventory\s*position|effective\s*inventory)(?:\s*(?:is|of|=|:|\b))\s*([0-9,]+(?:\.[0-9]+)?)",
            ], q)

            if avg_demand is not None:
                inputs["avg_demand"] = avg_demand
            else:
                missing.append("avg_demand (d)")

            if review_period is not None:
                inputs["review_period"] = review_period
            else:
                missing.append("review_period (R)")

            if lead_time is not None:
                inputs["lead_time"] = lead_time
            else:
                missing.append("lead_time (L)")

            if service_level is not None:
                inputs["service_level"] = service_level
            elif z_score is not None:
                inputs["z_score"] = z_score
            else:
                inputs["service_level"] = 0.95

            inputs["demand_std_dev"] = demand_std_dev if demand_std_dev is not None else 0.0

            # Reconcile on-hand, on-order, backorders into current_inventory
            if on_hand is not None:
                inputs["current_inventory"] = on_hand
            elif inventory_position is not None:
                inputs["current_inventory"] = inventory_position

            inputs["on_order_inventory"] = on_order if on_order is not None else 0.0
            inputs["backorders"] = backorders if backorders is not None else 0.0

        elif model == InventoryModelType.NEWSVENDOR:
            # Distinguish:
            # 1. Selling price (p)
            selling_price = self._extract_first([
                r"(?:selling\s*price|retail\s*price|sales\s*price|customer\s*price|revenue\s*per\s*unit|price\s*per\s*unit)(?:\s*(?:is|are|of|=|:))?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
                r"sells?\s*(?:at|for)\s*\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:per|each|\/)\s*unit",
                r"price(?:\s*(?:is|are|of|=|:))?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            ], q)

            # 2. Purchase cost (c)
            cost_price = self._extract_first([
                r"(?:purchase\s*cost|cost\s*price|procurement\s*cost|production\s*cost|acquisition\s*cost|cost\s*per\s*unit|unit\s*cost)(?:\s*(?:is|are|of|=|:))?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
                r"costs?\s*\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:per|each|\/)\s*unit\s*(?:to\s*(?:buy|produce|procure))?",
                r"cost(?:\s*(?:is|are|of|=|:))?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            ], q)

            # 3. Salvage value (s)
            salvage_value = self._extract_first([
                r"(?:salvage\s*value|salvage\s*price|markdown\s*price|markdown\s*value|scrap\s*value|residual\s*value|liquidation\s*value|salvage)(?:\s*(?:is|are|of|=|:))?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
                r"salvaged?\s*(?:at|for)\s*\$?([0-9,]+(?:\.[0-9]+)?)",
                r"markdown(?:\s*(?:to|is|are|of|=|:))?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            ], q)

            # 4. Underage / Shortage Cost (Cu)
            underage_cost = self._extract_first([
                r"(?:underage\s*cost|shortage\s*cost|cost\s*of\s*underage|cost\s*of\s*shortage|marginal\s*profit(?:\s*per\s*unit)?)(?:\s*(?:is|are|of|=|:))?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            ], q)

            # 5. Overage / Excess Cost (Co)
            overage_cost = self._extract_first([
                r"(?:overage\s*cost|excess\s*cost|cost\s*of\s*overage|cost\s*of\s*excess|cost\s*of\s*overstocking|marginal\s*loss(?:\s*per\s*unit)?)(?:\s*(?:is|are|of|=|:))?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            ], q)

            # 6. Penalty cost (g)
            penalty_cost = self._extract_first([
                r"(?:penalty\s*cost|lost\s*goodwill|goodwill\s*penalty|stockout\s*penalty)(?:\s*(?:is|are|of|=|:))?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            ], q)

            # 7. Mean Demand (mu)
            mean_demand = self._extract_first([
                r"(?:mean\s*demand|mean\s*seasonal\s*demand|expected\s*demand|forecasted\s*demand|average\s*demand)(?:\s*(?:is|are|of|=|:))?\s*([0-9,]+(?:\.[0-9]+)?)",
                r"demand\s*(?:is\s*normally\s*distributed\s*with\s*mean|averages?|is|are|of|=|:)\s*([0-9,]+(?:\.[0-9]+)?)",
                r"seasonal\s*mean\s*demand(?:\s*(?:is|are|of|=|:))?\s*([0-9,]+(?:\.[0-9]+)?)",
            ], q)

            # 8. Demand Standard Deviation (sigma)
            std_demand = self._extract_first([
                r"(?:demand\s*standard\s*deviation|standard\s*deviation\s*of\s*demand|demand\s*std\s*dev|sigma\s*of\s*demand)(?:\s*(?:is|are|of|=|:))?\s*([0-9,]+(?:\.[0-9]+)?)",
                r"(?:standard\s*deviation|std\s*dev|sigma)(?:\s*(?:is|are|of|=|:))?\s*([0-9,]+(?:\.[0-9]+)?)",
            ], q)

            if mean_demand is not None:
                inputs["mean_demand"] = mean_demand
            else:
                missing.append("mean_demand (mu)")

            if std_demand is not None:
                inputs["std_demand"] = std_demand
            else:
                missing.append("std_demand (sigma)")

            # Check if we have either (selling_price & cost_price) OR (underage_cost & overage_cost)
            if underage_cost is not None and overage_cost is not None:
                inputs["underage_cost"] = underage_cost
                inputs["overage_cost"] = overage_cost
                if selling_price is not None:
                    inputs["selling_price"] = selling_price
                if cost_price is not None:
                    inputs["cost_price"] = cost_price
            elif selling_price is not None and cost_price is not None:
                inputs["selling_price"] = selling_price
                inputs["cost_price"] = cost_price
            else:
                if selling_price is None and underage_cost is None:
                    missing.append("selling_price (p) or underage_cost (Cu)")
                if cost_price is None and overage_cost is None:
                    missing.append("cost_price (c) or overage_cost (Co)")

            inputs["salvage_value"] = salvage_value if salvage_value is not None else 0.0
            inputs["penalty_cost"] = penalty_cost if penalty_cost is not None else 0.0

        return inputs, missing

    # =========================================================================
    # STAGE 3: SELECT PYTHON TOOL
    # =========================================================================
    def select_tool(self, model: str) -> Tuple[str, Callable]:
        """Maps identified model to the appropriate deterministic inventory tool."""
        mapping = {
            InventoryModelType.EOQ: ("calculate_eoq", it.calculate_eoq),
            InventoryModelType.QUANTITY_DISCOUNT: (
                "calculate_quantity_discount",
                it.calculate_quantity_discount,
            ),
            InventoryModelType.SAFETY_STOCK_ROP: (
                "calculate_rop",
                it.calculate_rop,
            ),
            InventoryModelType.FIXED_PERIOD_REVIEW: (
                "calculate_periodic_review",
                it.calculate_periodic_review,
            ),
            InventoryModelType.NEWSVENDOR: (
                "calculate_newsvendor",
                it.calculate_newsvendor,
            ),
        }
        if model in mapping:
            return mapping[model]
        raise ValueError(f"Unknown inventory model: {model}")

    # =========================================================================
    # STAGE 4: CALCULATE
    # =========================================================================
    def calculate(self, tool_func: Callable, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute calculation tool with extracted parameters."""
        return tool_func(**inputs)

    # =========================================================================
    # STAGE 5: VERIFY
    # =========================================================================
    def verify(
        self, model: str, inputs: Dict[str, Any], results: Dict[str, Any]
    ) -> VerificationResult:
        """
        Audit calculation results for physical feasibility, mathematical validity,
        and inventory management boundary conditions.
        """
        res = results.get("results", {})
        checks = []
        warnings = []
        notes = []

        if model == InventoryModelType.EOQ:
            eoq = res.get("eoq", 0)
            holding_cost = res.get("annual_holding_cost", 0)
            ordering_cost = res.get("annual_ordering_cost", 0)

            if eoq > 0:
                checks.append(f"EOQ is strictly positive ({eoq} units).")
            else:
                warnings.append(f"EOQ must be strictly positive, calculated: {eoq}.")

            cost_diff = abs(holding_cost - ordering_cost)
            if cost_diff < 1.0 or (cost_diff / (holding_cost + ordering_cost + 1e-6) < 0.02):
                checks.append(
                    f"First-order optimality verified: Annual Holding Cost (${holding_cost:.2f}) ≈ Annual Ordering Cost (${ordering_cost:.2f})."
                )
            else:
                warnings.append(
                    f"Cost balance deviation detected: Holding=${holding_cost:.2f} vs Ordering=${ordering_cost:.2f}."
                )

            notes.append("EOQ minimizes Total Inventory Cost where d(TC)/dQ = 0.")

        elif model == InventoryModelType.QUANTITY_DISCOUNT:
            opt_q = res.get("optimal_order_quantity", 0)
            opt_tier = res.get("optimal_tier")
            min_tc = res.get("minimum_total_annual_cost", 0)
            tier_breakdown = res.get("tier_breakdown", [])

            all_costs = [t["total_annual_cost"] for t in tier_breakdown]
            if min_tc == min(all_costs):
                checks.append(
                    f"Global optimality verified: Selected Tier {opt_tier} has minimal cost (${min_tc:,.2f}) among all {len(tier_breakdown)} evaluated tiers."
                )
            else:
                warnings.append("Selected tier is not strictly the minimal cost tier.")

            for t in tier_breakdown:
                if t["tier_index"] == opt_tier:
                    min_q = t["min_qty"]
                    max_q = t["max_qty"]
                    if opt_q >= min_q and (max_q is None or opt_q <= max_q):
                        checks.append(
                            f"Tier feasibility verified: Order quantity {opt_q} satisfies tier bracket [{min_q}, {max_q or 'inf'}]."
                        )
                    else:
                        warnings.append(f"Order quantity {opt_q} outside bracket bounds.")

        elif model == InventoryModelType.SAFETY_STOCK_ROP:
            ss = res.get("safety_stock", 0)
            rop = res.get("reorder_point", 0)
            ltd = res.get("lead_time_demand", 0)
            z = res.get("z_score", 0)

            if abs(rop - (ltd + ss)) < 0.05:
                checks.append(
                    f"Conservation check passed: ROP ({rop:.2f}) = Lead Time Demand ({ltd:.2f}) + Safety Stock ({ss:.2f})."
                )
            else:
                warnings.append("Reorder point does not match LTD + SS.")

            if ss >= 0:
                checks.append(f"Safety stock is non-negative ({ss:.2f} units).")
            else:
                warnings.append("Safety stock cannot be negative.")

            notes.append(f"Z-score {z:.2f} corresponds to cumulative normal service level.")

        elif model == InventoryModelType.FIXED_PERIOD_REVIEW:
            prot_int = res.get("protection_interval", 0)
            target = res.get("target_inventory_level", 0)
            exp_dem = res.get("expected_protection_demand", 0)
            ss = res.get("safety_stock", 0)

            r = inputs.get("review_period", 0)
            l = inputs.get("lead_time", 0)
            if abs(prot_int - (r + l)) < 0.01:
                checks.append(
                    f"Protection interval verified: {prot_int} time units (R={r} + L={l})."
                )
            else:
                warnings.append("Protection interval mismatch.")

            if abs(target - (exp_dem + ss)) < 0.05:
                checks.append(
                    f"Target stock level verified: Target ({target:.2f}) = Demand ({exp_dem:.2f}) + SS ({ss:.2f})."
                )
            else:
                warnings.append("Target inventory equation mismatch.")

        elif model == InventoryModelType.NEWSVENDOR:
            cr = res.get("critical_ratio", 0)
            cu = res.get("underage_cost_Cu", 0)
            co = res.get("overage_cost_Co", 0)
            opt_q = res.get("optimal_order_quantity", 0)

            if 0 < cr < 1:
                checks.append(
                    f"Critical ratio probability valid: CR = {cr:.4f} (Underage Cu=${cu:.2f}, Overage Co=${co:.2f})."
                )
            else:
                warnings.append(f"Critical ratio must be in (0, 1), received {cr}.")

            if cu > 0 and co > 0:
                checks.append("Both underage and overage penalties are positive.")
            else:
                warnings.append("Non-positive underage or overage margin detected.")

            notes.append(f"Optimal order quantity {opt_q:.2f} balances marginal profit vs loss.")

        status = "PASSED" if not warnings else "WARNING"
        return VerificationResult(
            status=status,
            checks_passed=checks,
            warnings=warnings,
            audit_notes=notes,
        )

    # =========================================================================
    # STAGE 6: EXPLAIN
    # =========================================================================
    def explain(
        self,
        model: str,
        inputs: Dict[str, Any],
        results: Dict[str, Any],
        verification: VerificationResult,
    ) -> str:
        """
        Synthesize results into an 8-part pedagogical decision report for student verification.
        All numerical values, substitutions, and checks come strictly from the deterministic Python tool.
        """
        lines = []

        lines.append(f"### 📦 Supply Chain Inventory Analysis: {model}")
        lines.append("")

        # 6. Final Answer (Executive Decision upfront)
        if "final_answer" in results:
            lines.append("#### 🎯 6. Final Answer & Recommendation")
            lines.append(f"> {results['final_answer']}")
            lines.append("")

        # 1. Inputs Extracted
        lines.append("#### 📥 1. Inputs Extracted")
        for k, v in results.get("inputs", {}).items():
            if isinstance(v, list):
                lines.append(f"- **{k}**: `{len(v)} tiers defined`")
            elif isinstance(v, float):
                lines.append(f"- **{k}**: `{v:,.2f}`" if abs(v) >= 1 else f"- **{k}**: `{v}`")
            else:
                lines.append(f"- **{k}**: `{v}`")
        lines.append("")

        # 2. Variable Definitions
        if "variable_definitions" in results:
            lines.append("#### 📖 2. Variable Definitions")
            for var_sym, var_def in results["variable_definitions"].items():
                lines.append(f"- **{var_sym}**: {var_def}")
            lines.append("")

        # 3. Formula
        if "formula" in results:
            lines.append("#### 📐 3. Mathematical Formula")
            lines.append("```text")
            lines.append(results["formula"])
            lines.append("```")
            lines.append("")

        # 4. Formula with Substituted Values
        if "formula_with_substituted_values" in results:
            lines.append("#### 🔢 4. Formula with Substituted Values")
            lines.append("```text")
            lines.append(results["formula_with_substituted_values"])
            lines.append("```")
            lines.append("")

        # 5. Intermediate Calculations
        if "intermediate_calculations" in results:
            lines.append("#### ⚙️ 5. Intermediate Calculations")
            inter = results["intermediate_calculations"]
            if "evaluated_tiers" in inter:
                for t in inter["evaluated_tiers"]:
                    lines.append(
                        f"- **Tier {t['tier_index']}** (Price: ${t['unit_price']:.2f}, Range: {t['min_qty']}-{t['max_qty'] or '∞'}): "
                        f"Q_cand = `{t['candidate_q']:,}`, Annual Cost = `${t['total_annual_cost']:,.2f}` ({t['status']})"
                    )
            else:
                for k, v in inter.items():
                    if v is not None:
                        lines.append(f"- **{k.replace('_', ' ').title()}**: `{v}`")
            lines.append("")

        # 7. Verification Checks
        checks = results.get("verification_checks", [])
        lines.append("#### 🛡️ 7. Verification Checks")
        if checks:
            for chk in checks:
                status_icon = "✅" if chk.get("status") == "PASSED" else "⚠️"
                lines.append(f"- {status_icon} **{chk.get('check')}** (`{chk.get('condition')}`): {chk.get('calculated_value')}")
                if "student_note" in chk:
                    lines.append(f"  *Student Note:* _{chk['student_note']}_")
        else:
            for chk in verification.checks_passed:
                lines.append(f"- ✅ {chk}")
            for wrn in verification.warnings:
                lines.append(f"- ⚠️ {wrn}")
        lines.append("")

        # 8. Assumptions
        if "assumptions" in results:
            lines.append("#### 📋 8. Assumptions")
            for asm in results["assumptions"]:
                lines.append(f"- {asm}")

        return "\n".join(lines)

    # =========================================================================
    # END-TO-END PIPELINE ORCHESTRATOR
    # =========================================================================
    def process_query(self, query: str) -> AgentResponse:
        """
        Executes the complete 6-stage Supply Chain Decision Pipeline:
        1. Customer Question -> 2. Identify Model -> 3. Extract Inputs ->
        4. Select Python Tool -> 5. Calculate -> 6. Verify & Explain.
        """
        trace = []

        # 1. Question received
        trace.append({"stage": "1. Customer Question", "data": {"raw_query": query}})

        # 2. Identify Model
        model_name, confidence = self.identify_model(query)
        trace.append({
            "stage": "2. Identify Model",
            "data": {"model_name": model_name, "confidence": confidence},
        })

        # 3. Extract Inputs
        extracted_inputs, missing = self.extract_inputs(query, model_name)
        trace.append({
            "stage": "3. Extract Inputs",
            "data": {"extracted_inputs": extracted_inputs, "missing_inputs": missing},
        })

        if missing:
            msg = (
                f"### ⚠️ Missing Required Parameters for {model_name}\n\n"
                f"To perform accurate supply chain calculation, please provide:\n"
                + "\n".join([f"- **{m}**" for m in missing])
            )
            return AgentResponse(
                question=query,
                model_name=model_name,
                confidence=confidence,
                extracted_inputs=extracted_inputs,
                missing_inputs=missing,
                tool_name="N/A",
                calculation_results=None,
                verification=None,
                explanation=msg,
                pipeline_trace=trace,
            )

        # 4. Select Python Tool
        tool_name, tool_func = self.select_tool(model_name)
        trace.append({
            "stage": "4. Select Python Tool",
            "data": {"tool_name": tool_name},
        })

        # 5. Calculate
        try:
            results = self.calculate(tool_func, extracted_inputs)
            trace.append({
                "stage": "5. Calculate",
                "data": {"results": results.get("results", {})},
            })
        except Exception as e:
            return AgentResponse(
                question=query,
                model_name=model_name,
                confidence=confidence,
                extracted_inputs=extracted_inputs,
                missing_inputs=[],
                tool_name=tool_name,
                calculation_results=None,
                verification=VerificationResult(
                    status="FAILED", warnings=[f"Calculation error: {str(e)}"]
                ),
                explanation=f"Calculation Error in `{tool_name}`: {str(e)}",
                pipeline_trace=trace,
            )

        # 6. Verify
        verification = self.verify(model_name, extracted_inputs, results)
        trace.append({
            "stage": "6. Verify",
            "data": {
                "status": verification.status,
                "checks": verification.checks_passed,
                "warnings": verification.warnings,
            },
        })

        # 7. Explain
        explanation = self.explain(model_name, extracted_inputs, results, verification)
        trace.append({
            "stage": "7. Explain",
            "data": {"summary_length": len(explanation)},
        })

        return AgentResponse(
            question=query,
            model_name=model_name,
            confidence=confidence,
            extracted_inputs=extracted_inputs,
            missing_inputs=[],
            tool_name=tool_name,
            calculation_results=results,
            verification=verification,
            explanation=explanation,
            pipeline_trace=trace,
        )


if __name__ == "__main__":
    agent = InventoryDecisionAgent()
    sample_queries = [
        "Our annual demand is 10,000 units, each order costs $50, and holding cost is $2 per unit per year. What is the optimal EOQ?",
        "We have demand of 15,000 units. Setup cost is $100. Holding cost is 20%. Tier 1: 0-999 units at $10. Tier 2: 1000-2499 at $9. Tier 3: 2500+ at $8.",
        "Daily demand is 50 units with standard deviation of 8 units. Lead time is 5 days. We want a 95% service level. What is our safety stock and reorder point?",
        "We review inventory every 7 days. Replenishment lead time is 3 days. Average daily demand is 100 units with standard deviation of 15 units. We target 98% service level. Current on hand is 400 units.",
        "Selling price is $120, cost price is $70, salvage value is $30, mean seasonal demand is 500 units with standard deviation of 80 units. How many should we order?"
    ]

    print("=== Testing Inventory Decision Agent ===")
    for q in sample_queries:
        print("\n" + "=" * 80)
        print("QUESTION:", q)
        resp = agent.process_query(q)
        print("MODEL IDENTIFIED:", resp.model_name, f"(Confidence: {resp.confidence})")
        print("TOOL:", resp.tool_name)
        print("INPUTS:", resp.extracted_inputs)
        print("VERIFICATION:", resp.verification.status if resp.verification else "N/A")
        print("\nEXPLANATION:\n", resp.explanation)
