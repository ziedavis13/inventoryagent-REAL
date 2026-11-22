"""
Supply Chain Analytics Inventory Decision Agent - Web & CLI Application.

Architecture Pipeline:
Customer Question -> Identify Model -> Extract Inputs -> Select Python Tool -> Calculate -> Verify -> Explain
"""

import sys
import pandas as pd
import numpy as np
from scipy.stats import norm
import streamlit as st

import inventory_tools as it
from inventory_agent import (
    InventoryDecisionAgent,
    InventoryModelType,
    AgentResponse,
)


def render_pedagogical_verification_panel(res: dict, title: str = "🎓 Student Step-by-Step Pedagogical Verification"):
    """
    Renders all 8 required verification components directly from the deterministic tool output:
    1. Inputs Extracted
    2. Variable Definitions
    3. Formula
    4. Formula with Substituted Values
    5. Intermediate Calculations
    6. Final Answer
    7. Verification Checks
    8. Assumptions
    """
    if not res:
        return

    st.markdown(f"#### {title}")

    # 6. Final Answer banner upfront
    if "final_answer" in res:
        st.info(f"🎯 **Final Answer & Decision:** {res['final_answer']}")

    t1, t2, t3, t4, t5, t6, t7 = st.tabs([
        "📥 1. Inputs",
        "📖 2. Definitions",
        "📐 3. Formula",
        "🔢 4. Substitution",
        "⚙️ 5. Calculations",
        "🛡️ 6. Verification Checks",
        "📋 7. Assumptions",
    ])

    with t1:
        st.markdown("##### 1. Inputs Extracted")
        inputs_data = []
        for k, v in res.get("inputs", {}).items():
            if isinstance(v, list):
                val_str = f"{len(v)} tiers configured"
            elif isinstance(v, float):
                val_str = f"{v:,.4g}"
            else:
                val_str = str(v)
            inputs_data.append({"Input Parameter": str(k), "Value": str(val_str)})
        st.dataframe(pd.DataFrame(inputs_data), use_container_width=True, hide_index=True)

    with t2:
        st.markdown("##### 2. Variable Definitions")
        vardefs = res.get("variable_definitions", {})
        for sym, desc in vardefs.items():
            st.markdown(f"- **`{sym}`**: {desc}")

    with t3:
        st.markdown("##### 3. Mathematical Formula")
        st.code(res.get("formula", "N/A"), language="text")

    with t4:
        st.markdown("##### 4. Formula with Substituted Values")
        st.code(res.get("formula_with_substituted_values", "N/A"), language="text")

    with t5:
        st.markdown("##### 5. Intermediate Calculations")
        inter = res.get("intermediate_calculations", {})
        if "evaluated_tiers" in inter:
            st.dataframe(pd.DataFrame(inter["evaluated_tiers"]), use_container_width=True, hide_index=True)
        else:
            inter_rows = []
            for k, v in inter.items():
                if v is not None:
                    inter_rows.append({
                        "Step / Metric": str(k).replace("_", " ").title(),
                        "Calculated Value": str(f"{v:,.4g}" if isinstance(v, (int, float)) else v)
                    })
            st.dataframe(pd.DataFrame(inter_rows), use_container_width=True, hide_index=True)

    with t6:
        st.markdown("##### 6. Verification Checks")
        checks = res.get("verification_checks", [])
        if checks:
            checks_rows = []
            for c in checks:
                status_icon = "✅ PASSED" if c.get("status") == "PASSED" else "⚠️ FAILED"
                checks_rows.append({
                    "Status": str(status_icon),
                    "Check Name": str(c.get("check")),
                    "Mathematical Condition": str(c.get("condition")),
                    "Calculated Value": str(c.get("calculated_value")),
                    "Student Pedagogical Note": str(c.get("student_note", "")),
                })
            st.dataframe(pd.DataFrame(checks_rows), use_container_width=True, hide_index=True)

    with t7:
        st.markdown("##### 7. Mathematical & Operational Assumptions")
        for asm in res.get("assumptions", []):
            st.markdown(f"- {asm}")


def run_cli():
    """Run agent in interactive CLI mode."""
    print("=" * 80)
    print("SUPPLY CHAIN ANALYTICS: INVENTORY DECISION AGENT (CLI MODE)")
    print("Architecture: Customer Question -> Identify Model -> Extract Inputs ->")
    print("              Select Python Tool -> Calculate -> Verify -> Explain")
    print("=" * 80)

    agent = InventoryDecisionAgent()

    if len(sys.argv) >= 2 and sys.argv[1] != "--cli":
        query = " ".join(sys.argv[1:])
        resp = agent.process_query(query)
        print(f"\nModel: {resp.model_name} (Confidence: {resp.confidence:.0%})")
        print(f"Tool: {resp.tool_name}")
        print("\n" + resp.explanation)
        return

    print("\nType your inventory question below (or 'exit' to quit):\n")
    while True:
        try:
            query = input("SupplyChainQuery> ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("Exiting Supply Chain Decision Agent. Goodbye!")
                break

            resp = agent.process_query(query)
            print("\n" + "-" * 60)
            print(f"IDENTIFIED MODEL : {resp.model_name} ({resp.confidence * 100:.1f}% confidence)")
            print(f"PYTHON TOOL      : {resp.tool_name}")
            print(f"VERIFICATION     : {resp.verification.status if resp.verification else 'N/A'}")
            print("-" * 60)
            print(resp.explanation)
            print("-" * 60 + "\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


def render_pipeline_stage_card(step_num: int, title: str, subtitle: str, content: str, status: str = "done"):
    border_color = "#10B981" if status == "done" else ("#F59E0B" if status == "warning" else "#3B82F6")
    badge_bg = "#D1FAE5" if status == "done" else ("#FEF3C7" if status == "warning" else "#DBEAFE")
    badge_fg = "#065F46" if status == "done" else ("#92400E" if status == "warning" else "#1E40AF")

    st.markdown(
        f"""
        <div style="border-left: 4px solid {border_color}; background-color: #F9FAFB; padding: 12px 16px; border-radius: 4px; margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: 600; font-size: 1.05rem; color: #1F2937;">Step {step_num}: {title}</span>
                <span style="background-color: {badge_bg}; color: {badge_fg}; padding: 2px 8px; border-radius: 9999px; font-size: 0.75rem; font-weight: 600;">{status.upper()}</span>
            </div>
            <div style="font-size: 0.85rem; color: #4B5563; margin-top: 2px;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def process_and_render_csv_upload(uploaded_file):
    """
    Processes an uploaded CSV file, validates schema, and executes either:
    1. Multi-Criteria Inventory Classification (A, B, C, D) using Sales ($) and Hits
    2. Units & Valuation Analysis using Product, Category, Units, Unit Cost
    (Or both if the dataset includes all fields).
    """
    if uploaded_file is None:
        return

    st.markdown("---")
    st.header("📊 Inventory Data Analysis & Classification (Uploaded CSV)")

    # 1. Read CSV into a pandas dataframe
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ Failed to read CSV file: {e}")
        return

    # Dataset properties
    st.markdown("##### 📁 Uploaded Dataset Properties")
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    col_stat1.metric("Number of Rows", f"{len(df):,}")
    col_stat2.metric("Number of Columns", f"{len(df.columns):,}")
    col_stat3.info(f"**Column Names:** `{', '.join(df.columns.tolist())}`")

    # Detect Schemas
    has_sales = any("sale" in c.lower() for c in df.columns)
    has_hits = any("hit" in c.lower() for c in df.columns)
    has_classification_schema = has_sales and has_hits

    has_units = "Units" in df.columns or any("unit" in c.lower() and "cost" not in c.lower() for c in df.columns)
    has_unit_cost = any("unit cost" in c.lower() or "cost" in c.lower() for c in df.columns)
    has_valuation_schema = "Units" in df.columns and "Unit Cost" in df.columns

    if not has_classification_schema and not has_valuation_schema:
        st.error(
            "⚠️ **Unrecognized Inventory Schema**: The uploaded CSV does not match either supported inventory schema:\n\n"
            "- **Multi-Criteria Inventory Classification**: Requires `Item Number`, `Item Name`, `UoM`, `Sales ($)`, and `Hits`.\n"
            "- **Inventory Valuation & Category Analysis**: Requires `Product`, `Category`, `Units`, and `Unit Cost`.\n\n"
            f"**Columns found in uploaded file:** `{', '.join(df.columns.tolist())}`"
        )
        return

    # =========================================================================
    # 1. MULTICRITERIA INVENTORY CLASSIFICATION (A, B, C, D)
    # =========================================================================
    if has_classification_schema:
        st.subheader("🏷️ Multi-Criteria Inventory Classification (A, B, C, D)")
        st.markdown(
            """
            **Inventory Analytics Module Procedure:**
            - **Sales ($)**: Ranked descending (tied by Item Number ascending) $\\rightarrow$ Cumulative %: A ($\\le 60\\%$), B ($>60\\%$ to $\\le 80\\%$), C ($>80\\%$ to $\\le 90\\%$), D ($>90\\%$).
            - **Hits**: Ranked descending (tied by Item Number ascending) $\\rightarrow$ Cumulative %: A ($\\le 60\\%$), B ($>60\\%$ to $\\le 80\\%$), C ($>80\\%$ to $\\le 90\\%$), D ($>90\\%$).
            - **Scores**: $A = 4, B = 3, C = 2, D = 1$.
            - **Combined Weighted Score**: $0.65 \\times (\\text{Sales Score}) + 0.35 \\times (\\text{Hits Score})$.
            - **Final Inventory Class**: $A \\ge 3.25$, $B \\in [2.50, 3.25)$, $C \\in [1.75, 2.50)$, $D < 1.75$.
            """
        )

        try:
            class_res = it.classify_inventory_multicriteria(df)
            classified_df = class_res["classified_df"]
            summary_df = class_res["summary_df"]
            validation_messages = class_res["validation_messages"]
            warnings = class_res["warnings"]
            metrics = class_res["metrics"]
            sales_col = class_res["sales_col"]
            hits_col = class_res["hits_col"]

            # Display non-crash warning messages for blanks, non-numeric, or duplicates
            if warnings:
                with st.expander("⚠️ Data Quality & Hygiene Alerts", expanded=True):
                    for w in warnings:
                        st.warning(w)

            # 2. Required Classification Summary table
            st.markdown("##### 📋 Required Classification Summary")
            formatted_summary = summary_df.style.format({
                "Number of Items": "{:,}",
                "Total Sales ($)": "${:,.2f}",
            })
            st.dataframe(formatted_summary, use_container_width=True, hide_index=True)

            # 3. Validation confirmation messages
            st.markdown("##### 🛡️ Reconciliation & Validation Checks")
            for msg in validation_messages:
                if "PASSED" in msg:
                    st.success(f"✅ {msg}")
                else:
                    st.error(f"❌ {msg}")

            # Metric Cards
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Class A Items (>= 3.25)", f"{metrics['class_a_items']:,}", f"${metrics['class_a_sales']:,.2f}")
            c2.metric("Class B Items (2.50 - 3.24)", f"{metrics['class_b_items']:,}", f"${metrics['class_b_sales']:,.2f}")
            c3.metric("Class C Items (1.75 - 2.49)", f"{metrics['class_c_items']:,}", f"${metrics['class_c_sales']:,.2f}")
            c4.metric("Class D Items (< 1.75)", f"{metrics['class_d_items']:,}", f"${metrics['class_d_sales']:,.2f}")

            # 4. Top 5 items by Sales within each final class
            st.markdown("##### 🏆 Top 5 Items by Sales within Each Final Class")
            tab_a, tab_b, tab_c, tab_d = st.tabs([
                "Class A (Top 5)",
                "Class B (Top 5)",
                "Class C (Top 5)",
                "Class D (Top 5)",
            ])

            # Prepare column format dictionary
            format_dict = {}
            if sales_col in classified_df.columns:
                format_dict[sales_col] = "${:,.2f}"
            if hits_col in classified_df.columns:
                format_dict[hits_col] = "{:,.0f}"
            if "Sales Cumulative %" in classified_df.columns:
                format_dict["Sales Cumulative %"] = "{:.2f}%"
            if "Hits Cumulative %" in classified_df.columns:
                format_dict["Hits Cumulative %"] = "{:.2f}%"
            if "Sales Score" in classified_df.columns:
                format_dict["Sales Score"] = "{:d}"
            if "Hits Score" in classified_df.columns:
                format_dict["Hits Score"] = "{:d}"
            if "Combined Weighted Score" in classified_df.columns:
                format_dict["Combined Weighted Score"] = "{:.2f}"

            for tab_obj, class_name in zip([tab_a, tab_b, tab_c, tab_d], ["A", "B", "C", "D"]):
                with tab_obj:
                    subset = classified_df[classified_df["Final Inventory Class"] == class_name]
                    if subset.empty:
                        st.info(f"No items classified into Class {class_name}.")
                    else:
                        top_5 = subset.sort_values(by=sales_col, ascending=False).head(5)
                        st.dataframe(top_5.style.format(format_dict), use_container_width=True)

            # 1. Full Classified DataFrame Display
            st.markdown("##### 📑 Full Classified Inventory Dataset")
            st.markdown(
                "Complete item master with original attributes, separate criterion cumulative rankings, "
                "scores, combined weighted score, and final inventory class:"
            )
            st.dataframe(classified_df.style.format(format_dict), use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error during multi-criteria inventory classification: {e}")

    # =========================================================================
    # 2. UNITS & VALUATION ANALYSIS (If Product, Category, Units, Unit Cost present)
    # =========================================================================
    if has_valuation_schema:
        st.subheader("📦 Inventory Valuation & Category Breakdown")

        # Convert numeric columns cleanly
        if not pd.api.types.is_numeric_dtype(df["Units"]):
            df["Units"] = pd.to_numeric(df["Units"].astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")
        else:
            df["Units"] = pd.to_numeric(df["Units"], errors="coerce")

        if not pd.api.types.is_numeric_dtype(df["Unit Cost"]):
            df["Unit Cost"] = pd.to_numeric(
                df["Unit Cost"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False).str.strip(),
                errors="coerce",
            )
        else:
            df["Unit Cost"] = pd.to_numeric(df["Unit Cost"], errors="coerce")

        # Calculate Inventory Value = Units * Unit Cost
        df["Inventory Value"] = df["Units"] * df["Unit Cost"]
        total_units = df["Units"].sum()
        avg_unit_cost = df["Unit Cost"].mean()

        # Product with highest units
        if not df.empty and df["Units"].notna().any():
            max_idx = df["Units"].idxmax()
            prod_col = "Product" if "Product" in df.columns else ("Item Name" if "Item Name" in df.columns else df.columns[0])
            highest_units_product = df.loc[max_idx, prod_col]
            highest_units_val = df.loc[max_idx, "Units"]
        else:
            highest_units_product = "N/A"
            highest_units_val = 0

        # Group data by Category if present
        if "Category" in df.columns:
            units_by_category = df.groupby("Category", as_index=False)["Units"].sum()
            units_by_category.columns = ["Category", "Total Units"]

        total_inventory_value = df["Inventory Value"].sum()

        st.markdown("##### 📈 Key Inventory Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Units", f"{total_units:,.0f}")
        m2.metric("Average Unit Cost", f"${avg_unit_cost:,.2f}")
        m3.metric("Product with Highest Units", f"{highest_units_product} ({highest_units_val:,.0f})")
        m4.metric("Total Inventory Value", f"${total_inventory_value:,.2f}")

        if "Category" in df.columns:
            st.markdown("##### 📦 Total Units by Category")
            c_col1, c_col2 = st.columns([1, 1])
            with c_col1:
                st.dataframe(
                    units_by_category.style.format({"Total Units": "{:,.0f}"}),
                    use_container_width=True,
                    hide_index=True,
                )
            with c_col2:
                st.bar_chart(units_by_category.set_index("Category"))

        st.markdown("##### 📋 Uploaded Dataframe (with Calculated Inventory Value)")
        formatted_df = df.style.format({
            "Units": "{:,.0f}",
            "Unit Cost": "${:,.2f}",
            "Inventory Value": "${:,.2f}",
        })
        st.dataframe(formatted_df, use_container_width=True)

    st.markdown("---")


def run_streamlit_app():
    st.set_page_config(
        page_title="Supply Chain Analytics Inventory Agent",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    agent = InventoryDecisionAgent()

    # Sidebar
    st.sidebar.title("📦 Supply Chain Agent")
    st.sidebar.markdown(
        """
        **Autonomous Inventory Decision Engine**
        
        *Supported Models:*
        1. 🏷️ **Economic Order Quantity (EOQ)**
        2. 📉 **Quantity Discount Analysis**
        3. 🛡️ **Safety Stock & Reorder Point**
        4. ⏱️ **Fixed-Period Review (R, S)**
        5. 📰 **Newsvendor Model**
        """
    )
    st.sidebar.divider()

    mode = st.sidebar.radio(
        "Navigation Mode:",
        ["🤖 Natural Language Agent", "🧮 Interactive Model Calculators", "📐 Architecture & Theory"],
    )

    # DATA UPLOAD (Last section in the sidebar)
    st.sidebar.divider()
    st.sidebar.subheader("Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

    # If CSV is uploaded, process and display results on the main page
    if uploaded_file is not None:
        process_and_render_csv_upload(uploaded_file)

    if mode == "🤖 Natural Language Agent":
        st.title("📦 Supply Chain Analytics: Inventory Decision Agent")
        st.markdown(
            """
            This agent processes natural-language inventory questions through a deterministic 6-stage architecture:
            **Customer Question → Identify Model → Extract Inputs → Select Python Tool → Calculate → Verify → Explain**
            """
        )

        # Sample queries accordion
        with st.expander("💡 Load a Pre-Configured Supply Chain Query", expanded=False):
            sample_scenarios = {
                "EOQ: Standard Manufacturing Replenishment": (
                    "Our manufacturing plant has an annual demand of 12,000 electric motors. "
                    "Each order setup costs $75, and the inventory holding cost is $3 per motor per year. "
                    "The lead time is 4 days. What is our optimal EOQ, how often should we order, and what is our reorder point?"
                ),
                "Quantity Discount: Multi-Tier Wholesale Purchasing": (
                    "Annual demand for microcontrollers is 15,000 units with an ordering cost of $100 per PO. "
                    "Holding cost rate is 20% of unit price. The vendor offers price breaks: "
                    "Tier 1: 0-999 units at $10.00. Tier 2: 1000-2499 units at $9.00. Tier 3: 2500+ units at $8.00. "
                    "Which quantity bracket minimizes our total annual purchasing and inventory cost?"
                ),
                "Safety Stock & ROP: Retail Distribution Buffer": (
                    "A distribution center faces average daily demand of 80 units with a standard deviation of 12 units. "
                    "Replenishment lead time is 6 days. We require a 95% cycle service level. "
                    "What safety stock should we carry, and at what inventory level should we trigger a replenishment reorder?"
                ),
                "Fixed-Period Review: Periodic Supermarket Ordering": (
                    "A grocery chain reviews dairy stock every 7 days, and replenishment lead time takes 3 days. "
                    "Average daily demand is 120 cartons with a standard deviation of 18 cartons. "
                    "Target service level is 98%. Current on-hand inventory is 450 cartons. "
                    "Calculate the target order-up-to level and how many cartons we should order today."
                ),
                "Newsvendor: Seasonal Holiday Apparel": (
                    "We are procuring a seasonal winter coat. Retail selling price is $150, unit purchase cost is $65, "
                    "and any unsold coats at season end are salvaged for $25. "
                    "Forecasted seasonal demand averages 600 coats with a standard deviation of 120 coats. "
                    "Determine the critical ratio and the optimal quantity to procure."
                ),
            }
            selected_sample = st.selectbox(
                "Choose a scenario to inspect:", list(sample_scenarios.keys())
            )
            default_query = sample_scenarios[selected_sample]

        query_input = st.text_area(
            "Enter your inventory question in natural language:",
            value=default_query,
            height=110,
            help="Describe your demand, costs, lead times, service levels, or discount tiers.",
        )

        col_run, col_clear = st.columns([1, 6])
        run_clicked = col_run.button("🚀 Run Agent Analysis", type="primary")

        if run_clicked or query_input:
            resp: AgentResponse = agent.process_query(query_input)

            st.divider()
            st.subheader("🔍 Execution Pipeline Trace")

            # Stage 1 & 2: Identification
            col_m1, col_m2, col_m3 = st.columns([3, 2, 2])
            with col_m1:
                st.metric("1. Identified Model", resp.model_name)
            with col_m2:
                conf_pct = int(resp.confidence * 100)
                st.metric("Confidence Score", f"{conf_pct}%")
            with col_m3:
                st.metric("Selected Tool", resp.tool_name)

            if resp.missing_inputs:
                st.warning(f"⚠️ Missing Inputs Required: {', '.join(resp.missing_inputs)}")
                st.markdown(resp.explanation)
                return

            # Stage 3: Extracted Inputs
            st.markdown("##### 2. Extracted Input Parameters")
            inputs_display = []
            for k, v in resp.extracted_inputs.items():
                if k == "discount_tiers":
                    v_str = f"{len(v)} tiers defined"
                else:
                    v_str = f"{v:,}" if isinstance(v, (int, float)) else str(v)
                inputs_display.append({"Parameter Name": k, "Extracted Value": v_str})
            st.dataframe(pd.DataFrame(inputs_display), use_container_width=True, hide_index=True)

            # Stage 4 & 5: Calculation Results
            st.markdown("##### 3. Deterministic Python Calculation Output")
            res_dict = resp.calculation_results.get("results", {}) if resp.calculation_results else {}

            if resp.model_name == InventoryModelType.EOQ:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Optimal Batch (EOQ)", f"{res_dict.get('optimal_order_quantity_int'):,} units")
                c2.metric("Orders Per Year", f"{res_dict.get('orders_per_year')}")
                c3.metric("Cycle Interval", f"{res_dict.get('cycle_time_days')} days")
                c4.metric("Total Inventory Cost", f"${res_dict.get('total_inventory_cost'):,.2f}")

                # EOQ Cost Curve Chart
                demand = resp.extracted_inputs["demand"]
                s = resp.extracted_inputs["order_cost"]
                h = resp.extracted_inputs["holding_cost"]
                eoq_val = res_dict.get("eoq", 100)

                q_range = np.linspace(max(10, eoq_val * 0.2), eoq_val * 2.2, 100)
                df_eoq = pd.DataFrame({
                    "Order Quantity (Q)": q_range,
                    "Annual Holding Cost ($)": (q_range / 2.0) * h,
                    "Annual Ordering Cost ($)": (demand / q_range) * s,
                    "Total Cost ($)": (q_range / 2.0) * h + (demand / q_range) * s,
                }).set_index("Order Quantity (Q)")

                st.line_chart(df_eoq, use_container_width=True)

            elif resp.model_name == InventoryModelType.QUANTITY_DISCOUNT:
                c1, c2, c3 = st.columns(3)
                c1.metric("Recommended Tier", f"Tier {res_dict.get('optimal_tier')}")
                c2.metric("Optimal Order Quantity", f"{res_dict.get('optimal_order_quantity'):,} units")
                c3.metric("Min Total Annual Cost", f"${res_dict.get('minimum_total_annual_cost'):,.2f}")

                tiers_df = pd.DataFrame(res_dict.get("tier_breakdown", []))
                if not tiers_df.empty:
                    st.dataframe(
                        tiers_df[[
                            "tier_index", "min_qty", "max_qty", "unit_price",
                            "candidate_q", "annual_purchase_cost", "annual_holding_cost",
                            "annual_ordering_cost", "total_annual_cost"
                        ]],
                        use_container_width=True,
                        hide_index=True,
                    )
                    chart_df = tiers_df[["tier_index", "total_annual_cost"]].set_index("tier_index")
                    st.bar_chart(chart_df, use_container_width=True)

            elif resp.model_name == InventoryModelType.SAFETY_STOCK_ROP:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Reorder Point (ROP)", f"{res_dict.get('reorder_point_int'):,} units")
                c2.metric("Safety Stock (SS)", f"{res_dict.get('safety_stock_int'):,} units")
                c3.metric("Lead Time Demand", f"{res_dict.get('lead_time_demand'):,} units")
                c4.metric("Z-Score", f"{res_dict.get('z_score')}")

                # Demand Distribution Normal Curve
                ltd = res_dict.get("lead_time_demand", 100)
                sigma = res_dict.get("sigma_lead_time_demand", 10)
                if sigma > 0:
                    x = np.linspace(max(0, ltd - 3.5 * sigma), ltd + 3.5 * sigma, 100)
                    y = norm.pdf(x, ltd, sigma)
                    df_norm = pd.DataFrame({"Lead Time Demand": x, "Probability Density": y}).set_index("Lead Time Demand")
                    st.line_chart(df_norm, use_container_width=True)

            elif resp.model_name == InventoryModelType.FIXED_PERIOD_REVIEW:
                c1, c2, c3 = st.columns(3)
                c1.metric("Target Level (S)", f"{res_dict.get('target_inventory_level_int'):,} units")
                c2.metric("Safety Stock", f"{res_dict.get('safety_stock_int'):,} units")
                if res_dict.get("order_quantity_int") is not None:
                    c3.metric("Order Quantity to Place", f"{res_dict.get('order_quantity_int'):,} units")
                else:
                    c3.metric("Protection Interval", f"{res_dict.get('protection_interval')} periods")

            elif resp.model_name == InventoryModelType.NEWSVENDOR:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Optimal Order (Q*)", f"{res_dict.get('optimal_order_quantity_int'):,} units")
                c2.metric("Critical Ratio", f"{res_dict.get('critical_ratio_pct')}%")
                c3.metric("Expected Sales", f"{res_dict.get('expected_sales'):,} units")
                c4.metric("Expected Profit", f"${res_dict.get('expected_profit'):,.2f}")

            # Stage 6: Student Verification Panel
            if resp.calculation_results:
                st.divider()
                render_pedagogical_verification_panel(
                    resp.calculation_results,
                    title="🎓 Step-by-Step Mathematical Verification for Students"
                )

            # Stage 7: Advisory Explanation
            st.divider()
            st.markdown("##### 5. Executive Advisory Report")
            st.markdown(resp.explanation)

    elif mode == "🧮 Interactive Model Calculators":
        st.title("🧮 Interactive Supply Chain Calculators")
        st.markdown("Fine-tune model parameters and observe immediate impact on inventory policies.")

        calc_tab1, calc_tab2, calc_tab3, calc_tab4, calc_tab5 = st.tabs([
            "EOQ", "Quantity Discount", "Safety Stock & ROP", "Fixed-Period Review", "Newsvendor"
        ])

        with calc_tab1:
            st.subheader("Economic Order Quantity (EOQ)")
            col1, col2, col3 = st.columns(3)
            d = col1.number_input("Annual Demand (D)", value=10000.0, step=500.0)
            s = col2.number_input("Order Setup Cost (S, $)", value=50.0, step=5.0)
            h = col3.number_input("Annual Holding Cost per Unit (H, $)", value=2.0, step=0.25)
            lead_d = col1.number_input("Lead Time (Days)", value=5.0, step=1.0)
            work_d = col2.number_input("Work Days / Year", value=250, step=10)

            res = it.calculate_eoq(demand=d, order_cost=s, holding_cost=h, lead_time_days=lead_d, work_days_per_year=work_d)
            render_pedagogical_verification_panel(res, title="EOQ Verification & Calculations")

        with calc_tab2:
            st.subheader("Quantity Discount Model")
            st.markdown("Evaluates all-units price break tiers against holding vs purchasing savings.")
            col1, col2 = st.columns(2)
            d_qd = col1.number_input("Annual Demand", value=15000.0, step=1000.0, key="qd_d")
            s_qd = col2.number_input("Ordering Cost ($)", value=100.0, step=10.0, key="qd_s")
            h_rate = col1.number_input("Holding Rate (% of Price)", value=20.0, step=1.0, key="qd_hr") / 100.0

            tiers_sample = [
                {"min_qty": 0, "max_qty": 999, "unit_price": 10.0},
                {"min_qty": 1000, "max_qty": 2499, "unit_price": 9.0},
                {"min_qty": 2500, "max_qty": None, "unit_price": 8.0},
            ]
            st.write("Configured Tiers: 0-999 @ $10.00 | 1000-2499 @ $9.00 | 2500+ @ $8.00")
            res_qd = it.calculate_quantity_discount(demand=d_qd, order_cost=s_qd, discount_tiers=tiers_sample, holding_cost_rate=h_rate)
            render_pedagogical_verification_panel(res_qd, title="Quantity Discount Verification & Calculations")

        with calc_tab3:
            st.subheader("Safety Stock & Reorder Point (ROP)")
            col1, col2, col3 = st.columns(3)
            avg_d = col1.number_input("Average Daily Demand (d)", value=50.0, step=5.0)
            lt = col2.number_input("Lead Time in Days (L)", value=5.0, step=1.0)
            csl = col3.slider("Target Service Level (CSL)", min_value=0.50, max_value=0.999, value=0.95, step=0.01)
            sig_d = col1.number_input("Demand Std Dev (sigma_d)", value=8.0, step=1.0)
            sig_lt = col2.number_input("Lead Time Std Dev (sigma_L)", value=0.0, step=0.5)

            res_rop = it.calculate_rop(
                avg_demand=avg_d, lead_time=lt, service_level=csl,
                demand_std_dev=sig_d, lead_time_std_dev=sig_lt
            )
            st.info(f"**Identified Case:** {res_rop['case'].title()}")
            render_pedagogical_verification_panel(res_rop, title="Safety Stock & ROP Verification & Calculations")

        with calc_tab4:
            st.subheader("Fixed-Period Review (Periodic Review / (R, S))")
            col1, col2, col3 = st.columns(3)
            fp_d = col1.number_input("Daily Demand (d)", value=100.0, step=10.0, key="fp_d")
            fp_r = col2.number_input("Review Period (R, days)", value=7.0, step=1.0, key="fp_r")
            fp_l = col3.number_input("Lead Time (L, days)", value=3.0, step=1.0, key="fp_l")
            fp_sig = col1.number_input("Daily Demand Std Dev", value=15.0, step=1.0, key="fp_sig")
            fp_csl = col2.slider("Service Level", min_value=0.50, max_value=0.999, value=0.98, step=0.01, key="fp_csl")
            fp_curr = col3.number_input("Current On-Hand Inventory", value=400.0, step=20.0, key="fp_curr")

            res_fp = it.calculate_periodic_review(
                avg_demand=fp_d, review_period=fp_r, lead_time=fp_l,
                demand_std_dev=fp_sig, service_level=fp_csl, current_inventory=fp_curr
            )
            render_pedagogical_verification_panel(res_fp, title="Fixed-Period Review Verification & Calculations")

        with calc_tab5:
            st.subheader("Newsvendor Model (Single-Period Stochastic)")
            col1, col2 = st.columns(2)
            nv_p = col1.number_input("Unit Selling Price (p, $)", value=120.0, step=5.0)
            nv_c = col2.number_input("Unit Procurement Cost (c, $)", value=70.0, step=5.0)
            nv_s = col1.number_input("Salvage Value (s, $)", value=30.0, step=5.0)
            nv_g = col2.number_input("Goodwill Penalty Cost (g, $)", value=0.0, step=5.0)
            nv_mu = col1.number_input("Mean Seasonal Demand (mu)", value=500.0, step=50.0)
            nv_sig = col2.number_input("Demand Std Dev (sigma)", value=80.0, step=10.0)

            res_nv = it.calculate_newsvendor(
                selling_price=nv_p, cost_price=nv_c, salvage_value=nv_s,
                penalty_cost=nv_g, mean_demand=nv_mu, std_demand=nv_sig
            )
            render_pedagogical_verification_panel(res_nv, title="Newsvendor Verification & Calculations")

    elif mode == "📐 Architecture & Theory":
        st.title("📐 Architecture & Supply Chain Theory")
        st.markdown(
            """
            ### 6-Stage Pipeline Architecture
            
            ```text
            +--------------------------------------------------------------------+
            | 1. Customer Question                                               |
            |    "Our demand is 10k, order cost is $50, holding is $2... EOQ?"  |
            +---------------------------------+----------------------------------+
                                              |
                                              v
            +---------------------------------+----------------------------------+
            | 2. Identify Model                                                  |
            |    NLP Classifier: Economic Order Quantity (EOQ)                   |
            +---------------------------------+----------------------------------+
                                              |
                                              v
            +---------------------------------+----------------------------------+
            | 3. Extract Inputs                                                  |
            |    D = 10,000, S = 50, H = 2.0                                     |
            +---------------------------------+----------------------------------+
                                              |
                                              v
            +---------------------------------+----------------------------------+
            | 4. Select Python Tool                                              |
            |    Tool: calculate_eoq() in inventory_tools.py                     |
            +---------------------------------+----------------------------------+
                                              |
                                              v
            +---------------------------------+----------------------------------+
            | 5. Calculate                                                       |
            |    EOQ = sqrt(2*D*S / H) = 707.11 units                            |
            +---------------------------------+----------------------------------+
                                              |
                                              v
            +---------------------------------+----------------------------------+
            | 6. Verify                                                          |
            |    Check Holding Cost == Ordering Cost; EOQ > 0; Feasibility       |
            +---------------------------------+----------------------------------+
                                              |
                                              v
            +--------------------------------------------------------------------+
            | 7. Explain                                                         |
            |    Executive summary, managerial trade-offs & policy guidance      |
            +--------------------------------------------------------------------+
            ```
            """
        )


if __name__ == "__main__":
    import streamlit.runtime
    is_streamlit_running = streamlit.runtime.exists()

    if is_streamlit_running:
        run_streamlit_app()
    elif "--cli" in sys.argv or (len(sys.argv) > 1 and sys.argv[1] not in ["--web", "--gui"]):
        run_cli()
    else:
        # If user executed `python3 app.py` directly from their terminal without streamlit
        # Automatically launch the Streamlit web application
        try:
            from streamlit.web import cli as stcli
            print("🚀 Launching Streamlit web application...")
            sys.argv = ["streamlit", "run", __file__]
            sys.exit(stcli.main())
        except Exception as e:
            print(f"Direct Streamlit launch notice: {e}")
            print("\nPlease start Streamlit with either:")
            print(f'  python3 -m streamlit run "{__file__}"')
            print(f'  streamlit run "{__file__}"')
