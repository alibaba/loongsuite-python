"""Synthetic tools for the e-commerce customer service example."""

import json
from collections.abc import Callable
from typing import Any

from langchain_core.tools import tool

PRODUCTS = {
    "P-DEMO-001": {
        "name": "CloudStep commuter shoes",
        "category": "shoes",
        "material": "water-resistant woven upper and rubber outsole",
        "use_cases": ["daily commute", "light walking"],
        "sizes": ["39", "40", "41", "42", "43"],
    },
    "P-DEMO-002": {
        "name": "Breeze insulated bottle",
        "category": "drinkware",
        "material": "food-grade stainless steel",
        "use_cases": ["office", "short trips"],
        "sizes": ["500ml"],
    },
}

INVENTORY = {
    ("P-DEMO-001", "39"): 8,
    ("P-DEMO-001", "40"): 12,
    ("P-DEMO-001", "41"): 6,
    ("P-DEMO-001", "42"): 4,
    ("P-DEMO-001", "43"): 0,
    ("P-DEMO-002", "500ml"): 15,
}

ORDERS = {
    "DEMO-1001": {
        "product_id": "P-DEMO-001",
        "size": "42",
        "status": "delivered",
        "delivered_days_ago": 1,
    },
    "DEMO-1002": {
        "product_id": "P-DEMO-002",
        "size": "500ml",
        "status": "in_transit",
        "delivered_days_ago": None,
    },
}

POLICIES = {
    "quality_issue": {
        "window_days": 30,
        "resolution": "inspection followed by replacement or refund when confirmed",
        "required_evidence": "order ID and a text description of the issue",
    },
    "change_of_mind": {
        "window_days": 7,
        "resolution": "return when the product is unused and complete",
        "required_evidence": "order ID and product condition",
    },
}


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def _safe_lookup(operation: Callable[[], Any]) -> str:
    """Keep synthetic tool failures explainable instead of crashing the agent."""
    try:
        return _json({"ok": True, "data": operation()})
    except (KeyError, TypeError, ValueError) as exc:
        return _json({"ok": False, "error": str(exc)})


@tool
def search_product_catalog(query: str) -> str:
    """Search the synthetic product catalog by product name, category, or use case."""

    def search() -> list[dict[str, Any]]:
        normalized = query.casefold()
        matches = []
        for product_id, product in PRODUCTS.items():
            searchable = " ".join(
                [
                    product_id,
                    product["name"],
                    product["category"],
                    *product["use_cases"],
                ]
            ).casefold()
            if not normalized or normalized in searchable:
                matches.append(
                    {
                        "product_id": product_id,
                        "name": product["name"],
                        "category": product["category"],
                    }
                )
        return matches

    return _safe_lookup(search)


@tool
def query_product_knowledge(product_id: str) -> str:
    """Return specifications and suitable use cases for a synthetic product ID."""

    def lookup() -> dict[str, Any]:
        if product_id not in PRODUCTS:
            raise KeyError(f"unknown product: {product_id}")
        return {"product_id": product_id, **PRODUCTS[product_id]}

    return _safe_lookup(lookup)


@tool
def check_inventory(product_id: str, size: str) -> str:
    """Check synthetic inventory for one product and size."""

    def lookup() -> dict[str, Any]:
        if product_id not in PRODUCTS:
            raise KeyError(f"unknown product: {product_id}")
        if (product_id, size) not in INVENTORY:
            raise KeyError(f"unknown size {size} for product {product_id}")
        quantity = INVENTORY[(product_id, size)]
        return {
            "product_id": product_id,
            "size": size,
            "available": quantity > 0,
            "quantity": quantity,
        }

    return _safe_lookup(lookup)


@tool
def lookup_order_history(order_id: str) -> str:
    """Look up a synthetic order. Only DEMO-prefixed IDs exist in this example."""

    def lookup() -> dict[str, Any]:
        if order_id not in ORDERS:
            raise KeyError(f"order not found: {order_id}")
        return {"order_id": order_id, **ORDERS[order_id]}

    return _safe_lookup(lookup)


@tool
def query_after_sales_policy(issue_type: str) -> str:
    """Return the synthetic policy for quality_issue or change_of_mind."""

    def lookup() -> dict[str, Any]:
        if issue_type not in POLICIES:
            raise KeyError(f"unsupported issue type: {issue_type}")
        return {"issue_type": issue_type, **POLICIES[issue_type]}

    return _safe_lookup(lookup)


@tool
def assess_issue(order_id: str, description: str) -> str:
    """Provide a bounded next step using a synthetic order and text description."""

    def assess() -> dict[str, Any]:
        if order_id not in ORDERS:
            raise KeyError(f"order not found: {order_id}")
        if not description.strip():
            raise ValueError("issue description is required")
        order = ORDERS[order_id]
        if order["status"] != "delivered":
            next_step = "wait for delivery or contact logistics support"
        else:
            next_step = (
                "submit the order ID and issue description for inspection"
            )
        return {
            "order_id": order_id,
            "assessment": "manual verification required",
            "next_step": next_step,
        }

    return _safe_lookup(assess)


PRESALES_TOOLS = [
    search_product_catalog,
    query_product_knowledge,
    check_inventory,
]

AFTERSALES_TOOLS = [
    lookup_order_history,
    query_after_sales_policy,
    assess_issue,
]
