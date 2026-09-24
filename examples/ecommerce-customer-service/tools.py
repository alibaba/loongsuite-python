"""电商客服示例使用的虚构工具和数据。"""

import json
from collections.abc import Callable
from typing import Any

from langchain_core.tools import tool

PRODUCTS = {
    "P-DEMO-001": {
        "name": "云步通勤鞋",
        "category": "鞋履",
        "material": "防泼水织物鞋面和橡胶外底",
        "use_cases": ["日常通勤", "轻度步行"],
        "sizes": ["39", "40", "41", "42", "43"],
    },
    "P-DEMO-002": {
        "name": "清风保温杯",
        "category": "杯具",
        "material": "食品级不锈钢",
        "use_cases": ["办公室", "短途出行"],
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
        "status": "已签收",
        "delivered_days_ago": 1,
    },
    "DEMO-1002": {
        "product_id": "P-DEMO-002",
        "size": "500ml",
        "status": "运输中",
        "delivered_days_ago": None,
    },
}

POLICIES = {
    "quality_issue": {
        "window_days": 30,
        "resolution": "核验属实后可换货或退款",
        "required_evidence": "订单号和问题文字描述",
    },
    "change_of_mind": {
        "window_days": 7,
        "resolution": "商品未使用且配件完整时可以退货",
        "required_evidence": "订单号和商品状态说明",
    },
}


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def _safe_lookup(operation: Callable[[], Any]) -> str:
    """将虚构工具异常转为可解释的结果，避免中断 Agent。"""
    try:
        return _json({"ok": True, "data": operation()})
    except (KeyError, TypeError, ValueError) as exc:
        message = (
            exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
        )
        return _json({"ok": False, "error": str(message)})


@tool
def search_product_catalog(query: str) -> str:
    """按商品名称、分类或使用场景搜索虚构商品目录。"""

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
    """按虚构商品编号查询规格和适用场景。"""

    def lookup() -> dict[str, Any]:
        if product_id not in PRODUCTS:
            raise KeyError(f"未找到商品：{product_id}")
        return {"product_id": product_id, **PRODUCTS[product_id]}

    return _safe_lookup(lookup)


@tool
def check_inventory(product_id: str, size: str) -> str:
    """查询指定虚构商品和尺码的库存。"""

    def lookup() -> dict[str, Any]:
        if product_id not in PRODUCTS:
            raise KeyError(f"未找到商品：{product_id}")
        if (product_id, size) not in INVENTORY:
            raise KeyError(f"商品 {product_id} 没有尺码 {size}")
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
    """查询虚构订单；本示例只包含 DEMO- 前缀的订单号。"""

    def lookup() -> dict[str, Any]:
        if order_id not in ORDERS:
            raise KeyError(f"未找到订单：{order_id}")
        return {"order_id": order_id, **ORDERS[order_id]}

    return _safe_lookup(lookup)


@tool
def query_after_sales_policy(issue_type: str) -> str:
    """查询 quality_issue（质量问题）或 change_of_mind（无理由退货）政策。"""

    def lookup() -> dict[str, Any]:
        if issue_type not in POLICIES:
            raise KeyError(f"不支持的问题类型：{issue_type}")
        return {"issue_type": issue_type, **POLICIES[issue_type]}

    return _safe_lookup(lookup)


@tool
def assess_issue(order_id: str, description: str) -> str:
    """根据虚构订单和文字问题描述给出有限的下一步建议。"""

    def assess() -> dict[str, Any]:
        if order_id not in ORDERS:
            raise KeyError(f"未找到订单：{order_id}")
        if not description.strip():
            raise ValueError("必须提供问题描述")
        order = ORDERS[order_id]
        if order["status"] != "已签收":
            next_step = "等待商品送达，或联系物流客服查询"
        else:
            next_step = "提交订单号和问题描述，等待人工核验"
        return {
            "order_id": order_id,
            "assessment": "需要人工核验",
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
