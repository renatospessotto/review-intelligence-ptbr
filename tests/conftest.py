import pandas as pd
import pytest

from review_intel.core.schema import ASPECTS


def make_analysis(**overrides):
    asp = {a: "ausente" for a in ASPECTS}
    asp.update(overrides)
    return {"aspectos": asp, "motivo_principal": "m", "acao_sugerida": "a"}


@pytest.fixture
def raw_dir(tmp_path):
    """Mini Olist com duplicatas, texto vazio e títulos, para testar o prepare."""
    n = 60
    reviews = pd.DataFrame({
        "review_id": [f"r{i}" for i in range(n)] + ["r0"],
        "order_id": [f"o{i}" for i in range(n)] + ["o0"],
        "review_score": [(i % 5) + 1 for i in range(n)] + [1],
        "review_comment_title": ["Título"] + [None] * (n - 1) + [None],
        "review_comment_message": [f"texto número {i}\nlinha" for i in range(n - 2)] + [None, "Ok"] + ["dup"],
        "review_creation_date": "2018-01-01",
        "review_answer_timestamp": "2018-01-02",
    })
    orders = pd.DataFrame({
        "order_id": [f"o{i}" for i in range(n)],
        "customer_id": [f"c{i}" for i in range(n)],
        "order_status": "delivered",
        "order_purchase_timestamp": "2018-01-01",
        "order_approved_at": "2018-01-01",
        "order_delivered_carrier_date": "2018-01-02",
        "order_delivered_customer_date": ["2018-01-10" if i % 2 else "2018-01-03" for i in range(n)],
        "order_estimated_delivery_date": "2018-01-05",
    })
    customers = pd.DataFrame({"customer_id": [f"c{i}" for i in range(n)], "customer_unique_id": "u",
                              "customer_zip_code_prefix": 1, "customer_city": "x", "customer_state": "SP"})
    reviews.to_csv(tmp_path / "olist_order_reviews_dataset.csv", index=False)
    orders.to_csv(tmp_path / "olist_orders_dataset.csv", index=False)
    customers.to_csv(tmp_path / "olist_customers_dataset.csv", index=False)
    return tmp_path
