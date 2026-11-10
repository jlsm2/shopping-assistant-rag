"""Read-only product catalog.

`Catalog` is a Protocol because a real store integration will be an
`HttpCatalog` calling the store's own API. Whatever the source returns, only
`public_view()` output ever reaches the model or the HTTP response.
"""

import json
from pathlib import Path
from typing import Any, Protocol

from app.retrieval import tokenize

NAME_WEIGHT = 2


def public_view(product: dict[str, Any]) -> dict[str, Any]:
    """Allowlist of fields. Numeric stock, cost and anything internal stay out."""
    return {
        "id": product["id"],
        "name": product["name"],
        "price": product["price"],
        "in_stock": product.get("stock", 0) > 0,
        "description": product.get("description", ""),
        "url": product["url"],
    }


class Catalog(Protocol):
    def search(self, query: str, max_price: float | None = None) -> list[dict[str, Any]]: ...

    def get(self, product_id: str) -> dict[str, Any] | None: ...


class JsonCatalog:
    def __init__(self, path: Path) -> None:
        products = json.loads(path.read_text(encoding="utf-8"))
        self._products = {product["id"]: product for product in products}
        self._terms = {
            product["id"]: (
                set(tokenize(product["name"] + " " + " ".join(product.get("tags", [])))),
                set(tokenize(product.get("description", ""))),
            )
            for product in products
        }

    def get(self, product_id: str) -> dict[str, Any] | None:
        return self._products.get(product_id)

    def search(self, query: str, max_price: float | None = None) -> list[dict[str, Any]]:
        """Best matches first; in-stock and cheaper products break ties."""
        terms = set(tokenize(query))
        ranked = []
        for product in self._products.values():
            if max_price is not None and product["price"] > max_price:
                continue
            name_terms, description_terms = self._terms[product["id"]]
            score = NAME_WEIGHT * len(terms & name_terms) + len(terms & description_terms)
            if score or not terms:
                ranked.append((-score, product.get("stock", 0) <= 0, product["price"], product))
        ranked.sort(key=lambda item: item[:3])
        return [item[3] for item in ranked]
