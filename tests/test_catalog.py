from app.catalog import public_view

INTERNAL_PRODUCT = {
    "id": "nude-curto",
    "name": "Set Nude Curto",
    "price": 59.9,
    "stock": 12,
    "cost": 11.5,
    "supplier": "ACME",
    "description": "Nude.",
    "tags": ["nude"],
    "url": "https://lojademo.example/produtos/nude-curto",
}


def test_public_view_exposes_only_allowlisted_fields():
    view = public_view(INTERNAL_PRODUCT)

    assert set(view) == {"id", "name", "price", "in_stock", "description", "url"}
    assert view["in_stock"] is True


def test_public_view_never_leaks_numeric_stock_or_cost():
    assert "12" not in str(public_view(INTERNAL_PRODUCT))
    assert "11.5" not in str(public_view(INTERNAL_PRODUCT))


def test_public_view_marks_zero_stock_as_unavailable():
    assert public_view({**INTERNAL_PRODUCT, "stock": 0})["in_stock"] is False


def test_get_returns_product_or_none(catalog):
    assert catalog.get("nude-curto")["name"] == "Set Nude Curto"
    assert catalog.get("nao-existe") is None


def test_search_finds_by_name_ignoring_accents_and_plural(catalog):
    assert catalog.search("francesinhas classicas")[0]["id"] == "french-classica"


def test_search_filters_by_max_price(catalog):
    results = catalog.search("nude", max_price=60)

    assert [product["id"] for product in results][0] == "nude-curto"
    assert all(product["price"] <= 60 for product in results)


def test_search_puts_in_stock_products_first_on_ties(catalog):
    results = catalog.search("francesinha")

    assert [product["id"] for product in results[:2]] == ["french-classica", "french-baby"]


def test_search_without_match_returns_nothing(catalog):
    assert catalog.search("geladeira") == []


def test_search_with_only_stopwords_lists_cheapest_first(catalog):
    results = catalog.search("o que tem", max_price=20)

    assert [product["id"] for product in results] == ["adesivos-dupla-face", "cola-extra-forte"]
