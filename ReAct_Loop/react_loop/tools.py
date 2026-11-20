SUPPLIERS = {
    "Acme Supply": {
        "laptop": 899.0,
        "monitor": 249.0,
    },
    "Northstar Wholesale": {
        "laptop": 875.5,
        "keyboard": 79.0,
    },
    "Metro Office": {
        "monitor": 229.5,
        "keyboard": 72.0,
    },
}


def search_suppliers(product: str) -> list[str]:
    return [
        supplier
        for supplier, catalog in SUPPLIERS.items()
        if product in catalog
    ]


def get_quote(supplier: str, product: str) -> float:
    if supplier not in SUPPLIERS:
        raise ValueError(f"Unknown supplier: {supplier}")
    if product not in SUPPLIERS[supplier]:
        raise ValueError(f"{supplier} does not sell {product}")
    return SUPPLIERS[supplier][product]
