import logging
from meta_generate.utils import generate_mock_data

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

generated_mock = """
def generate_mock_data(n: int):
    import random
    from datetime import datetime, timedelta

    def mock_users(n):
        users = []
        first_names = [
            "James",
            "Mary",
            "John",
            "Patricia",
            "Robert",
            "Jennifer",
            "Michael",
            "Linda",
            "William",
            "Elizabeth",
        ]
        last_names = [
            "Smith",
            "Johnson",
            "Williams",
            "Brown",
            "Jones",
            "Garcia",
            "Miller",
            "Davis",
            "Rodriguez",
            "Martinez",
        ]
        domains = [
            "gmail.com",
            "yahoo.com",
            "hotmail.com",
            "outlook.com",
            "example.com",
        ]

        base_date = datetime(2020, 1, 1)

        for i in range(n):
            first_name = random.choice(first_names)
            last_name = random.choice(last_names)
            days_offset = random.randint(0, 1460)
            created_at = base_date + timedelta(days=days_offset)

            user = {
                "user_id": i + 1,
                "first_name": first_name,
                "last_name": last_name,
                "email": f"{first_name.lower()}.{last_name.lower()}{random.randint(1,999)}@{random.choice(domains)}",
                "age": random.randint(18, 80),
                "is_active": random.choice([True, False]),
                "created_at": created_at.isoformat(),
                "last_login": (
                    created_at + timedelta(days=random.randint(0, 30))
                ).isoformat(),
            }
            users.append(user)
        return users

    def mock_products(n):
        products = []
        categories = [
            "Electronics",
            "Clothing",
            "Home",
            "Sports",
            "Books",
            "Toys",
            "Food",
            "Beauty",
        ]
        adjectives = [
            "Premium",
            "Basic",
            "Deluxe",
            "Standard",
            "Professional",
            "Essential",
            "Advanced",
            "Classic",
        ]
        items = [
            "Widget",
            "Gadget",
            "Tool",
            "Device",
            "Set",
            "Kit",
            "Package",
            "Bundle",
        ]

        for i in range(n):
            price = round(random.uniform(9.99, 999.99), 2)

            product = {
                "product_id": i + 1,
                "name": f"{random.choice(adjectives)} {random.choice(categories)} {random.choice(items)}",
                "category": random.choice(categories),
                "price": price,
                "stock": random.randint(0, 500),
                "sku": f"SKU-{random.randint(100000, 999999)}",
                "description": f"High-quality {random.choice(categories).lower()} product.",
                "weight": round(random.uniform(0.1, 50.0), 2),
                "in_stock": random.choice([True, False]),
            }
            products.append(product)
        return products

    def mock_orders(n, user_count, product_count):
        orders = []
        statuses = ["pending", "processing", "shipped", "delivered", "cancelled"]
        base_date = datetime(2023, 1, 1)

        for i in range(n):
            days_offset = random.randint(0, 365)
            order_date = base_date + timedelta(days=days_offset)

            order = {
                "order_id": i + 1,
                "user_id": random.randint(1, user_count),
                "product_id": random.randint(1, product_count),
                "quantity": random.randint(1, 10),
                "order_date": order_date.isoformat(),
                "status": random.choice(statuses),
                "total_amount": round(random.uniform(20.0, 2000.0), 2),
                "shipping_address": f"{random.randint(100, 999)} Main St, City, State {random.randint(10000, 99999)}",
                "tracking_number": (
                    f"TN{random.randint(1000000000, 9999999999)}"
                    if random.choice([True, False])
                    else None
                ),
            }
            orders.append(order)
        return orders

    def mock_categories(n):
        categories = []
        descriptions = [
            "Various electronic devices and accessories",
            "Clothing items for men, women, and children",
            "Home furniture and decor items",
            "Sports equipment and accessories",
            "Books across various genres",
            "Toys and games for all ages",
            "Food and grocery items",
            "Beauty and personal care products",
        ]

        for i in range(n):
            category = {
                "category_id": i + 1,
                "name": f"Category {i + 1}",
                "description": (
                    descriptions[i % len(descriptions)]
                    if i < len(descriptions)
                    else f"Description for category {i + 1}"
                ),
                "is_active": random.choice([True, False]),
                "display_order": random.randint(1, 100),
            }
            categories.append(category)
        return categories

    records_per_table = n // 4
    remainder = n % 4

    users_data = mock_users(records_per_table + (1 if remainder >= 1 else 0))
    products_data = mock_products(records_per_table + (1 if remainder >= 2 else 0))
    orders_data = mock_orders(
        records_per_table + (1 if remainder >= 3 else 0),
        len(users_data),
        len(products_data),
    )
    categories_data = mock_categories(records_per_table)
    combined_data = users_data + products_data + orders_data + categories_data
    return combined_data
"""


def run():
    logging.info("Testing generated mock data execution...")
    res = generate_mock_data(generated_mock, 10)
    logging.info(res)
