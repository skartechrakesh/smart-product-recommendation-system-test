import pandas as pd
import sqlite3
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

# =====================================================
# 🔥 GLOBAL CACHE
# =====================================================
_model = None
_user_index = None
_product_index = None
_interaction_matrix = None
_products_df = None


# =====================================================
# 🔥 LOAD DATA
# =====================================================
def load_data():
    conn = sqlite3.connect("database.db")

    users = pd.read_sql("SELECT user_id, age, gender FROM users", conn)

    products = pd.read_sql("""
        SELECT product_id, category, brand, price,
               purchase_frequency,
               customer_satisfaction,
               purchase_intent
        FROM products
    """, conn)

    purchases = pd.read_sql("SELECT user_id, product_id FROM purchases", conn)

    conn.close()

    products = products.rename(columns={
        "customer_satisfaction": "satisfaction"
    })

    return users, products, purchases


# =====================================================
# 🔥 BUILD INTERACTION MATRIX
# =====================================================
def build_matrix(users, products, purchases):
    # Merge
    df = purchases.merge(products, on="product_id", how="left")

    # Interaction strength (you can tune this)
    df["interaction"] = (
        1 +
        df["purchase_frequency"] * 0.5 +
        df["satisfaction"] * 0.5 +
        df["purchase_intent"] * 0.5
    )

    matrix = df.pivot_table(
        index="user_id",
        columns="product_id",
        values="interaction",
        fill_value=0
    )

    return matrix


# =====================================================
# 🔥 TRAIN MODEL (SVD)
# =====================================================
def train_model():
    global _model, _user_index, _product_index, _interaction_matrix, _products_df

    users, products, purchases = load_data()

    if purchases.empty:
        return None

    matrix = build_matrix(users, products, purchases)

    if matrix.empty:
        return None

    # Save indices
    _user_index = list(matrix.index)
    _product_index = list(matrix.columns)

    _interaction_matrix = matrix

    # SVD (latent factors)
    svd = TruncatedSVD(n_components=5, random_state=42)
    user_factors = svd.fit_transform(matrix)
    product_factors = svd.components_.T

    _model = {
        "svd": svd,
        "user_factors": user_factors,
        "product_factors": product_factors
    }

    _products_df = products

    return _model


# =====================================================
# 🔥 GET USER VECTOR
# =====================================================
def get_user_vector(user_id):
    if _model is None:
        train_model()

    if user_id not in _user_index:
        return None

    idx = _user_index.index(user_id)
    return _model["user_factors"][idx]


# =====================================================
# 🔥 RECOMMENDATION (ML BASED)
# =====================================================
def ml_recommend(user_id, top_n=5):

    if _model is None:
        train_model()

    if _model is None:
        return fallback_recommend()

    user_vector = get_user_vector(user_id)

    # New user → fallback
    if user_vector is None:
        return fallback_recommend()

    product_vectors = _model["product_factors"]

    # 🔥 PREDICT SCORES (dot product)
    scores = product_vectors @ user_vector

    rec_df = pd.DataFrame({
        "product_id": _product_index,
        "score": scores
    })

    # Merge with product info
    rec_df = rec_df.merge(_products_df, on="product_id", how="left")

    # Remove already purchased
    user_purchased = _interaction_matrix.loc[user_id]
    purchased_ids = user_purchased[user_purchased > 0].index.tolist()

    rec_df = rec_df[~rec_df["product_id"].isin(purchased_ids)]

    return rec_df.sort_values(by="score", ascending=False).head(top_n)


# =====================================================
# 🔥 FALLBACK
# =====================================================
def fallback_recommend():
    _, products, _ = load_data()

    rec = products.sort_values(
        by=["purchase_intent", "purchase_frequency"],
        ascending=False
    )

    return rec.head(20)


# =====================================================
# 🔥 BUDGET FILTER
# =====================================================
def ml_budget_recommend(user_id, budget, top_n=100):

    # 🔥 Step 1: Get large pool from ML
    rec = ml_recommend(user_id, top_n=100)

    # 🔥 Step 2: Apply budget filter
    rec_budget = rec[rec["price"] <= budget]

    # 🔥 Step 3: If enough → return
    if len(rec_budget) >= top_n:
        return rec_budget.sort_values(by="price", ascending=False).head(top_n)

    # 🔥 Step 4: Expand using ALL products within budget
    _, products, _ = load_data()
    budget_products = products[products["price"] <= budget]

    # Remove already selected products
    budget_products = budget_products[~budget_products["product_id"].isin(rec_budget["product_id"])]

    # Sort fallback products by popularity
    budget_products = budget_products.sort_values(
        by=["purchase_frequency", "purchase_intent"],
        ascending=False
    )

    # 🔥 Combine ML + fallback
    final = pd.concat([rec_budget, budget_products])

    # Remove duplicates
    final = final.drop_duplicates("product_id")

    # 🔥 Step 5: Still empty → cheapest products
    if final.empty:
        final = products.sort_values(by="price").head(top_n)

    # 🔥 Final sort (best UX)
    return final.sort_values(by="price", ascending=False).head(top_n)
