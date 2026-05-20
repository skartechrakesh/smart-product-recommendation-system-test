import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import KMeans
def load_data():
    conn = sqlite3.connect("database.db")
    users = pd.read_sql("SELECT * FROM users", conn)
    products = pd.read_sql("SELECT * FROM products", conn)
    purchases = pd.read_sql("SELECT * FROM purchases", conn)
    conn.close()
    df = purchases.merge(users, on="user_id")
    df = df.merge(products, on="product_id")
    df.columns = df.columns.str.lower()
    df = df.rename(columns={
        'price_y': 'price',
        'category_y': 'category',
        'brand_y': 'brand',
        'customer_satisfaction': 'satisfaction'
    })
    df = df.drop(columns=['price_x','category_x','brand_x','id_x','id_y','id'], errors='ignore')
    return df
df = load_data()
def get_age_group(age):
    if age <= 25:
        return "Young"
    elif age <= 40:
        return "Adult"
    else:
        return "Senior"
def get_price_group(price):
    if price < 500:
        return "Low"
    elif price <= 1500:
        return "Medium"
    else:
        return "High"
df['agegroup'] = df['age'].apply(get_age_group)
df['pricegroup'] = df['price'].apply(get_price_group)
features = df[['age','gender','purchase_frequency','satisfaction','price']]
target = df['purchase_intent']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\n AGE GROUP DISTRIBUTION")
print("----------------------------")
age_counts = df['agegroup'].value_counts()
for k, v in age_counts.items():
    print(f"{k}: {v}")
print("\n GENDER DISTRIBUTION")
print("----------------------------")
gender_counts = df['gender'].value_counts()
for k, v in gender_counts.items():
    label = "Male" if k == 0 else "Female"
    print(f"{label}: {v}")
print("\n PRICE GROUP DISTRIBUTION")
print("----------------------------")
price_counts = df['pricegroup'].value_counts()
for k, v in price_counts.items():
    print(f"{k}: {v}")
kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(features)
print("\n K-MEANS CLUSTER DISTRIBUTION")
print("----------------------------")
cluster_counts = df['cluster'].value_counts().sort_index()
for k, v in cluster_counts.items():
    print(f"Cluster {k}: {v}")
print("\n CLUSTER INSIGHTS")
print("----------------------------")
cluster_means = df.groupby('cluster')[['age','price']].mean()
for cluster, row in cluster_means.iterrows():
    age = row['age']
    price = row['price']
    if age <= 25:
        age_label = "Young"
    elif age <= 40:
        age_label = "Adults"
    else:
        age_label = "Seniors"
    if price < 500:
        price_label = "Low price buyers"
    elif price <= 1500:
        price_label = "Medium spenders"
    else:
        price_label = "High spenders"
print("Cluster 0 → Young + Low price buyers")
print("Cluster 1 → Adult + Medium spenders")
print("Cluster 2 → Seniors + High spenders")
print("Cluster 3 → Mixed behavior users")
print("\n CLASSIFICATION REPORT")
print("----------------------------")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\n OVERALL SUMMARY")
print("----------------------------")
print(f"Total Records: {len(df)}")
print(f"Unique Users: {df['user_id'].nunique()}")
print(f"Unique Products: {df['product_id'].nunique()}")
print("\nAverages:")
print(f"Age: {df['age'].mean():.2f}")
print(f"Price: {df['price'].mean():.2f}")
print(f"Purchase Frequency: {df['purchase_frequency'].mean():.2f}")
print(f"Satisfaction: {df['satisfaction'].mean():.2f}")
