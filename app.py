from flask import Flask, render_template, request, redirect, session, jsonify
import sqlite3
from recommendation_model import load_data, ml_recommend, ml_budget_recommend
from recommendation_model import train_model
train_model()
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
app.secret_key = "secret_key"
def generate_user_id():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT MAX(CAST(SUBSTR(user_id, 4) AS INTEGER)) FROM users
    """)
    db_max = cursor.fetchone()[0]
    conn.close()
    if db_max is None:
        db_max = 0
    return f"UID{str(db_max + 1).zfill(5)}"
@app.route('/')
def index():
    return render_template("index.html")
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            name = request.form['name']
            email = request.form['email']
            age = int(request.form['age'])
            gender = int(request.form['gender'])
            password = request.form['password']
            confirm_password = request.form['confirm_password']
            if password != confirm_password:
                return "Passwords do not match"
            conn = sqlite3.connect("database.db")
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE email=?", (email,))
            if cursor.fetchone():
                conn.close()
                return "Email already registered"
            user_id = generate_user_id()
            cursor.execute("""
                INSERT INTO users (user_id, name, email, age, gender, password)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, name, email, age, gender, password))
            conn.commit()
            conn.close()
            return redirect(f"/register?registered=true&uid={user_id}")
        except Exception as e:
            return str(e)
    return render_template("register.html")
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_id = request.form['user_id']
        password = request.form['password']
        conn = sqlite3.connect("database.db")
        cursor = conn.cursor()
        cursor.execute("""
            SELECT user_id, name FROM users
            WHERE user_id=? AND password=?
        """, (user_id, password))
        user = cursor.fetchone()
        conn.close()
        if user:
            session['user_id'] = user[0]
            session['name'] = user[1]
            return redirect('/dashboard')
        return "Invalid credentials"
    return render_template("login.html")
@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect('/login')
    user_id = session['user_id']
    name = session['name']
    budget = request.args.get('budget')
    try:
        if budget:
            budget = float(budget)
            results = ml_budget_recommend(user_id, budget)
        else:
            results = ml_recommend(user_id)
    except Exception as e:
        print("Recommendation Error:", e)
        results = []
    recommendations = results.to_dict(orient='records') if len(results) > 0 else []
    return render_template("dashboard.html", name=name, recommendations=recommendations)
@app.route('/buy_product', methods=['POST'])
def buy_product():
    if 'user_id' not in session:
        return redirect('/login')
    user_id = session['user_id']
    product_id = request.form.get('product_id')
    if not product_id:
        return "Product ID missing"
    try:
        conn = sqlite3.connect("database.db")
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 1 FROM purchases WHERE user_id=? AND product_id=?
        """, (user_id, product_id))
        if cursor.fetchone():
            conn.close()
            return redirect('/dashboard')
        cursor.execute("""
            SELECT category, brand, price FROM products WHERE product_id=?
        """, (product_id,))
        product = cursor.fetchone()
        if not product:
            conn.close()
            return "Product not found"
        category, brand, price = product
        cursor.execute("""
            INSERT INTO purchases (user_id, product_id, category, brand, price)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, product_id, category, brand, price))
        conn.commit()
        conn.close()
    except Exception as e:
        return f"Error: {str(e)}"
    return redirect('/dashboard')
@app.route('/analytics')
def analytics():
    return render_template("analytics.html")
def get_merged_df():
    users, products, purchases = load_data()
    users['user_id'] = users['user_id'].astype(str)
    purchases['user_id'] = purchases['user_id'].astype(str)
    df = purchases[['user_id', 'product_id']]
    df = df.merge(users[['user_id', 'age', 'gender']], on='user_id', how='inner')
    df = df.merge(products[['product_id', 'price', 'purchase_frequency', 'satisfaction']],
                  on='product_id', how='inner')
    return df
def get_clustered_df():
    df = get_merged_df()
    if df.empty:
        return df
    features = df[['age', 'gender', 'purchase_frequency', 'satisfaction', 'price']]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(scaled)
    return df
@app.route('/api/age_distribution')
def age_distribution():
    users, _, _ = load_data()
    users['agegroup'] = users['age'].apply(
        lambda age: "Young" if age <= 25 else "Adult" if age <= 40 else "Senior"
    )
    return jsonify(users['agegroup'].value_counts().to_dict())
@app.route('/api/gender_distribution')
def gender_distribution():
    users, _, _ = load_data()
    return jsonify(users['gender'].value_counts().to_dict())
@app.route('/api/price_distribution')
def price_distribution():
    _, products, _ = load_data()
    products['pricegroup'] = products['price'].apply(
        lambda p: "Low" if p < 500 else "Medium" if p <= 1500 else "High"
    )
    return jsonify(products['pricegroup'].value_counts().to_dict())
@app.route('/api/cluster_distribution')
def cluster_distribution():
    df = get_clustered_df()
    if df.empty:
        return jsonify({})
    return jsonify(df['cluster'].value_counts().to_dict())
@app.route('/api/cluster_insights')
def cluster_insights():
    df = get_clustered_df()
    if df.empty:
        return jsonify([])
    cluster_stats = df.groupby('cluster').agg({
        'price': 'mean',
        'purchase_frequency': 'mean',
        'satisfaction': 'mean',
        'age': 'mean'
    }).reset_index()
    cluster_stats = cluster_stats.sort_values(by='cluster')
    labels = [
        "Budget Users",
        "Frequent Buyers",
        "Premium Users",
        "Loyal Customers"
    ]
    result = []
    for i, row in cluster_stats.iterrows():
        label = labels[i % len(labels)]
        result.append({
            "cluster": int(row['cluster']),
            "type": label
        })
    return jsonify(result)
@app.route('/api/scatter_data')
def scatter_data():
    df = get_merged_df()
    if df.empty:
        return jsonify({"age": [], "price": []})
    sample = df.sample(min(300, len(df)), random_state=42)
    return jsonify({
        "age": sample['age'].tolist(),
        "price": sample['price'].tolist()
    })
@app.route('/api/histogram_data')
def histogram_data():
    users, products, _ = load_data()
    return jsonify({
        "age": users['age'].tolist(),
        "price": products['price'].tolist()
    })
@app.route('/api/table_data')
def table_data():
    users, products, _ = load_data()
    users['agegroup'] = users['age'].apply(
        lambda age: "Young" if age <= 25 else "Adult" if age <= 40 else "Senior"
    )
    products['pricegroup'] = products['price'].apply(
        lambda p: "Low" if p < 500 else "Medium" if p <= 1500 else "High"
    )
    return jsonify({
        "age": users['agegroup'].value_counts().to_dict(),
        "gender": users['gender'].value_counts().to_dict(),
        "price": products['pricegroup'].value_counts().to_dict()
    })
@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')
if __name__ == "__main__":
    app.run(debug=True)
