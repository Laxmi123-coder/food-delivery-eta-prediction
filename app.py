import streamlit as st
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

st.set_page_config(page_title="Food Delivery ETA Predictor", page_icon="🍕", layout="wide")

# ───────────────────────── STYLING ─────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');

* { font-family: 'Poppins', sans-serif; }

.stApp {
    background: radial-gradient(circle at top left, #1a0a00 0%, #0d0d0d 50%);
    color: #f0f0f0;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a1a, #0d0d0d);
    border-right: 1px solid #ff6b2b33;
}

[data-testid="stSidebar"] h3 {
    color: #ff6b2b;
    font-weight: 800;
}

.stButton > button {
    background: linear-gradient(135deg, #ff6b2b, #ff8c4b);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 14px 28px;
    font-size: 16px;
    font-weight: 700;
    width: 100%;
    box-shadow: 0 4px 20px #ff6b2b55;
    transition: all 0.2s;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 28px #ff6b2b88;
}

h1 { letter-spacing: -1px; }

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: #1a1a1a;
    padding: 6px;
    border-radius: 12px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #999;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: #ff6b2b22 !important;
    color: #ff6b2b !important;
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ───────────────────────── HELPER ─────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))


# ───────────────────────── LOAD & TRAIN ─────────────────────────
@st.cache_resource
def load_and_train():
    df = pd.read_csv("data/food_delivery.csv")
    df['Time_taken'] = df['Time_taken(min)'].str.extract(r'(\d+)').astype(float)
    df['Delivery_person_Ratings'] = pd.to_numeric(df['Delivery_person_Ratings'], errors='coerce')
    df['distance_km'] = df.apply(
        lambda r: haversine(r['Restaurant_latitude'], r['Restaurant_longitude'],
                            r['Delivery_location_latitude'], r['Delivery_location_longitude']), axis=1)
    df = df[df['distance_km'] <= 30]

    df['Weatherconditions'] = df['Weatherconditions'].str.strip().str.replace('conditions ', '', regex=False)
    df['Road_traffic_density'] = df['Road_traffic_density'].str.strip()
    df = df.dropna()

    traffic_map = {v: i for i, v in enumerate(sorted(df['Road_traffic_density'].unique()))}
    weather_map = {v: i for i, v in enumerate(sorted(df['Weatherconditions'].unique()))}
    vehicle_map = {v: i for i, v in enumerate(sorted(df['Type_of_vehicle'].unique()))}
    order_map = {v: i for i, v in enumerate(sorted(df['Type_of_order'].unique()))}

    df['Road_traffic_density'] = df['Road_traffic_density'].map(traffic_map)
    df['Weatherconditions'] = df['Weatherconditions'].map(weather_map)
    df['Type_of_vehicle'] = df['Type_of_vehicle'].map(vehicle_map)
    df['Type_of_order'] = df['Type_of_order'].map(order_map)
    df = df.dropna()

    features = ['Delivery_person_Age', 'Delivery_person_Ratings', 'distance_km',
                'Road_traffic_density', 'Weatherconditions', 'Type_of_vehicle', 'Type_of_order']

    X = df[features]
    y = df['Time_taken']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, traffic_map, weather_map, vehicle_map, order_map


model, traffic_map, weather_map, vehicle_map, order_map = load_and_train()


# ───────────────────────── HEADER ─────────────────────────
st.markdown("""
<div style='text-align:center; padding:30px 0 10px;'>
    <div style='display:inline-block; background:#ff6b2b22; border:1px solid #ff6b2b55;
                color:#ff6b2b; font-size:11px; letter-spacing:3px; text-transform:uppercase;
                padding:6px 16px; border-radius:20px; margin-bottom:16px;'>
        🤖 Machine Learning · Random Forest
    </div>
    <h1 style='font-size:46px; font-weight:800; margin:0;
               background: linear-gradient(135deg, #fff, #ff6b2b);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
        🍕 Food Delivery ETA Predictor
    </h1>
    <p style='color:#888; font-size:15px; margin-top:8px;'>
        Real-time delivery time prediction trained on 43,000+ real orders
    </p>
</div>
""", unsafe_allow_html=True)
st.divider()


# ───────────────────────── SIDEBAR ─────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Delivery Details")
    age = st.slider("Delivery Person Age", 18, 50, 28)
    rating = st.slider("Rating", 1.0, 5.0, 4.5, 0.1)
    st.markdown("---")
    restaurant_lat = st.number_input("Restaurant Latitude", value=12.9716)
    restaurant_lon = st.number_input("Restaurant Longitude", value=77.5946)
    delivery_lat = st.number_input("Delivery Latitude", value=13.0012)
    delivery_lon = st.number_input("Delivery Longitude", value=77.6412)
    distance = haversine(restaurant_lat, restaurant_lon, delivery_lat, delivery_lon)
    st.markdown(f"📍 **Distance:** `{distance:.2f} km`")
    st.markdown("---")
    traffic = st.selectbox("Traffic", list(traffic_map.keys()))
    weather = st.selectbox("Weather", list(weather_map.keys()))
    vehicle = st.selectbox("Vehicle", list(vehicle_map.keys()))
    order_type = st.selectbox("Order Type", list(order_map.keys()))
    predict_btn = st.button("🚀 Predict ETA")


# ───────────────────────── MAIN ─────────────────────────
col1, col2 = st.columns([1, 1.6], gap="large")

with col1:
    st.markdown("### 📦 Prediction")
    if predict_btn:
        input_data = pd.DataFrame({
            'Delivery_person_Age': [age],
            'Delivery_person_Ratings': [rating],
            'distance_km': [distance],
            'Road_traffic_density': [traffic_map[traffic]],
            'Weatherconditions': [weather_map[weather]],
            'Type_of_vehicle': [vehicle_map[vehicle]],
            'Type_of_order': [order_map[order_type]],
        })
        eta = round(model.predict(input_data)[0])

        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1a0a00, #2a1400);
                    border: 2px solid #ff6b2b; border-radius:20px;
                    padding:32px; text-align:center;
                    box-shadow: 0 0 40px #ff6b2b33;'>
            <div style='font-size:13px; color:#ff6b2b; letter-spacing:2px;
                        text-transform:uppercase; margin-bottom:8px;'>
                🛵 Estimated Delivery Time
            </div>
            <div style='font-size:72px; font-weight:900; color:#fff;
                        line-height:1; text-shadow: 0 0 30px #ff6b2b88;'>
                {eta}<span style='font-size:24px; color:#ff6b2b;'> min</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        low, high = max(5, eta - 5), eta + 5
        st.markdown(f"""
        <div style='text-align:center; color:#888; font-size:13px; margin-top:10px;'>
            Likely between <strong style='color:#ff6b2b'>{low}–{high} min</strong>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if traffic == 'Jam':
            st.warning("🚦 Heavy traffic detected — adding significant delay")
        if weather in ['Stormy', 'Fog']:
            st.warning(f"🌧️ {weather} weather slows delivery by ~10–15 min")
        if distance > 8:
            st.info("📍 Long distance — expect non-linear time increase")

    else:
        st.markdown("""
        <div style='background:#1e1e1e; border:1px dashed #333;
                    border-radius:16px; padding:44px; text-align:center; color:#555;'>
            <div style='font-size:42px;'>🛵</div>
            <div style='margin-top:12px; font-size:14px;'>
                Fill in the sidebar and click<br>
                <strong style='color:#ff6b2b'>Predict ETA</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Distance", f"{distance:.1f} km")
    with m2:
        st.metric("Rating", f"{rating}")
    with m3:
        traffic_icons = {'Low': '🟢', 'Medium': '🟡', 'High': '🟠', 'Jam': '🔴'}
        st.metric("Traffic", f"{traffic_icons.get(traffic,'')} {traffic}")


with col2:
    st.markdown("### 📈 Data Insights")
    tab1, tab2, tab3 = st.tabs(["Traffic Impact", "Weather Impact", "Feature Importance"])

    plt.style.use('dark_background')

    with tab1:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        fig.patch.set_facecolor('#0d0d0d')
        ax.set_facecolor('#0d0d0d')
        orig = pd.read_csv("data/food_delivery.csv")
        orig['Time_taken'] = orig['Time_taken(min)'].str.extract(r'(\d+)').astype(float)
        orig['Road_traffic_density'] = orig['Road_traffic_density'].str.strip()
        orig = orig.dropna(subset=['Road_traffic_density', 'Time_taken'])
        order = [k for k in ['Low', 'Medium', 'High', 'Jam'] if k in orig['Road_traffic_density'].values]
        data = [orig[orig['Road_traffic_density'] == t]['Time_taken'].values for t in order]
        bp = ax.boxplot(data, tick_labels=order, patch_artist=True,
                        medianprops=dict(color='white', linewidth=2))
        for patch, color in zip(bp['boxes'], ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        ax.set_title('Traffic vs Delivery Time', color='white', pad=10)
        ax.set_ylabel('Minutes', color='#aaa')
        ax.tick_params(colors='#aaa')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333')
        st.pyplot(fig); plt.close()

    with tab2:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        fig.patch.set_facecolor('#0d0d0d')
        ax.set_facecolor('#0d0d0d')
        orig2 = pd.read_csv("data/food_delivery.csv")
        orig2['Time_taken'] = orig2['Time_taken(min)'].str.extract(r'(\d+)').astype(float)
        orig2['Weatherconditions'] = orig2['Weatherconditions'].str.strip().str.replace('conditions ', '', regex=False)
        orig2 = orig2.dropna(subset=['Weatherconditions', 'Time_taken'])
        avg = orig2.groupby('Weatherconditions')['Time_taken'].mean().sort_values()
        avg.plot(kind='barh', ax=ax, color='#ff6b2b', alpha=0.85)
        ax.set_title('Avg Delivery Time by Weather', color='white', pad=10)
        ax.tick_params(colors='#aaa')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333')
        st.pyplot(fig); plt.close()

    with tab3:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        fig.patch.set_facecolor('#0d0d0d')
        ax.set_facecolor('#0d0d0d')
        labels = ['Age', 'Rating', 'Distance', 'Traffic', 'Weather', 'Vehicle', 'Order']
        importance = pd.Series(model.feature_importances_, index=labels).sort_values()
        colors = ['#ff6b2b' if v == importance.max() else '#555' for v in importance.values]
        importance.plot(kind='barh', ax=ax, color=colors)
        ax.set_title('What Affects ETA Most?', color='white', pad=10)
        ax.tick_params(colors='#aaa')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333')
        st.pyplot(fig); plt.close()


# ───────────────────────── FOOTER ─────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center; color:#444; font-size:12px; padding:10px 0 20px;'>
    Built with Python · Scikit-learn · Streamlit &nbsp;|&nbsp;
    <span style='color:#ff6b2b;'>Food Delivery ETA Prediction — Portfolio Project</span>
</div>
""", unsafe_allow_html=True)