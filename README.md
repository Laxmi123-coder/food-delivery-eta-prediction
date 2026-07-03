# 🍕 Food Delivery ETA Prediction

## 🚀 Live Demo
👉 [Click here to try the app](https://yourusername-food-delivery-eta.streamlit.app)

---

## Problem Statement
Food delivery platforms lose customers when ETA estimates are wrong.
This project predicts delivery time in minutes using Machine Learning
based on distance, traffic, weather and delivery person details.

## Dataset
- Source: Kaggle — Food Delivery Dataset
- Records: 43,000+ real delivery orders
- Target: Time taken (minutes)

## Models Trained
| Model             | MAE (min) | R² Score |
|-------------------|-----------|----------|
| Linear Regression | 6.39      | 0.26     |
| Random Forest     | 3.75      | 0.74     |

✅ Random Forest predicts within 3.75 minutes accuracy!

## Key Findings
- Jam traffic adds ~15 extra minutes vs Low traffic
- Stormy weather increases delivery time significantly
- Distance and traffic are the top 2 factors affecting ETA

## Tools Used
- Python, Pandas, NumPy
- Scikit-learn (Random Forest)
- Streamlit (Web App)
- Matplotlib, Seaborn

## How to Run Locally
pip install -r requirements.txt
streamlit run app.py

## Project Structure
food-delivery-eta/
├── data/
│   └── food_delivery.csv
├── notebooks/
│   └── analysis.ipynb
├── images/
├── app.py
└── requirements.txt