import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap # N'oubliez pas le pip install shap

import warnings
warnings.filterwarnings('ignore') # Pour garder la console propre

# =========================================================
# ÉTAPE 1 : CHARGEMENT ET SPLIT TEMPOREL (LA CIBLE)
# =========================================================
print("1/4 - Création de la Target (Split Temporel)...")
df_trans = pd.read_csv('transactions.csv') 

df_trans['invoice_date'] = pd.to_datetime(df_trans['invoice_date'])
df_trans['customer_id'] = pd.to_numeric(df_trans['customer_id'], errors='coerce')
df_trans = df_trans.dropna(subset=['customer_id', 'invoice_date']).copy()

if 'line_total' not in df_trans.columns:
    df_trans['line_total'] = df_trans['quantity'] * df_trans['unit_price']

date_max = df_trans['invoice_date'].max()
snapshot_date = date_max - pd.DateOffset(months=12)

df_observation = df_trans[df_trans['invoice_date'] <= snapshot_date].copy()
df_cible = df_trans[df_trans['invoice_date'] > snapshot_date].copy()

clients_actifs = pd.DataFrame({'customer_id': df_observation['customer_id'].unique()})
target_clv = df_cible.groupby('customer_id')['line_total'].sum().reset_index()
target_clv.rename(columns={'line_total': 'target_12m_value'}, inplace=True)

df_ml_base = pd.merge(clients_actifs, target_clv, on='customer_id', how='left')
df_ml_base['target_12m_value'] = df_ml_base['target_12m_value'].fillna(0)


# =========================================================
# ÉTAPE 2 : FEATURE ENGINEERING (LES VARIABLES)
# =========================================================
print("2/4 - Calcul des 15 Features Métier...")
df_observation['is_peak_season'] = df_observation['invoice_date'].dt.month.isin([11, 12]).astype(int)
df_observation['invoice_month'] = df_observation['invoice_date'].dt.to_period('M')

features = df_observation.groupby('customer_id').agg(
    last_purchase=('invoice_date', 'max'),
    first_purchase=('invoice_date', 'min'),
    frequency=('invoice_id', 'nunique'),
    monetary=('line_total', 'sum'),
    unique_products=('product_code', 'nunique'),
    total_items=('quantity', 'sum'),
    peak_season_purchases=('is_peak_season', 'sum'),
    active_months=('invoice_month', 'nunique'),
    country=('country', 'first')
).reset_index()

features['recency'] = (snapshot_date - features['last_purchase']).dt.days
features['tenure_days'] = (snapshot_date - features['first_purchase']).dt.days
features['avg_basket'] = features['monetary'] / features['frequency']
features['first_purchase_month'] = features['first_purchase'].dt.month
features['peak_season_prop'] = (features['peak_season_purchases'] / features['total_items']).fillna(0)

features = features.drop(columns=['last_purchase', 'first_purchase', 'peak_season_purchases'])

top_countries = features['country'].value_counts().nlargest(3).index.tolist()
features['country_clean'] = features['country'].apply(lambda x: x if x in top_countries else 'Other')
country_dummies = pd.get_dummies(features['country_clean'], prefix='country')
features = pd.concat([features.drop(columns=['country', 'country_clean']), country_dummies], axis=1)


# =========================================================
# ÉTAPE 3 : MODÉLISATION ET ÉVALUATION
# =========================================================
print("3/4 - Entraînement des modèles (LR, RF, XGBoost)...")
df_final = pd.merge(features, df_ml_base[['customer_id', 'target_12m_value']], on='customer_id', how='inner')

# CORRECTIF DES VALEURS INFINIES (Division par zéro)
df_final = df_final.replace([np.inf, -np.inf], np.nan).fillna(0)

X = df_final.drop(columns=['customer_id', 'target_12m_value'])
y = df_final['target_12m_value']

# Split temporel sans mélange (shuffle=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

models = {
    "1. Régression Linéaire": LinearRegression(),
    "2. Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "3. XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

results = {}
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = np.maximum(0, model.predict(X_test)) # CLV >= 0
    predictions[name] = y_pred
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results[name] = {'RMSE': rmse, 'R²': r2_score(y_test, y_pred)}

print("\n--- PERFORMANCES ---")
print(pd.DataFrame(results).T.round(2))


# =========================================================
# ÉTAPE 4 : INTERPRÉTABILITÉ AVEC SHAP
# =========================================================
print("\n4/4 - Calcul de l'interprétabilité SHAP sur XGBoost...")
best_model = models["3. XGBoost"] 
explainer = shap.TreeExplainer(best_model)
shap_values = explainer(X_test)

# --- A. Importance Globale (Beeswarm) ---
plt.figure(figsize=(10, 6))
shap.plots.beeswarm(shap_values, show=False)
plt.title("Importance globale des Features sur la CLV (XGBoost)", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('shap_beeswarm_global.png', bbox_inches='tight', dpi=150)
plt.clf()

# --- B. Analyse Individuelle ---
preds_xgb = predictions["3. XGBoost"]

# Client VIP
vip_idx = np.argmax(preds_xgb)
vip_id = df_final.iloc[X_test.index[vip_idx]]['customer_id']

# Client Churner (faible prédiction)
churner_idx = np.argsort(preds_xgb)[10] 
churner_id = df_final.iloc[X_test.index[churner_idx]]['customer_id']

print(f"\n--- EXEMPLES D'ANALYSES INDIVIDUELLES ---")
print(f"🥇 VIP (ID: {vip_id}) - CLV Prédite : {preds_xgb[vip_idx]:.2f} €")
print(f"📉 À Risque (ID: {churner_id}) - CLV Prédite : {preds_xgb[churner_idx]:.2f} €")

# Waterfall VIP
plt.figure(figsize=(8, 5))
shap.plots.waterfall(shap_values[vip_idx], show=False)
plt.title(f"Client VIP (Prédit: {preds_xgb[vip_idx]:.0f} €)", fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('shap_waterfall_vip.png', bbox_inches='tight', dpi=150)
plt.clf()

# Waterfall Churner
plt.figure(figsize=(8, 5))
shap.plots.waterfall(shap_values[churner_idx], show=False)
plt.title(f"Client à Risque (Prédit: {preds_xgb[churner_idx]:.0f} €)", fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('shap_waterfall_churner.png', bbox_inches='tight', dpi=150)
plt.close()

print("\n🚀 C'EST TERMINÉ ! Tous les graphiques SHAP ont été générés avec succès.")