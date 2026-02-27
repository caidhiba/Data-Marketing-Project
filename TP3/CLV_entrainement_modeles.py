import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =========================================================
# ÉTAPE 1 : CHARGEMENT ET SPLIT TEMPOREL (LA CIBLE)
# =========================================================
print("1. Chargement et Split Temporel...")
# Assurez-vous que le nom du fichier correspond à vos données
df_trans = pd.read_csv('transactions.csv') 

# Préparation des dates et montants
df_trans['invoice_date'] = pd.to_datetime(df_trans['invoice_date'])
df_trans['customer_id'] = pd.to_numeric(df_trans['customer_id'], errors='coerce')
df_trans = df_trans.dropna(subset=['customer_id', 'invoice_date']).copy()

if 'line_total' not in df_trans.columns:
    df_trans['line_total'] = df_trans['quantity'] * df_trans['unit_price']

# Définition du Snapshot (1 an avant la dernière date)
date_max = df_trans['invoice_date'].max()
snapshot_date = date_max - pd.DateOffset(months=12)

# Séparation stricte (Prévention du Data Leakage)
df_observation = df_trans[df_trans['invoice_date'] <= snapshot_date].copy()
df_cible = df_trans[df_trans['invoice_date'] > snapshot_date].copy()

# Création de la Target
clients_actifs = pd.DataFrame({'customer_id': df_observation['customer_id'].unique()})
target_clv = df_cible.groupby('customer_id')['line_total'].sum().reset_index()
target_clv.rename(columns={'line_total': 'target_12m_value'}, inplace=True)

df_ml_base = pd.merge(clients_actifs, target_clv, on='customer_id', how='left')
df_ml_base['target_12m_value'] = df_ml_base['target_12m_value'].fillna(0)


# =========================================================
# ÉTAPE 2 : FEATURE ENGINEERING (LES VARIABLES)
# =========================================================
print("2. Création des Features...")
df_observation['is_peak_season'] = df_observation['invoice_date'].dt.month.isin([11, 12]).astype(int)
df_observation['invoice_month'] = df_observation['invoice_date'].dt.to_period('M')

# Agrégations de base
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

# Features dérivées
features['recency'] = (snapshot_date - features['last_purchase']).dt.days
features['tenure_days'] = (snapshot_date - features['first_purchase']).dt.days
features['avg_basket'] = features['monetary'] / features['frequency']
features['first_purchase_month'] = features['first_purchase'].dt.month
features['peak_season_prop'] = (features['peak_season_purchases'] / features['total_items']).fillna(0)

# Nettoyage des dates et variables temporaires
features = features.drop(columns=['last_purchase', 'first_purchase', 'peak_season_purchases'])

# Encodage du Pays (Garder le Top 3, le reste en 'Other')
top_countries = features['country'].value_counts().nlargest(3).index.tolist()
features['country_clean'] = features['country'].apply(lambda x: x if x in top_countries else 'Other')
country_dummies = pd.get_dummies(features['country_clean'], prefix='country')
features = pd.concat([features.drop(columns=['country', 'country_clean']), country_dummies], axis=1)


# =========================================================
# ÉTAPE 3 : MODÉLISATION ET ÉVALUATION
# =========================================================
print("3. Entraînement des modèles en cours...\n")

# Fusion finale
df_final = pd.merge(features, df_ml_base[['customer_id', 'target_12m_value']], on='customer_id', how='inner')

# --- LE CORRECTIF EST ICI ---
# On remplace les infinis par du vide (NaN), puis on remplit tous les vides par 0
df_final = df_final.replace([np.inf, -np.inf], np.nan).fillna(0)

X = df_final.drop(columns=['customer_id', 'target_12m_value'])
y = df_final['target_12m_value']

# Split temporel (shuffle=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Initialisation
models = {
    "1. Régression Linéaire": LinearRegression(),
    "2. Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "3. XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

results = {}
predictions = {}

# Entraînement
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred = np.maximum(0, y_pred) # La CLV ne peut pas être négative
    predictions[name] = y_pred
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'RMSE': rmse, 'MAE': mae, 'R²': r2}

print("--- RÉSULTATS SUR LE TEST SET ---")
print(pd.DataFrame(results).T.round(2))

# =========================================================
# VISUALISATION DES RÉSULTATS
# =========================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True, sharex=True)
colors = ['skyblue', 'hotpink', 'purple']

for ax, (name, y_pred), color in zip(axes, predictions.items(), colors):
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, color=color, edgecolor='w', s=40, ax=ax)
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([0, max_val], [0, max_val], color='black', linestyle='--', linewidth=1.5)
    
    ax.set_title(f"{name}\nR²: {results[name]['R²']:.2f}", color=color, fontweight='bold', fontsize=13)
    ax.set_xlabel("CLV Réelle (€)", color='purple', fontsize=11)
    if ax == axes[0]:
        ax.set_ylabel("CLV Prédite (€)", color='purple', fontsize=11)
    
    ax.grid(True, linestyle='--', alpha=0.3, color='purple')
    ax.tick_params(colors='purple')
    for spine in ax.spines.values():
        spine.set_color('purple')

plt.suptitle("Évaluation des Modèles : Prédictions vs CLV Réelle à 12 mois", color='purple', fontsize=16, fontweight='bold', y=1.05)
plt.tight_layout()

plt.savefig('clv_models_comparison_colors.png', bbox_inches='tight', dpi=150)
plt.close()

print("\nGraphique de comparaison sauvegardé sous 'clv_models_comparison_colors.png'.")
