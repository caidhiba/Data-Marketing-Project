import pandas as pd
import numpy as np

# --- 0. RÉCUPÉRATION DES DONNÉES DE L'ÉTAPE 1 ---
# On part du principe que df_observation est déjà chargé et prêt
# (Sinon : df_observation = pd.read_csv('votre_fichier.csv') filtré sur la période d'observation)
snapshot_date = pd.to_datetime('2010-12-31') # Remplacez par votre vraie date de snapshot calculée à l'étape 1

# On s'assure des formats
df_observation['invoice_date'] = pd.to_datetime(df_observation['invoice_date'])
df_observation['invoice_month'] = df_observation['invoice_date'].dt.to_period('M')
df_observation['is_peak_season'] = df_observation['invoice_date'].dt.month.isin([11, 12]).astype(int)

# --- 1. AGRÉGATION DE BASE (RFM + Temporel + Géo) ---
features = df_observation.groupby('customer_id').agg(
    # RFM (Base)
    last_purchase=('invoice_date', 'max'),
    frequency=('invoice_id', 'nunique'),
    monetary=('line_total', 'sum'),
    
    # Comportemental
    unique_products=('product_code', 'nunique'), # Remplace les catégories si non disponibles
    total_items=('quantity', 'sum'),
    peak_season_purchases=('is_peak_season', 'sum'),
    
    # Temporel
    first_purchase=('invoice_date', 'min'),
    active_months=('invoice_month', 'nunique'),
    
    # Géographique
    country=('country', 'first') # On prend le pays principal du client
).reset_index()


# --- 2. CALCUL DES FEATURES COMPLEXES DERIVÉES ---

# Feature 1 : Récence (en jours) par rapport au snapshot
features['recency'] = (snapshot_date - features['last_purchase']).dt.days

# Feature 2 : Panier moyen (Avg Basket)
features['avg_basket'] = features['monetary'] / features['frequency']

# Feature 3 : Ancienneté (Tenure en jours)
features['tenure_days'] = (snapshot_date - features['first_purchase']).dt.days

# Feature 4 : Mois de la première acquisition (capture la saisonnalité d'acquisition)
features['first_purchase_month'] = features['first_purchase'].dt.month

# Feature 5 : Proportion d'achats en haute saison (Peak season prop)
# (Nombre d'articles achetés en Nov/Dec divisé par le total)
features['peak_season_prop'] = features['peak_season_purchases'] / features['total_items']
features['peak_season_prop'] = features['peak_season_prop'].fillna(0)


# --- 3. CALCUL DE LA RÉGULARITÉ (Écart-type des délais inter-achats) ---
# On trie les transactions par client et par date
df_sorted = df_observation.drop_duplicates(subset=['customer_id', 'invoice_id']).sort_values(['customer_id', 'invoice_date'])

# On calcule la différence en jours entre chaque commande
df_sorted['days_since_prior_order'] = df_sorted.groupby('customer_id')['invoice_date'].diff().dt.days

# On calcule l'écart-type de ce délai pour chaque client (régularité)
regularity = df_sorted.groupby('customer_id')['days_since_prior_order'].std().reset_index()
regularity.rename(columns={'days_since_prior_order': 'purchase_regularity_std'}, inplace=True)

# S'il n'a qu'une commande, ou des commandes le même jour, l'écart-type est nul (on remplit par 0)
regularity['purchase_regularity_std'] = regularity['purchase_regularity_std'].fillna(0)

# On fusionne avec la table principale
features = pd.merge(features, regularity, on='customer_id', how='left')


# --- 4. CALCUL DE LA TENDANCE DU MONTANT (Dépense de + en + ou de - en - ?) ---
# Simplification : Comparaison du panier moyen du dernier mois d'activité vs panier moyen historique
# On récupère la dernière commande de chaque client
last_orders = df_sorted.groupby('customer_id').tail(1)[['customer_id', 'line_total']]
last_orders.rename(columns={'line_total': 'last_order_value'}, inplace=True)

features = pd.merge(features, last_orders, on='customer_id', how='left')

# Tendance = Valeur de la dernière commande / Panier moyen historique
# > 1 : En croissance (dépense plus qu'avant) | < 1 : En décroissance (dépense moins qu'avant)
features['spending_trend_ratio'] = features['last_order_value'] / features['avg_basket']
features['spending_trend_ratio'] = features['spending_trend_ratio'].fillna(1) # Si 1 seule commande, ratio = 1 (stable)


# --- 5. ENCODAGE GÉOGRAPHIQUE (One-Hot Encoding des Top Pays) ---
# On identifie les 3 pays principaux, le reste passe en "Other" pour éviter 50 colonnes inutiles
top_countries = ['United Kingdom', 'Germany', 'France'] # Adaptez selon votre EDA
features['country_clean'] = features['country'].apply(lambda x: x if x in top_countries else 'Other')

# Création des colonnes binaires (One-Hot Encoding)
country_dummies = pd.get_dummies(features['country_clean'], prefix='country')
features = pd.concat([features, country_dummies], axis=1)


# --- 6. NETTOYAGE FINAL DE LA TABLE DES FEATURES ---
# On supprime les colonnes de dates brutes et les variables de travail
cols_to_drop = ['last_purchase', 'first_purchase', 'country', 'country_clean', 'last_order_value', 'peak_season_purchases']
features = features.drop(columns=cols_to_drop)

# On vérifie les 15 features générées !
print("--- TABLEAU DES FEATURES (X) GÉNÉRÉ AVEC SUCCÈS ---")
print(f"Nombre de clients : {len(features)}")
print(f"Nombre de features : {len(features.columns) - 1} (hors customer_id)\n")
print(features.head())

# (Optionnel) Sauvegarder la table finale des features
# features.to_csv('features_clv_model.csv', index=False)