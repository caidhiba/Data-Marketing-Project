import pandas as pd
import numpy as np

# --- 1. CHARGEMENT DES TRANSACTIONS ---
df_trans = pd.read_csv('transactions.csv') # Adaptez le nom si besoin

# Formatage des dates et calcul du montant de chaque ligne
df_trans['invoice_date'] = pd.to_datetime(df_trans['invoice_date'])
df_trans['customer_id'] = pd.to_numeric(df_trans['customer_id'], errors='coerce')
df_trans = df_trans.dropna(subset=['customer_id', 'invoice_date']).copy()

# On s'assure d'avoir le montant de chaque ligne (Quantité * Prix unitaire)
if 'line_total' not in df_trans.columns:
    df_trans['line_total'] = df_trans['quantity'] * df_trans['unit_price']


# --- 2. LE SPLIT TEMPOREL (Découpage strict) ---
# On identifie la date la plus récente du dataset
date_max = df_trans['invoice_date'].max()

# On définit la date de "Snapshot" : exactement 1 an (12 mois) avant la date max
snapshot_date = date_max - pd.DateOffset(months=12)

print(f"Date de début des données : {df_trans['invoice_date'].min()}")
print(f"Date de Snapshot (Séparation) : {snapshot_date}")
print(f"Date de fin des données : {date_max}\n")

# Séparation physique des données pour éviter tout Data Leakage
df_observation = df_trans[df_trans['invoice_date'] <= snapshot_date].copy()
df_cible = df_trans[df_trans['invoice_date'] > snapshot_date].copy()


# --- 3. CRÉATION DE LA BASE CLIENTS (Features futures) ---
# On ne garde QUE les clients qui existaient pendant la période d'observation
# On ne peut pas prédire l'avenir d'un client qu'on ne connaît pas encore !
clients_actifs = pd.DataFrame({'customer_id': df_observation['customer_id'].unique()})


# --- 4. CALCUL DE LA TARGET (CLV à 12 mois) ---
# On calcule combien chaque client a dépensé DANS LA PÉRIODE CIBLE
target_clv = df_cible.groupby('customer_id')['line_total'].sum().reset_index()
target_clv.rename(columns={'line_total': 'target_12m_value'}, inplace=True)

# On associe cette cible à notre base de clients actifs
df_ml_base = pd.merge(clients_actifs, target_clv, on='customer_id', how='left')

# ÉTAPE CRUCIALE : Les clients qui n'ont rien acheté pendant les 12 derniers mois 
# auront des valeurs nulles (NaN). En réalité, leur CLV sur cette période est de 0 €.
df_ml_base['target_12m_value'] = df_ml_base['target_12m_value'].fillna(0)


# --- 5. VÉRIFICATION ---
print("--- APERÇU DE LA BASE POUR LE MACHINE LEARNING ---")
print(f"Nombre de clients à prédire : {len(df_ml_base)}")
print(f"Clients ayant racheté (Target > 0) : {len(df_ml_base[df_ml_base['target_12m_value'] > 0])}")
print(f"Clients inactifs (Target = 0) : {len(df_ml_base[df_ml_base['target_12m_value'] == 0])}\n")

print(df_ml_base.head())

