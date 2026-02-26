import pandas as pd

# 1. Charger le fichier CSV
df = pd.read_csv('customers.csv')
lignes_initiales = len(df)
print(f"Lignes initiales : {lignes_initiales}")

# --- ÉTAPE 1 : Nettoyage des doublons ---
df_cleaned = df.drop_duplicates().copy()

# --- ÉTAPE 2 : Standardisation des zones géographiques ---
country_mapping = {
    'EIRE': 'Ireland',
    'RSA': 'South Africa',
    'West Indies': 'Caribbean',
    'Channel Islands': 'United Kingdom', 
    'European Community': 'Europe'
}
df_cleaned['country'] = df_cleaned['country'].replace(country_mapping)

# --- ÉTAPE 3 : Suppression des pays inconnus ("Unspecified") ---
df_cleaned = df_cleaned[df_cleaned['country'] != 'Unspecified']

# --- ÉTAPE 4 : Conversion des types de données ---
df_cleaned['customer_id'] = df_cleaned['customer_id'].astype(int)
df_cleaned['country'] = df_cleaned['country'].astype(str)

df_cleaned['first_purchase'] = pd.to_datetime(df_cleaned['first_purchase'], errors='coerce')
df_cleaned['last_purchase'] = pd.to_datetime(df_cleaned['last_purchase'], errors='coerce')

colonnes_decimales = ['n_orders', 'total_spent', 'avg_basket', 'recency_days', 'tenure_days']
for col in colonnes_decimales:
    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce').astype(float)


# --- ÉTAPE 5 : Vérification de la cohérence des dates ---
# first_purchase <= last_purchase
condition_dates = (df_cleaned['first_purchase'] <= df_cleaned['last_purchase']) | df_cleaned['first_purchase'].isna() | df_cleaned['last_purchase'].isna()
df_cleaned = df_cleaned[condition_dates]

# --- ÉTAPE 6 : Vérification de la cohérence Récence / Ancienneté ---
# recency_days <= tenure_days
condition_recence = (df_cleaned['recency_days'] <= df_cleaned['tenure_days']) | df_cleaned['recency_days'].isna() | df_cleaned['tenure_days'].isna()
df_cleaned = df_cleaned[condition_recence]

# --- ÉTAPE 7 : Vérification de la cohérence des montants ---
# total_spent >= avg_basket
condition_montants = (df_cleaned['total_spent'] >= df_cleaned['avg_basket']) | df_cleaned['total_spent'].isna() | df_cleaned['avg_basket'].isna()
df_cleaned = df_cleaned[condition_montants]

lignes_finales = len(df_cleaned)
print(f"Lignes finales conservées (fichier propre) : {lignes_finales}")
print(f"Total des lignes aberrantes ou inconnues supprimées : {lignes_initiales - lignes_finales}")

# --- ÉTAPE 8 : Sauvegarde ---
df_cleaned.to_csv('customers_cleaned_final.csv', index=False)
print("\nLe fichier formaté et finalisé a été sauvegardé sous 'customers_cleaned_final.csv'.")