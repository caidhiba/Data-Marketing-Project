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

# Conversion des dates
df_cleaned['first_purchase'] = pd.to_datetime(df_cleaned['first_purchase'], errors='coerce')
df_cleaned['last_purchase'] = pd.to_datetime(df_cleaned['last_purchase'], errors='coerce')

# Conversion des décimales
colonnes_decimales = ['n_orders', 'total_spent', 'avg_basket', 'recency_days', 'tenure_days']
for col in colonnes_decimales:
    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce').astype(float)


# --- ÉTAPE 5 : Vérification de la cohérence des dates ---
# On garde les lignes où first_purchase <= last_purchase (ou si l'une des dates est vide)
condition_dates_coherentes = (df_cleaned['first_purchase'] <= df_cleaned['last_purchase']) | df_cleaned['first_purchase'].isna() | df_cleaned['last_purchase'].isna()
df_cleaned = df_cleaned[condition_dates_coherentes]


# --- ÉTAPE 6 : Vérification de la cohérence Récence / Ancienneté ---
# On veut s'assurer que recency_days <= tenure_days (ou tolérer les valeurs vides si elles existent)
lignes_avant_filtre_recence = len(df_cleaned)

condition_recence_coherente = (df_cleaned['recency_days'] <= df_cleaned['tenure_days']) | df_cleaned['recency_days'].isna() | df_cleaned['tenure_days'].isna()
df_cleaned = df_cleaned[condition_recence_coherente]

lignes_finales = len(df_cleaned)
print(f"Lignes supprimées pour récence > ancienneté : {lignes_avant_filtre_recence - lignes_finales}")
print(f"Lignes finales conservées (fichier propre) : {lignes_finales}")
print(f"Total des lignes supprimées au cours du processus : {lignes_initiales - lignes_finales}")

# --- ÉTAPE 7 : Sauvegarde ---
df_cleaned.to_csv('customers_cleaned_final.csv', index=False)
print("\nLe fichier formaté et finalisé a été sauvegardé sous 'customers_cleaned_final.csv'.")