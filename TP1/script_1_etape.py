import pandas as pd

# 1. Charger le fichier CSV
df = pd.read_csv('customers.csv')

# --- ÉTAPE 1 : Nettoyage des doublons (script précédent) ---
df_cleaned = df.drop_duplicates()

print(f"Lignes initiales : {len(df)}")
print(f"Lignes après suppression des doublons : {len(df_cleaned)}")


# --- ÉTAPE 2 : Conversion des types de données ---

# customer_id au format entier (int)
df_cleaned['customer_id'] = df_cleaned['customer_id'].astype(int)

# country au format texte (string/object)
df_cleaned['country'] = df_cleaned['country'].astype(str)

# first_purchase et last_purchase au format date (datetime)
# errors='coerce' permet de transformer les valeurs invalides en "NaT" (valeur nulle pour les dates)
df_cleaned['first_purchase'] = pd.to_datetime(df_cleaned['first_purchase'], errors='coerce')
df_cleaned['last_purchase'] = pd.to_datetime(df_cleaned['last_purchase'], errors='coerce')

# Dictionnaire de remplacement pour standardiser les pays
country_mapping = {
    'EIRE': 'Ireland',
    'RSA': 'South Africa',
    'West Indies': 'Caribbean',  # ou une autre appellation qui vous convient
    'Channel Islands': 'United Kingdom', # ou 'Channel Islands' si vous souhaitez les garder séparés
    'European Community': 'Europe', 
    'Unspecified': 'Unknown'
}

# Appliquer le remplacement
df_cleaned['country'] = df_cleaned['country'].replace(country_mapping)

# n_orders, total_spent, avg_basket, recency_days, tenure_days au format décimal (float)
colonnes_decimales = ['n_orders', 'total_spent', 'avg_basket', 'recency_days', 'tenure_days']
for col in colonnes_decimales:
    # On force la conversion en numérique d'abord (en cas de caractères bizarres), puis en float
    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce').astype(float)


# --- ÉTAPE 3 : Vérification et Sauvegarde ---

# Afficher les nouveaux types de données pour vérifier
print("\nTypes de données après conversion :")
print(df_cleaned.dtypes)

# Sauvegarder dans un nouveau fichier final
df_cleaned.to_csv('customers_cleaned_typed.csv', index=False)
print("\nLe fichier formaté a été sauvegardé sous 'customers_cleaned_typed.csv'.")
