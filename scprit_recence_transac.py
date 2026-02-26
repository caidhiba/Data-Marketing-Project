import pandas as pd

# 1. Charger les données
df = pd.read_csv('customers.csv')

# 2. S'assurer que la colonne last_purchase est bien au format Date
df['last_purchase'] = pd.to_datetime(df['last_purchase'], errors='coerce')

# 3. Définir la date de référence (Snapshot Date)
# On prend la date de transaction la plus récente de tout le fichier
date_reference = df['last_purchase'].max()
print(f"La date de référence pour le calcul est le : {date_reference}")

# 4. Calculer la Récence
# On soustrait la date du dernier achat à la date de référence, et on extrait le nombre de jours (.dt.days)
df['recence_calculee'] = (date_reference - df['last_purchase']).dt.days

# 5. Afficher un aperçu pour comparer votre ancienne colonne et la nouvelle
colonnes_a_afficher = ['customer_id', 'last_purchase', 'recency_days', 'recence_calculee']
print("\nAperçu des calculs :")
print(df[colonnes_a_afficher].head(10))

# (Optionnel) Sauvegarder le résultat
# df.to_csv('customers_avec_recence.csv', index=False)