import pandas as pd
import matplotlib.pyplot as plt

# 1. CHARGEMENT ET NETTOYAGE
df = pd.read_csv('customers.csv')

# Doublons et pays inconnus
df = df.drop_duplicates()
df = df[df['country'] != 'Unspecified']

# Conversions
colonnes_decimales = ['n_orders', 'total_spent', 'avg_basket', 'recency_days', 'tenure_days']
for col in colonnes_decimales:
    df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)

df['first_purchase'] = pd.to_datetime(df['first_purchase'], errors='coerce')
df['last_purchase'] = pd.to_datetime(df['last_purchase'], errors='coerce')

# Règles de cohérence (dates, récence, montants)
cond_dates = (df['first_purchase'] <= df['last_purchase']) | df['first_purchase'].isna()
cond_recence = (df['recency_days'] <= df['tenure_days']) | df['recency_days'].isna()
cond_montants = (df['total_spent'] >= df['avg_basket']) | df['total_spent'].isna()

# Données finales propres
df_clean = df[cond_dates & cond_recence & cond_montants].copy()


# 2. CALCUL DES PROPORTIONS
clients_uniques = len(df_clean[df_clean['n_orders'] <= 1.0])
clients_recurrents = len(df_clean[df_clean['n_orders'] > 1.0])


# 3. CRÉATION DU GRAPHIQUE (Camembert / Pie Chart)
# Labels pour le graphe
labels = ['Transaction Unique\n(≤ 1 commande)', 'Clients Récurrents\n(> 1 commande)']
tailles = [clients_uniques, clients_recurrents]

# Couleurs pour chaque part (rouge pastel et bleu ciel)
couleurs = ['#ff9999', '#66b3ff']

# "Explode" permet de détacher légèrement la première part (Transaction unique) pour la mettre en valeur
separation = (0.1, 0) 

fig, ax = plt.subplots(figsize=(8, 6))

# Paramétrage du diagramme circulaire
ax.pie(
    tailles, 
    explode=separation, 
    labels=labels, 
    colors=couleurs, 
    autopct='%1.1f%%', # Formatage du pourcentage avec une décimale
    shadow=True, 
    startangle=140, # Angle de départ pour la rotation du graphe
    textprops={'fontsize': 12}
)

# Assure que le camembert est un cercle parfait (et non une ellipse)
ax.axis('equal') 

# Titre du graphique
plt.title(
    "Répartition des Clients : Transaction Unique vs Récurrents\n(Sur base nettoyée)", 
    fontsize=14, 
    pad=20
)

# Sauvegarde de l'image
plt.savefig('repartition_clients.png', bbox_inches='tight')
plt.close(fig) # On ferme la figure pour libérer la mémoire

print("Le graphique a été généré et sauvegardé sous le nom 'repartition_clients.png'.")
