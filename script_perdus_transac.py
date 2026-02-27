import pandas as pd
import matplotlib.pyplot as plt

# --- 1. CHARGEMENT ET PRÉPARATION ---
df_clean = pd.read_csv('customers.csv')

# Uniformisation des noms de colonnes en français
colonnes_a_renommer = {
    'recency_days': 'Récence',
    'n_orders': 'Fréquence',
    'total_spent': 'Montant Total'
}
df_clean = df_clean.rename(columns=colonnes_a_renommer)

# Sécurisation des données
for col in ['Récence', 'Fréquence', 'Montant Total']:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
df_clean = df_clean.dropna(subset=['Récence', 'Fréquence', 'Montant Total'])


# --- 2. CALCUL DES SCORES RFM PAR QUINTILES ---
df_clean['R_Score'] = pd.qcut(df_clean['Récence'].rank(method='first'), 5, labels=[5, 4, 3, 2, 1])
df_clean['F_Score'] = pd.qcut(df_clean['Fréquence'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
df_clean['M_Score'] = pd.qcut(df_clean['Montant Total'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])

# Création du score textuel global (ex: '111')
df_clean['RFM_Score'] = df_clean['R_Score'].astype(str) + df_clean['F_Score'].astype(str) + df_clean['M_Score'].astype(str)


# --- 3. DÉFINITION DES CLIENTS "PERDUS" (Score exact 111) ---
clients_perdus = df_clean[df_clean['RFM_Score'] == '111']

# Calculs globaux
total_clients = len(df_clean)
total_ca = df_clean['Montant Total'].sum()

# Calculs pour les clients perdus
nb_perdus = len(clients_perdus)
ca_perdus = clients_perdus['Montant Total'].sum()

print(f"--- RÉSULTATS : CLIENTS PERDUS (Score 111) ---")
print(f"Nombre de clients 'morts' : {nb_perdus} (soit {nb_perdus/total_clients*100:.1f} % de la base)")
print(f"CA historique généré par eux : {ca_perdus:,.2f} € (soit {ca_perdus/total_ca*100:.1f} % du CA total)")


# --- 4. GRAPHIQUE (Pie Chart des clients Perdus vs Reste de la base) ---
plt.figure(figsize=(8, 8))

labels = ['Base Perdue\n(Score 111)', 'Reste de la base']
tailles_clients = [nb_perdus, total_clients - nb_perdus]

# Création du Camembert avec les couleurs Rose et Bleu Ciel
plt.pie(
    tailles_clients, 
    labels=labels, 
    colors=['hotpink', 'skyblue'], 
    autopct='%1.1f%%', 
    startangle=140,
    textprops={'fontsize': 12, 'color': 'purple', 'fontweight': 'bold'},
    explode=(0.1, 0), # Fait ressortir la part perdue
    shadow=True
)

# Textes en Violet
plt.title(
    "Proportion de la Base Clients 'Perdue' (Score RFM 111)", 
    fontsize=14, color='purple', fontweight='bold', pad=20
)

# Sauvegarde de l'image
plt.savefig('rfm_perdus_colors.png', bbox_inches='tight', dpi=150)
plt.close()

print("\nLe graphique de la base perdue a été sauvegardé avec succès sous 'rfm_perdus_colors.png' !")