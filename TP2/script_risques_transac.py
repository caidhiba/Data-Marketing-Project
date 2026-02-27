import pandas as pd
import matplotlib.pyplot as plt

# --- 1. CHARGEMENT ET PRÉPARATION ---
df_clean = pd.read_csv('customers.csv')

# Uniformisation des noms de colonnes en français (au cas où)
colonnes_a_renommer = {
    'recency_days': 'Récence',
    'n_orders': 'Fréquence',
    'total_spent': 'Montant Total'
}
df_clean = df_clean.rename(columns=colonnes_a_renommer)

for col in ['Récence', 'Fréquence', 'Montant Total']:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
df_clean = df_clean.dropna(subset=['Récence', 'Fréquence', 'Montant Total'])


# --- 2. CALCUL DES SCORES RFM PAR QUINTILES ---
df_clean['R_Score'] = pd.qcut(df_clean['Récence'].rank(method='first'), 5, labels=[5, 4, 3, 2, 1])
df_clean['F_Score'] = pd.qcut(df_clean['Fréquence'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
df_clean['M_Score'] = pd.qcut(df_clean['Montant Total'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])


# --- 3. DÉFINITION DES CLIENTS "À RISQUE" ---
# Critère : Score R faible (1 ou 2) ET forte fidélité historique (Score F >= 4 OU Score M >= 4)
clients_a_risque = df_clean[
    df_clean['R_Score'].isin([1, 2]) & 
    (df_clean['F_Score'].isin([4, 5]) | df_clean['M_Score'].isin([4, 5]))
]

# Calculs globaux
total_ca = df_clean['Montant Total'].sum()
total_clients = len(df_clean)

# Calculs pour les clients à risque
ca_a_risque = clients_a_risque['Montant Total'].sum()
nb_a_risque = len(clients_a_risque)

print(f"--- RÉSULTATS : CLIENTS À RISQUE ---")
print(f"Nombre de clients en passe de nous quitter : {nb_a_risque} (soit {nb_a_risque/total_clients*100:.1f} % de la base)")
print(f"CA historique menacé : {ca_a_risque:,.2f} € (soit {ca_a_risque/total_ca*100:.1f} % du CA total)")


# --- 4. GRAPHIQUE (Pie Chart des CA menacés vs sécurisés) ---
plt.figure(figsize=(8, 8))

labels = ['Part du CA menacée\n(Clients "À risque")', 'CA sécurisé\n(Autres clients)']
tailles_ca = [ca_a_risque, total_ca - ca_a_risque]

# Création du Camembert avec les couleurs Bleu Ciel et Rose
plt.pie(
    tailles_ca, 
    labels=labels, 
    colors=['hotpink', 'skyblue'], 
    autopct='%1.1f%%', 
    startangle=140,
    textprops={'fontsize': 12, 'color': 'purple', 'fontweight': 'bold'},
    explode=(0.1, 0), # Fait ressortir la part menacée
    shadow=True
)

# Textes en Violet
plt.title(
    "Part du Chiffre d'Affaires menacée par l'attrition", 
    fontsize=14, color='purple', fontweight='bold', pad=20
)

# Sauvegarde de l'image
plt.savefig('rfm_a_risque_ca_colors.png', bbox_inches='tight', dpi=150)
plt.close()

print("\nLe graphique de répartition a été sauvegardé avec succès !")
