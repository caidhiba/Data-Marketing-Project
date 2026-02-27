import pandas as pd
import matplotlib.pyplot as plt

# --- 1. CHARGEMENT ET PRÉPARATION ---
df = pd.read_csv('customers.csv')

# On s'assure que la colonne est bien numérique et on enlève les valeurs vides
df['tenure_days'] = pd.to_numeric(df['tenure_days'], errors='coerce')
df = df.dropna(subset=['tenure_days']).copy()

# On s'assure qu'il n'y a pas d'ancienneté négative (erreur de saisie CRM)
df = df[df['tenure_days'] >= 0]

# Pour éviter l'écrasement visuel dû à quelques clients pionniers (ex: depuis 1990), 
# on filtre à 95% des données normales
q_tenure = df['tenure_days'].quantile(0.95)
df_plot = df[df['tenure_days'] <= q_tenure]


# --- 2. CRÉATION DU GRAPHIQUE (Histogramme) ---
plt.figure(figsize=(10, 6))

# Histogramme (barres) en BLEU CIEL avec contours en VIOLET
plt.hist(
    df_plot['tenure_days'], bins=40, 
    color='skyblue', edgecolor='purple', alpha=0.8
)

# Ajout des lignes pour la moyenne et la médiane en ROSE
moyenne_tenure = df_plot['tenure_days'].mean()
mediane_tenure = df_plot['tenure_days'].median()

plt.axvline(moyenne_tenure, color='hotpink', linestyle='dashed', linewidth=2.5, 
            label=f'Moyenne ({moyenne_tenure:.0f} jours)')
plt.axvline(mediane_tenure, color='hotpink', linestyle='dotted', linewidth=2.5, 
            label=f'Médiane ({mediane_tenure:.0f} jours)')

# Titres et labels en VIOLET
plt.title(
    "Distribution de l'Ancienneté des Clients (Tenure)\n(Sous le 95e percentile)", 
    fontsize=15, pad=15, color='purple', fontweight='bold'
)
plt.xlabel("Ancienneté (Nombre de jours depuis le premier achat)", fontsize=12, color='purple')
plt.ylabel("Nombre de Clients", fontsize=12, color='purple')

# Grille et axes en VIOLET léger
plt.grid(True, linestyle='--', alpha=0.3, color='purple')

ax = plt.gca()
ax.tick_params(colors='purple')
for spine in ax.spines.values():
    spine.set_color('purple')

# Légende
plt.legend(loc='upper right', frameon=True, labelcolor='purple')

# Sauvegarde de l'image
plt.savefig('distribution_anciennete_colors.png', bbox_inches='tight', dpi=150)
plt.close()

print("Graphique de l'ancienneté généré avec succès !")
