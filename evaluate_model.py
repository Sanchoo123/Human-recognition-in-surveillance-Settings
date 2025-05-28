import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Matriz: [[TN, FP], [FN, TP]]
conf_matrix = np.array([[116, 34],   # Linha 0: impostores
                        [34, 116]])  # Linha 1: genuínos

plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Impostor', 'Genuíno'],  # Labels para colunas
            yticklabels=['Impostor', 'Genuíno'])  # Labels para linhas
plt.title('Matriz de Confusão')
plt.show()  