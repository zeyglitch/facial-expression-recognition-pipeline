# Facial Expression Recognition — Pipeline classique

Pipeline complet de reconnaissance d'expressions faciales sur la base RAF-DB (15 339 images, 7 émotions), développé dans le cadre du cours GTI771 à l'ÉTS Montréal (Automne 2026).

## Résultats

| Modèle | F1 macro (validation) |
|---|---|
| Random Forest — jeu rééquilibré par augmentation | **0.8908** |
| Logistic Regression — sans rééquilibrage | 0.2513 |
| SVM RBF — sans rééquilibrage | 0.2548 |
| Random Forest (`class_weight=balanced`) — sans rééquilibrage | 0.1896 |

Évaluation sur une validation stratifiée 80/20 issue du jeu original non augmenté (2 451 exemples représentatifs de la distribution réelle).

## Démarche (Labs 1 → 3)

1. **Nettoyage** — 12 271 → 12 254 images (règles métier + statistiques, Lab 1)
2. **Prétraitement** — égalisation CLAHE, redimensionnement 128×128
3. **Rééquilibrage** — augmentation ciblée sur les classes minoritaires (Fear : 281 images, Disgust : 717 images)
4. **Extraction de primitives** — GLCM/Haralick avec zonage 4×4 → vecteurs de **320 dimensions** par image, stockés en HDF5 (Lab 2)
5. **Classification supervisée** — Régression logistique, Random Forest, SVM RBF (ce dépôt, Lab 3)
6. **Évaluation** — F1 macro, matrice de confusion, rapport de classification par classe

## Points méthodologiques clés

- **Pas de fuite de données** : `StandardScaler` intégré dans un `Pipeline` scikit-learn (`fit` sur le train uniquement, `transform` sur la validation).
- **Validation représentative** : le split train/val est effectué sur le jeu *original non rééquilibré*, pour conserver la distribution réelle du problème (Happiness majoritaire à ~39 %).
- **Métrique adaptée** : F1 macro plutôt qu'accuracy — sur un jeu déséquilibré, un modèle qui prédit systématiquement Happiness obtient ~39 % d'accuracy sans rien apprendre.
- **Réserve sur l'écart RF augmenté** : l'écart de +70 points de F1 entre `class_weight='balanced'` et le jeu rééquilibré par augmentation est atypique. Une explication plausible est une contamination partielle train/val si des paires (original, augmentation) se retrouvent de part et d'autre du split. Une validation `GroupKFold` par identité d'image originale permettrait de trancher.

## Stack technique

Python · scikit-image · scikit-learn · h5py · NumPy · Pandas · Matplotlib · Seaborn

## Structure

```
├── P1_Lab1_Lab2_GTI771_A26_pipeline.ipynb   # à uploader maintenant
├── P1_Lab3_GTI771_A26_classification.ipynb # Notebook principal (Labs 1+2+3)
├── README.md
└── LICENSE
```

## Reproduire

Les fichiers `.h5` de primitives (~50 MB) ne sont pas versionnés — RAF-DB est distribué sous licence académique. Pour les régénérer, exécuter les Labs 1 et 2 du cours GTI771 sur la base RAF-DB.

Le notebook est conçu pour Google Colab (montage Drive). Pour une exécution locale, commenter la cellule `drive.mount` et adapter les chemins `PATH_TRAIN_NORM`, `PATH_TRAIN_REEQ`, `PATH_TEST`.

## Auteur

**Mathieu Jonniaux** — ISIS Castres (partenaire INSA), FIE4 DSIA
[github.com/zeyglitch](https://github.com/zeyglitch) · [LinkedIn](https://www.linkedin.com/in/mathieu-jonniaux)
