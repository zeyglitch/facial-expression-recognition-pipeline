# Reconnaissance d'expressions faciales : pipeline classique (RAF-DB)

Pipeline de reconnaissance d'expressions faciales sur RAF-DB (15 339 images, 7 émotions) : nettoyage, prétraitement, extraction de primitives GLCM/Haralick, puis classification. Projet personnel, réalisé à partir des énoncés et des supports du cours GTI771 de l'ÉTS Montréal (automne 2026).

## Résultats

| Modèle                                                 | F1 macro (validation) |
| ------------------------------------------------------ | --------------------- |
| Régression logistique                                  | 0,2784                |
| Random Forest (`class_weight=balanced`)                | 0,2691                |
| SVM RBF (`class_weight=balanced`)                      | 0,2536                |
| Random Forest, train enrichi par augmentation d'images | 0,2395                |

Validation stratifiée de 2 451 images originales, jamais augmentées.

Les scores sont faibles (toujours répondre « Happiness » donnerait environ 0,08) et très proches les uns des autres. Disgust n'est presque jamais reconnue.

L'augmentation d'images n'améliore pas le Random Forest : son F1 baisse de 0,03 (intervalle à 95 % par bootstrap sur la validation : de −0,0489 à −0,0095). Cet intervalle ne tient pas compte du hasard de l'entraînement ni du découpage : le même Random Forest avait fait 0,19 sur mon premier découpage et 0,27 sur celui-ci. Je n'ai pas testé l'augmentation avec la régression logistique ni le SVM.

Les images synthétiques se distinguent des vraies par leurs primitives (AUC de 0,82 à 0,85 pour les séparer, à l'intérieur d'une même classe), sans que je sache encore d'où vient cette différence.

## Une fuite de données, détectée et corrigée

Ma première version annonçait un F1 macro de 0,8908. Ce chiffre était faux : j'avais découpé train et validation après l'augmentation. 80,0 % des vecteurs de validation se retrouvaient à l'identique dans l'entraînement, avec leurs copies retournées ou tournées (pour la classe Fear, la distance médiane au plus proche voisin passait de 8,36 à 2,69).

J'ai corrigé en découpant avant d'augmenter, et en sauvegardant les noms de fichiers dans les `.h5` pour pouvoir le vérifier. Il reste 2 vecteurs de validation identiques à un vecteur du train sur 2 451 (0,08 %). Il s'agit d'images originales des classes Anger et Neutral, dont les fichiers sont identiques dans l'archive RAF-DB. Avec la même mesure que dans la première version (distance médiane au plus proche voisin), Fear passe maintenant de 10,56 à 10,13 : la distance ne s'effondre plus.

## Démarche

1. **Nettoyage** : 12 271 → 12 254 images (taille, ratio, luminosité).
2. **Prétraitement** : CLAHE, redimensionnement 128×128.
3. **Découpage** train 9 803 / validation 2 451 (stratifié, graine 42), avant toute augmentation.
4. **Rééquilibrage** : retournements et rotations de ±15° pour amener chaque classe du train au niveau de la plus grosse (3 813 images). Fear passe de 224 à 3 813 et Disgust de 573 à 3 813. Les images synthétiques ne dérivent que du train.
5. **Primitives** : GLCM/Haralick sur une grille 4×4, soit 320 valeurs par image, stockées en HDF5.
6. **Classification** : régression logistique, Random Forest, SVM RBF, évalués au F1 macro.

## Quelques points de méthode

- La normalisation est dans un `Pipeline` scikit-learn, apprise sur le train uniquement.
- J'utilise le F1 macro et pas l'accuracy : Happiness fait environ 39 % des images.
- Les paramètres du nettoyage (luminosité) sont calculés sur tout le train original, validation comprise. Ça ne concerne que 17 images exclues.

## Stack

Python · OpenCV · scikit-image · scikit-learn · h5py · NumPy · Pandas · Matplotlib · Seaborn

## Structure

```
├── 01_GTI771_Preparation_Donnees.ipynb
├── 02_GTI771_Classification.ipynb
├── requirements.txt
├── .gitignore
├── README.md
└── LICENSE
```

## Reproduire

Les images, les fichiers de labels et les `.h5` ne sont pas versionnés : RAF-DB est distribué sous licence académique.

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # Windows PowerShell
pip install -r requirements.txt
```

Placer les données de RAF-DB dans le dossier de travail (`train_labels.csv`, `original-train.zip`, le dossier `original-test`). Dans la première cellule de chaque notebook, `BASE_DIR` et `DATASET_DIR` valent `.` en local : à adapter si besoin. Le notebook 01 génère les trois `.h5` d'entraînement et de validation ainsi que celui du test, le notebook 02 les utilise. Les deux notebooks détectent Google Colab automatiquement.

## Licence

Code sous licence MIT (voir `LICENSE`). Les données RAF-DB restent soumises à leur propre licence.

## Auteur

Mathieu Jonniaux, ISIS Castres (partenaire INSA), FIE4 DSIA
[github.com/zeyglitch](https://github.com/zeyglitch) · [LinkedIn](https://www.linkedin.com/in/mathieu-jonniaux)
