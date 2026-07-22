# Utilisation de base

Ce tutoriel décrit le déroulement complet d'une analyse, du chargement des
données à l'exploration des cartographies T2.

## 1. Lancer le pipeline

```bash
pixi run sphene
```

Options disponibles (`python main.py --help`) :

| Flag | Effet |
|------|-------|
| `--device {cpu,gpu}` | Device utilisé pour le filtrage spatial. `gpu` nécessite l'environnement `gpu` (`pixi install -e gpu`) ; repli automatique sur CPU sinon, avec message explicite. |
| `--compare-filters` | Compare les stratégies de filtrage (`none`, `gaussian_spatial`, `savgol_temporal`) sur la coupe centrale, affiche R²/RMSE moyens, et ne lance pas le visualiseur. |

## 2. Charger un fichier NIfTI

Une fenêtre de sélection de fichier s'ouvre. Choisissez votre fichier
`.nii` ou `.nii.gz` (volume 4D : x, y, z, temps d'écho).

## 3. Définir les temps d'écho (TE)

Trois options sont proposées :

- **Load ACQP file** : sélectionner le fichier `acqp` associé à l'acquisition
  Bruker. Les temps d'écho sont extraits automatiquement du champ
  `##$ACQ_echo_time`.
- **Generate TE from NIfTI shape** : génère des TE synthétiques (3, 6, 9, …
  ms), utile pour une exploration rapide sans fichier ACQP.
- **Enter TE manually** : saisie manuelle des valeurs, séparées par des
  espaces ou virgules.

## 4. Filtrage optionnel du signal

Avant le calcul du masque et l'ajustement des modèles, un filtre peut être
appliqué au signal brut (voir [`filter_data`](../api/utils.md)) :

- `gaussian_spatial` : lissage spatial, par coupe et par écho indépendamment.
- `savgol_temporal` : lissage le long de l'axe des temps d'écho, voxel par
  voxel.

Le filtre actif se règle via `FILTER_METHOD` dans `main.py` (`"none"` par
défaut). Pour comparer objectivement les méthodes avant de choisir, utiliser
`pixi run filter-compare` (voir [`compare_filters`](../api/utils.md)).

## 5. Masque tissulaire

Un masque binaire est calculé automatiquement avec la méthode **Rician**
(seuillage basé sur le bruit de fond, recommandé pour les données IRM en
magnitude), sur les données déjà filtrées le cas échéant. Voir
[`compute_mask`](../api/utils.md) pour les autres méthodes disponibles
(`otsu`, `histogram`).

## 6. Interface interactive

La fenêtre principale affiche le volume avec deux curseurs :

- **Slice** : navigation dans l'axe z
- **Echo (ms)** : navigation entre les temps d'écho

### Boutons d'analyse

| Bouton    | Action |
|-----------|--------|
| **Fit**   | Active le mode clic-pour-ajuster. Cliquer sur un voxel affiche les 4 modèles de décroissance avec leurs AIC, le meilleur étant mis en évidence (★). |
| **Mono Map** | Cartographies T2 mono-exponentielles (standard, avec offset, avec offset fixe global). |
| **Bi Map**   | Cartographies T2 bi-exponentielles (composantes courte/longue, T2 effectif). |
| **Utils**    | Carte de sélection de modèle (AIC), fractions d'eau, cartes I0. |
| **Error**    | Cartes R² et RMSE, voxel-wise et avec modèle global. |
| **Noise**    | Histogramme du bruit de fond, cartes de l'offset C (zéros exclus du calcul de moyenne, seuil à 1.0 pour écarter les solutions bornées). |
| **3D View**  | Rendu volumique interactif (ouvre dans le navigateur). |
| **Export**   | Exporte toutes les cartes déjà calculées pour la coupe affichée en un CSV unique, format long (voir ci-dessous). |
| **Reset**    | Réinitialise les curseurs et l'état des boutons. |

!!! tip "Cache par slice"
    Chaque calcul (Mono Map, Bi Map, etc.) est mis en cache par slice. Changer
    de slice puis revenir ne relance pas le calcul.

!!! tip "Lecture des coordonnées"
    Sur le visualiseur principal et les cartes calculées (Mono Map, Bi Map,
    Error, I0), la barre d'outils matplotlib affiche `x=... y=...`
    directement alignés sur l'indexation utilisée dans les CSV exportés —
    pas besoin d'inverser les coordonnées manuellement.

## 7. Export des résultats

Un clic sur **Export** génère `exports/T2_slice_z<N>.csv` : une ligne par
voxel du masque tissulaire, une colonne par grandeur calculée (T2, C, R²,
AIC, ...). Format compatible Excel français : séparateur `;`, décimales à
la virgule. Chaque écriture est relue et comparée aux données en mémoire
avant de confirmer le succès (bouton vert) ou signaler un écart précis
(bouton rouge, détail en console).

## Exemple d'analyse voxel par voxel

1. Cliquer sur **Fit** (le bouton devient vert)
2. Cliquer sur un voxel dans l'image
3. Une fenêtre s'ouvre avec la courbe de décroissance et les 4 modèles
   ajustés ; le terminal affiche un tableau récapitulatif (I0, T2, AIC, R²,
   RMSE) avec le meilleur modèle marqué ★
