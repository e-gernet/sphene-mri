# Utilisation de base

Ce tutoriel décrit le déroulement complet d'une analyse, du chargement des
données à l'exploration des cartographies T2. Pour comprendre **comment** les
T2 sont calculés et **pourquoi** on peut leur faire confiance, voir la page
[Méthode et validation](../methode.md).

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

## 4. Réglages de `main.py`

Deux réglages se trouvent en tête de `main.py`.

**`FILTER_METHOD`** (`"none"` par défaut) : filtre appliqué au signal brut avant
le masque et l'ajustement (voir [`filter_data`](../api/utils.md)) :

- `gaussian_spatial` : lissage spatial, par coupe et par écho indépendamment.
- `savgol_temporal` : lissage le long de l'axe des temps d'écho, voxel par
  voxel.

!!! warning "Garder « none » pour les ajustements T2"
    Le calcul de T2 modélise le bruit des images brutes en module (loi de Rice).
    Un filtre change ce bruit : le niveau de bruit est alors mesuré sur les
    données filtrées et le programme affiche un avertissement. Pour comparer
    objectivement les filtres avant de choisir, utiliser `pixi run filter-compare`
    (voir [`compare_filters`](../api/utils.md)).

**`SKIP_FIRST_ECHOES`** (`0` par défaut) : nombre de premiers échos laissés de côté.
Sur certaines séquences, les tout premiers échos s'écartent un peu de la
décroissance attendue (impulsions de refocalisation imparfaites, échos stimulés).
La fenêtre **Noise** (voir plus bas) affiche en console l'écart de chacun des trois
premiers échos, en pourcentage du signal, et prévient à partir de 5 %. Les simulations
du rapport de validation (section « Premiers échos perturbés ») donnent la règle :

- **écart de quelques pour cent (jusqu'à 5 % environ) : laisser `0`.** Un décalage de 3 %
  du deuxième écho change T2 de −0,3 % seulement ;
- **écart de 12 % ou plus : l'option devient utile** (T2 faussé d'environ −1,2 % et, si le
  bruit est faible, un modèle bi retenu à tort dans 14 % des voxels) ;
- **dans ce cas, ignorer tous les échos décalés (la console dit combien), jamais un seul de
  moins.** Ignorer un seul écho alors que le suivant est lui aussi décalé est pire que de
  n'en ignorer aucun : un écho trop bas et un écho trop haut se compensent en partie, et en
  retirer un seul laisse l'autre sans contrepartie. Exemple avec le motif mesuré sur l'acanthe
  (1<sup>er</sup> écho à −8,7 %, 2<sup>e</sup> à +10,5 %), simulé sur une zone de 64 voxels
  (tissu mono de T2 30 ms, SNR 20) :

    | Échos ignorés | Biais sur T2 | Bi retenu à tort |
    |---------------|--------------|------------------|
    | aucun | +0,7 % | 0 % |
    | 1 seul | **−3,8 %** | **100 % des zones** |
    | 2 | +0,04 % | 1 % |
    | 3 | −0,04 % | 0 % |

    Sur un voxel isolé, ces écarts sont du même ordre que l'erreur due au bruit (près de 5 %) :
    l'option sert surtout à « Zone fit », où le bruit s'efface et où le défaut apparaît.

- **ignorer des échos a un prix** : l'incertitude sur T2 est multipliée par 1,3 (T2 de
  120 ms), 1,4 (30 ms) et 2,3 (12 ms). Ne pas l'activer « par précaution ».

## 5. Masque tissulaire

Un masque binaire est calculé automatiquement avec la méthode **Rician**
(seuillage basé sur le bruit de fond, recommandé pour les données IRM en
magnitude), sur les données déjà filtrées le cas échéant. Voir
[`compute_mask`](../api/utils.md) pour les autres méthodes disponibles
(`otsu`, `histogram`).

Le niveau de bruit **σ** est mesuré une fois pour toutes dans le fond de
l'image (voir [`estimate_sigma`](../api/utils.md)) ; il est affiché en console
avec un contrôle : le fond doit se comporter comme du bruit de Rayleigh (c'est
ce que la loi de Rice donne quand il n'y a aucun signal).

## 6. Interface interactive

La fenêtre principale affiche le volume avec deux curseurs :

- **Slice** : navigation dans l'axe z (la première coupe est la **1**)
- **Echo (ms)** : navigation entre les temps d'écho (le premier écho est le **1**)

### Boutons d'analyse

Tous les boutons d'analyse lisent **le même calcul**, fait une seule fois par
coupe : le modèle mono et le modèle bi sont ajustés sur chaque voxel du masque,
puis le programme garde, voxel par voxel, le modèle que les données justifient
(critère AICc, voir [Méthode et validation](../methode.md)). Le premier bouton
cliqué sur une coupe lance le calcul (le bouton passe à l'orange) ; les suivants
sont instantanés.

| Bouton    | Action |
|-----------|--------|
| **Fit**   | Active le mode clic-pour-ajuster. Cliquer sur un voxel ajuste les modèles mono et bi, trace les deux courbes et affiche un tableau (I0, T2, f, AICc, χ² réduit) ; le modèle retenu est marqué ★ et tracé en gras. |
| **Mono Map** | Carte du T2 mono-exponentiel de la coupe, avec sa distribution ; la console donne ses statistiques (voir plus bas). |
| **Bi Map**   | T2c (composante courte), T2l (longue), fraction d'eau f et T2 effectif f·T2c + (1−f)·T2l, **uniquement dans les voxels où les données justifient deux composantes** ; gris clair = mono suffit. La console donne les statistiques de chaque carte. |
| **Utils**    | Modèle retenu dans chaque voxel, preuve en faveur du bi (ΔAICc), fraction d'eau, cartes I0 (voxel par voxel et globale), image lissée de référence. |
| **Error**    | Cartes de χ² réduit, RMSE et R² (modèle retenu par voxel, et un modèle unique pour toute la coupe), avec leurs distributions. |
| **Noise**    | Histogramme du fond comparé à la loi du bruit pur, masque, niveau de bruit σ écho par écho, et résidu moyen de chaque écho (voir plus bas). |
| **3D View**  | Rendu volumique interactif (ouvre dans le navigateur). |
| **Select**   | Ouvre le panneau de sélection de zones (voir la section suivante). |
| **Restore mask** | Revient au masque automatique sur toutes les coupes (annule Exclude et Fit only), efface les drapeaux capillaires et vide les cartes en cache. |
| **Export**   | Exporte toutes les cartes de la coupe affichée en un CSV unique, format long (voir ci-dessous). |
| **Reset**    | Réinitialise les curseurs et l'état des boutons. Ne touche pas au masque : utiliser **Restore mask**. |

!!! tip "Cache par slice"
    Le calcul d'une coupe est mis en cache. Changer de coupe puis revenir ne
    relance rien. **Exclude**, **Fit only** et **Restore mask** vident le cache
    de la ou des coupes concernées.

!!! tip "Lecture des coordonnées"
    Sur le visualiseur principal et les cartes calculées, la barre d'outils
    matplotlib affiche `x=... y=...` directement alignés sur l'indexation
    utilisée dans les CSV exportés — pas besoin d'inverser les coordonnées
    manuellement.

### Statistiques dans la console

Quatre actions écrivent aussi des statistiques dans la console : nombre de voxels
(n), moyenne, écart-type (SD), médiane, minimum et maximum. Ce sont les mêmes
calculs et la même présentation que ceux de `quickstats.py` (voir la section 8).

| Action | Ce que la console écrit |
|--------|-------------------------|
| **Mono Map** | une ligne : le T2 mono de la coupe affichée |
| **Bi Map** | une ligne par carte (T2c, T2l, f, T2 effectif), calculée **seulement sur les voxels où le bi est retenu** |
| **Show signal** | le signal moyen au premier écho, au dernier écho et le plancher de bruit, puis une ligne avec le T2 mono de chaque voxel de la zone |
| **Zone fit** | le tableau de l'ajustement d'ensemble (I0, T2, f, AICc, χ² réduit), puis la même ligne de T2 mono voxel par voxel |

Exemple (Mono Map, valeurs d'illustration) :

```text
[Stats] slice 8
  map                   n     mean       SD   median      min      max
  T2 mono (ms)       3834     21.7      5.3     21.6      2.9     41.6
```

L'écart-type est celui **d'échantillon** (division par n − 1), comme dans la
plupart des articles et des logiciels (R, Excel, pandas). NumPy, lui, divise par
n par défaut : un calcul refait à la main avec `np.std` donne un résultat un peu
plus petit. Les valeurs non finies (voxels où l'ajustement a échoué) sont laissées
de côté.

### Lire la fenêtre Noise

- **Histogramme du fond** : doit épouser la courbe de la loi de Rayleigh (bruit
  pur, sans signal). Un écart net signale une image filtrée, un fond qui contient
  du signal, ou une acquisition à plusieurs antennes.
- **σ écho par écho** : doit être à peu près constant. Le σ utilisé est la médiane.
- **Résidu moyen de chaque écho** : « mesuré moins attendu » en unités de bruit,
  moyenné sur les voxels de la coupe. Si le modèle explique l'écho, la valeur est 0
  (les barres sont ± 2 erreurs-types). La console donne aussi l'écart des trois
  premiers échos en pourcentage du signal et prévient si l'un d'eux atteint 5 % :
  voir `SKIP_FIRST_ECHOES` ci-dessus.

### Sélection de zones (bouton Select)

Le bouton **Select** fait apparaître trois réglages à gauche de l'image :

- **Shape** : forme du tracé, *Lasso* (à main levée), *Box* (rectangle) ou
  *Circle* (cercle ou ellipse).
- **Action** : ce qu'on fait de la zone tracée (liste ci-dessous).
- **All slices** : si coché, l'action s'applique à toutes les coupes au lieu de
  la coupe affichée.

Le mode reste actif après chaque tracé (le bouton **Select** reste vert) :
on peut enchaîner plusieurs zones. Il se ferme en cliquant sur un autre
bouton. Seuls les voxels du masque tissulaire comptent ; le fond n'est
jamais pris en compte.

| Action | Effet |
|--------|-------|
| **Zone fit** | Ajuste les modèles mono et bi sur **tous les voxels de la zone ensemble** (une seule vraisemblance : les courbes ne sont pas moyennées avant l'ajustement, ce qui évite de déformer le plancher de bruit ; chaque voxel garde sa propre amplitude, seule la forme de la décroissance est commune). Une fenêtre montre la courbe moyenne et les ajustements, la console affiche le tableau (I0, T2, f, AICc, χ² réduit) avec le modèle retenu ★, puis les statistiques du T2 mono de chaque voxel de la zone. Les coordonnées de la zone sont conservées (voir Export). |
| **Exclude** | Retire les voxels de la zone du masque. Les cartes et l'export suivants ne les contiennent plus (cartes de la coupe concernée à recalculer). |
| **Fit only** | Inverse d'Exclude : garde **uniquement** les voxels de la zone, tous les autres sortent du masque. Une nouvelle zone *Fit only* remplace la précédente. Les voxels déjà retirés par *Exclude* restent exclus. |
| **Flag capillary** | Marque les voxels comme capillaire (tube de référence). L'export ajoute une colonne `is_capillary` (1 = capillaire), seulement si au moins un voxel est marqué sur la coupe. |
| **Show signal** | Affiche le signal brut de la zone, **sans aucun ajustement** : une courbe par voxel (gris), la moyenne (noir), une bande ± 1 écart-type et le plancher de bruit attendu σ√(π/2) (pointillé orange). Deux graphiques : échelle linéaire, et échelle logarithmique où une décroissance mono-exponentielle est une droite (une courbure indique plusieurs composantes). La console donne le signal moyen aux deux extrémités et les statistiques du T2 mono de chaque voxel. |

!!! warning "Un bi sur une grande zone n'est pas une preuve"
    Si la zone mélange des voxels de T2 différents (deux tissus voisins, un
    dégradé), l'ajustement d'ensemble peut préférer le bi alors que chaque voxel
    est mono. Regarder la **Mono Map** et **Show signal** voxel par voxel permet de
    trancher. Ce cas est mesuré dans la section « Zone hétérogène » du rapport de
    validation.

!!! tip "Revenir en arrière"
    **Restore mask** annule *Exclude*, *Fit only* et *Flag capillary* d'un
    coup. **Reset** ne touche pas au masque.

!!! note "Coordonnées"
    `x` est la ligne et `y` la colonne du tableau, comme dans les CSV et dans
    la barre d'outils matplotlib ; ils sont comptés à partir de 0. Les coupes et les
    échos affichés (curseurs, titres, console, CSV) sont comptés à partir de 1.

## 7. Export des résultats

Un clic sur **Export** génère `exports/T2_slice_<N>.csv` (N = numéro de la coupe
affichée, à partir de 1, comme le curseur Slice) : une ligne par
voxel du masque tissulaire, une colonne par grandeur calculée. Les cartes de la
coupe doivent avoir été calculées avant (cliquer d'abord sur un bouton d'analyse).

| Colonne | Contenu |
|---------|---------|
| `slice`, `x`, `y` | Position du voxel : `slice` = numéro de coupe **à partir de 1** (comme le curseur Slice) ; `x` = ligne et `y` = colonne, **à partir de 0**. |
| `T2_mono` | T2 du modèle mono (ms), calculé dans tous les voxels. |
| `T2c_bi`, `T2l_bi`, `f_bi`, `T2eff_bi` | Résultats du modèle bi ; **vides là où le mono suffit**. |
| `bi_kept` | 1 si les données justifient le bi, 0 si le mono suffit. |
| `delta_AICc` | Preuve en faveur du bi : AICc(mono) − AICc(bi). Le bi est retenu au-dessus de 2. |
| `I0`, `chi2_red`, `R2`, `RMSE` | Amplitude et qualité d'ajustement du modèle retenu. `chi2_red` vaut environ 1 si le modèle explique les mesures au niveau du bruit. |
| `T2_at_limit` | 1 si un T2 du modèle retenu est au bord de la plage de recherche : sa valeur n'est pas fiable. |
| `x_mm`, `y_mm`, `z_mm` | Position physique (si la taille de voxel est connue) ; la première coupe est à `z_mm` = 0. |
| `is_capillary` | 1 = capillaire (seulement si des voxels ont été marqués). |

Format : séparateur de colonnes `;`, décimales avec un point
(lecture directe avec `pandas.read_csv(..., sep=";")` ; dans Excel, passer par
l'import de texte avec le séparateur `;`). Chaque écriture est relue et
comparée aux données en mémoire avant de confirmer le succès (bouton vert) ou
signaler un écart précis (bouton rouge, détail en console).

Si des régions ont été analysées avec **Select > Zone fit**, Export écrit
aussi `exports/T2_ROIs_voxels.csv` (coordonnées de chaque voxel de chaque
région : `roi`, `slice`, `x`, `y`, numérotés comme ci-dessus) et `exports/T2_ROIs_fits.csv` (une ligne par région et par modèle :
`roi`, `scope`, `n_voxels`, `model`, `kept`, `I0`, `T2`, `T2c`, `T2l`, `f`,
`loglik`, `aicc`, `delta_aicc`, `chi2_red`, `at_limit`).

Les régions ne dépendent d'aucune carte. Si aucune carte n'a été calculée sur la
coupe affichée, Export écrit **seulement** ces deux fichiers : la console le dit
et le bouton passe au vert. S'il n'y a ni carte ni région, rien n'est écrit : le
bouton passe au rouge et la console indique quoi calculer d'abord.

## Exemple d'analyse voxel par voxel

1. Cliquer sur **Fit** (le bouton devient vert)
2. Cliquer sur un voxel dans l'image
3. Une fenêtre s'ouvre avec la courbe de décroissance et les deux modèles
   ajustés ; le terminal affiche un tableau récapitulatif (I0, T2, f, AICc,
   χ² réduit) avec le modèle retenu marqué ★ et la valeur de ΔAICc.

## 8. Statistiques rapides en console (sans interface)

Pour obtenir les statistiques de la carte T2 sans ouvrir la fenêtre, avec les
mêmes calculs que le bouton **Mono Map** (masque, bruit, ajustement par maximum
de vraisemblance) :

```bash
pixi run stats scan.nii                       # une image, coupe centrale
pixi run stats scan.nii --acqp scan.acqp      # avec son fichier ACQP
pixi run stats donnees/                       # tous les .nii d'un dossier, un par un
pixi run stats donnees/ --acqp dossier_acqp/  # ACQP retrouvés par leur nom
pixi run stats scan.nii --all-slices          # toutes les coupes + toutes les coupes ensemble
pixi run stats scan.nii --all-slices --exclude-capillary   # idem, sans le capillaire
```

Pour chaque image, la console donne le nombre de voxels, la moyenne, l'écart-type
(SD), la médiane, le minimum et le maximum du T2 (en ms), et la part des voxels où
le modèle bi est retenu. L'écart-type est l'écart-type d'échantillon (division
par n − 1), celui de la plupart des articles. Les boutons Mono Map et Bi Map de
la fenêtre affichent les mêmes statistiques, dans le même ordre, dans la console. Pour un dossier, un tableau récapitulatif suit.

**Temps d'écho.** Pour chaque image, le premier cas qui fonctionne est utilisé :

1. le fichier donné avec `--acqp` ;
2. dans le dossier donné avec `--acqp`, le fichier de même nom
   (`scan.nii` → `scan.acqp`) ;
3. à côté de l'image, le fichier de même nom ;
4. des TE synthétiques (3, 6, 9, … ms ; `--te-step` pour changer le pas).

Avec des TE synthétiques, les T2 sont à une échelle près : comparer des voxels
ou des zones reste exact, mais les valeurs en ms ne sont qu'approximatives.
La console le signale.

**Options utiles.**

| Option | Effet |
|--------|-------|
| `--all-slices` | Toutes les coupes : une ligne par coupe, plus une ligne « all slices » calculée sur l'ensemble des voxels de toutes les coupes (et non une moyenne des lignes). |
| `--slice N` | Une coupe précise, numérotée depuis 1 comme dans l'application (la première coupe est la 1), au lieu de la coupe centrale. |
| `--exclude-capillary` | Retire le capillaire (le tube de référence à côté de l'échantillon) des statistiques. Le programme le cherche **une seule fois dans toute l'image en 3D** : c'est un objet rond, plus petit que l'échantillon et qui ne le touche pas. Il est ensuite retiré de chaque coupe analysée, donc l'option marche avec la coupe centrale, `--slice N` et `--all-slices` (une coupe de bord qui ne contient que le capillaire apparaît avec n = 0). La console écrit ce qui est retiré (voxels, coupes, position). Si rien n'est trouvé (un seul objet, objet trop gros ou pas assez rond), elle le dit et ne retire rien. Le programme ne sait pas ce que contient le tube : il reconnaît seulement un objet rond plus petit que l'échantillon, à vérifier dans la note. Dans le CSV, la colonne `n_capillary` donne le nombre de voxels retirés par coupe (vide si l'option n'est pas utilisée). |
| `--exclude-unreliable` | Écarte les voxels dont le T2 est tombé sur une borne de recherche (valeur non fiable). Par défaut ils restent comptés, comme dans la fenêtre, et leur nombre est indiqué. |
| `--csv fichier` | Écrit aussi les résultats dans un CSV (séparateur `;`). La colonne `slice` donne le numéro de coupe à partir de 1 (comme l'application), ou `all` pour toutes les coupes ensemble. |
| `--recursive` | Avec un dossier : cherche aussi dans les sous-dossiers. |
| `-v` | Affiche aussi les messages détaillés (masque, bruit) et les barres de progression. |

## Pour aller plus loin : vérifier le calcul

```bash
pixi run validate-quick     # rapport de validation, 1 à 2 minutes
pixi run validate           # version complète (environ 8 minutes), à citer dans un document
```

Le rapport (`validation_report/rapport.html`) contient des graphes, des
statistiques et des explications ; voir [Méthode et validation](../methode.md).
