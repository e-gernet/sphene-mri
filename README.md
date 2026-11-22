> 🏠 [Retour à l'accueil du projet](https://github.com/e-gernet/sphene-mri) — description des versions disponibles (`mpl` / `pyqt`).

# Sphene MRI

Pipeline d'analyse de relaxométrie T2 pour des données IRM de plantes
(graines, algues, tissus végétaux).

## Fonctionnalités

- Chargement de fichiers NIfTI et lecture des temps d'écho (fichiers ACQP
  Bruker, saisie manuelle, ou génération synthétique)
- Filtrage optionnel du signal avant masquage et fit (spatial ou temporel),
  avec un mode de comparaison objective (R², RMSE) et accélération GPU
  facultative pour le filtre spatial
- Génération de masques tissulaires (Rician, Otsu, histogramme)
- Ajustement de modèles de décroissance T2 par maximum de vraisemblance, avec
  la loi de bruit des images en module (loi de Rice) et le niveau de bruit
  mesuré dans le fond de l'image : mono-exponentiel et bi-exponentiel, sans
  paramètre « offset »
- Choix du modèle par le critère AICc : le bi n'est retenu que là où les
  données le justifient
- Cartographies paramétriques voxel par voxel (T2, I0, fractions, χ² réduit,
  erreurs)
- Sélection interactive (lasso / rectangle / cercle) pour exclure des voxels
  du masque, les flaguer (ex: capillaires), ou ajuster ensemble tous les voxels
  d'une région
- Rapport de validation par simulation (`pixi run validate`) : graphes,
  statistiques et explications montrant que les T2 sont justes
- Export des cartes calculées en tableau CSV (une ligne par voxel), filtré
  sur le masque tissulaire, avec coordonnées physiques (mm) et
  auto-vérification à l'écriture
- Visualisation interactive 2D (matplotlib) et 3D (Plotly), à l'échelle
  physique réelle

## Prérequis

Avant d'installer le projet, vous devez avoir :

- Git
- Ce projet utilise [Pixi](https://pixi.sh/) pour la gestion de l'environnement.

## Installation

### 1. Installer Git

#### Windows
Télécharger depuis :
https://git-scm.com/download/win

ou via winget :
```bash
winget install Git.Git
```

#### Linux
```bash
sudo apt update
sudo apt install git
```

#### macOS
```bash
brew install git
```

### 2. Installer Pixi

Pixi est compatible Windows / Linux / macOS.

#### Windows (PowerShell)
```bash
winget install prefix-dev.pixi
```

#### Linux / macOS
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### 3. Installer Sphene MRI

```bash
git clone https://github.com/e-gernet/sphene-mri.git
cd sphene-mri
git checkout mpl
pixi install
```

Environnements optionnels :

```bash
pixi install -e dev   # lint, formatage, documentation locale
pixi install -e gpu   # accélération GPU du filtrage (NVIDIA/CUDA uniquement)
```

## Utilisation

```bash
pixi run sphene
```

Avec accélération GPU :

```bash
pixi run -e gpu sphene-gpu
```

Comparer les stratégies de filtrage avant de choisir :

```bash
pixi run filter-compare
```

Vérifier le calcul (rapport HTML avec graphes, statistiques et explications) :

```bash
pixi run validate-quick     # 1 à 2 minutes
pixi run validate           # version complète (environ 8 minutes), à citer dans un document
pixi run simulate           # jeux simulés .nii + .acqp avec vérité connue
```

## Documentation

La documentation complète (guide d'installation, tutoriel d'utilisation,
référence API) est disponible ici :
**[https://e-gernet.github.io/sphene-mri/](https://e-gernet.github.io/sphene-mri/)**

Pour la consulter en local :

```bash
pixi run -e dev docs
```

## Statut du projet

Projet en développement actif, dans le cadre d'un travail de recherche INRAE
sur l'analyse de données IRM de tissus végétaux. Les fonctionnalités de
diffusion (DWI) seront ajoutées dans une prochaine version. La version PyQt
est en préparation.

## Licence

À définir.