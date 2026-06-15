# Sphene MRI

Pipeline d'analyse de relaxométrie T2 pour des données IRM de plantes
(graines, algues, tissus végétaux).

## Fonctionnalités

- Chargement de fichiers NIfTI et lecture des temps d'écho (fichiers ACQP
  Bruker, saisie manuelle, ou génération synthétique)
- Génération de masques tissulaires (Rician, Otsu, histogramme)
- Ajustement de modèles de décroissance T2 : mono-exponentiel, mono-exponentiel
  avec offset, bi-exponentiel, bi-exponentiel avec offset
- Sélection de modèle par critère d'information d'Akaike (AIC)
- Cartographies paramétriques voxel par voxel (T2, I0, fractions, erreurs)
- Visualisation interactive 2D (matplotlib) et 3D (Plotly)

## Installation

Ce projet utilise [Pixi](https://pixi.sh/) pour la gestion de l'environnement.

```bash
git clone git@github.com:e-gernet/sphene-mri.git
cd sphene-mri
pixi install
```

## Utilisation

```bash
pixi run python main.py
```

## Documentation

La documentation complète (guide d'installation, tutoriel d'utilisation,
référence API) est disponible ici :
**[https://e-gernet.github.io/sphene-mri/](https://e-gernet.github.io/sphene-mri/)**

Pour la consulter en local :

```bash
pixi run mkdocs serve
```

## Statut du projet

Projet en développement actif, dans le cadre d'un travail de recherche INRAE
sur l'analyse de données IRM de tissus végétaux. Les fonctionnalités de
diffusion (DWI) seront ajoutées dans une prochaine version.

## Licence

À définir.