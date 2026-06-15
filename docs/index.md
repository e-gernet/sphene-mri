# Sphene MRI

**Pipeline d'analyse de relaxométrie T2 pour des données IRM de plantes**

Ce projet fournit un pipeline complet pour charger, traiter et visualiser des
données IRM 4D (volume + temps d'écho), dans le cadre de l'étude de tissus
végétaux (graines, algues, plantes) par relaxométrie T2.

## Fonctionnalités

- Chargement de fichiers NIfTI et lecture des temps d'écho depuis des fichiers
  ACQP (format Bruker ParaVision)
- Génération de masques tissulaires (seuillage Rician, Otsu, histogramme)
- Ajustement de modèles de décroissance T2 : mono-exponentiel, mono-exponentiel
  avec offset, bi-exponentiel, bi-exponentiel avec offset
- Sélection de modèle par critère d'information d'Akaike (AIC)
- Cartographies paramétriques voxel par voxel (T2, I0, fractions, erreurs)
- Visualisation interactive 2D (matplotlib) et 3D (Plotly)

## Démarrage rapide

```bash
pixi run python main.py
```

Voir la page [Installation](installation.md) pour la mise en place complète
de l'environnement, et [Utilisation de base](tutorials/usage.md) pour un
guide pas à pas.

## Structure du projet

```
sphene-mri/
├── main.py              # point d'entrée
├── functions/
│   ├── io.py            # chargement NIfTI / ACQP, dialogues
│   ├── model.py         # modèles de décroissance T2
│   ├── utils.py         # métriques, bruit, masques
│   ├── mapping.py        # calcul voxel-wise parallélisé
│   └── display.py        # interface interactive
├── pixi.toml             # dépendances (reproductible)
├── pixi.lock
└── docs/                  # cette documentation
```

## Licence et contexte

Projet open science développé dans le cadre d'un CDD INRAE, portant sur
l'analyse de données IRM de tissus végétaux (T2, et à terme diffusion).