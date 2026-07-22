# Sphene MRI

**Pipeline d'analyse de relaxométrie T2 pour des données IRM de plantes**

Ce projet fournit un pipeline complet pour charger, traiter et visualiser des
données IRM 4D (volume + temps d'écho), dans le cadre de l'étude de tissus
végétaux (graines, algues, plantes) par relaxométrie T2.

## Fonctionnalités

- Chargement de fichiers NIfTI et lecture des temps d'écho depuis des fichiers
  ACQP (format Bruker ParaVision)
- Filtrage optionnel du signal avant masquage et fit, spatial (gaussien) ou
  temporel (Savitzky-Golay le long de l'axe TE), avec un mode comparaison
  pour évaluer objectivement le gain (R², RMSE)
- Génération de masques tissulaires (seuillage Rician, Otsu, histogramme)
- Ajustement de modèles de décroissance T2 : mono-exponentiel, mono-exponentiel
  avec offset, bi-exponentiel, bi-exponentiel avec offset
- Sélection de modèle par critère d'information d'Akaike (AIC)
- Cartographies paramétriques voxel par voxel (T2, I0, fractions, erreurs)
- Export des cartes calculées en tableau long (CSV, une ligne par voxel),
  filtré sur le masque tissulaire, avec auto-vérification à l'écriture
- Visualisation interactive 2D (matplotlib) et 3D (Plotly)
- Accélération GPU (cupy) optionnelle pour le filtrage spatial, avec repli
  automatique sur CPU si aucun GPU CUDA n'est disponible

## Démarrage rapide

```bash
pixi run sphene
```

Voir la page [Installation](installation.md) pour la mise en place complète
de l'environnement (y compris les environnements optionnels `dev` et `gpu`),
et [Utilisation de base](tutorials/usage.md) pour un guide pas à pas.

## Structure du projet

```
sphene-mri/
├── main.py                # point d'entrée (flags --device, --compare-filters)
├── functions/
│   ├── io.py               # chargement NIfTI / ACQP, dialogues, export CSV
│   ├── model.py             # modèles de décroissance T2
│   ├── utils.py              # métriques, bruit, masques, filtrage (CPU/GPU)
│   ├── filter_compare.py      # comparaison objective des stratégies de filtrage
│   ├── mapping.py              # calcul voxel-wise parallélisé
│   └── display.py               # interface interactive
├── pixi.toml                      # dépendances et environnements (default/dev/gpu)
├── pixi.lock
└── docs/                            # cette documentation
```

## Licence et contexte

Projet open science développé dans le cadre d'un CDD INRAE, portant sur
l'analyse de données IRM de tissus végétaux (T2, et à terme diffusion).
