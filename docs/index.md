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
- Ajustement de modèles de décroissance T2 par maximum de vraisemblance, avec
  la loi de bruit des images en module (loi de Rice) et le niveau de bruit
  mesuré dans le fond de l'image : mono-exponentiel et bi-exponentiel, sans
  paramètre « offset »
- Choix du modèle par le critère AICc : le bi n'est retenu que là où les
  données le justifient
- Cartographies paramétriques voxel par voxel (T2, I0, fractions, χ² réduit,
  erreurs)
- Rapport de validation par simulation : graphes, statistiques et explications
  (voir [Méthode et validation](methode.md))
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
├── validation.py           # rapport de validation par simulation (pixi run validate)
├── make_simulated.py        # jeux simulés .nii + .acqp avec vérité connue
├── functions/
│   ├── io.py                 # chargement NIfTI / ACQP, dialogues, export CSV
│   ├── model.py               # modèles T2 mono/bi, vraisemblance de Rice, choix AICc
│   ├── utils.py                # métriques, bruit, masques, filtrage (CPU/GPU)
│   ├── filter_compare.py        # comparaison objective des stratégies de filtrage
│   ├── mapping.py                # calcul voxel-wise parallélisé
│   ├── display.py                 # interface interactive
│   ├── simulation.py               # tirages simulés, limite de Cramér-Rao
│   └── report.py                    # rapport HTML (texte, figures, contrôles)
├── pixi.toml                        # dépendances et environnements (default/dev/gpu)
├── pixi.lock
└── docs/                              # cette documentation
```

## Licence et contexte

Projet open science développé dans le cadre d'un CDD INRAE, portant sur
l'analyse de données IRM de tissus végétaux (T2, et à terme diffusion).
