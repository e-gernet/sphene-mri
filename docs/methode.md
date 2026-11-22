# Méthode et validation

Cette page explique, en mots simples, **comment Sphene-MRI calcule T2**, pourquoi ce
calcul remplace un ajustement classique, et **comment vérifier soi-même** qu'il est
juste. Tous les chiffres viennent du rapport de validation (`pixi run validate`,
version complète, graine 2026) : chacun peut le relancer, rien n'est à croire sur parole.

!!! abstract "En trois phrases"
    - Une image IRM est une image de **modules** : le bruit ne peut pas faire passer une
      mesure sous zéro, donc quand le signal devient faible les mesures s'arrêtent sur un
      « plancher » de bruit (environ 1,25 σ). Un ajustement classique prend ce plancher
      pour du signal et **surestime T2** (jusqu'à +107 % dans nos tests).
    - Sphene-MRI mesure le bruit σ dans le fond de l'image, puis cherche le T2 qui rend les
      mesures **les plus probables** avec la loi exacte du bruit (loi de Rice). Résultat :
      pas de biais notable, et une précision égale à la limite que la physique autorise.
    - Le programme n'affiche un **bi** (deux composantes) que là où les données le
      justifient ; sinon il répond **mono**.

## 1. Ce que l'on cherche

Dans chaque voxel, le signal diminue quand le temps d'écho TE augmente. **T2** est la
vitesse de cette décroissance, en millisecondes : plus T2 est grand, plus le signal dure.
Deux descriptions sont possibles.

| Modèle | Ce qu'il dit | Paramètres |
|--------|--------------|------------|
| **mono** | une seule décroissance | I0 (signal à TE = 0), T2 |
| **bi** | deux décroissances mélangées | I0, T2c (courte), T2l (longue), f (part de la courte) |

T2 est cherché entre **la moitié du premier TE et cinq fois le dernier** (pour le grain :
de 1,5 à 467 ms). Ces bornes ne dépendent que de l'acquisition, jamais du voxel ajusté.
Si un T2 finit sur une borne, sa valeur n'est pas fiable : le programme le signale
(`T2_at_limit` dans l'export).

## 2. Le bruit d'une image IRM : un plancher

Une image IRM est une image de **modules** : le programme voit la longueur d'un vecteur,
jamais un nombre avec un signe. Le bruit ne peut donc pas descendre sous zéro : quand le
vrai signal devient petit (échos tardifs), la mesure reste autour d'un **plancher**
positif, qui vaut en moyenne **1,25 σ** (σ = niveau de bruit). Sur 1 000 000 de mesures
simulées sans signal, le rapport trouve 1,2531 σ, la théorie donne 1,2533 σ.

σ est mesuré **dans le fond de l'image**, où il n'y a que du bruit : écho par écho, puis
médiane sur les échos (grain de blé : σ = 513 ; acanthe : σ = 545). C'est la seule
quantité que le calcul demande en plus des images et des temps d'écho.

![Le plancher de bruit et ce qu'il fait à un ajustement classique](img/validation/01_bruit_plancher.png)

## 3. Pourquoi pas les moindres carrés

Un ajustement classique (moindres carrés) ne connaît pas ce plancher : il le lit comme du
signal, et trouve un T2 trop long. Mesuré sur 300 voxels simulés par case (erreur
typique = médiane des erreurs) :

| T2 vrai | SNR | moindres carrés | + constante libre | **Sphene** |
|---------|-----|-----------------|-------------------|------------|
| 12 ms | 10 | **+107 %** | −25 % | +1,1 % |
| 12 ms | 30 | +7,6 % | −10,7 % | +0,2 % |
| 120 ms | 10 | +4,4 % | −17 % | +2,0 % |
| 120 ms | 30 | +0,3 % | −4,2 % | +0,05 % |

La « constante libre » (l'ancienne option **offset**) ne répare rien : elle se
**confond avec T2** (une queue lente peut aussi bien être un T2 long qu'une constante),
d'où des erreurs dans l'autre sens, de −4 à −25 %.

## 4. La méthode retenue : la vraisemblance de Rice

Le principe, sans formule : parmi toutes les courbes possibles, on garde celle qui rend les
mesures **les plus probables**, **sachant exactement comment le bruit se comporte** (loi de
Rice). C'est la méthode du **maximum de vraisemblance**.

Comme σ est connu, le plancher fait partie du modèle : **aucun paramètre en plus** (plus
d'offset), **aucun seuil à régler**. Tout le calcul tient dans une seule fonction
(`functions/model.py`) : un seul ajustement, deux modèles (mono, bi), les mêmes pour un
voxel et pour une zone.

Résultats (rapport, section 2) :

- erreur typique **2,0 % au plus** (au SNR 10, le pire cas testé), moins de 1 % dès le
  SNR 20 ;
- biais moyen : 0,3 % au plus à partir du SNR 30 ; un léger biais positif apparaît aux
  SNR les plus bas (jusqu'à 3,2 % au SNR 10, 1,3 % au SNR 20 pour T2 = 12 ms), et c'est
  attendu : un T2 est l'inverse d'une vitesse de décroissance, donc connu à 14 % près il est
  en moyenne trop grand d'environ 0,14 × 0,14 ≈ 2 %, pour n'importe quelle méthode ;
- dispersion = **limite théorique** (borne de Cramér-Rao : la meilleure précision
  qu'aucune méthode ne peut battre avec ce bruit) : rapport de 1,14 au plus, et 1,06 au
  plus dès le SNR 20.

![Biais et dispersion de T2 mono](img/validation/02_mono_biais_dispersion.png)

!!! question "Pourquoi ne pas se servir du bruit mesuré pour régler le seuil de l'offset ?"
    C'était l'idée : mesurer σ, en déduire le plancher, et décider avec ce seuil si
    l'offset est utile. Mais un seuil est un **oui / non**, alors que l'effet du bruit varie
    **en continu** le long de la courbe : fort quand le signal est faible, presque nul quand
    il est fort. Une constante ne peut pas décrire cela, un seuil non plus. La
    vraisemblance de Rice, elle, **utilise le bruit mesuré à chaque point de chaque
    courbe**. C'est strictement plus fin, et il n'y a plus rien à régler. On avait d'ailleurs
    essayé de fixer l'offset à la valeur du bruit sur des données réelles : T2 se décalait de
    3 à 18 ms, car l'offset absorbait un mélange d'effets, pas seulement le bruit.

    Même chose pour le seuil du choix mono / bi (ΔAICc > 2) : il n'a pas besoin d'être adapté
    au bruit, car l'AICc est calculé avec la vraisemblance, donc déjà avec σ. Les fausses
    alarmes restent autour de 1 % (0 à 2,7 % selon la case, sans tendance) du SNR 10 au SNR 150.

## 5. Mono ou bi ?

Le bi a deux paramètres de plus : il colle **toujours** au moins aussi bien aux données,
même quand il n'y a que du bruit à expliquer. Le programme compare donc les deux modèles
avec le critère **AICc**, qui retire de la vraisemblance « le prix » des paramètres en
plus. Le bi n'est retenu que si son avantage dépasse **2** (ΔAICc > 2).

- **Fausses alarmes** (tissu mono classé bi à tort) : **0,9 %** des voxels (65 sur 7 200),
  2,7 % au pire dans une case (un T2 et un SNR).
- **Détection** (tissu bi classé bi) : 100 % à SNR 150 dès que T2l ≥ 3 × T2c. Au bruit
  du grain, cela dépend du couple : 100 % pour 12/60 ms, 62 % pour 12/36 ms, 16 % pour
  25/50 ms.
- **Il n'existe pas de rapport T2l / T2c universel** (comme la règle « 4 × T2c < T2l »
  parfois citée) : le rapport nécessaire passe de 6 (SNR 15) à 2 (SNR 80 et plus).

![Le seuil ΔAICc = 2 : fausses alarmes et détection](img/validation/04b_seuil_aicc.png)

Détecter un bi ne veut pas dire mesurer précisément T2c et T2l. Pour 12/60 ms au SNR 30,
le bi est détecté dans 100 % des voxels, mais **un voxel isolé ne donne T2c qu'à ±37 % et
T2l qu'à ±25 % au mieux**. C'est la physique, pas le programme (voir § 7).

## 6. Une zone entière (« Zone fit »)

Pour gagner en précision, « Zone fit » ajuste **tous les voxels de la zone ensemble**,
dans la même vraisemblance. Les courbes ne sont **pas moyennées avant** : moyenner des
modules garde le plancher de bruit dans la moyenne et fausse T2. Les voxels partagent la
**forme** de la décroissance (T2, ou T2c, T2l et f) mais chacun garde **sa propre
amplitude**, car deux voxels du même tissu ont rarement le même niveau de signal.

Cette amplitude propre est importante : avec une amplitude commune, un tissu dont le signal
varie de ±25 % d'un voxel à l'autre donnait 4,5 % (64 voxels) à 9 % (256 voxels) de faux bi ;
avec une amplitude par voxel on retombe à 1,0 % (4 sur 400 zones de 4 voxels ou plus).

Résultats (tissu mono, T2 = 30 ms, SNR 30, amplitude variable de ±25 %) :

| Voxels | Biais Sphene | Biais « moyenne + moindres carrés » | Dispersion / limite |
|--------|--------------|-------------------------------------|---------------------|
| 4 | −0,10 % | +2,05 % | 1,05 |
| 64 | −0,07 % | +2,03 % | 0,98 |
| 256 | +0,01 % | +2,06 % | 0,95 |

La précision s'améliore comme 1/√N et suit la limite théorique ; la moyenne des modules,
elle, garde un biais qui ne baisse jamais : plus on ajoute de voxels, plus on est précis…
sur une mauvaise valeur.

![Zone : ajustement joint contre moyenne](img/validation/05_zone.png)

!!! warning "Un bi sur une zone étendue n'est pas une preuve"
    Si la zone mélange des voxels de T2 différents (deux tissus voisins, un dégradé),
    l'ajustement d'ensemble préfère le bi alors que chaque voxel est mono. Le rapport le
    montre (section « Zone hétérogène ») : un mélange 50/50 de voxels 12 et 36 ms est classé bi
    sur la zone entière, alors que **99 % des voxels isolés** sont mono. Regarder la **Mono Map**
    et **Show signal** voxel par voxel permet de trancher.

## 7. Jusqu'où peut-on aller : la limite de précision

Avec un bruit donné, aucune méthode ne peut dépasser une précision limite (Cramér-Rao). Elle
décide de ce qui est **mesurable**, quel que soit le programme. Un couple T2c / T2l est dit
**séparable** quand cette limite donne moins de 25 % d'erreur sur les deux T2.

SNR nécessaire (signal à TE = 0 divisé par σ, temps d'écho du grain de blé, f = 0,5) pour
atteindre **±30 %** puis **±10 %** sur T2c et T2l :

| Couple (ms) | 1 voxel : ±30 % / ±10 % | 16 voxels | 64 voxels |
|-------------|-------------------------|-----------|-----------|
| 12 / 36 | 57 / 169 | 16 / 43 | 9 / 23 |
| 12 / 60 | 37 / 111 | 10 / 28 | 6 / 14 |
| 12 / 120 | 29 / 86 | 7 / 21 | 4 / 11 |
| 8 / 40 | 38 / 112 | 11 / 29 | 6 / 15 |
| 20 / 100 | 59 / 176 | 15 / 44 | 8 / 22 |
| 25 / 50 | 179 / 538 | 45 / 134 | 23 / 67 |

Lecture : le grain de blé a un SNR proche de 25 ; à un voxel on détecte le bi des couples bien
séparés mais on ne mesure pas T2c et T2l ; sur une zone de 64 voxels, ±10 % devient accessible
pour la plupart des couples. Le couple 25 / 50 (rapport 2) ne se résout presque jamais.
Pour votre propre acquisition : `python validation.py --acqp votre.acqp --t2c 12 --t2l 60 --f 0.5` (point décimal dans la commande).

## 8. Les premiers échos

Sur les données réelles, les tout premiers échos s'écartent parfois de la décroissance
(échos stimulés, refocalisation imparfaite). Sur les deux jeux de test du dépôt, le décalage
moyen des trois premiers échos est de −1,3 %, +4,7 %, −2,0 % du signal pour le grain de blé
et de −8,7 %, +10,5 %, −0,1 % pour l'acanthe (le reste de la courbe est bien décrit :
résidu moyen de 0,3 unité de bruit au plus).

Effet mesuré par simulation (T2 = 120 ms, le cas où il est le plus visible) :

| Décalage du 2<sup>e</sup> écho | Biais sur T2 (SNR 150) | Bi retenu à tort (SNR 150) |
|-----------------------------|------------------------|----------------------------|
| 3 % | −0,3 % | 0,5 % |
| 6 % | −0,6 % | 6,0 % |
| 12 % | −1,2 % | 14,5 % |

Ignorer les trois premiers échos (`SKIP_FIRST_ECHOES = 3` dans `main.py`) supprime l'effet, mais
**coûte de la précision** : la dispersion de T2 est multipliée par 1,3 (T2 = 120 ms), 1,4
(30 ms) et 2,3 (12 ms). L'option est donc à **0 par défaut** ; ne l'activer que si le décalage
mesuré sur vos données dépasse 5 % environ (le rapport et la console le disent).

![Premiers échos perturbés](img/validation/07_premiers_echos.png)

**Ignorer tous les échos décalés, ou aucun.** Sur l'acanthe, seuls les deux premiers échos sont
décalés (−8,7 % et +10,5 %) ; le troisième ne l'est plus (−0,1 %). Le rapport simule ce motif exact
(tissu mono de T2 = 30 ms, SNR 20) sur une zone de 64 voxels dont l'amplitude varie de ±25 % (comme
« Zone fit »), 100 zones par ligne, et sur un voxel isolé (200 tirages) :

| Cas | Biais sur T2, zone de 64 voxels | Bi retenu à tort (zone) | Biais sur T2, un voxel |
|-----|---------------------------------|-------------------------|------------------------|
| échos parfaits (référence), tous utilisés | −0,01 % | 3 % | −0,23 % |
| motif de l'acanthe, tous les échos utilisés | +0,71 % | 0 % | +0,50 % |
| motif de l'acanthe, **1 écho ignoré** | **−3,83 %** | **100 %** | −3,79 % |
| motif de l'acanthe, 2 échos ignorés | +0,04 % | 1 % | −0,10 % |
| motif de l'acanthe, 3 échos ignorés | −0,04 % | 0 % | +0,08 % |

Pourquoi : avec tous les échos, les deux défauts se compensent en partie (un écho trop bas, un écho
trop haut). Ignorer un seul écho retire le premier mais laisse le deuxième, le plus décalé
(+10,5 %), sans contrepartie : le modèle mono ne peut plus suivre la courbe, il sous-estime T2 et le
critère AICc préfère à tort le bi. Ignorer les deux échos décalés (ou les trois) ramène le biais à
0,04 % au plus.

Sur un voxel isolé, ces écarts sont du même ordre que l'erreur due au bruit (près de 5 %) ; ils
deviennent visibles sur une zone, où le bruit s'efface (dispersion de 0,7 %). L'option sert donc
surtout à « Zone fit ». Dans tous les cas : **tous les échos décalés, ou aucun**.

## 9. Hypothèses et limites

1. **σ doit venir d'un fond de bruit de Rice pur** : images brutes (non filtrées), une seule
   antenne de réception. Un filtre ou plusieurs antennes changent la loi du bruit. Le rapport
   vérifie le fond de vos images (quartiles de la loi de Rayleigh). Une erreur de 10 % sur σ
   change T2 de 0,3 % au plus et fait monter les fausses alarmes à 3,2 % au plus.
2. **Les couples T2c / T2l peu séparés ne se résolvent pas voxel par voxel** au bruit du grain
   (voir § 7). Le programme répond alors « mono » : c'est la réponse honnête, pas un défaut.
3. **Un bi sur une zone étendue peut venir d'un mélange de voxels** (§ 6).
4. **Le T2 mesuré est un T2 « apparent »** : il dépend de la séquence (échos stimulés,
   gradients), pas seulement du tissu.
5. **Les premiers échos** peuvent s'écarter du modèle (§ 8).
6. Les validations se font sur des **données simulées** : on y connaît la vérité. Sur des
   données réelles on ne vérifie que la cohérence (bruit de Rice, χ² réduit proche de 1,
   résidus) : χ² réduit médian de 1,12 pour le grain et de 1,20 pour l'acanthe (dont les
   premiers échos s'écartent de près de 10 %, voir § 8).

## 10. Relancer la validation

```bash
pixi run validate-quick     # environ 1 à 2 minutes : tendances et contrôles
pixi run validate           # environ 8 minutes : tirages complets, à citer dans un document

# avec vos données : le niveau de bruit et les temps d'écho sont les vôtres
python validation.py --nifti vos_donnees.nii --acqp vos_donnees.acqp

# qu'espérer d'un tissu donné avec votre acquisition ?
python validation.py --acqp vos_donnees.acqp --t2c 12 --t2l 60 --f 0.5 --snr 30 --couples
```

Le dossier `validation_report/` contient :

- `rapport.html` : le rapport (un seul fichier, figures incluses, à ouvrir dans un navigateur) ;
- `figures/*.png` : les mêmes figures, une par fichier, prêtes pour un diaporama ;
- `controles.csv` : chaque contrôle, sa valeur mesurée, son critère et son résultat.

Un contrôle est **RÉUSSI** quand l'écart observé reste dans une tolérance fixée à l'avance. Quand
le résultat dépend du hasard des tirages, la tolérance inclut une marge statistique (plusieurs
erreurs-types) ou un test binomial : un contrôle n'échoue donc que si l'écart est trop grand pour
être dû au hasard. **ATTENTION** signale une consigne de prudence sur VOS données (ce n'est pas un
échec du programme). Le code de sortie vaut 1 si un contrôle échoue, 0 sinon.

| Section du rapport | Question posée | Figure |
|--------------------|----------------|--------|
| 1. Le bruit | Pourquoi pas de simples moindres carrés ? | `01_bruit_plancher.png` |
| 2. T2 mono | Retrouve-t-on le bon T2 ? | `02_mono_biais_dispersion.png` |
| 3. T2 bi | Retrouve-t-on T2c, T2l et f ? | `03_bi_nuages.png` |
| 4. Choix mono / bi | Fausses alarmes, détection, pourquoi le seuil 2 | `04_choix_mono_bi.png`, `04b_seuil_aicc.png` |
| 5. Zone | Que gagne-t-on à ajuster ensemble ? | `05_zone.png` |
| 6. Erreur sur σ | Et si σ est un peu faux ? | `06_sensibilite_sigma.png` |
| 7. Premiers échos | Faut-il les ignorer ? | `07_premiers_echos.png` |
| 8. Cartes complètes | Tout comme dans le viewer, sur des disques de vérité connue | `08_cartes_*.png` |
| 9. Zone hétérogène | Que répond « Zone fit » à un mélange ? | `09_zone_heterogene*.png` |
| 10. Vos données | Le bruit est-il de Rice ? Le modèle explique-t-il les mesures ? | `10_donnees_reelles.png` |
| 11. Votre tissu attendu | Qu'espérer avec mon acquisition ? | tableaux |

## Glossaire

- **T2** : durée caractéristique de la décroissance du signal (ms).
- **mono / bi** : une ou deux composantes ; **T2c** courte, **T2l** longue, **f** part de la courte.
- **σ** : niveau de bruit ; **SNR** : signal à TE = 0 divisé par σ (le chiffre « SNR » affiché
  par le viewer, signal du 1<sup>er</sup> écho divisé par σ, est un peu plus bas).
- **plancher** : valeur moyenne d'une mesure sans signal, 1,25 σ.
- **biais** : erreur moyenne (devrait être nulle) ; **dispersion** : à quel point les résultats
  varient d'un voxel à l'autre.
- **limite de Cramér-Rao** : la meilleure précision qu'une méthode puisse atteindre avec ce bruit.
- **AICc / ΔAICc** : mesure de combien le bi explique mieux que le mono, une fois retiré le
  prix de ses paramètres en plus.
- **χ² réduit** : 1 si le modèle explique les mesures au niveau du bruit.
