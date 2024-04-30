# Historical Analysis

Ce projet reprend les travaux de Margot D. disponible sur le dépôt GitHub : https://github.com/margot-demol/historical_analysis

Le travail était d'optimiser l'ouverture et la lecture de fichiers. 

Pour cela la librairie [Pangeo Forge](https://pangeo-forge.org/) à été utilisé afin de créer un datacube des jeux de données AVISO et ERASTAR, afin de créer un point d'entrée unique et éviter d'avoir à manipuler l'ensemble des fichiers NetCDF. 

Ensuite afin de paralléliser l'extraction et le traitement des données (principalement des interpolations) la librairie Python [Dask](https://www.dask.org/) a été utilisé.

## Notebooks

Différents Notebooks ont été créés afin de découvrir pas à pas les concepts utilisés, voici la liste:

- `1_generate_datacube`: Génération d'un datacube à partir du jeu de données AVISO (permet d'avoir un unique endpoint pour le jeu de données, via un catalogue).
- `2_open_datacube`: Ouvrir le datacube avec dask et comparer les performances en termes de chunking sur des opérations simples.
- `3_colocation`: Colocaliser les observations avec le datacube et extraire un petit cube autour de chaque point d'observation.
- `4_interpolation`: Interpolation des données d'observations avec le datacube.
- `5_qualification`: Qualification des résultats en comparaison avec les sorties de Margot. 

Ces notebooks existe pour le jeu de données AVISO (dossier `./aviso`) et ERASTAR (dossier `./erastar`)

**Remarque :** Le notebook `1_generate_datacube` est fourni à titre d'exemple pour comprendre comment ont été construit les datacubes, mais il n'est pas nécessaire de régénérer ces derniers, ils sont accessible aux chemins suivants :

- `AVISO : /home/datawork-data-terra/odatis/data/aviso/datacube-year`
- `ERASTAR : /home/datawork-data-terra/odatis/data/erastar/datacube-year`

De plus, des scripts PBS ont été écrits afin de lancer les calculs sur Datarmor et pouvoir ainsi dimensionner librement les ressources de calculs nécessaires. 

## Environnements

Afin d'exécuter ces notebooks, 2 environnements sont nécessaire:
- `data-env`: Utiliser dans la quasi totalité des notebooks. 
- `pangeo-forge-recipes-0.9-env`: Ce dernier est nécessaire uniquement pour la génération du datacube, autrement dit pour les notebooks / scripts PBS `1_generate_datacube`.

Pour créer l'environnement, il suffit d'entrer la commande:

`mamba env create -f data-env.yaml`