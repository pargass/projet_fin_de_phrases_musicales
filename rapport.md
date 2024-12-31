# Rapport de projet - Prédiction de fins de phrases musicales

Loïcia Robart & Gaspar Henniaux



## Appréhension du projet

Nous avons commencé par bien étudier les données. Nous sommes allés lire les descriptions des différents attributs qui sont donnés dans le dataset pour avoir une vision globale des types de données disponibles et ce qu'elles représentent en terme de musique.  

On comprend que chaque ligne du jeu de données correspond à une mélodie, et chaque colone à une caracteristique de cette mélodie.
Dans une mélodie/chanson, on retrouve plusieurs motifs de notes et donc des départs et fin de mélodies, donc dans chacune de nos lignes (mélodies), on peut trouver plusieurs fin de phrases (une note marquée comme une fin de phrase, c'est à dire que c'est la dernière note d'une sous séquece mélodique).  
Nous avons ensuite réfléchis sur la notion de "fin de phrase" en terme de classification. Cela nous a permis de trouver des attributs tels que phrase_end, phrasepos, beatinphrase et beatinphrase_end qui pourraient nous aider dans cette classification.  

Une autre particularité des données qui nous sont données est que pour une ligne (une méloldie), on peut avoir pour un attribut donnée (une colonne), une valeur sous forme de liste. Ces listes correspondent en réalité à une représentation de la mélodie par notes. Pour prendre un exemple concret et en reprenant notre attribut cité plus haut 'phrase_end', on peut afficher, pour la première mélodie du jeu de donnée, sa valeur (df_phrase['phrase_end'][0]). On récupère alors une liste de booléans, chaque booléan indique pour chaque note de la mélodie (dans l'ordre des notes de la mélodie), si cette note est considérée comme une fin de phrase. Il faudra donc trouver une manière de gérer ces données sous forme de liste.  
Chaque liste a donc une taille équivalente au nombre de notes dans la mélodie dont il est question (ligne du dataset). 

Nous avons affiché pour une certaine ligne l'ensembles de ses valeurs par attribut pour avoir une vision d'ensemble.
Pour visualiser les différentes valeurs possibles d'un attribut, nous avons bouclé sur toutes les lignes du dataset et ajouté au fur et à mesure chaque nouvelle valeur qui aparaissait dans une liste (voir notebook). Cela nous permettait d'avoir une meilleur vision sur l'ensemble de valeurs qu'un attribut pouvait prendre.

## Préparation des données

### Tri manuel dans les features


De nombreux atributs sont directement liés les uns aux autres (voir détail dans le notebook), par exemple, duration, duration_frac et duration_fullname, qui se distinguent simplement par des différences dans la notation. On peut alors ne s'en tenir qu'à un seul. Ces corrélations entre attributs sont déductibles depuis la documentation fournie sur les features.  
D'autres attributs de par leur nature n'ont rien à voir avec les placements de fin de phrase et peuvent donc directement être évincées.  

Ce pré-tri nous a permis de sortir la liste d'attributs plus réduite suivante : 'midipitch', 'chromaticinterval', 'scaledegree', 'timesignature', 'beatstrength', 'metriccontour', 'imaweight', 'imacontour', 'duration', 'durationcontour', 'beatfraction', 'beat', 'restduration_frac', 'phrase_end'.

### Découpage en sous-séquences

L'objectif de notre modèle sera de prédire à partir d'une séquence de notes si elle est une séquence de fin de phrase musicale. Notre modèle doit donc s'entrainer sur des séquences de notes qui sont pour certaines des séquences de fin de phrase (target à 1) et d'autres non (target à 0) ([voir choix d'étiquetage plus bas](#préparation-des-données)).
Nous avons dans un premier temps réfléchis à un découpage basé sur la notion de mesures en musique (une unité de temps qui organise les rythmes dans une composition musicale). En effet, une phrase musicale s’étend généralement sur un certain nombre de mesures.
Les fins de phrases musicales coïncident souvent avec la fin d’une ou plusieurs mesures, renforçant le sentiment de structure, mais cela dépend du style et de la composition.  

Une autre approche était de définir un nombre de notes fixe, et de diviser chaque mélodie en plusieurs sous séquences au nombre de notes égal à ce nombre fixé.

La problématique d'aplatissement des données ([voir aplatissement des données plus bas](#préparation-des-données)) qui sont sous forme de liste nous implique de choisir cette seconde solution de découpage en sous-séquences par nombre de notes fixes. 
Au départ nous avions fixé notre nombre de notes à 8 mais nous sommes revenus en arrière pour tester l'impact de différentes valeurs pour ce choix de nombre de notes. 
Ce nombre de notes par sous-séquences peut ainsi être vu comme un paramètre à ajuster dans notre apprentissage pour optimiser les performances de modèles. Nous avonspu ainsi tester des sous séquences avec un total de 8, 12, 16, 20 et 24 notes.  

### Choix d'étiquetage 

L'étiquetage consister ici à décider pour chacune de nos sous-séquence d'apprentissage pour les modèles, si sa classe (colonne target -> la valeur que notre modèle devra prédire) est 1 (sous-sequence correspondant à une fin de phrase) ou 0 (la sous-séquence n'est pas une fin de phrase).  

La notion de séquence musicale qui correspond à une fin de phrase était assez floue pour nous. Nous avons ainsi décidé de tester plusieurs approches.  
Dans une première tentative, nous avons décidé d'étiqueter chaque sous-séquence contenant au moins une note marquée comme fin de phrase (phrase_end à true) à 1. Si la sous-séquence ne contient aucune fin de phrase parmis toute ses notes, la classe est notée à 0.  

Cette approche ne nous semblait pas optimale en terme de logique. En effet si la note marquant la fin de phrase est en début de sous-séquence, on peut considérer que toutes les notes suivantes dans cette sous-séquence correspondent en réalité à une progression de note pour le début de la phrase mélodique suivante. Cela brouille l'apprentissage car la sous-séquence aurait quand même été considérée comme une séquence de fin de phrase alors que la majorité des notes dans cette séquence auraient en réalité été en lien avec un début de phrase musicale.  

Nous avons donc décidé de revenir en arrière

### Transformation des données

### Applatissement des valeurs



### Equilibrages des classes



## Apprentissage des modeles

### Hyperparametres

### Scaler

## résultats

meilleuir resultat -> longueur -> 8 , fin de phrase derniere note de sequence

## lkimites difficultes rencontrees

aller-retours

---------------------------------------
