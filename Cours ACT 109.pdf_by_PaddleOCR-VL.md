CNAM

Cours 109

# Introduction au calcul statistique et modélisation

Guillaume Gorge

28 novembre 2025

Table des matières

Table des matières  
1 Introduction - ACT 109  
1.1 Petite Histoire de la science des données et des besoins en formations 1  
1.2 Objectif d'un cours Python 3  
1.3 Un cours en pratique 4  
1.4 A faire 4  
2 introduction à la programmation statistique 5  
2.1 Guide d'installation de Visual Studio Code et Python 5  
2.2 Installation de Python 5  
2.3 Installation de Visual Studio Code 5  
2.4 Configurer Python dans VS Code 6  
2.5 Introduction à Python et pandas 8  
2.6 Introduction à pandas 9  
2.7 Quelles alternatives à pandas pour les actuaires? 9  
2.8 Comment Utiliser ChatGPT (ou un autre LLM) pour la Programmation? 11  
2.9 Utilisations Principales 11  
2.10 Exemple Complet d'Utilisation 13  
3 Fraude 15  
3.1 Introduction aux modèles supervisés 15  
3.2 qu'est ce qu'un modèle de fraude 16  
3.3 Présentation du Modèle de Régression Logistique 17  
3.4 Régression Logistique en Python 19  
3.5 comment corriger le sur-apprentissage? 23  
3.6 Application au cas Indien 26  
3.7 à vous! 26  
3.8 Un autre cas de modèle logistique; les décès 26  
4 modèles GLM 29  
4.1 Introduction aux GLM 29  
4.2 qu'est ce qu'un modèle GLM? 29  
4.3 Exemples de GLM 29  
4.4 Implémentation en Python 30  
4.5 Interprétation des résultats 31  
4.6 Applications 32  
4.7 Mesurer l'efficacité du modèle 34  
4.8 Améliorer la performance par des modèles non linéaire : GAM et XGBoost 35  
5 Approche non supervisée 37  
5.1 La malédiction de l'assurance : HDLSS 37  
5.2 Approche non supervisée 38  
5.3 Analyse en Composantes Principales (ACP) 38  
5.4 Introduction 39  
5.5 t-SNE : t-Distributed Stochastic Neighbor Embedding 42  
5.6 quelle efficacité? 45  
5.7 Problèmes 46  
6 Modèles financiers stochastiques 47  
6.1 Modèle de Vasicek 47  
6.2 Probabilités P et Q risque neutre 48  
6.3 aller plus loin : le modèle HJM 51  
7 modèles de réserve 53  
7.1 Quelques éléments des comptes en Assurance 53

7.2 Indicateur de profitabilité ..... 53  
7.3 Méthodes de projection des provisions ..... 54  
7.4 Comment calculer l'incertitude des réserve : la méthode de Mack ..... 57  
7.5 Le bootstrapping ..... 58  
7.6 Exercices ..... 60  
8 Modèles actuariels vie ..... 63  
8.2 Diagramme de Lexis ..... 63  
8.3 Modèle de Lee-Carter ..... 65  
8.4 Introduction à SQL : langage d'accès aux données ..... 66  
9 Modèles censurés ..... 69  
9.1 Modèle de Cox vs Modèle Linéaire Généralisé (GLM) ..... 71  
10 Modèle Catastrophe ..... 73  
10.1 qu'est ce qu'un modèle catastrophe ..... 73  
10.2 les modèles OpenSource ..... 74  
10.3 quelques mathématiques catastrophe ..... 75  
10.4 simulation deux centenaires ..... 76  
10.5 Implémentation en Climada ..... 77  
11 Solvabilité ..... 83  
12 Best Practices ..... 85  
13 Validation ..... 87  
13.1 Sujets possibles ..... 89  
ANNEXES ..... 91  
Solutions ..... 93  
Bibliographie ..... 95

1.1 Petite Histoire de la science des données et des besoins en formations

I keep saying the sexiest job in the next ten years will be statisticians. People think I'm joking, but who would've guessed that computer engineers would've been the sexiest job of the 1990s? $ ^{1} $ 

Mais aujourd'hui, on est plus de 10 ans après. Les premiers data scientists, c'étaient les physiciens car ils avaient une aisance avec les grands volumes.

1.1 Petite Histoire de la science des données et des besoins en formations ..... 1  
Python et les LLM ..... 1  
1.2 Objectif d'un cours Python ..... 3  
1.3 Un cours en pratique ..... 4  
1: Hal Varian, Google's Chief Economist Jan 2009  
mais qui est important ..... 4  
1.4 A faire ..... 4

Quels sont les besoins? Pour les personnes qui ne font pas de stats? pour recycler les personnes qui quittent SAS?

• Requêter directement dans les bases (SQL, noSQL).

• Elaborer un bon dataset, le passer à la production (programmation à 4 mains)?

• aujourd’hui, des stats dans toutes les écoles d’ingénieur; collège de France

• comment faire des pipelines (2019-2023), code propre, les suivis.

• En série temp, disparition des modèles ARIMA, par des modèles de fondation.

• comprendre les transformers et l'IA générative

• penser dès le départ, la production.

DÉFINITION 1.1.1 - CULTURE DES DONNÉES. Difficile de parler d'une science des données du fait de la complexité à transmettre. Plutôt une culture avec sa caractéristique IT,...

PROPRIÉTÉ 1 - QUEL PROFIL RECRUTER. Il fallait de la curiosité pour s'intéresser à la Data Science. Aujourd'hui, du fait de la caractéristique des LLM, la capacité à dialoguer est essentiel.

Document L'évolution de travailler et générer des documents est que cela risque d'être la tendance long terme.

Python et les LLM

Les LLM (Large Language Model) sont particulierement adaptés aux languages. Lors de la première version de ChatGPT, les concepteurs ont été étonnés de découvrir sa capacité à parler catalan alors qu'ils n'avaient pas intégrés le catalan dans l'apprentissage formel. De même la traduction d'une langue à l'autre est fascinante. Aussi, très vite, on a testé ces LLM par rapport aux languages informatiques. Ceci marche d'autant mieux que le languagest largement utilisé, comme Python.

Python vs R, ...

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl//44453948-142c-485c-8827-16372c660a37/markdown_5/imgs/img_in_chart_box_455_167_1035_810.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-01-11T18%3A42%3A33Z%2F-1%2F%2Fabfc3c49027fb7bf777a9bc5b73887eee293d44d2e69da7a1e821b3b88822c37" alt="Image" width="48%" /></div>


<div style="text-align: center;">Fig. 1.1 : Evolution Python The Economist</div>


The Economist

Exemple 1.1.1 (Python could become the world's most popular coding language But its rivals are unlikely to disappear) The Economist - Sep 19th 2018

“I CERTAINLY didn’t set out to create a language that was intended for mass consumption,”

says Guido van Rossum, a Dutch computer scientist who devised Python, a programming language, in 1989.

But nearly three decades on, his invention has overtaken almost all of its rivals and brought coding to the fingertips of people who were once baffled by it. In the past 12 months Americans have searched for Python on Google more often than for Kim Kar-dashian, a reality-TV star. The number of queries has trebled since 2010, while those for other major programming languages have been flat or declining. The language's two main advantages are its simplicity and flexibility. Its straightforward syntax and use of indented spaces make it easy to learn, read and share. Its avid practitioners, known as Pythonistas, have uploaded 145,000 custom-built software packages to an online repository. These cover everything from game development to astronomy, and can be installed and inserted into a Python program in a matter of seconds. This versatility means that the Central Intelligence Agency has used it for hacking. Google for crawling webpages, Pixar for producing movies and Spotify for recommending songs. Some of the most popular packages harness "machine learning", by crunching large quantities of data to pick out patterns that would otherwise be imperceptible.

With such a rapidly growing user base and wide array of capabilities, Python might seem destined to become the lingua franca of coding, rendering all other competitors obsolete. That is unlikely, according to Grady Booch, IBM's chief software scientist, who compares programming languages to empires. Though at times a rising power might be poised for world domination, its rivals generally survive in the technical and cultural niches in which they emerged. Python will not replace C and C++, which are "lower-level options" that give the user more control over what is going on in a computer's processor. Nor will it kill off Java, which is popular for building

complicated applications, or JavaScript, which powers most web pages. Moreover, Pythonistas who take their language's supremacy for granted should beware. Fortran, Lisp and Ada were all highly popular in the 1980s and 1990s, according to the TIOBE index, which tracks coding practices among professional developers. Their use has plummeted, as more efficient options have become available. No empire, regardless of its might, can last forever.

Pour les langages comme SAS, "très spécialisés", il y a moins d'information et les performances ne sont pas toujours là.

le retour d'expérience des LLM pour le code

Exemple 1.1.2 (AI code is simpler, but not better) The Economist - 14 sept 2025

Programmers increasingly use artificial-intelligence tools to write code for software. But is it any good? A recent study by researchers at the University of Naples Federico II suggests not. The researchers compared more than 500,000 software functions written in Python and Java, two popular programming languages. Humans wrote some of the samples; popular ai models such as ChatGPT, DeepSeek-Coder and Qwen created the rest. The scientists found that ai tends to write shorter, simpler programs. But its output was riddled with errors and showed more serious security weaknesses, particularly in Java. The identified vulnerabilities were serious: they could have allowed hackers to insert nefarious commands at an operating-system level, or left credentials such as usernames and passwords unprotected. In contrast, the human-written code was more complex, making it less vulnerable to hackers but harder to troubleshoot. The researchers suggested that ai coders can be useful—but only with a rigorous check.

1.2 Objectif d'un cours Python

Objectif du cours Traditionnellement, un cours de Python nécessite plusieurs dizaines d'heures. Ce cours ne vise pas à remplacer ces cours mais plutôt à favoriser une approche "apprendre en faisant". Cela est possible grâce aux LLM qui ont largement favorisé l'accès à la programmation.

Apprendre Python de façon traditionnelle Par exemple :

• Cours Python du Michigan : complet environ 80 heures de cours.

Objectif du cours

• mettre les "mains dans la boue" de la programmtion

• Qu'est ce qu'un modèle

• résidu d'un modèle

• Comment les LLM peuvent nous aider ou pas

Principe du cours

avant chaque cours, faire les exercices Pendant le cours, corriger les exercices, voir la théorie

Plan

• Introduction à Python

• projet fraude

• projet pricing

• projet calibration de table de mortalité sur données titanic

• projet de modelisation des taux d'intêret (Vasicek...)

• projet de réduction de dimension (essentiel en assurance) : zonier, véhiculier, profil santé,...

• projet de modélisation catastrophe

• utilisation de l'API d'un LLM (par exemple pour tester la tonalité d'une conversation, traduire un texte...)

Pour chaque exemple, je vous donnerai une base et un objectif, à faire pour le cours suivant.

Premier cours Installer Python

1.3 Un cours en pratique

Ce que nous ne couvrons pas mais qui est important

• souveraineté et gouvernance de la donnée (quelles sont les données que l'on peut exposer...)

• Aspect écologique : ressource de l'eau...

• les aspects les plus avancés (n8n, agents,...)

1.4 A faire

Comment faire un cours basé sur des données (par définition open source): Utilisation de bases de données publiques

La plupart des exemples seront sur CASDataSets.

Mais on peut aussi utiliser d'autres données :

· Kaggle

• Diabete

· triangle

• Titanic

SYNTHÈSE 1.4.1 Un cours de Python avec LLM, c'est un peu comme du beau- jolais nouveau, cela vieillit mal car les sous-jacents évoluent très vite.

# introduction à la programmation statistique

2.1 Guide d'installation de Visual Studio Code et Python

SYNTHÈSE 2.1.1 · Installer Visual Studio Code
• ouvrir VS code
• Dans les extensions, installer python, pip et jupyter and French Language Pack for Visual Studio Code.

Ce guide explique comment installer Visual Studio Code (VS Code) et Python sur votre machine. Ces outils sont essentiels pour le développement en Python et offrent une expérience de programmation efficace grâce à leurs nombreuses fonctionnalités.

2.2 Installation de Python

Étape 1 : Télécharger Python

• Rendez-vous sur le site officiel de Python : https://www.python.org/downloads/.

• Téléchargez la version adaptée à votre système d'exploitation (Windows, macOS ou Linux).

Étape 2 : Installer Python

• Lancez l'exécutable téléchargé.

• Cochez l'option Add Python to PATH avant de cliquer sur Install Now.

• Suivez les étapes de l'installation jusqu'à la fin.

Étape 3 : Vérifier l'installation

Ouvrez un terminal (ou l'invite de commandes sous Windows) et exécutez :

python --version

Cela doit afficher la version de Python installée, par exemple: Python 3.11.6.

2.3 Installation de Visual Studio Code

Étape 1 : Télécharger Visual Studio Code

• Rendez-vous sur le site officiel de VS Code: https://code.visualstudio.com/.

• Téléchargez l'installateur correspondant à votre système d'exploitation.

Étape 2 : Installer Visual Studio Code

• Lancez l'installateur téléchargé.

• Cochez les options suivantes lors de l'installation :

- Add "Open with Code" action to Windows Explorer.

- Add to PATH (available after restart).

• Cliquez sur Install.

Étape 3 : Lancer VS Code

Ouvrez VS Code une fois l'installation terminée.

2.4 Configurer Python dans VS Code

Installer l'extension Python

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl//44453948-142c-485c-8827-16372c660a37/markdown_9/imgs/img_in_image_box_440_672_1051_849.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-01-11T18%3A42%3A35Z%2F-1%2F%2F015f863cb28356ba5c8c14b6c6251ea370a3f9929beac8fb089dcf7e4b2c8965" alt="Image" width="51%" /></div>


<div style="text-align: center;">FIG. 2.1 : A gauche, choisissez le gestionnaire d'extension puis Python.</div>


Dans Visual Studio Code :

1. Cliquez sur l'icône des extensions (dans la barre latérale gauche).

2.

3. Recherchez Python.

4. Installez l'extension officielle développée par Microsoft.

autres extensions utiles

Installez de la même façon :

· pip

• jupyter

• French Language Pack for Visual Studio Code

Étape 3 : Tester la configuration

Créez un fichier test.py avec le contenu suivant :

print("Installation réussie !")

Exécutez le fichier directement dans VS Code en cliquant sur le bouton Run (ou via le terminal intégré).

Vous avez maintenant installé Python et Visual Studio Code, et conﬁguré votre environnement pour développer en Python.

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl//44453948-142c-485c-8827-16372c660a37/markdown_10/imgs/img_in_image_box_289_155_598_349.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-01-11T18%3A42%3A35Z%2F-1%2F%2Fcaf07587c369bc8d96962fca0225ac0d379c414b6a9d4433fc8290ce7b874376" alt="Image" width="25%" /></div>


Autres outils utiles pour la programmation

• Github permet de sauvegarder et partager le code

• Docker est un outil de conteneurisation. Il permet d'emballer une application avec tout son environnement (système d'exploitation, bibliothèques, dépendances, etc.) dans un conteneur. Ces conteneurs sont portables et garantissent que votre application fonctionnera de manière cohérente, quelle que soit la machine ou l'environnement où elle est déployée.

• Poetry est un outil de gestion de projets Python. Il simplifie la gestion des dépendances, l'installation des packages, et la création d'environnement isolés pour Python. Poetry facilite aussi la gestion des versions et la publication de packages Python sur PyPI.

pendant qu"on est dans de l'installation de VS Code, Github Copilot Chat

Dans un premier temps, nous n"allons pas installer Copilot Chat mais n"hésitez pas à le faire si vous le souhaitez!

1. Se connecter à GitHub dans VSCode

Ouvrir VSCode, cliquer sur l'icône de compte (en bas à gauche) → Se connecter avec GitHub.

2. Installer l'extension GitHub Copilot

Aller dans l'onglet Extensions (Ctrl+Shift+X), chercher GitHub Copilot, cliquer sur Installer.

3. Installer l'extension GitHub Copilot Chat

Chercher GitHub Copilot Chat dans les extensions. Cette extension permet d'ouvrir une fenêtre de chat dans VSCode (disponible si vous avez une licence Copilot).

4. Utiliser Copilot Chat

Ouvrir le panneau de commande (Ctrl+Shift+P), taper Copilot Chat: Open Chat pour commencer à discuter.

Étapes pour accéder à ChatGPT

1. Aller sur https://chat.openai.com

2. Créer un compte ou se connecter.

3. Vous pouvez utiliser la version gratuite ou souscrire à ChatGPT Plus pour accéder à GPT-4 et ses fonctionnalités avancées (dont l'interpréteur de code, l'analyse de fichiers, etc.).

Recommandation pour l'apprentissage Utilisez Copilot Chat dans VSCode pendant que vous codez, et ChatGPT dans votre navigateur pour poser des questions plus générales, obtenir des explications ou tester des blocs de code indépendants.

<div style="text-align: center;">Fig. 2.2: Data wrangler</div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl//44453948-142c-485c-8827-16372c660a37/markdown_11/imgs/img_in_image_box_594_153_898_336.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-01-11T18%3A42%3A36Z%2F-1%2F%2F6b0e5a94c13f376287830eda51d22ff7b04f8ac6a7c0f394df87d776beadd67b" alt="Image" width="25%" /></div>


Bonnes pratiques

• Posez des questions claires : “Peux-tu m’expliquer cette erreur?” ou “Écris-moi une fonction Python pour calculer une prime nette avec une loi exponentielle.”

• Corrigez votre code en demandant: "Améliore ce code" ou "Ajoute des commentaires explicatifs".

- Utilisez Copilot Chat pour le débogage en direct et ChatGPT pour des explications approfondies.

Data wrangler un outil très pratique pour l'exploration de données.

2.5 Introduction à Python et pandas

Concepts de base

Variables et Types de Données Python est un langage interprété, simple et lisible. Voici les principaux types de données :

· int : entier (x = 5)

· float : nombre à virgule flottante (pi = 3.14)

· str: chaîne de caractères (nom = "Guillaume")

• bool : booléen (est vrai = True)

· list: liste(ma_liste = [1, 2, 3])

dict:dictionnaire(mon_dico = {"clé": "valeur"})

Contrôle de Flux

Conditions :

if x > 0:
    print("Positif")
elif x == 0:
    print("Zéro")
else:
    print("Négatif")

Boucles for : , pour parcourir des objets :

for i in range(5):  # De 0 à 4
    print(i)

Les boucles for sont particulièrement utiles en actuariat. Par exemple :

mes_annees = [2024, 2025, 2026]
for annee in mes_annees:
    print(annee)

Si nous avons plusieurs objets, l'instruction enumerate est elle aussi très utile.

mes_annees = [2024, 2025, 2026]
mes_coefficients_developpement = [1, 1.1, 1.2]
for index, annee in enumerate(mes_annees):
    print(annee, mes_coefficients_developpement[index])

indx est créé par enumerate et donne l'index (0,1,2) de la variables années en cours.

Fonctions Une fonction regroupe un bloc de code réutilisable :

def addition(a, b):
    return a + b
print(addition(2, 3)) # 5

2.6 Introduction à pandas

Pandas est le package de gestion de base de données de python.

Il faut écrire sous Windows!pip install pandas. Voir aussi Poetry.

Installation

Pour installer la bibliothèque pandas, exécutez :

pip install pandas

Les Structures de pandas

Series Une Series est une colonne de données :

import pandas as pd

# Création d'une Series
s = pd.Series([10, 20, 30, 40])
print(s)

DataFrame Un DataFrame est une table avec des lignes et des colonnes :

data = {"Nom": ["Alice", "Bob"], "Age": [25, 30]}
df = pd.DataFrame(data)
print(df)

2.7 Quelles alternatives à pandas pour les actuaires?

pandas est la bibliothèque de référence pour la manipulation de données en Python. Cependant, d'autres solutions offrent de meilleures performances ou une meilleure adaptabilité selon les cas d'usage, notamment pour le traitement de gros volumes de données ou l'exécution distribuée. Voici un panorama des principales alternatives, avec leurs points forts, leurs limites et leurs cas d'usage spécifiques dans le contexte actuariel.

Tableau comparatif des alternatives à pandas

Tab. 2.1: Comparatif des alternatives à pan-das pour les cas d'usage actuariels. L'approche via SQL peut être intéressante


<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Bibliothèque</td><td style='text-align: center;'>Objectif</td><td style='text-align: center;'>Avantages</td><td style='text-align: center;'>Limites</td></tr><tr><td style='text-align: center;'>Polars</td><td style='text-align: center;'>Manipulation rapide de données (Rust)</td><td style='text-align: center;'>Très rapide, support du multi-threading, évaluation paresseuse (lazy)</td><td style='text-align: center;'>Moins mature que pandas, compatibilité limitée avec certaines lib Python</td></tr><tr><td style='text-align: center;'>Dask</td><td style='text-align: center;'>pandas distribuée</td><td style='text-align: center;'>Compatible avec l&#x27;API pandas, gestion des jeux de données plus gros que la RAM</td><td style='text-align: center;'>Moins performant pour de petits jeux de données, nécessite une infrastructure adaptée</td></tr><tr><td style='text-align: center;'>Vaex</td><td style='text-align: center;'>Analyse interactive de très gros fichiers</td><td style='text-align: center;'>Lecture mémoire-mappée, très faible empreinte mémoire</td><td style='text-align: center;'>API limitée, pas de support pour toutes les opérations pandas</td></tr><tr><td style='text-align: center;'>Modin</td><td style='text-align: center;'>Accélérer pandas sans changer de code</td><td style='text-align: center;'>API identique, parallèlisation automatique</td><td style='text-align: center;'>Support partiel, instable dans certains cas</td></tr><tr><td style='text-align: center;'>PyArrow</td><td style='text-align: center;'>Format colonne hautement performant</td><td style='text-align: center;'>Parfait pour lire/écrire du Parquet, interopérabilité avec d&#x27;autres langages</td><td style='text-align: center;'>Pas une alternative directe pour manipuler des Dataframes</td></tr><tr><td style='text-align: center;'>datatable</td><td style='text-align: center;'>Inspiré de data-table de R</td><td style='text-align: center;'>Très rapide, syntaxe familière pour les utilisateurs de R</td><td style='text-align: center;'>API différente, moins intégrée à l&#x27;écosystème Python</td></tr><tr><td style='text-align: center;'>cuDF (NVIDIA)</td><td style='text-align: center;'>Manipulation de Data-frames sur GPU</td><td style='text-align: center;'>Très rapide avec GPU, API proche de pandas</td><td style='text-align: center;'>Nécessite un GPU NVIDIA, complexiété technique</td></tr><tr><td style='text-align: center;'>DuckDB</td><td style='text-align: center;'>SQL in-memory performant</td><td style='text-align: center;'>Requêtes SQL rapides sur fichiers Parquet, très bon en lecture/agrégation</td><td style='text-align: center;'>Syntaxe SQL à maîtriser, nécessite adaptation pour les utilisateurs 100% Python</td></tr></table>

Opérations de Base

Privilégier parquet au lieu de csv pour la sauvegarde de bases volumineuses. Cela se fait via l'instruction pd.read_parquet

Lire un fichier

df = pd.read_csv("chemin_fichier.csv")

Aperçu des données

print(df.head())  # Les 5 premières lignes
print(df.info())  # Informations sur le DataFrame
print(df.describe())  # Statistiques descriptives

Accéder aux données

Par colonne :

print(df["Nom"]) # Accès à la colonne "Nom"

Par ligne :

print(df.iloc[0]) # Première ligne
print(df.loc[0]) # Idem si l'index est numérique

Filtrer les données

# Sélectionner les lignes où Âge > 25
print(df[df["Âge"] > 25])

Ajouter ou Modifier des colonnes

df["Salaire"] = [3000, 4000]
print(df)

Groupement et Résumés

grouped = df.groupby("Nom").sum()
print(grouped)

Exporter les données

df.to_csv("nouveau_fichier.csv", index=False)

Quelques instructions utiles pour voir la puissance de Python Une fonction utile de Pandas est df.columns qui donne une liste de toutes les variables.

list_variables = df.columns

# List des noms de variables contenant "md_"
variables_main_driver = [var for var in list_variables if "md_" in var]

# Afficher les résultats
print(variables_main_driver)

Pour apprendre réellement Python

AI Python for Beginners Cours coursera (payant mais raisonnable) pour apprendre python dans une logique d'IA, par Andrew Ng.

2.8 Comment Utiliser ChatGPT (ou un autre LLM) pour la Programmation?

ChatGPT peut être un outil puissant pour aider dans la programmation, que ce soit pour apprendre de nouveaux concepts, résoudre des problèmes, ou automatiser des tâches.

2.9 Utilisations Principales

Apprendre les Concepts de Base

Posez des questions sur des langages ou des bibliothèques spécifiques, comme Python, pandas, ou SQL. Par exemple :

• Exemple : « Peux-tu expliquer les différences entre une liste et un dictionnaire en Python? »

• Exemple : « Montre-moi comment utiliser une boucle for pour parcourir une liste. »

Résoudre des Problèmes

Décrivez précisément votre problème ou comportement attendu. Vous pouvez également inclure votre code :

import pandas as pd

data = {"A": [1, 2, 3]}

df = pd.DataFrame(data)

print(df["B'])  # Cela renvoie une erreur KeyError

Exemple de question : « Mon code lève une exception KeyError dans pandas lorsque j"essaie d'accéder à une colonne. Que puis-je vérifier? »

Générer du Code

Demandez à ChatGPT de générer rapidement des morceaux de code. Par exemple :

« Écris un script Python pour lire un fichier CSV, calculer la moyenne d'une colonne, et sauvegarder les résultats dans un nouveau fichier. »

Optimiser et Refactoriser

Faites vérifier ou améliorer votre code pour le rendre plus lisible ou performant :

my_list = [3, 1, 4, 1, 5, 9]
sorted_list = sorted(my_list)
print(sorted_list)

Exemple de question : « Peux-tu optimiser ce code pour trier une liste ? »

Automatiser des Tâches

Demandez de l'aide pour automatiser des processus courants, par exemple :

« Écris un script Python pour télécharger un fichier à partir d'une URL et l'enregistrer dans un dossier local. »

Déboguer

Soumettez votre script avec une explication des erreurs rencontrées. Par exemple :

for i in range(10):
    if i % 2 == 0:
        print(i)

Exemple de question : « Ce script doit afficher les nombres pairs de 1 à 10, mais il ne fonctionne pas. »

Comprendre les Bibliothèques et Frameworks

Posez des questions sur des outils comme pandas, NumPy, Flask ou Django. Par exemple :

« Comment utiliser pandas pour grouper des données par une colonne et calculer la somme? »

Tester et Valider

Demandez des conseils pour écrire des tests unitaires avec des frameworks comme pytest :

def addition(a, b):
    return a + b

Exemple de question : « Comment écrire un test unitaire avec pytest pour cette fonction ? »

Documentation et Tutoriels

ChatGPT peut rédiger des tutoriels complets ou des guides pédagogiques. Par exemple :

« Rédige un guide pour débutants sur l'utilisation de pandas pour manipuler des données. »

Par exemple, cette partie du cours a été produite en grande partie par ChatGPT.

Bonnes Pratiques avec ChatGPT

• Soyez clair et précis : Décrivez bien ce que vous voulez, comme le comportement attendu, les erreurs rencontrées ou les restrictions.

• Demandez des explications : Si une réponse est trop concise, demandez plus de détails.

• Testez toujours les solutions : Même si ChatGPT donne des réponses utiles, vérifiez et testez le code dans votre contexte.

- Itérez : Si la réponse initiale n"est pas parfaite, ajustez votre question ou demandez des corrections.

2.10 Exemple Complet d'Utilisation

Question : « Aide-moi à écrire un script Python pour lire un fichier Excel, filtrer les lignes où une colonne "Âge" est supérieure à 30, et sauvegarder le résultat dans un nouveau fichier CSV. »

Réponse :

import pandas as pd

# Charger le fichier Excel
df = pd.read_excel("fichier.xlsx")

# Filter les lignes où "Âge" est supérieur à 30
df_filtre = df[df["Âge"] > 30]

# Sauvegarder le résultat dans un fichier CSV
df_filtre.to_csv("fichier_filtre.csv", index=False)

print("Fichier filtré sauvegardé avec succès.")

Exercice 1 Voici un code :

/* Étape 1 : Création d'un jeu de données temporaire */
data ventes;
input Region $ Vendeur $ Montant;
datalines;
Nord Dupont 1500
Nord Martin 2000
Sud Durand 1800
Sud Lefevre 2200
Est Bernard 1600
Est Petit 1900
;
run;
/* Étape 2 : Calcul du total des ventes par région */
proc summary data=ventes nway;
class Region;
var Montant;
output out=total_ventes sum=Total_Ventes;
run;
/* Étape 3 : Affichage des résultats */
proc print data=total_ventes;
title "Total des ventes par région";
run;

je veux la traduction en python.

Exercice 2 Voici un dictionnaire python sous format Akur8, proche du format PMML :

{"coefficients": {"ratio_diff_price_ornikarassurances": {"-91": -1.127970440173, "-95": -1.127970440173, "350": 1.424014998816, "592": 1.424014998816, "230": 1.424014998816, "110": 1.424014998816, "112": 1.424014998816, "354": 1.424014998816, "115": 1.424014998816, "237": 1.424014998816, "117": 1.424014998816, "119": 1.424014998816, "10": 1.324431327202, "11": 1.324431327202, "12": 1.324431327202, "13": 1.334165928943, "14": 1.334165928943, "15": -0.246083733298, "-195": -0.179037512714, "3059": 0.816095613805, "2882": 0.816095613805, "-516": -0.246083733298, "-757": -0.340631724663, "-878": -0.340631724663, "5238": 0.816095613805, "-633":

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl//44453948-142c-485c-8827-16372c660a37/markdown_17/imgs/img_in_image_box_435_158_1051_304.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-01-11T18%3A42%3A39Z%2F-1%2F%2F552edf2a3642e4072d9e89520c7b6c8bbf069990490ef54f3360b844eade27cf" alt="Image" width="51%" /></div>


<div style="text-align: center;">je veux un code m"indiquant toutes les variables retenues.</div>


3

objectif du cours

• qu'est ce qu'un modèle supervisé;

• qu'est ce qu'on modèle de Fraude

Cette partie s'inspire du mémoire d'actuariat CNAM 2024 de Cécile ROUGIER Améliorer la détection de la fraude à l'assurance automobile en Inde grâce à des modèles d'apprentissage statistique.

• modèle logistique

• métriques de performance (AUC, lift...)

• résidu

• sur-apprentissage

3.1 Introduction aux modèles supervisés

DÉFINITION 3.1.1 - MODÈLE SUPERVISÉ. Un modèle supervisé est un type de modèle d'apprentissage automatique qui apprend à partir d'un ensemble de données étiquetées. L'objectif principal est de trouver une relation ou une fonction qui associe les entrées (appelées X ou variables explicatives) aux sorties (appelées Y ou variable à expliquer), afin de pouvoir prédire Y pour de nouvelles données non étiquetées.

Cela signifie que, pour chaque exemple d'apprentissage, une entrée  $ (X) $  (caractéristiques du dossier : lieu, date, ....) est associée à une sortie  $ (Y) $  connue (c'est une fraude, ce n'est pas une fraude).

Principes des modèles supervisés

Les principes des modèles supervisés peuvent être résumés comme suit :

• Données étiquetées : Les données d'apprentissage contiennent des paires d'entrées et de sorties  $ (X, Y) $ .

• Entraînement : Le modèle est entraîné pour minimiser l'erreur entre les prédictions ( $ \hat{Y} $ ) et les sorties réelles (Y).

• Généralisation : Une fois entraîné, le modèle est capable de faire des prédictions sur de nouvelles données.

Types de tâches supervisées

Les modèles supervisés sont utilisés dans deux grandes catégories de problèmes :

- Classification : Le but est de prédire une catégorie ou une classe parmi un ensemble fini de classes possibles.

– Exemple : Prédire si une personne va décéder ou non, fraude vs non fraude.

- Algorithms courants : régression logistique, arbres de décision, forêts aléatoires.

• Régression : Le but est de prédire une valeur continue.

- Exemple : Nombre de sinistres, coût du sinistre

- Algorithms courants : GLM, réseaux de neurones.

Composants d'un modèle supervisé

Un modèle supervisé comprend plusieurs composants :

1. Ensemble de données d'entraînement : Contient les exemples étiquetés utilisés pour entraîner le modèle.

2. Fonction d'apprentissage : Méthode qui ajuste les paramètres du modèle pour minimiser l'erreur.

3. Fonction de perte : Mesure de l'écart entre les prédictions du modèle  $ (\hat{Y}) $  et les sorties réelles  $ (Y) $ .

4. Évaluation : Mesurer les performances du modèle sur des données de test.

3.2 qu'est ce qu'un modèle de fraude

Importance de la fraude en assurance

on considère généralement que 5% des personnes peuvent être tentés par la fraude dans des cas où le risque est faible. Montant qui peut paraître faible mais qui rapporté à la fréquence sinistre, qui est du même ordre, peut rendre l'assurance prohibitive.

Les types de fraudes

Le sujet ici porte sur la détection de la fraude externe, c'est-à-dire commise par des clients, bénéficiaires ou autres personnes assurées. La fraude interne, c'est-à-dire perpétrée par des collaborateurs de la société d'assurance (agents, ...) ne sera pas traitée dans cette étude, comme celle des intermédiaires externes (courtiers, ...) ou les prestataires/fournisseurs (les garages).

Généralement on distingue la fraude en fonction du moment où elle se produit dans la vie du contrat :


<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>A la souscription</td><td style='text-align: center;'>- Fausse déclaration :o Risque présenté minoré, avec un usage non exact (mauvais conducteur principal)o Demande de couverture pour un risque déjà existant au jour de la souscrip-tion</td></tr><tr><td style='text-align: center;'>Durant la vie du contrat</td><td style='text-align: center;'>- Fraude à la déclaration du retrait du permis de conduire-Non déclaration de sinistre : &quot;l&#x27;assuré est obligé de donner avis à l&#x27;assureur [...] de tout sinistre de nature à entraîner la garantie de l&#x27;assureur&quot;, d&#x27;après l&#x27;article L113-2 du Code des assurances. C&#x27;est-à- dire qu&#x27;un sinistre même très réduit doit être déclaré à l&#x27;assureur228</td></tr><tr><td style='text-align: center;'>Après la surveillance du sinistre</td><td style='text-align: center;'>- Déclaration d&#x27;un faux sinistre : Déclaration d&#x27;un sinistre qui n&#x27;a pas eu lieu, déclaration d&#x27;un sinistre perpétrée par le client contre ses propres intérêts assurés, avec ou sans l&#x27;aide de tierces personnes, ...o Faux vol,o Faux accidento Faux incendioo ...- Inexactitude dans la déclaration du sinistreoDéclaration d&#x27;un sinistre supérieur à celui réel :Demande de remboursement d&#x27;un montant supérieur aux dommages réelsoDéclaration de dommages qui n&#x27;ont pas de lien de causalité avec le sinistre déclaré (dommage préexistant par exemple)o Déclaration d&#x27;un autre conducteur au volant lors du sinistroo Responsabilité attribuée à tort à un tierso ...</td></tr></table>

Fraude sinistre

Dans le cas de la fraude sinistre, il est essentiel de comprendre comment fonctionne le processus de détection de la fraude.

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl//44453948-142c-485c-8827-16372c660a37/markdown_20/imgs/img_in_image_box_139_152_746_459.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-01-11T18%3A42%3A41Z%2F-1%2F%2Fc929a5994e1d16274d5f89eb9bbd23a7ac62401d55b4ebf49260369183cc0f60" alt="Image" width="50%" /></div>


Fig. 3.1 : Exemple de processus de lutte contre la fraude sinistre.

Variables explicatives de la Fraude

En France, les indicateurs classiques de fraude sont :

• des paiements en cash

• une demande d'indemnisation moins d'un mois après le début de couverture du contrat

• le véhicule était garé dans la rue alors qu'une protection comme un garage privé et/ou sécurisé était disponible.

• imprécisions ou contradiction des déclarations de l'assuré sur les circonstances de l'accident

• le nombre de sinistres déclarés par l'assuré

• la souscription d'assurances multiples

• la répétition de sinistres similaires

• un sinistre parfaitement documenté et préparé


<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>N.</td><td style='text-align: center;'>Elément déclenchant une revue dans la lutte anti-fraude</td></tr><tr><td style='text-align: center;'>1</td><td style='text-align: center;'>Signatures différentes</td></tr><tr><td style='text-align: center;'>2</td><td style='text-align: center;'>Délai dans la déclaration du sinistre</td></tr><tr><td style='text-align: center;'>3</td><td style='text-align: center;'>Proimité entre la date de survenance du sinistre et le début de la police d’assurance</td></tr><tr><td style='text-align: center;'>4</td><td style='text-align: center;'>Rupture dans le contrat</td></tr><tr><td style='text-align: center;'>5</td><td style='text-align: center;'>Conducteur différent du propriétaire du véhicule et sans relation entre les deux</td></tr><tr><td style='text-align: center;'>6</td><td style='text-align: center;'>Documents originaux non disponibles pour vérification</td></tr><tr><td style='text-align: center;'>7</td><td style='text-align: center;'>Lors de l’inspection, les dommages semblent anciens et/ou multiples</td></tr><tr><td style='text-align: center;'>8</td><td style='text-align: center;'>Causes du dommage non cohérents avec les dommagesconstatés</td></tr><tr><td style='text-align: center;'>9</td><td style='text-align: center;'>Perte totale du véhicule</td></tr><tr><td style='text-align: center;'>10</td><td style='text-align: center;'>Atelier de réparation sous surveillance</td></tr><tr><td style='text-align: center;'>11</td><td style='text-align: center;'>Dommages mécaniques qui semblent frauduleux</td></tr><tr><td style='text-align: center;'>12</td><td style='text-align: center;'>Date et heure de l’accident</td></tr><tr><td style='text-align: center;'>13</td><td style='text-align: center;'>Parc de véhicules professionnels</td></tr><tr><td style='text-align: center;'>14</td><td style='text-align: center;'>Véhicule en stationnement et des dommagessupérieurs à 20% de la valeur du véhicule</td></tr><tr><td style='text-align: center;'>15</td><td style='text-align: center;'>Nombre de kilomètre du véhicule élevé étant donnel’âge du véhicule</td></tr><tr><td style='text-align: center;'>16</td><td style='text-align: center;'>Autocollant sur le véhicule accidenté pour des callscenters, compagnies de taxis, auto-école</td></tr><tr><td style='text-align: center;'>17</td><td style='text-align: center;'>Absence de preuve du trajet comme des reçus, carnets de route</td></tr><tr><td style='text-align: center;'>18</td><td style='text-align: center;'>Présence de sang, de fluides corporels ou de cheveux sur le pare- brise</td></tr><tr><td style='text-align: center;'>19</td><td style='text-align: center;'>Incendie du véhicule</td></tr><tr><td style='text-align: center;'>20</td><td style='text-align: center;'>Absence de certificat médicaux</td></tr><tr><td style='text-align: center;'>21</td><td style='text-align: center;'>Nom du conducteur non confirmé dans le rapport depolice</td></tr><tr><td style='text-align: center;'>22</td><td style='text-align: center;'>Lieu de l’accident par rapport au lieu de résidence/travail de l’assure</td></tr><tr><td style='text-align: center;'>23</td><td style='text-align: center;'>Permis de conduire expiré</td></tr><tr><td style='text-align: center;'>24</td><td style='text-align: center;'>Cas de vol</td></tr><tr><td style='text-align: center;'>25</td><td style='text-align: center;'>Proimité de la date de vol et de l’émission du permis de conduire</td></tr></table>

3.3 Présentation du Modèle de Régression Logistique

La régression logistique est un modèle statistique utilisé pour modéliser une variable binaire, c'est-à-dire une variable qui prend deux valeurs, typique-ment notées 0 et 1 (par exemple 1 : Fraude). Elle permet d'estimer la proba-

bilité qu'un événement survienne en fonction d'un ensemble de variables explicatives.

1. Fonction de Modèle

En passant par le log, le modèle peut être écrit comme :

DÉFINITION 3.3.1 - RÉGRESSION LOGISTIQUE. Le modèle logistique s'exprime (sous sa forme log):

 $$ \ln\left(\frac{P(Y=1\mid\mathsf{X})}{1-P(Y=1\mid\mathsf{X})}\right)=\beta_{0}+\mathsf{X}\cdot\beta $$ 

où:

•  $ P(Y = 1 \mid \mathbf{X}) $  est la probabilité que la variable cible Y prenne la valeur 1 pour un vecteur de variables explicatives X.

• X est le vecteur des variables explicatives :  $ (X_{1}, X_{2}, \ldots, X_{p}) $ .

•  $ \beta = (\beta_{1}, \beta_{2}, \ldots, \beta_{p}) $  est le vecteur des coefficients associés aux variables explicatives.

⋅ β0 est l'ordonnée à l'origine ou le biais.

• ln ( ( P ( 1-P ) est appelé le logit de la probabilité P.

• La relation est linéaire dans les paramètres  $ \beta_{0} $  et  $ \beta $ .

3. Estimation des Paramètres

Les paramètres  $ \beta_{0} $  et  $ \beta $  sont estimés par maximisation de la vraisemblance, une approche essentielle en calcul statistique. Il ne s'agit pas ici de développer cette méthode.

Le logarithme de la vraisemblance (log-vraisemblance) est utilisé pour simplifier les calculs :

 $$ \ell(\beta)=\sum_{i=1}^{n}\left[Y_{i}\mathsf{l n}(P(Y_{i}\mid\mathsf{X}_{i}))+(1-Y_{i})\mathsf{l n}(1-P(Y_{i}\mid\mathsf{X}_{i}))\right] $$ 

La maximisation de  $ \ell(\beta) $  se fait généralement par des algorithms numériques comme la descente de gradient ou des méthodes de quasi-Newton.

4. Interprétation des Coefficients

• Chaque coefficient  $ \beta_{j} $  représente l'effet marginal d'une unité d'augmentation de  $ X_{j} $  sur le logit de  $ P(Y=1) $ , toutes choses égales par ailleurs.

5. Applications

La régression logistique est largement utilisée dans des domaines variés :

• Diagnostic médical : prédire la probabilité de présence d'une maladie.

• Assurance : modéliser la probabilité de sinistres ou de fraude.

• Marketing : prédire la probabilité qu’un client achète un produit.

• Sciences sociales : analyser des phénomènes binaires, tels que voter ou ne pas voter.

6. Avantages et Limites

Avantages :

• Modèle simple et interprétable.

• Facile à implémenter et à généraliser.

• Bien adapté aux variables explicatives continues et catégoriques.

Limites : Ce sont les limites classiques des modèles linaires :

• Hypothèse de linéarité.

• Sensible à la multicolinéarité entre les variables explicatives.

3.4 Régression Logistique en Python

Voici les étapes pour programmer une régression logistique avec Python, en utilisant la variable binaire fraud_reported comme cible.

1. Importer les bibliothèques nécessaires

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

sklearn se pronounce ess-key-learn car c'est l'abréviation de scikit-learn.

Sklearn est la principale librairie statistique de Python. Numpy permet de gérer des matrices. Il est compatible avec Pandas.

2. Charger et explorer les données

# Charger les données
df = pd.read_csv("votre_fichier.csv")

# Visualiser les premières lignes
print(df.head())

# Vérifier les types de données
print(df.info())

3. Préparer les données

Comment gérer les variables catégorielles - get_dummies Nous avons souvent à travailler avec des variables catégorielles, par exemple un code postal: le fait que le code postal 75011 soit entre le 75012 et le 75010 ne veut pas dire que le risque dans le 11ème arrondissement de Paris a quelque chose à voir avec ceux du 10ème et 12ème. Pour le gérer, on crée une variable par modalité, ce qui se fait aisément avec get_dummies de la librairie sklearn. On supprime la première modalité car sinon, on a une infinité de combinaisons de résultats.

# Définir la variable cible et les variables explicatives
X = df.drop(columns=['fraud_reported', 'policy_number']) # Toutes les colonnes sauf
la cible et les variables qui ne doivent avoir aucun lien avec la cible (policy_number...)
y = df['fraud_reported'] # Variable cible

# Si les variables explicatives contiennent des variables catégoriques
X = pd.get_dummies(X, drop_first=True)

# Vérifier les classes de la variable cible
print(y.value_counts())

4. Diviser les données en ensembles d'entraînement et de test

C'est un élément essentiel des statistiques récentes : on ne va apprendre qu'avec une partie des données (70% par exemple), le reste servant à tester si le modèle est prédictif. Dans la plupart des cas, on peut faire un tirage aléatoire (en utilisant train_test_split).

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

5. Ajuster le modèle de régression logistique

le modèle s'appelle  $ log\_reg $ . On le définit d'abord (instanciation) puis on estime les paramètres à partir des données d'entrainement.

# Instancier le modèle
log_reg = LogisticRegression(max_iter=1000, random_state=42)

# Ajuster le modèle
log_reg.fit(X_train, y_train)

# Prédire sur les données de test
y_pred = log_reg.predict(X_test)

# Afficher les coefficients du modèle
coefficients = pd.DataFrame({
    'Variable': X.columns,
    'Coefficient': log_reg.coef_[0]
})
print(coefficients)

Le résultat présente (une partie des) 2152 modalités :

Variable Coefficient
months_as_customer 1.453070e-03
age -2.218752e-02
policy_deductable 9.820905e-05
policy_annual_premium -2.093427e-04
umbrella_limit 7.121870e-08
... ... ...
2147 auto_model_Tahoe 2.135301e-03
2148 auto_model_Ultima -1.353144e-03
2149 auto_model_Wrangler -2.129602e-03
2150 auto_model_X5 8.372722e-04
2151 auto_model_X6 1.524687e-03

Comment cela s'interprète

6. Évaluer le modèle

Afin d'étudier si le modèle est efficace, il nous faut regarder son efficacité. Pour cela, on utilise une matrice de confusion et une sortie spécifique de sklearn, le rapport de classification.

# Matrice de confusion

conf_matrix = confusion_matrix(y_test, y_pred)
print("Matrice de confusion:")
print(conf_matrix)

# Rapport de classification
print("Rapport de classification:")
print(classification_report(y_test, y_pred))

Précision et Rappel : Définitions et Interprétation Dans un rapport de classification, les métiriques précision (precision) et rappel (recall) permettent d'évaluer les performances d'un modèle de classification.


<table border=1 style='margin: auto; width: max-content;'><tr><td colspan="5">Rapport de classification:</td></tr><tr><td style='text-align: center;'></td><td style='text-align: center;'>precision</td><td style='text-align: center;'>recall</td><td style='text-align: center;'>f1-score</td><td style='text-align: center;'>support</td></tr><tr><td style='text-align: center;'>N</td><td style='text-align: center;'>0.79</td><td style='text-align: center;'>0.96</td><td style='text-align: center;'>0.86</td><td style='text-align: center;'>226</td></tr><tr><td style='text-align: center;'>Y</td><td style='text-align: center;'>0.62</td><td style='text-align: center;'>0.20</td><td style='text-align: center;'>0.31</td><td style='text-align: center;'>74</td></tr><tr><td style='text-align: center;'>accuracy</td><td style='text-align: center;'></td><td style='text-align: center;'></td><td style='text-align: center;'>0.77</td><td style='text-align: center;'>300</td></tr><tr><td style='text-align: center;'>macro avg</td><td style='text-align: center;'>0.71</td><td style='text-align: center;'>0.58</td><td style='text-align: center;'>0.59</td><td style='text-align: center;'>300</td></tr><tr><td style='text-align: center;'>weighted avg</td><td style='text-align: center;'>0.75</td><td style='text-align: center;'>0.77</td><td style='text-align: center;'>0.73</td><td style='text-align: center;'>300</td></tr></table>

FIG. 3.2 : Rapport de classification. Precision et Recall sont calculés directement par la matrice de confusion.

Précision (Precision) La précision mesure la proportion de prédictions positives correctes parmi toutes les prédictions positives effectuées par le modèle. Elle est définie par :

 $$ \mathrm{Précision}=\frac{\mathrm{Vrais Positifs}\left(\mathrm{VP}\right)}{\mathrm{Vrais Positifs}\left(\mathrm{VP}\right)+\mathrm{Faux Positifs}\left(\mathrm{FP}\right)} $$ 

Interprétation :

• Une précision élevée signifie que le modèle fait peu de faux positifs, c'est-à-dire qu'il ne classe pas incorrectement des exemples négatifs comme positifs.

• Utile lorsque les faux positifs ont un coût élevé (par exemple, dans le diagnostic médical ou la détection de fraude).

Rappel (Recall) Le rappel mesure la proportion de véritables exemples positifs qui ont été correctement identifiés par le modèle. Il est défini par :

 $$ Rappel=\frac{Vrais Positifs\left(VP\right)}{Vrais Positifs\left(VP\right)+Faux Négatifs\left(FN\right)} $$ 

Interprétation :

• Un rappel élevé signifie que le modèle détecte la majorité des exemples positifs, avec peu de faux négatifs.

• Utile lorsque les faux négatifs ont un coût élevé (par exemple, dans la détection des maladies graves).

Relation entre Précision et Rappel Il existe souvent un compromis entre précision et rappel. Un modèle peut avoir :

• Une précision élevée mais un rappel faible, ce qui signifie qu'il est très sélectif pour prédire des exemples positifs.

• Un rappel élevé mais une précision faible, ce qui signifie qu'il détecte la majorité des exemples positifs mais avec de nombreux faux positifs.

F-score ( $ F_{1} $ -score) Pour équilibrer précision et rappel, on utilise le  $ F_{1} $ -score, qui est la moyenne harmonique de ces deux métriques :

 $$ F_{1}=2\cdot\frac{Précision\cdot Rappel}{Précision+Rappel} $$ 

Le  $ F_{1} $ -score est particulièrement utile lorsque vous avez besoin d'un équilibre entre précision et rappel.

Courbe ROC et AUC La courbe ROC - Receiver Operating Characteristic consiste à classer les vrais positifs en ordonnées et les faux positifs en ordonnées selon le seuil de décision. Donc, si la courbe est proche de la diagonale, c'est que l'on a autant de vrais positifs que de faux positifs, bref, notre modèle est inutile.

DÉFINITION 3.4.1 - AUC. AUC signifie Area Under the Curve (aire sous la courbe). C'est une métrique utilisée principalement dans l'évaluation des modèles de classification. L'AUC mesure l'aire sous une courbe spécifique, souvent la courbe ROC (Receiver Operating Characteristic). L'objectif est d'évaluer la capacité d'un modèle à distinguer entre classes positives et négatives.

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Probabilities prédites
y_pred_prob = log_reg.predict_proba(X_test[:, 1]
# Calcul de la courbe ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Tracer la courbe ROC
plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_pred_prob):.2f}')
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbe ROC')
plt.legend()
plt.show()

Ce qui est intéressant ici, c'est que toute les métriques sont calculés sur l'échantillon de test qui n'a pas servi à l'apprentissage.

DÉFINITION 3.4.2 - GINI. Le critère de Gini est aussi utilisé :

Gini = 2 × AUC - 1

On cherche donc à avoir un critère de Gini le plus haut possible, signe de la plus forte segmentation. Le coefficient de Gini sera une valeur entre -1 (classement inverse) et 1 (classement parfait), où 0 indique un classement aléatoire.

import numpy as np

def gini_coefficient(y_true, y_pred):
    """
    Calcul du coefficient de Gini.
    :param y_true: Liste ou tableau des valeurs réelles (0 ou 1).
    :param y_pred: Liste ou tableau des probabilités prédites.
    :return: Coefficient de Gini (entre -1 et 1).
    """
    assert len(y_true) == len(y_pred), "Les tailles de y_true et y_pred doivent être identiques."

    # Créer une matrice contenant les valeurs réelles et prédites
    data = np.array(list(zip(y_true, y_pred)))

    # Trier par probabilités prédites décroissantes, en cas d'égalité par les vraies valeurs
    data = data[np.lexsort((data[:, 0], -data[:, 1])])

    # Calcul des positions cumulées
    cum_pos = np.cumsum(data[:, 0])  # Cumul des valeurs vraies (positives)
    total_positives = cum_pos[-1]  # Total des positives

    # Calcul de la Gini brute
    gini_sum = np.sum(cum_pos) / total_positives

    # Nombre total d'observations
    n = len(y_true)

    # Calcul du Gini normalisé
    gini = (2 * gini_sum - total_positives - n) / n

    return gini

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl//44453948-142c-485c-8827-16372c660a37/markdown_26/imgs/img_in_chart_box_145_163_741_635.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-01-11T18%3A42%3A45Z%2F-1%2F%2F48edd39da6b97d1271871c4cf6a74b211f1ea4c61e9424c3039264600c587da7" alt="Image" width="50%" /></div>


<div style="text-align: center;">Fig. 3.3 : Résultat du Gini - modèle log_reg</div>


Le modèle, malgré ces 2151 paramètres est décevant, avec une courbe de Gini proche de la diagonale. Nous sommes dans un cas classique de surapprentissage : les données d'entraînement sont parfaitement répliqués mais la capacité prédictive du modèle est faible.

3.5 comment corriger le sur-apprentissage?

Introduction En régression logistique, il est souvent nécessaire de réduire le nombre de variables explicatives pour améliorer la performance et l'interprétabilité du modèle. Deux approches courantes pour sélectionner les variables sont :

• Lasso : Réduction basée sur la régularisation L1, qui force certains co-efficients à devenir nuls.

- Stepwise : Méthode itérative qui ajoute ou retire des variables en fonction de critères statistiques.

Ce document présente comment implémenter ces deux méthodes en Python.

1. Réduction via Lasso Le Lasso (Least Absolute Shrinkage and Selection Operator) utilise la régularisation L1 pour réduire le nombre de variables en contraignant certains coefficients à zéro. En régression logistique ou régression linéaire, la fonction de coût sans régularisation est définie comme :

 $$ J(\beta)=\frac{1}{n}\sum_{i=1}^{n}\mathcal{L}(y_{i},\hat{y}_{i}) $$ 

où:

• n : nombre d'observations.

•  $ \mathcal{L}(y_{i},\hat{y}_{i}) $  : la fonction de perte, par exemple l'erreur quadratique (mean squared error) ou la log-vraisemblance pour la régression logistique,

• β : vecteur des coefficients du modèle.

Avec la régularisation L1, une pénalité est ajoutée pour contrôler la taille des coefficients. La fonction de coût devient :

 $$ J(\beta)=\frac{1}{n}\sum_{i=1}^{n}\mathcal{L}(y_{i},\hat{y}_{i})+\lambda\|\beta\|_{1} $$ 

où:

•  $ \|\beta\|_{1}=\sum_{j=1}^{p}|\beta_{j}| $ : la norme L1 des coefficients (p étant le nombre de variables explicatives).

•  $ \lambda \geq 0 $  : hyperparamètre de régularisation qui contrôle l'intensité de la pénalisation.

La pénalisation  $ \lambda\|\beta\|_{1} $  pousse certains coefficients  $ \beta_{j} $  vers zéro, ce qui entraîne une sélection automatique des variables.

Exemple avec Scikit-learn En poursuivant sur le même exemple que précédemment, on peut modifier notre approche pour introduire la régression lasso :

Listing 3.1 : Régression logistique avec Lasso

# Instancier le modèle
lasso_model = LogisticRegression(max_iter=10000, random_state=42, penalty='l1', solver='liblinear', C=0.1)
lasso_model.fit(X_train, y_train)
# Ajuster le modèle
lasso_model.fit(X_train, y_train)

# Prédire sur les données de test
y_pred_lasso = lasso_model.predict(X_test)

# Afficher les coefficients du modèle
coefficients_lasso = pd.DataFrame({
    'Variable': X.columns,
    'Coefficient': lasso_model.coef_[0]
})

coefficients_lasso_sans_zero = coefficients_lasso[coefficients_lasso.Coefficient != 0]

selected_variables = coefficients_lasso_sans_zero
display("Variables sélectionnées :")
display(selected_variables)
display(selected_variables.shape)

# Prédictions et évaluation
y_pred = lasso_model.predict(X_test)
print(classification_report(y_test, y_pred))

Le modèle lasso_model se distingue du modèle précédent log_reg :

· penalty='l1': Applique la régularisation L1.

- solver='liblinear': Adapté pour les petites bases avec régularisation L1.

• C : Contrôle la force de la régularisation. Une valeur plus petite applique une régularisation plus forte (moins de variables). Essayez de modifier ce coefficient pour voir l'effet.

Les coefficients peuvent s'afficher : Le résultat sur la base de test est bien meilleur comme le montre la courbe de Gini :

2. Réduction via Stepwise La méthode Stepwise sélectionne ou élimine des variables itérativement en fonction de critères statistiques tels que l'AIC, le BIC, ou la p-valeur.

Listing 3.2 : Régression logistique avec sélection Stepwise. il est nécessaire de programmer la fonction

# Initialisation des variables
remaining = list(X_train.columns)  # Variables disponibles pour la sélection
selected = []  # Variables sélectionnées
best_score = 0  # Meilleur score initial

# Modèle logistique avec régularisation minimale
model = LogisticRegression(penalty='none', solver='lbfgs', max_iter=1000)  # Pas de régularisation


<table border=1 style='margin: auto; width: max-content;'><tr><td colspan="3">Variable</td></tr><tr><td style='text-align: center;'>0</td><td style='text-align: center;'>months_as_customer</td><td style='text-align: center;'>1.708583e-03</td></tr><tr><td style='text-align: center;'>1</td><td style='text-align: center;'>age</td><td style='text-align: center;'>-2.096739e-02</td></tr><tr><td style='text-align: center;'>2</td><td style='text-align: center;'>policy_deductable</td><td style='text-align: center;'>8.648700e-05</td></tr><tr><td style='text-align: center;'>3</td><td style='text-align: center;'>policy_annual_premium</td><td style='text-align: center;'>7573542e-05</td></tr><tr><td style='text-align: center;'>4</td><td style='text-align: center;'>umbrella_limit</td><td style='text-align: center;'>7.744030e-08</td></tr><tr><td style='text-align: center;'>5</td><td style='text-align: center;'>insured_zip</td><td style='text-align: center;'>5.068363e-07</td></tr><tr><td style='text-align: center;'>6</td><td style='text-align: center;'>capital_gains</td><td style='text-align: center;'>2.173899e-06</td></tr><tr><td style='text-align: center;'>7</td><td style='text-align: center;'>capital_loss</td><td style='text-align: center;'>-1.491745e-07</td></tr><tr><td style='text-align: center;'>8</td><td style='text-align: center;'>incident_hour_of_the_day</td><td style='text-align: center;'>-3.657532e-03</td></tr><tr><td style='text-align: center;'>11</td><td style='text-align: center;'>witnesses</td><td style='text-align: center;'>4.008135e-02</td></tr><tr><td style='text-align: center;'>12</td><td style='text-align: center;'>total_claim_amount</td><td style='text-align: center;'>8.704893e-06</td></tr><tr><td style='text-align: center;'>13</td><td style='text-align: center;'>injury_claim</td><td style='text-align: center;'>-5.066854e-06</td></tr><tr><td style='text-align: center;'>14</td><td style='text-align: center;'>property_claim</td><td style='text-align: center;'>-2.547334e-06</td></tr><tr><td style='text-align: center;'>15</td><td style='text-align: center;'>vehicle_claim</td><td style='text-align: center;'>1.670705e-05</td></tr><tr><td style='text-align: center;'>16</td><td style='text-align: center;'>auto_year</td><td style='text-align: center;'>-8.335546e-04</td></tr><tr><td style='text-align: center;'>995</td><td style='text-align: center;'>insured_hobbies_chess</td><td style='text-align: center;'>1.926617e+00</td></tr><tr><td style='text-align: center;'>996</td><td style='text-align: center;'>insured_hobbies_cross-fit</td><td style='text-align: center;'>7.493423e-01</td></tr><tr><td style='text-align: center;'>1080</td><td style='text-align: center;'>incident_severity_Minor Damage</td><td style='text-align: center;'>-1.551291e+00</td></tr><tr><td style='text-align: center;'>1081</td><td style='text-align: center;'>incident_severity_Total Loss</td><td style='text-align: center;'>-1.298107e+00</td></tr><tr><td style='text-align: center;'>2097</td><td style='text-align: center;'>property_damage_NO</td><td style='text-align: center;'>-8.825028e-02</td></tr></table>

<div style="text-align: center;">Tab. 3.1 : coefficient après Lasso // C=0.1</div>


<div style="text-align: center;">Courbe ROC</div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl//44453948-142c-485c-8827-16372c660a37/markdown_28/imgs/img_in_chart_box_145_544_740_993.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-01-11T18%3A42%3A45Z%2F-1%2F%2F79ff30036fd110ff14934758e555633192cbcbca7fe0d37dc1388a57fff19b8b" alt="Image" width="49%" /></div>


<div style="text-align: center;">FIG. 3.4 : Coefficient de Gini Post Lasso</div>


# Forward Stepwise Selection
while remaining:
    scores_with_candidates = []
    for candidate in remaining:
        # Test avec chaque variable restante
        candidate_vars = selected + [candidate]
        model.fit(X_train[candidate_vars], y_train)
        y_pred = model.predict(X_test[candidate_vars])
        score = accuracy_score(y_test, y_pred)
        scores_with_candidates.append((score, candidate))

# Sélection de la meilleure variable
scores_with_candidates.sort(reverse=True)  # Trier les scores
best_new_score, best_candidate = scores_with_candidates[0]

# Si le score s'améliore, on ajoute la variable
if best_new_score > best_score:
    remaining.remove(best_candidate)
    selected.append(best_candidate)
    best_score = best_new_score
    print(f"Ajoutée : {best_candidate}, Nouveau score : {best_new_score:.4f}")
else:
    break

print(f"Variables sélectionnées : {selected}")

# Réentraînement final du modèle avec les variables sélectionnées
final_model = LogisticRegression(penalty='none', solver='lbfgs', max_iter=1000)
final_model.fit(X_train[selected], y_train)
y_pred_final = final_model.predict(X_test[selected])

# Évaluation finale

final_score = accuracy_score(y_test, y_pred_final)
print(f"Score final avec les variables sélectionnées : {final_score:.4f}")

3.6 Application au cas Indien

L'application d'un modèle supervisé sur un faible échantillon donne de très bon résultats(tab. 3.5). Grâce à cette étude, de nombreuses fraudes ont pu être identifiés, laissant espérer de bons résultats en production.

Fig. 3.5 : Revue des 120 sinistres du POC par des experts métiers. Le modèle a appris sur 7 sinistres, tous prédits.


<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="2" colspan="2"></td><td colspan="2">Prédiction</td></tr><tr><td style='text-align: center;'>Fraude</td><td style='text-align: center;'>Non Fraude</td></tr><tr><td rowspan="2">Réalité</td><td style='text-align: center;'>Fraude</td><td style='text-align: center;'>35</td><td style='text-align: center;'>1</td></tr><tr><td style='text-align: center;'>Non Fraude</td><td style='text-align: center;'>7</td><td style='text-align: center;'>77</td></tr></table>

Malheureusement, la mise en production est décevante, avec un taux de faux positifs (des cas que le modèle dit être vrais et qui s'avèrent faux) trop important et qui démobilise les équipes.


<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="2" colspan="2"></td><td colspan="2">Prédiction</td></tr><tr><td style='text-align: center;'>Fraude</td><td style='text-align: center;'>Non Fraude</td></tr><tr><td rowspan="2">Réalité</td><td style='text-align: center;'>Fraude</td><td style='text-align: center;'>148</td><td style='text-align: center;'>287</td></tr><tr><td style='text-align: center;'>Non Fraude</td><td style='text-align: center;'>15146</td><td style='text-align: center;'>83671</td></tr></table>

Fig. 3.6 : Revue des résultats après le 1er cal-

librage en production

Le modèle de Gini ne fait pas la distinction entre faux positif et vrai négatif (des vraies fraudes qui ne sont pas détectés). Pourtant, opérationnellement, les deux sont très différents.

3.7 à vous!

Données Indiennes dataset from Kaggle

Code Python

Données US Data américaines : Il faut expliquer fraud_reported.

3.8 Un autre cas de modèle logistique; les décès

Le modèle logistique sert aussi pour les modèles de décès (0-1): Les décès sont adaptés aux modèles de déces car on peut les modéliser par 0/1 (vivant / mort), avec la probabilité d de décès qui est modélisé.

Un exemple, modéliser la survie du Titanic : l'assurance des passagers de l'Olympic, sister-ship du Titanic Petit exercice de programmation à partir d'une table de mortalité un peu spéciale mais qui sert à tous les apprentis data-scientists, la base kaggle titanic $ ^{1} $ .

1: La base titanic est aussi accessible dans le package python seaborn

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl//44453948-142c-485c-8827-16372c660a37/markdown_30/imgs/img_in_image_box_139_278_745_647.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-01-11T18%3A42%3A46Z%2F-1%2F%2F4c81511dffe9da60e6033977de39950a082f2b265bf429a494a467950301a9e8" alt="Image" width="50%" /></div>


FIG. 3.7 : Police d'assurance protégeant le Titanic. Suisse de Re. Police émise deux semaines avant la disparition du Titanic

Exercice 3 Après le Titanic qui a coulé, vous voulez proposer aux personnes qui traversent l'Atlantique de s'assurer contre la mort. Un client va prendre l'Olympic, le sister-ship du Titanic. Vous considerez qu'il y a 10% de chance que ce bateau coule. Vous voulez utiliser les statistiques du Titanic pour calculer la prime à demander à deux clients :

Créez une table de mortalité "traversée de l'Atlantique" en tenant compte des statistiques du titanic.

Cette méthode suppose que l'âge et le sexe sont les informations les plus importantes pour discriminer la mortalité. Mais on peut supposer que la classe et le fait de voyager seul ou non sont des informations importantes. Essayez d'estimer le risque de mortalité en utilisant ces informations. Pour cela, vous pouvez vous inspirer des codes du kaggle titanic.

<div style="text-align: center;">Tab. 3.2: Jack et Rose</div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl//44453948-142c-485c-8827-16372c660a37/markdown_30/imgs/img_in_image_box_789_877_879_984.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-01-11T18%3A42%3A46Z%2F-1%2F%2F8d59736609809bc0f604633c83a26cae949f644a2f9a7ce65a65694368b370d9" alt="Image" width="7%" /></div>


15 ans

part de Sou-

thampton

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl//44453948-142c-485c-8827-16372c660a37/markdown_30/imgs/img_in_image_box_928_897_1015_982.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-01-11T18%3A42%3A46Z%2F-1%2F%2F894e03e6bd735f20f009632d266923dc9a789cd1a33d20859fd8db9a241dceab" alt="Image" width="7%" /></div>


Powered by TCPDF (www.tcpdf.org)

Lors du chapitre précédent, nous avons vu le cas de la modélisation d'une classe 0/1 pour la Fraude ou le décès. En assurance, nous avons souvent à modéliser le nombre de sinistres ou le coût de ce sinistre. Pour cela, on utilise un modèle GLM.

4.1 Introduction aux GLM

DÉFINITION 4.1.1 - GLM. Les modèles linéaires généralisés (Generalized Linear Models ou GLM) étendent la régression linéaire pour les distributions dites exponentielles (Gaussienne, Binomiale, Poisson).

La régression logistique est un cas particulier des GLM.

Les modèles linéaires généralisés (Generalized Linear Models ou GLM) étendent la régression linéaire en permettant :

• Une fonction de lien qui relie la moyenne de la variable réponse à une combinaison linéaire des prédicteurs; Cela permet en particulier, en prenant une fonction lien en log d'avoir un modèle multiplication et non additif.

• Une distribution de la famille exponentielle (ex. : Gaussienne, Binomiale, Poisson) pour la variable réponse.

- Un offset, qui permet de faire passer directement une information sans passer par la transformation de distribution. Cela est essentiel pour gérer ce qu'on appelle l'exposition, concept que nous verrons plus loin.

Structure d'un GLM

Un GLM comprend trois composants principaux :

1. La composante aléatoire : la distribution de la variable réponse Y, supposée appartenir à la famille exponentielle.

2. La composante systématique : une combinaison linéaire des prédicteurs X, notée  $ \eta = X\beta $ .

3. La fonction de lien : une fonction g qui relie l'espérance de Y à la composante systématique  $ \eta $ ,  $ g(\mathbb{E}[Y]) = \eta $ .

4.2 qu'est ce qu'un modèle GLM?

4.3 Exemples de GLM

• Régression linéaire :  $ Y \sim \mathcal{N}(\mu, \sigma^{2}) $ , fonction de lien  $ g(\mu) = \mu $ .

 $$ \cdot\operatorname{R e g r e s s i o n l o g i s t i q u e}:Y\sim\operatorname{B i n o m i a l e}(n,p),g(p)=\log\left(\frac{p}{1-p}\right). $$ 

• Régression de Poisson :  $ Y \sim \operatorname{Poisson}(\lambda) $ ,  $ g(\lambda) = \log(\lambda) $ .

• il y a enfin le modèle Tweedie, non présenté ici mais très utile pour modéliser la sinistralité.

4.4 Implémentation en Python

il existe aussi une librairie dédiée stats-models qui est plus facile mais il préférable en python d'utiliser sklearn, qui a beaucoup plus d'utilisateurs

Voici un exemple d'implémentation d'un GLM en Python avec la bibliothèque sklearn.

Régression de Poisson

Supposons que nous modélisions le nombre moyen d'accidents en fonction de l'âge et du sexe.

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Example de données
data = pd.DataFrame({
    'age': [25, 30, 35, 40, 45, 50, 55, 60],
    'sex': [0, 1, 0, 1, 0, 1, 0, 1],
    'accidents': [2, 1, 3, 2, 4, 3, 5, 4],
    'exposure': [1, 0, 5, 0.5, 0.5, 1, 0, 3, 0.2, 1],
    'Variable d'exposition' (durée exprimé en année)
})

# Transformation de l'exposure en logarithme pour être utilisé comme offset
data['log_exposure'] = np.log(data['exposure'])

L'objectif est de prédire le nombre d'accidents Y à partir des variables explicatives X.

Listing 4.1 : Classification supervisée avec scikit-learn

# Variables explicatives (X) et réponse (y)
X = data[['age', 'sex']]
y = data['accidents']
offset = data['log_exposure']  # Logarithme de l'exposure

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test, offset_train, offset_test = train_test_split(X, y, offset, test_size=0.25, random_state=42)

# Modèle de régression de Poisson
model = PoissonRegressor(alpha=1e-12, max_iter=1000)  # alpha contrôle la régularisation (L2)
model.fit(X_train, y_train, sample_weight=np.exp(offset_train))  # Offset géré via sample_weight

Pour étudier le modèle, on peut calculer différentes métriques.

Listing 4.2 : description du modèle

# Prédictions
y_pred = model.predict(X_test) + np.exp(offset_test)  # Réintégration de l'offset
dans les prédictions

# Évaluation des performances
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("Coefficients :", model.coef_)
print("Intercept :", model.intercept_)
print("Mean Squared Error :", mse)
print("Mean Absolute Error :", mae)

# Affichage des prédictions
predictions = pd.DataFrame({
    'Age': X_test['age'],
    'Sex': X_test['sex_female'],
    'True_Accidents': y_test,
    'Predicted_Accidents': y_pred,
    'Exposure': np.exp(offset_test)
})
print(predictions)

Exposition Dans la plupart des cas en assurance, l'exposition n'est pas constante. Par exemple, un client qui a une voiture pendant seulement un mois aura (normalement) 12 fois moins d'accidents qu'une personne ayant la même voiture pendant 12 mois. On tient compte de cela via l'exposure.

DÉFINITION 4.4.1 - EXPOSITION. L'exposition désigne la mesure ou la quantité de risque auquel un assureur est exposé pendant une période donnée. Elle reflète la durée ou le degré d'exposition d'un contrat d'assurance ou d'un assuré à un risque couvert. Synonyme : année-police

Pratiquement, gestion de l'exposition dans sklearn (Offset): · L'offset est inclus dans le modèle comme un facteur multiplicatif à travers sample_weight.

• log(exposure) : Convertit l'exposition en logarithme pour respecter l'hypothèse du modèle de Poisson.

2. Paramètre sample_weight dans PoissonRegressor :

• Utilisé pour pondérer chaque observation en fonction de l'exposition.

3. Réintégration de l'offset dans les prédictions :

• Comme le modèle a utilisé un offset logarithmique, les prédictions doivent être multipliées par  $ exp(\text{offset}) $  pour obtenir le taux ajusté.

4.5 Interprétation des résultats

Le tableau des coefficients produit par le modèle GLM inclut :

• Les coefficients estimés ( $ \beta $ );

• Les erreurs standards (standard errors):

• Les valeurs-p pour tester l'hypothèse nulle  $ H_{0}:\beta=0 $ 

métriques mesurant la qualité globale du modèle

DÉFINITION 4.5.1 - MEAN SQUARED ERROR (MSE.) une métrique utilisée pour évaluer la qualité des modèles de régression. Il mesure la moyenne des carrés des erreurs, c'est-à-dire des écarts entre les valeurs prédites  $ (\hat{y}_{i}) $  et les valeurs réelles  $ (y_{i}) $ .

 $$ \mathsf{MSE}=\frac{1}{n}\sum_{i=1}^{n}(y_{i}-\hat{y}_{i})^{2} $$ 

où:

• y i : la valeur réelle pour l'observation i,

•  $ \hat{y}_{i} $  : la valeur prédite pour l'observation i,

• n : le nombre total d'observations.

Avantages

• Sensible aux grandes erreurs : En mettant les erreurs au carré, le MSE pénalise fortement les erreurs importantes, ce qui peut être utile pour détecter les anomalies.

• Différentiable : Convient bien pour les algorithmes d'optimisation basés sur le gradient.

Inconvénients

• Sensible aux outliers : Les erreurs importantes (outliers) peuvent dominer la métrique et biaiser l'évaluation.

• Unité non intuitive : Le MSE est exprimé dans l'unité au carré de la variable cible, ce qui peut compliquer son interprétation.

DÉFINITION 4.5.2 - MEAN ABSOLUTE ERROR (MAE.) métrique utilisée pour évaluer la qualité des modèles de régression. Elle mesure la moyenne des valeurs absolues des erreurs :

 $$ \mathsf{MAE}=\frac{1}{n}\sum_{i=1}^{n}\left|y_{i}-\hat{y}_{i}\right| $$ 

où:

•  $ y_{i} $  : la valeur réelle pour l'observation i,

•  $ \hat{y}_{i} $  : la valeur prédite pour l'observation i,

• n : le nombre total d'observations.

Avantages

• Robuste aux outliers : Contrairement au MSE, le MAE ne donne pas plus de poids aux grandes erreurs.

• Interprétation intuitive : Le MAE est exprimé dans la même unité que la variable cible, ce qui le rend facile à comprendre.

Inconvénients

• Non différentiable: Le MAE n'est pas différentiable au point  $ y_{i} = \hat{y}_{i} $ , ce qui peut compliquer son utilisation dans certains algorithmes.

• Moins sensible aux grandes erreurs : Cela peut être un désavantage si l'identification des grandes erreurs est cruciale pour l'application.

Comparaison entre MSE et MAE

MSE est préférable si l'accent est mis sur la pénalisation des grandes erreurs, par exemple pour détecter des anomalies.

• MAE est préférable si les erreurs doivent être traitées de manière équitable, sans accorder plus de poids aux outliers.

4.6 Applications

La base CASACT La base Casact regroupent des données réelles actuarielles de tout type/ Elle sont disponibles sous une librairie R (un peu plus compliqué en python, raison pour laquelle on trouvera la plupart des données intéressantes sur MOODLE).

Les données FREnchMOTOR Les ensembles de données sont des ensembles de données issus de la même base de données d'un portefeuille d'assurance automobile privé français, observé entre janvier 2003 et décembre 2004. Il s'agit respectivement

• des bases de données sur la fréquence des sinistres, fremotor1freq0304a/b/c

• et les primes, fremotor1prem0304a/b/c

La dernière lettre (a, b ou c) distingue les échantillons aléatoires d'une même série de données. Notez que certains enregistrements sont communs entre les différentes versions issues de rééchantillonnages.

PROPRIÉTÉ 2 Lequel choisir?

• Votre nom commence par A-G : choisissez les fichiers a,

• votre nom commence par H-Q, choisissez le fichiers b,

• sinon, choisissez c.

Les ensembles de données fremotor1freq0304a/b/c contiennent 64 234 en-registrements avec des variables explicatives relatives aux contrats (incluant potentiellement plusieurs véhicules assurés sous un même numéro de contrat). Les ensembles de données fremotor1prem0304a/b/c comprennent 51 949 enregistrements relatifs au nombre de sinistres (par contrat) sur les années 2003 et 2004.

Description des ensembles de données

fremotor1prem0304 contient 30 colonnes :


<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Nom de la colonne</td><td style='text-align: center;'>Description</td></tr><tr><td style='text-align: center;'>IDpol</td><td style='text-align: center;'>L&#x27;identifiant du contrat (utilisé pour faire le lien avec les données de si- nistres).</td></tr><tr><td style='text-align: center;'>DrivAge</td><td style='text-align: center;'>L&#x27;âge du conducteur, en années.</td></tr><tr><td style='text-align: center;'>DrivGender</td><td style='text-align: center;'>Le genre du conducteur (en tant que &quot;facteur&quot;).</td></tr><tr><td style='text-align: center;'>MaritalStatus</td><td style='text-align: center;'>Le statut marital (en tant que &quot;facteur&quot;).</td></tr><tr><td style='text-align: center;'>BonusMalus</td><td style='text-align: center;'>Le bonus/malus, entre 50 et 350 : &lt; 100 signifie un bonus, &gt; 100 signifie un malus en France.</td></tr><tr><td style='text-align: center;'>LicenceNb</td><td style='text-align: center;'>Le nombre de permis (au moins un permis requis).</td></tr><tr><td style='text-align: center;'>PayFreq</td><td style='text-align: center;'>La fréquence de paiement (en tant que &quot;facteur&quot;).</td></tr><tr><td style='text-align: center;'>JobCode</td><td style='text-align: center;'>Le code de profession (en tant que &quot;facteur&quot;).</td></tr><tr><td style='text-align: center;'>VehAge</td><td style='text-align: center;'>L&#x27;âge du véhicule, en années.</td></tr><tr><td style='text-align: center;'>VehClass</td><td style='text-align: center;'>La classe du véhicule (en tant que &quot;facteur&quot;).</td></tr><tr><td style='text-align: center;'>VehPower</td><td style='text-align: center;'>La puissance du véhicule (en tant que &quot;facteur&quot;), allant de &quot;P2&quot; (moins puissant) à &quot;P15&quot; (plus puissant).</td></tr><tr><td style='text-align: center;'>VehGas</td><td style='text-align: center;'>Le type de carburant du véhicule (Diesel ou essence, en tant que &quot;facteur&quot;).</td></tr><tr><td style='text-align: center;'>VehUsage</td><td style='text-align: center;'>L&#x27;utilisation du véhicule (en tant que &quot;facteur&quot;).</td></tr><tr><td style='text-align: center;'>Garage</td><td style='text-align: center;'>Le type de garage (en tant que &quot;facteur&quot;).</td></tr><tr><td style='text-align: center;'>Area</td><td style='text-align: center;'>Le code de la zone (en tant que &quot;facteur&quot;, catégorie inconnue).</td></tr><tr><td style='text-align: center;'>Region</td><td style='text-align: center;'>Les régions des contrats en France.</td></tr><tr><td style='text-align: center;'>Channel</td><td style='text-align: center;'>Le code de distribution du canal (en tant que &quot;facteur&quot;, catégorie inconnue).</td></tr><tr><td style='text-align: center;'>Marketing</td><td style='text-align: center;'>Le code marketing (en tant que &quot;facteur&quot;, catégorie inconnue).</td></tr><tr><td style='text-align: center;'>PremWindscreen</td><td style='text-align: center;'>La prime pour la garantie bris de glace (base annuelle).</td></tr><tr><td style='text-align: center;'>PremDamAll</td><td style='text-align: center;'>La prime pour la garantie dommages tous accidents (base annuelle).</td></tr><tr><td style='text-align: center;'>PremFire</td><td style='text-align: center;'>La prime pour la garantie incendie (base annuelle).</td></tr><tr><td style='text-align: center;'>PremAcc1</td><td style='text-align: center;'>La prime pour la garantie accident type 1 (base annuelle).</td></tr><tr><td style='text-align: center;'>PremAcc2</td><td style='text-align: center;'>La prime pour la garantie accident type 2 (base annuelle).</td></tr><tr><td style='text-align: center;'>PremLegal</td><td style='text-align: center;'>La prime pour la garantie protection juridique (base annuelle).</td></tr><tr><td style='text-align: center;'>PremTPLM</td><td style='text-align: center;'>La prime pour la garantie responsabilité civile obligatoire (base annuelle).</td></tr><tr><td style='text-align: center;'>PremTPLV</td><td style='text-align: center;'>La prime pour la garantie responsabilité civile volontaire (base annuelle).</td></tr><tr><td style='text-align: center;'>PremServ</td><td style='text-align: center;'>La prime pour la garantie service (base annuelle).</td></tr><tr><td style='text-align: center;'>PremTheft</td><td style='text-align: center;'>La prime pour la garantie vol (base annuelle).</td></tr><tr><td style='text-align: center;'>PremTot</td><td style='text-align: center;'>La prime totale (base annuelle).</td></tr><tr><td style='text-align: center;'>Year</td><td style='text-align: center;'>L&#x27;année (valeur numérique).</td></tr></table>

fremotor1freq0304 contient 6 colonnes :

<div style="text-align: center;">Tab. 4.2 : Colonnes de fremotor1freq0304</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Nom de la colonne</td><td style='text-align: center;'>Description</td></tr><tr><td style='text-align: center;'>IDpol</td><td style='text-align: center;'>L&#x27;identifiant du contrat.</td></tr><tr><td style='text-align: center;'>Year</td><td style='text-align: center;'>L&#x27;année de souscription.</td></tr><tr><td style='text-align: center;'>Damage</td><td style='text-align: center;'>Le nombre de sinistres pour la garantie dommages.</td></tr><tr><td style='text-align: center;'>Fire</td><td style='text-align: center;'>Le nombre de sinistres pour la garantie incendie.</td></tr><tr><td style='text-align: center;'>Other</td><td style='text-align: center;'>Le nombre de sinistres pour les autres garanties.</td></tr><tr><td style='text-align: center;'>Theft</td><td style='text-align: center;'>Le nombre de sinistres pour la garantie vol.</td></tr><tr><td style='text-align: center;'>TPL</td><td style='text-align: center;'>Le nombre de sinistres pour la garantie responsabilité civile.</td></tr><tr><td style='text-align: center;'>Windscreen</td><td style='text-align: center;'>Le nombre de sinistres pour la garantie bris de glace.</td></tr></table>

• Quel modèle allez-vous choisir?

• Est il possible avec ces données de corriger de l'exposition ? Considerer si ce n'est pas possible que toutes les polices font un an?

• Programmez le code?

• Quelle est la variable la plus importante?

4.7 Mesurer l'efficacité du modèle

Nous avons vu p. 22 la courbe ROC. On ne peut pas utiliser cette courbe dans les cas GLM hors logistique car elle est adaptée pour les cas où la réponse Y est booléenne (0 ou 1). Une extension de la courbe ROC existe néanmoins pour les GLM : la courbe de Lorenz.

DÉFINITION 4.7.1 - COURBE DE LORENZ. La courbe de Lorenz décrit la qualité des prédictions du modèle. Elle est calculée de la manière suivante :

• Trier toutes les observations, de la prédiction la plus élevée à la plus faible;

• Tracer la valeur observée cumulée en fonction des prédictions triées.

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl//44453948-142c-485c-8827-16372c660a37/markdown_37/imgs/img_in_chart_box_436_864_1044_1336.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-01-11T18%3A42%3A50Z%2F-1%2F%2F1a23271ed28d82d5deeabc58bb94e5fb08f0d03da5ff07cce50b2e37bc0de146" alt="Image" width="51%" /></div>


Fig. 4.1: Exemple de courbe de Lorenz pour un modèle de prime pure. Par exemple, avec 20% du portefeuille, on trouver 45% de la prime pure totale. Le modèle est fait sur 5 tirages (folds) de la base pour valider la robustesse du modèle.

import numpy as np
import matplotlib.pyplot as plt

def lorenz_curve(data, data_predict):
    """
    Fonction pour calculer la courbe de Lorenz.
    Parameters:
        - data : liste ou tableau numpy contenant les données (revenus, etc.)
    Returns:
        ___
    """
    return data

- cumulative_population : pourcentage cumulé de la population
- cumulative_income : pourcentage cumulé des revenus

data_df = pd.DataFrame(data, columns=['Damage']).reset_index()
data_predict_df = pd.DataFrame(data_predict, columns=['y_pred']).reset_index()
data_df = pd.concat((data_df, data_predict_df), axis='columns')
display(data_df)

# Trier y_test en utilisant les indices triés
sorted_data = data_df.sort_values(by='y_pred', ascending=True, inplace=False)
# sorted_data = np.sort(data) # Trier les données dans l'ordre croissant
data_df[['cumulative_income']] = data_df["Damage"].cumsum() # Somme cumulée des données
data_df[['cumulative_income']] = data_df["cumulative_income"] / data_df["cumulative_income"].iloc[-1] # # iloc[-1] permet de normaliser à 1
data_df[['cumulative_population']] = np.linspace(0, 1, len(data_df)) # Population cumulée normalisée
return data_df

# Calcul de la courbe de Lorenz
data_df = Lorenz_curve(y_test, y_pred)

# Tracé de la courbe
plt.figure(figsize=(8, 8))
plt.plot(data_df[['cumulative_population']], data_df[['cumulative_income']], label="Courbe de Lorenz", color="red", linewidth=2)
plt.plot([0, 1], [0, 1], label="Égalité parfaite", color="blue", linestyle="--") # Ligne d'égalité parfaite
plt.title("Courbe de Lorenz")
plt.xlabel("% de la population cumulée")
plt.ylabel("% cumulé de la variable à expliquer")
plt.legend()
plt.grid(True)
plt.show()

Pourquoi est-ce important de comprendre le code quand on utilise ChatGPT et équivalent? Quand on demande à ChatGPT un code pour calculer la courbe de Lorenz, le code fourni est faux. Il est donc important de le comprendre pour le corriger.

4.8 Améliorer la performance par des modèles non linéaire : GAM et XGBoost

Powered by TCPDF (www.tcpdf.org)

# Approche non supervisée

Objectif du cours

• HDLSSS

• Approche linéaire : réduction de dimension par ACP

• Approche linéaire : réduction de dimension par ACM et approche mixte

• Réseau de neurones (auto-encoder)

• comment mesurer la performance

5.1 La malédiction de l'assurance : HDLSSS

DÉFINITION 5.1.1 - HDLSSS. Le concept de HDLSSS (High Dimensional Low Sample Size Setting) se réfère à une situation caractérisée par :

• Un grand nombre de dimensions  $ (p) $  : Les données contiennent un très grand nombre de variables explicatives ou de caractéristiques. Typiquement,  $ p \gg n $ , où n est la taille de l'échantillon.

• Une petite taille d'échantillon (n): Le nombre d'observations est très limité par rapport à la dimensionnalité des données.

Cette situation apparaît fréquemment dans des domaines comme :

• Biostatistique : Analyse de données génomiques ou protéomiques, où des milliers de gènes sont mesurés pour un échantillon de patients relativement faible.

• Assurance : peu de sinistres, beaucoup d'informations.

• Texte et NLP : Analyse de corpus textuels où chaque mot ou combinaison de mots est une dimension.

Défis du HDLSSS

· Sur-ajustement (overfitting) : La haute dimensionnalité conduit souvent à des modèles qui s'ajustent parfaitement aux données d'entraînement, mais qui généralisent mal.

· Singularité des matrices de covariance : Les matrices de covariance deviennent singulières lorsque p > n, rendant certains algorithmes comme les analyses factorielles ou les modèles linéaires non applicables sans régularisation.

· Interprétation : L'identification des variables pertinentes dans un grand espace de caractéristiques est complexe.

Approches pour gérer le HDLSSS

- Réduction de dimension : Techniques comme l'Analyse en Composantes Principales (ACP), t-SNE.

• Régularisation : Utilisation de pénalités L1 (Lasso) ou L2 (Ridge Regression) pour réduire le sur-ajustement.

- Sélection de variables : Identification et utilisation uniquement des variables les plus informatives.

• Augmentation de données : Générer artificiellement plus de données (par exemple, en vision par ordinateur ou en NLP... mais pas en assurances).

Méthodes Bayésiennes : Appliquer des approches bayésiennes pour incorporer des priors et éviter le sur-ajustement.

5.2 Approche non supervisée

5.3 Analyse en Composantes Principales (ACP)

DÉFINITION 5.3.1 - ACP. L'Analyse en Composantes Principales (ACP) est une technique statistique utilisée pour réduire la dimensionnalité des données tout en conservant autant que possible leur variance. Elle transforme les données initiales en un nouvel ensemble de variables non corrélées, appelées composantes principales.

1. Définition mathématique

Soit X une matrice de données de dimension  $ n \times p $ , où n est le nombre d'observations et p le nombre de variables.

• Objectif : Trouver une transformation linéaire Y = XW, où W est une matrice orthogonale ( $ W^{T}W = I $ ) telle que les colonnes de Y maximisent la variance des données transformées.

• Les nouvelles variables  $ Y_{1}, Y_{2}, \ldots, Y_{p} $  sont appelées les composantes principales.

2. Méthode de calcul

1. Centrer les données: X ← X—X, où X est la moyenne de chaque colonne.

2. Calculer la matrice de covariance:  $ S = \frac{1}{n-1} X^{\top} X $ .

3. Décomposer la matrice de covariance : Trouver les valeurs propres

 $ \lambda_{1}, \lambda_{2}, \ldots, \lambda_{p} $  et les vecteurs propres  $ v_{1}, v_{2}, \ldots, v_{p} $  de S.

4. Construire les composantes principales : Les composantes principales sont les projections des données sur les vecteurs propres :

 $$ \mathsf{Y}=\mathsf{X}\mathsf{V}, $$ 

où V est la matrice contenant les vecteurs propres en colonnes.

3. Propriétés d'optimalité de l'ACP

L'ACP est optimale dans les sens suivants :

1. Maximisation de la variance : La première composante principale  $ Y_{1} $  est celle qui maximise la variance des données projetées :

 $$ \mathsf{Y}_{1}=\mathsf{a r g}\operatorname*{m a x}_{\|\mathsf{w}\|=1}\mathsf{V a r}(\mathsf{X}\mathsf{w}), $$ 

sous la contrainte que w est un vecteur unitaire.

2. Orthogonalité : Les composantes principales sont mutuellement orthogonales :

 $$ \mathsf{Cov}(\mathsf{Y}_{i},\mathsf{Y}_{j})=0\quad\forall i\neq j. $$ 

3. Reconstruction optimale : Si seules k < p composantes principales sont conservées, elles minimisent l'erreur quadratique de reconstruction :

 $$ \mathsf{E r r e u r}=\|\mathsf{X}-\hat{\mathsf{X}}\|_{F}^{2}, $$ 

ou  $ \hat{X} $  est la reconstruction des données à partir des k premières composantes, et  $ \|\cdot\|_{F} $  est la norme de Frobenius.

4. Conservation de la variance cumulée : La variance totale des données est égale à la somme des variances des composantes principales :

 $$ \mathsf{Var}(\mathsf{X})=\sum_{i=1}^{p}\lambda_{i}, $$ 

où $\lambda_{i}$ sont les valeurs propres de la matrice de covariance. En conservant les $k$ premières composantes, on retient la proportion suivante de la variance :

 $$ \frac{\sum_{i=1}^{k}\lambda_{i}}{\sum_{i=1}^{p}\lambda_{i}}. $$ 

4. Applications

L'ACP est utilisée dans divers domaines, notamment :

• Réduction de dimension : Simplifier les modèles en conservant uniquement les composantes les plus informatives.

• Visualisation des données : Projeter les données dans un espace de faible dimension (2D ou 3D) pour faciliter l'interprétation.

· Compression de données : Réduire la taille des jeux de données tout en conservant leur information essentielle.

• Prétraitement des données : Supprimer la corrélation entre les variables pour simplifier l'analyse.

5. Limits

• L'ACP est une méthode linéaire et ne capture pas les relations non linéaires entre les variables.

• Les composantes principales sont des combinaisons linéaires des variables d'origine, ce qui peut rendre leur interprétation difficile.

• Elle est sensible à l'échelle des variables; il est donc nécessaire de standardiser les données avant son application.

5.4 Introduction

L'Analyse en Composantes Principales (ACP) est une technique de réduction de dimensionnalité permettant d'extraire les relations structurelles entre les variables d'un jeu de données.

Étape 1 : Préparation des données

Nous sélectionnons les colonnes numériques nécessaires et standardisons les données.

Listing 5.1 : Chargement et normalisation des données

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Charger les données

file_path = "premotor1prem0304.csv"  # Remplace par le chemin réel du fichier
data = pd.read_csv(file_path)

# Sélectionner les colonnes numériques pour l'ACP
numeric_columns = [
    'DrivAge', 'BonusMaUs', 'VehAge',
    'PremWindscreen', 'PremDamAll', 'PremFire',
    'PremAcc1', 'PremAcc2', 'PremLegal',
    'PremTPLM', 'PremTPLV', 'PremServ', 'PremTheft', 'PremTot'
]

X = data[numeric_columns]

# Standardiser les données (centrer et réduire)
scaler = standardScaler()

X_scaled = scaler.fit_transform(X)

Étape 2 : Réaliser l'ACP

Nous utilisons la classe PCA de scikit-learn pour calculer les deux premières composantes principales.

Listing 5.2 : Application de l'ACP

from sklearn.decomposition import PCA

# Initialiser l'ACP pour 2 composantes principales
pca = PCA(n_components=2)

# Appliquer l'ACP
X_pca = pca.fit_transform(X_scaled)

# Résultats de l'ACP
print("Variance expliquée par chaque composante:", pca.explained_variance_ratio_)
print("Variance cumulée:", pca.explained_variance_ratio_.sum())

Étape 3 : Visualisation des résultats

Nous visualisons les données projetées sur les deux premières composantes principales.

Listing 5.3 : Visualisation des données

import matplotlib.pyplot as plt

# Créer un DataFrame avec les deux composantes principales
pca_df = pd.DataFrame(X_pca, columns=['Composante 1', 'Composante 2'])

# Visualisation des données projetées
plt.figure(figsize=(10, 6))

plt.scatter(pca_df['Composante 1'], pca_df['Composante 2'], alpha=0.5, color='blue')
plt.title("Projection des données sur les deux premières composantes principales")
plt.xlabel("Composante 1")
plt.ylabel("Composante 2")
plt.grid()
plt.show()

Étape 4 : Analyse de la variance expliquée

Nous examinons la proportion de variance expliquée par chaque composante pour évaluer la pertinence de l'ACP.

Listing 5.4 : Analyse de la variance expliquée

# Variance expliquée cumulée
import numpy as np

plt.figure(figsize=(8, 5))
plt.plot(
    range(1, len(pca.explained_variance_ratio_) + 1),
    np.cumsum(pca.explained_variance_ratio_),
    marker='o')
plt.title("Variance expliquée cumulée")
plt.xlabel("Nombre de composantes principales")
plt.ylabel("Variance expliquée cumulée")
plt.grid()
plt.show()

Étape 5 : Interprétation des résultats

Pour comprendre la contribution de chaque variable dans les composantes principales, nous utilisons les coefficients associés.

# Composantes principales
print("Composantes principales (coefficients):")
print(pd.DataFrame(pca.components_, columns=numeric_columns))

Que faire quand on a des données numériques et catégorielles? La FAMD

Quand on a des données purement catégorielle, on peut utiliser une méthode dite ACM (analyse en composante multiple). Dans la pratique, on a souvent des données à la fois nécessaire d'utiliser les deux. Une méthode existe, AFDM (Analyse Factorielle des Données Mixtes) opu FAMD (Factorial Analysis of Mixed Data). Elle combine les principes de l'Analyse en Composantes Principales (ACP) pour les variables quantitatives et de l'Analyse des Correspondances Multiples (ACM) pour les variables qualitatives. Si elle est populaire en France, c'est une méthode relativement peu répandue en dehors de la France et qui n'a été développée qu'en R.

Pour aller plus loiAppeler du code R depuis Python Via rpy2, on peut lancer du code R, comme ici une AFDM.

import rpy2.objects as ro
from rpy2.objects import pandas2ri

# Activer la conversion automatique entre pandas et R
pandas2ri.activate()

# Importer un dataframe Pandas en Python
import pandas as pd

# Charger les données dans R
ro.globalenv['data'] = pandas2ri.py2rpy(df_with_claims[['Year', 'DrivAge', 'DrivGender', 'MaritalStatus', 'BonusMalus', 'LicenceNb', 'PayFreq', 'JobCode', 'VehAge', 'VehClass', 'VehPower', 'VehGas', 'VehUsage', 'Garage', 'Area', 'Region', 'Channel', 'Marketing']].dropna(axis=0))

# Exécuter le code R
ro.r(...
    library(FactoMineR)
    library(factoextra)
    result <- FAMD(data, ncp = 2, graph = FALSE)

# Sauvegarder le graphique des individus --- ne marche pas
png("Users/guillaumegeorge/Downloads/famd_individus.png", width = 800, height = 600)
fviz_famd_ind(result, repel = TRUE) # Graphe des individus
dev.off()

# Sauvegarder le graphique des variables --- ne marche pas
png("/Users/guillaumegeorge/Downloads/famd_variables.png", width = 800, height = 600)
fviz_famd_var(result, repel = TRUE) # Graphe des variables
dev.off()

res <- summary(result)
result.df <- as.data.frame(result$ind$coord) # Récupérer les coordonnées des individus
result.var_df <- as.data.frame(result$var$coord) # Récupérer les coordonnées des individus
...))

# Récupérer les résultats en Python
result_df = ro.r('result_df')
result_var_df = ro.r('result_var_df')
res = ro.r('res')
print(res)

result_df = pandas2ri.rpy2py(result_df)
result_var_df = pandas2ri.rpy2py(result_var_df)
#print("Résultats de la FAMD :\n", result_df)

5.5 t-SNE: t-Distributed Stochastic Neighbor Embedding

Le t-SNE (t-Distributed Stochastic Neighbor Embedding) est une méthode de réduction de dimension non linéaire utilisée principalement pour la visualisation de données en haute dimension. Cette technique a été introduite par van der Maaten et Hinton en 2008.

Objectif principal

t-SNE vise à projeter des données de haute dimension  $ (p) $  dans un espace de plus faible dimension  $ (d, $  souvent 2 ou 3 $  tout en :

• Préservant la structure locale des données : les points proches dans l'espace original restent proches dans l'espace réduit.

• Maximisant la séparation des clusters pour une meilleure interprétation visuelle.

Applications

t-SNE est principalement utilisé pour la visualisation des données. Quelques domaines d'application incluent :

• Biologie : Analyse de données génomiques ou protéomiques.

• Traitement du langage naturel (NLP): Visualisation d'espaces d'embed-ding de mots.

• Vision par ordinateur : Exploration de caractéristiques extraites de modèles convolutifs.

• Clustering : Analyse exploratoire pour identifier des groupes naturels dans les données.

Avantages et inconvénients

Avantages :

• Capacité à capturer la structure locale des données.

• Génération de visualisations interprétables dans des espaces de faible dimension.

Paramètres importants pour ajuster un t-SNE :

• Perplexity : Définit le nombre effectif de voisins considérés. Une valeur typique est comprise entre 5 et 50.

• Learning rate : Contrôle la vitesse d'apprentissage pendant l'optimisation.

• Iterations : Augmenter le nombre d'itéractions peut améliorer la convergence.

Inconvénients :

• Non déterministe : Les résultats peuvent varier à chaque exécution (selon l'initialisation).

- Sensibilité aux hyperparamètres : Notamment le perplexity, qui influence la balance entre structure locale et globale.

• Pas d'interpolation possible : t-SNE est conçu uniquement pour la visualisation, pas pour une application généralisée.

Exemple des données Si on a les données X qu'on souhaite réduire en deux dimensions :

# Appliquer t-SNE sur les données réduites
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)

# X est ici le résultat de l'ACP
data_tsne = tsne.fit_transform(X)

# Création d'un DataFrame pour visualisation
tsne_df = pd.DataFrame(data_tsne, columns=['t-SNE1', 't-SNE2'])

L'utilité de la méthode est surtout graphique :

# Ajout d'une colonne pour une éventuelle couleur (ex: clustering ou cible)
tsne_df['target'] = df_with_claims['TPL']  # Exemple de variable cible

# Tracer les données t-SNE
plt.figure(figsize=(10, 7))
scatter = plt.scatter(tsne_df['t-SNE1'], tsne_df['t-SNE2'], c=tsne_df['target'], cmap='viridis_r', alpha=0.7)
plt.colorbar(scatter, label='Damage (Siniestres)')
plt.title('Visualisation t-SNE des données réduites par l\\'ACP')
plt.xlabel('t-SNE1')
plt.ylabel('t-SNE2')
plt.show()

On a ainsi une représentation des zones avec sinistres (fig. 5.1).

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl//44453948-142c-485c-8827-16372c660a37/markdown_46/imgs/img_in_chart_box_140_642_746_1106.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-01-11T18%3A42%3A54Z%2F-1%2F%2F6e324f54643fad9c1672c6affe925257b9fa176fc49e4649332904759db203d0" alt="Image" width="50%" /></div>


Les Auto-Encoders

Un Auto-Encoder (AE) est un type de réseau de neurones conçu pour apprendre une représentation comprimée (ou codée) des données d'entrée. Il se compose de deux parties principales :

• Encodeur : Transforme les données d'entrée en une représentation de dimension inférieure (latent space).

Décodeur : Reconstruit les données d'entrée à partir de cette représentation codée.

1. Architecture d'un Auto-Encoder

Un Auto-Encoder est constitué de trois parties :

• Couche d'entrée : Représente les données originales  $ x \in R^{n} $ 

• Couche cachée (bottleneck): Représente la version comprimée  $ z \in R^{k} $ , où k < n.

• Couche de sortie : Produit la reconstruction des données  $ \hat{x} \in R^{n} $ 

La tâche d'un Auto-Encoder est de minimiser l'erreur de reconstruction :

 $$ L=\|\boldsymbol{x}-\hat{\boldsymbol{x}}\|^{2} $$ 

2. Applications des Auto-Encoders

• Réduction de dimension : Apprentissage d'une représentation compacte des données.

• Débruitage : Suppression du bruit dans les données.

• Prétraitement : Extraction de caractéristiques pour d'autres modèles.

- Génération de données : Utilisation dans des architectures génératives (par exemple, Variational Auto-Encoders).

3. Exemple pratique en Python

L'exemple suivant montre comment implémenter un Auto-Encoder simple avec Keras.

Listing 5.6 : Implémentation d'un Auto-Encoder avec Keras

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Chargement des données MNIST
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((x_train.shape[0], -1)) # (60000, 784)
x_test = x_test.reshape((x_test.shape[0], -1)) # (10000, 784)

# Dimensions de l'entrée et du codage
input_dim = x_train.shape[1]
encoding_dim = 32 # Dimension de l'espace latent

# Construction de l'Auto-Encoder
input_img = Input(shape=(input_dim,))
# Encodeur
encoded = Dense(encoding_dim, activation='relu')(input_img)
# Décodeur
 decoded = Dense(input_dim, activation='relu')(encoded)

# Modèle Auto-Encoder
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Entrainement
autoencoder.fit(x_train, x_train,
                 epochs=50,
                 batch_size=256,
                 shuffle=True,
                 validation_data=(x_test, x_test))

# Encodeur uniquement
encoder = Model(input_img, encoded)

# Visualisation des résultats
encoded_imgs = encoder.predict(x_test)

encoded_imgs = autoencoder.predict(x_test)

# Visualiser les données originales et reconstruites
n = 10 # Nombre d'images à afficher
plt.figure(figsize=(20, 4))
for i in range(n):
    # Données originales
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.axis('off')

    # Reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(encoded_imgs[i].reshape(28, 28), cmap='gray')
    plt.axis('off')

plt.show()

5.6 quelle efficacité?

On verra dans le code joint, l'efficacité d'un auto-encoder par rapport à l'ACP sur de la reconnaissance de caractères (données MNIST).

5041921314

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl//44453948-142c-485c-8827-16372c660a37/markdown_48/imgs/img_in_image_box_139_1276_747_1529.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-01-11T18%3A42%3A54Z%2F-1%2F%2Fdd79fe3d4352915f50fd1b136abd03be4205e6343c69c646f15a673d2b2a2ae0" alt="Image" width="51%" /></div>


504/921314

Fig. 5.2: Comparaison de la capacité de l'AutoEncoder (ligne 2) et ACP (ligne 4) à répliquer des chiffres - données MNIST (ligne 1 et 3)

5.7 Problèmes

Exercice 4 On peut lire dans un mémoire d'actuariat, le problème à faire des clusters :

Dans le cas de la variable "statut-type" dont les modalités sont Locataire d'Appartement (LA), Locataire de Maison (LM), Propriétaire d'Appartement (PA), Propriétaire de Maison (PM), le modèle associe une valeur numérique 'a chaque modalité 1,2,3,4 pour en sortir une moyenne et calculer avec pondération le centroïde de cette variable. Cette moyenne n'est pas interprétable puisqu'il n'y a pas d'ordre de risque ou de taille entre ces modalités.

Qu'en pensez vous?

# Modèles financiers stochastiques

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl//44453948-142c-485c-8827-16372c660a37/markdown_50/imgs/img_in_image_box_784_133_825_197.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-01-11T18%3A42%3A55Z%2F-1%2F%2Fa5cc2ec1c14f52635e1baddb63d99f6b9813e9221162abe370e925d110059f05" alt="Image" width="3%" /></div>


6.1 Modèle de Vasicek

Présentation du modèle de Vasicek

Le premier modèle proposé dans ce cours est le modèle dit de Vasicek $ ^{1} $ 

1 : Modèle de Vasicek

DÉFINITION 6.1.1 - LE MODÈLE DE VASICEK. Le modèle de Vasicek utilise le processus d'Ornstein-Uhlenbeck pour modéliser les taux d'intérêt court terme  $  r(t)  $  au cours du temps.

 $$ d r(t)=\kappa(\theta-r(t))d t+\sigma d W(t) $$ 

• Le paramètre θ donne la moyenne à long terme

•  $ \kappa > 0 $  donne la vitesse à laquelle le processus va converger vers cet équilibre.

• W est un mouvement brownien.

On peut trouver les données pour calibrer les modèles par exemple sur Euro 3 mois

Approximation discrète du modèle de Vasicek :

 $$ \Delta r_{t}=\kappa(\theta-r_{t})\Delta t+\sigma\epsilon_{t}\sqrt{\Delta t} $$ 

Signification des termes :

⋅ ∆r t = r t+1 − r t : variation discrète du taux d'intérêt,

• κ : vitesse de réversion.

• θ : taux moyen de long terme.

• r t : taux d'intérêt au temps t ,

•  $ \Delta t $  : incrément de temps (par exemple,  $ \frac{1}{252} $  pour des données quotidiennes),

• σ : volatilité.

 $$ \epsilon_{t}\sim\mathcal{N}(0,1):bruit blanc gaussien. $$ 

Si on fait une régression linéaire, en régressant  $ \Delta r_{t} $  :

 $$ Y=aX+b+\epsilon $$ 

Si on prend  $ Y = \Delta r_{t} $  et  $ X = r_{t}\Delta t = r_{t} $ , alors  $ a = -\kappa $  et  $ \theta = -\frac{b}{a} $ .

Voici un code pour estimer les paramètres :

# Calcul des différences
delta_r = np.diff(rates)
r_t = rates[:-1]

# Régression pour estimer a et b
from sklearn.linear_model import LinearRegression

# Variable cible : Delta r
y = delta_r

# Variable explicative : r_t
X = r_t.reshape(-1, 1)

# Régression linéaire
model = LinearRegression()
model.fit(X, y)

# Estimation des paramètres
kappa = -model.coef_[0]  # vitesse de réversion
theta = -model.intercept_ / a  # taux moyen
sigma = np.std(y - model.predict(X))  # volatilité

print(f"kappa (vitesse de réversion) = {kappa:.4f}")
print(f"theta (tau moyen) = {theta:.4f}")
print(f"sigma (volatilité) = {sigma:.4f}")

Puis pour resimuler les données :

# Paramètres de simulation
n_simulations = 100  # nombre de simulations
n_steps = 252  # nombre de jours (par exemple, 1 an de données quotidiennes)
dt = 1 / 252  # incrément de temps

# Initialisation
r0 = rates[-1]  # dernier taux observé
simulated_paths = np.zeros((n_steps, n_simulations))

# Simulation
for i in range(n_simulations):
    r_t = r0
    path = []
    for _ in range(n_steps):
        dr = kappa * (theta - r_t) * dt + sigma * np.sqrt(dt) * np.random.normal()
        r_t += dr
        path.append(r_t)
        simulated_paths[:, i] = path

# Visualisation
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
for i in range(n_simulations):
    plt.plot(simulated_paths[:, i], alpha=0.3)

plt.title("Simulations de trajectoires du taux d'intérêt (Modèle de Vasicek)")
plt.xlabel("Temps (jours)")
plt.ylabel("Taux d'intérêt (%)")
plt.grid()
plt.show()

<div style="text-align: center;">Fig. 6.1: Modèle de Vasicek - On considère le taux d'intérêt actuel  $ r(0) $  est de 0.5%, que  $ \theta $ , le taux d'intérêt long terme est de 0.15, et que la volatilité du processus brownien est  $ \sigma = 0.05 $ . On calcule sur 200 périodes, sachant que la volatilité a été mesurée sur 10 périodes. La retour à la moyenne k se fait avec trois valeurs 0.002, 0.02 et 0.2:</div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl//44453948-142c-485c-8827-16372c660a37/markdown_51/imgs/img_in_chart_box_447_776_1047_1176.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-01-11T18%3A42%3A55Z%2F-1%2F%2Ff3d6b4b1b22ca2a6b0669c902a2826b3e468da83e48899f19f4b8887c7b45e41" alt="Image" width="50%" /></div>


<div style="text-align: center;">On voit que les taux d'intérêt peuvent être négatifs.</div>


6.2 Probabilités P et Q risque neutre

Dans les modèles de finance actuarielle, on distingue deux types de probabilités : les probabilités réelles (ou historiques), notées P, et les probabilités neutres au risque, notées Q.

Probabilités réelles (P)

Les probabilités réelles reflètent les comportements observés sur le marché. Elles permettent de modéliser l'évolution des prix en fonction des don

nées historiques. Par exemple, une matrice de transition P peut être définie comme suit :

 $$ P=\begin{pmatrix}p_{uu}&p_{ud}\\ p_{du}&p_{dd}\end{pmatrix}, $$ 

où chaque élément  $ p_{ij} $  représente la probabilité de transition de l'état i vers l'état j sous P.

Probabilités neutres au risque (Q)

Les probabilités neutres au risque sont des probabilités artificielles utilisées pour la valorisation d'actifs financiers. Elles sont construites de manière à refléter un monde où tous les actifs financiers croissent au taux sans risque. Une matrice de transition sous Q peut être définie comme suit :

 $$ Q=\begin{pmatrix}q_{uu}&q_{ud}\\ q_{du}&q_{dd}\end{pmatrix}, $$ 

où chaque élément  $ q_{ij} $  est obtenu en éliminant la prime de risque associée à l'actif.

Relation entre P et Q

La relation entre les deux types de probabilités repose sur le concept de risque-neutralité ou d'absence d'arbitrage. Sous Q, l'espérance de l'évolution du prix d'un actif est donnée par :

 $$ \mathbb{E}_{Q}[S_{t+1}]=S_{t}(1+r), $$ 

où  $ S_{t} $  est le prix de l'actif à l'instant t, et r est le taux sans risque.

Exemple 6.2.1 (Exemple simplifié du lien entre S et Q) Considérons un actif financier avec un prix initial  $ S_{t}=100 $  et un taux sans risque r=0.05 (soit 5%). Le prix futur  $ S_{t+1} $  peut évoluer selon deux scénarios :

• Une hausse où $S_{t+1}=110$ avec une probabilité $q=0.6$ sous la probabilité neutre au risque $Q$.

• Une baisse où  $ S_{t+1}=90 $  avec une probabilité 1-q=0.4 sous Q.

Calcul de l’espérance sous Q

Sous la probabilité neutre au risque Q, l'espérance du prix futur doit être égale au prix actuel multiplié par  $ 1 + r $  (le taux sans risque). Ainsi, nous avons la relation suivante :

 $$ \mathbb{E}_{Q}[S_{t+1}]=S_{t}\cdot(1+r). $$ 

En remplaçant les valeurs :

 $$ \mathbb{E}_{Q}[S_{t+1}]=100\cdot(1+0.05)=105. $$ 

En utilisant les probabilités sous Q, l'espérance s'écrit également comme :

 $$ \mathbb{E}_{Q}[S_{t+1}]=q\cdot110+(1-q)\cdot90. $$ 

Substituons les valeurs :

 $$ \mathbb{E}_{Q}[S_{t+1}]=0.6\cdot110+0.4\cdot90=66+36=105. $$ 

Passer de P à Q

Imaginons que nous ayons 2 actifs  $ S_{1} $  et  $ S_{2} $  dont nous connaissons le prix et nous voulons trouver la matrice Q.

1. Notations

Nous utilisons les notations suivantes :

•  $ S_{1,t} $  et  $ S_{2,t} $  : prix actuels des actifs 1 et 2 respectivement.

• S1,t+1 et S2,t+1 : vecteurs des prix futurs possibles pour les actifs 1 et 2.

• Q = [qi j] : matrice des probabilités neutres au risque à déterminer.

• r : taux sans risque (exprimé en fraction, par exemple r = 0.05 pour 5%).

2. Condition de neutralité au risque

La neutralité au risque impose que l'espérance des prix futurs, calculée avec Q, corresponde au prix actuel actualisé au taux sans risque. Formulée mathématiquement, cette condition est donnée pour chaque actif k (k = 1, 2) par :

 $$ \mathbb{E}_{Q}[S_{k,t+1}]=S_{k,t}\cdot(1+r), $$ 

où:

 $$ \mathbb{E}_{Q}[S_{k,t+1}]=\sum_{j}q_{i j}\cdot S_{k,t+1,j}. $$ 

Pour chaque actif, cette condition doit être satisfaite afin de garantir que l'évolution des prix sous Q reflète un rendement au taux sans risque.

3. Système linéaire à résoudre

La neutralité au risque pour chaque actif permet d'écrire un système linéaire. Pour n états et deux actifs, ce système est :

 $$ \begin{aligned}&q_{1}\cdot S_{1,t+1,1}+q_{2}\cdot S_{1,t+1,2}+\cdots+q_{n}\cdot S_{1,t+1,n}=S_{1,t}\cdot(1+r),\\&q_{1}\cdot S_{2,t+1,1}+q_{2}\cdot S_{2,t+1,2}+\cdots+q_{n}\cdot S_{2,t+1,n}=S_{2,t}\cdot(1+r).\\ \end{aligned} $$ 

Cela donne un système linéaire où les inconnues sont les probabilités neutres au risque  $ q_{ij} $ .

4. Contraintes sur Q

Les probabilités neutres au risque doivent respecter les propriétés suivantes :

• qij ≥ 0 pour tout i, j (les probabilités sont positives);

•  $ \sum_{j}q_{ij}=1 $  pour chaque état i (les probabilités de transition sortant d'un état i doivent somme à 1).

5. Résolution

Pour résoudre ce système, on peut utiliser les techniques suivantes :

• Matrices et algèbre linéaire : Exprimer le système sous la forme  $ A_{q}=b $ , où A est une matrice des prix futurs, q est le vecteur des probabilités neutres au risque, et b est le vecteur des espérances actualisées.

- Outils numériques: Utiliser un solveur comme numpy.linalg.solve en Python pour calculer q.

import numpy as np

# Données
S1_t = 100  # Prix actuel actif 1
S2_t = 200  # Prix actuel actif 2
S1_t1 = [110, 90]  # Prix futurs actifs 1
S2_t1 = [220, 150]  # Prix futurs actifs 2
r = 0.05  # Taux sans risque

# Matrice de transition réelle P (elle ne sert à rien pour calculer P)
P = np.array([[0.7, 0.3],
                   [0.4, 0.6]])

# Système pour trouver Q
A = np.array([
    [S1_t1[0], S1_t1[1]],  # Neutralité au risque pour actif 1
    [S2_t1[0], S2_t1[1]]     # Neutralité au risque pour actif 2
])
b = np.array([
    S1_t * (1 + r),  # Espérance actualisée actif 1
    S2_t * (1 + r)     # Espérance actualisée actif 2
])

# Résolution du système
q = np.ilalg.solve(A, b)

# Construction de Q à partir de q
Q = np.array([[q[0], 1 - q[0]],
                  [q[0], 1 - q[0]]])  # Simplification : même probas pour chaque état

# Affichage
print("Matrice sous probabilité P :")
print(P)
print("\nMatrice sous probabilité Q :")
print(Q)

6.3 aller plus loin : le modèle HJM

3. Le modèle de taux HJM et l'ACP : une approche par facteurs

Le modèle Heath-Jarrow-Morton (HJM) est un cadre général pour modéliser l'évolution des courbes de taux d'intérêt dans le temps. Contrairement aux modèles courts (Vasicek, CIR), le HJM spécifie directement la dynamique de la courbe des taux forward.

Formulation générale du modèle HJM

Soit  $  f(t, T)  $  le taux forward au temps t pour maturité T, le modèle HJM pos- tule :

 $$ d f(t,T)=\mu(t,T)d t+\sum_{i=1}^{d}\sigma_{i}(t,T)d W_{t}^{(i)} $$ 

où:

•  $ \mu(t,T) $  est la dérive (imposée par l'absence d'arbitrage),

• σi(t,T) sont les volatilités forward associées à chaque facteur,

•  $  W_{t}^{(i)}  $  sont des mouvements browniens indépendants.

Lien avec l'analyse en composantes principales (ACP)

L'ACP appliquée aux variations historiques des taux permet d'identifier les facteurs de risque dominants :

• 1 $ ^{er} $  facteur : déplacement parallèle de la courbe (niveau),

•  $ 2^{e} $  facteur : pente (steepness),

•  $ 3^{e} $  facteur : courbure (curvature).

En pratique, on approxime les volatilités  $ \sigma_{i}(t,T) $  à l'aide des composantes principales issues de l'ACP sur les variations de taux forward.

Exemple en Python : simulation d'une courbe avec 3 facteurs

import numpy as np
import matplotlib.pyplot as plt

# Paramètres
n_maturities = 30
n_factors = 3
dt = 1 / 252
n_steps = 252

# Maturités de 1 à 30 ans
T = np.linspace(1, 30, n_maturities)

# Courbe initiale:  $ f(0, T) $ 
f0 = 0.02 + 0.002 * np.log(T)

# 3 profils de volatilité issus de 'lACP
sigma = np.array([
    np.ones(n_maturities), # Niveau
    T - T.mean(), # Pente
    (T - T.mean())**2 - 1 # Courbure
])
sigma = sigma / np.linalg.norm(sigma, axis=1, keepdims=True) * 0.01 # calibrage

# Simulation
f_t_T = np.zeros((n_steps, n_maturities))
f_t_T[0] = f0

for t in range(1, n_steps):
    dW = np.random.normal(0, np.sqrt(dt), size=n_factors)
    df = sigma.T @ dW
    f_t_T[t] = f_t_T[t-1] + df # sans dérive pour simplifier

# Visualisation
plt.figure(figsize=(10, 5))
for i in range(0, n_steps, 50):
    plt.plot(T, f_t_T[i], label=f"t = {i}")
    plt.title("Simulation de courbes de taux forward (HJM + ACP)")
    plt.xlabel("Maturité (années)")
    plt.ylabel("Taux forward f(t, T)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

PROPRIÉTÉ 3 • On utilise ici un modèle HJM simplifié avec volatilités constantes. Ceci est très loin de la réalité.

• L'ACP permet d'interpréter économiquement les chocs : niveau, pente, courbure.

• Ce type de simulation est utile pour générer des scénarios économiques (ESG) ou évaluer des produits sensibles aux taux.

DÉFINITION 7.0.1 - CYCLE INVERSÉ DE PRODUCTION. A la différence d'un cycle de production classique ("on produit puis on vend"), l'assurance se caractérise par un cycle inversé de production : l'assureur vend puis ensuite "produit" sous forme de remboursement de sinistres.

L'incertitude est au coeur de l'assurance On ne connaît donc son gain ou sa perte que bien après avoir vendu ("vision ultime").

Quelques conséquences

• pilotage "non cash". Par construction, l'assureur bénéficie de trésorerie, sans qu'elle ne représente un indicateur de santé de la société

• Besoin d'établir une vision la plus juste des engagements futurs : les provisions techniques

• En plus de ces provisions, une activité à risque : besoin en capital.

Par des calculs actuariels, on essaye d'avoir le plus tôt possible une idée la plus juste ou une idée prudente de cette vision ultime (selon le régime comptable) de ce que l'on devra payer et l'on établira les provisions nécessaires.

7.1 Quelques éléments des comptes en Assurance

Quelques notions comptables doivent être introduites et seront développées ultérieurement :

On trouva dans le répertoire de la bibliothèque Chain Ladder, de nombreuses bases de données : lien vers les données

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl//44453948-142c-485c-8827-16372c660a37/markdown_56/imgs/img_in_image_box_150_918_743_1231.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-01-11T18%3A42%3A58Z%2F-1%2F%2F739e0db5fbaa9af2cc0733bd42cb6ee43cdc14c8ddda2613f4f79435810b9979" alt="Image" width="49%" /></div>


<div style="text-align: center;">Fig. 7.1: Comptabilisation des primes</div>


7.2 Indicateur de profitabilité

DÉFINITION 7.2.1 - PRIME ÉMISE. La prime émise est souvent désignée comme la prime brute parce qu'elle est calculée avant toute déduction, comme les frais de réassurance. Elle représente le montant brut que l'assureur doit recevoir pour la couverture des risques.

DÉFINITION 7.2.2 - PRIME ACQUISE / NON ACQUISE. La prime acquise (en anglais, prime gagnée "earned") est la partie de la prime émise qu'une compagnie d'assurance a effectivement "gagnée" pour avoir couvert le risque sur une période donnée. Elle représente la fraction de la prime d'assurance correspondant à la période de couverture déjà écoulée. En d'autres termes, c'est la portion de la prime émise qui peut être comptabilisée comme un revenu parce que le risque a déjà été porté par l'assureur.

Le reporting peut être fait :

• Année d'accident - Accident year (sinistre) -

• Année de souscription - Underwriting year (prime)

• Année calendaire - Calendar year (comptabilité : delta par rapport à l'année précédente)

DÉFINITION 7.2.3 - CHARGE SINISTRE. La charge de sinistres (ou parfois appelée coût des sinistres) est une notion clé en assurance, représentant l'ensemble des dépenses engagées par une compagnie d'assurance pour couvrir les sinistres déclarés par ses assurés. En d'autres termes, c'est le montant que l'assureur doit payer pour indemniser les sinistres survenus au cours d'une période donnée.

Charge de sinistres = Prestations versées + Provisions pour sinistres + Frais de gestion des sinistres

Composition de la charge de sinistres :

Prestations versées : Cela correspond aux indemnités que l'assureur verse directement aux assurés ou aux bénéficiaires, pour les sinistres couverts par les contrats d'assurance. Cela inclut les remboursements pour des dommages matériels, des frais médicaux, des décès, des invalidités, etc. Une partie importante de la charge de sinistres concerne les provisions pour sinistres. Il s'agit des montants que l'assureur met de côté pour faire face aux sinistres survenus mais non encore réglés, ou pour lesquels le montant exact n'est pas encore connu. Il ne s'agit pas seulement des prestations versées, mais aussi des frais associés à la gestion des sinistres. Ces frais incluent les coûts administratifs liés au traitement des demandes de sinistres, tels que les enquêtes, l'expertise, et le traitement administratif des dossiers. On calcule aussi des Provisions sur ces frais de gestion (PFGS).

7.3 Méthodes de projection des provisions

Principe général L'idée de la plupart des méthodes de provisions est de considérer qu'un certain nombre d'éléments sont stables :

1: le montant des paiements à un moment donné par rapport à ce qu'on paiera à l'ultime

• le montant des réserves à un moment donné par rapport à ce qu'on paiera à l'ultime

Chaque méthode est basée sur ses propres hypothèses mais généralement, l'hypothèse de base est l'homogénéité des sinistres, d'où le calcul de toute méthode sur des sous-ensembles de sinistres que l'on estime homogène.

Example 7.31 Nous allons travailler sur les données dites RAA, très connues :

raa_df = pd.read_csv(
    "https://raw.githubusercontent.com/casact/chainladder-python/master/chainladder/utils/data/raa.csv"
)

raa_df['development'] = raa_df['development'] - raa_df['origin']

raa_df.head(20)

# Structuration sous forme de triangle de développement (DataFrame croisé)
triangle = raa_df.pivot(index="origin", columns="development", values="values")
triangle_formatted = triangle.style.format(lambda x: f"{int(x):", "replace", "") if pd.notnull(x) else "")

display(triangle_formatted)

Méthode la plus classique en IARD pour calculer la vision à l'ultime : Chain Ladder

La méthode Chain Ladder est une technique classique utilisée pour estimer les montants futurs des sinistres en cours. Elle repose sur l'hypothèse que le développement des sinistres observé par le passé reflète celui qui se produira à l'avenir. Cette méthode est largement utilisée dans l'évaluation des provisions pour sinistres des compagnies d'assurances.

Tableau des triangles de sinistres

La méthode se base sur la création d'un triangle de développement des sinistres. Voici un exemple simple d'un tel triangle où les lignes correspondent aux années d'origine (ou d'exercice) et les colonnes aux périodes de développement.


<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Année gine</td><td style='text-align: center;'>d&#x27;ori-</td><td style='text-align: center;'>Dév 1</td><td style='text-align: center;'>Dév 2</td><td style='text-align: center;'>Dév 3</td><td style='text-align: center;'>Dév 4</td></tr><tr><td style='text-align: center;'>2019</td><td style='text-align: center;'>1000</td><td style='text-align: center;'>1500</td><td style='text-align: center;'>1800</td><td style='text-align: center;'>1900</td><td style='text-align: center;'></td></tr><tr><td style='text-align: center;'>2020</td><td style='text-align: center;'>1100</td><td style='text-align: center;'>1600</td><td style='text-align: center;'>2000</td><td style='text-align: center;'></td><td style='text-align: center;'></td></tr><tr><td style='text-align: center;'>2021</td><td style='text-align: center;'>1200</td><td style='text-align: center;'>1700</td><td style='text-align: center;'></td><td style='text-align: center;'></td><td style='text-align: center;'></td></tr><tr><td style='text-align: center;'>2022</td><td style='text-align: center;'>1300</td><td style='text-align: center;'></td><td style='text-align: center;'></td><td style='text-align: center;'></td><td style='text-align: center;'></td></tr></table>

TAB. 7.1 : Triangle de développement des si-

nistres

On mesure après chaque période (mois, année,...) combien on a payé de sinistres Dev : développement

<div style="text-align: center;">On peut aussi le représenter graphiquement.</div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl//44453948-142c-485c-8827-16372c660a37/markdown_58/imgs/img_in_chart_box_143_1142_738_1482.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-01-11T18%3A42%3A59Z%2F-1%2F%2Fb0c6f8fdb91af52d4bba105cb0b664728961512b78a2271be39b4dde2cc7bbe1" alt="Image" width="49%" /></div>


<div style="text-align: center;">Un exemple classique (auto):</div>


<div style="text-align: center;">Fig. 7.2 : développement des paiements (points blancs) et des provisions dossier/dossier (point noir) au cours du temps</div>


Facteurs de développement

Pour chaque année d'origine, on calcule les facteurs de développement entre deux périodes consécutives. Par exemple, le facteur de développement entre les périodes 1 et 2 est donné par :

 $$ f_{1}=\frac{\sum_{i=1}^{n-1}C_{i,2}}{\sum_{i=1}^{n-1}C_{i,1}} $$ 

où  $ C_{i,j} $  représente les sinistres cumulés à la période j pour l'année d'origine i, et n est le nombre total d'années observées.

Listing 7.2 : Méthode Chain Ladder. Calculé les facteurs de développement selon la méthode de Chain-Ladder en tenant compte des valeurs manquantes (NaN)

Listing 7.1 : méthode de facteur moyen des ratios - ALR

Le facteur de développement global pour chaque période peut ensuite être calculé, puis utilisé pour estimer les sinistres futurs.

Pratiquement, cela se calcule assez facilement pour la moyenne des ratios (Average Link Ratio Method- ALR):

def calculate_factors(triangle):
    factors = []
    for i in range(triangle.shape[1] - 1):
        ratio = triangle.iloc[:, i + 1] / triangle.iloc[:, i]
        factors.append(ratio.mean(skipna=True))  # moyenne des ratios
    return factors

La méthode Chain-Ladder est un peu plus compliquée à programmer car il s'agit d'une somme pour les lignes développées :

def calculate_factors_chain_ladder(triangle):
    factors = []
    for i in range(triangle.shape[1] - 1):  # Parcourt les colonnes sauf la dernière
        # Sélectionner uniquement les lignes où les deux colonnes comparées ne sont pas NaN
        valid_rows = triangle.iloc[:, [i, i + 1]].dropna()
        numerator = valid_rows.iloc[:, 1].sum()  # Somme de la colonne suivante
        denominator = valid_rows.iloc[:, 0].sum()  # Somme de la colonne actuelle
        factors.append(numerator / denominator)
    return factors

PROPRIÉTÉ 4 la moyenne des ratios n'est pas égale au ratio des moyennes!!! En général, la moyenne des ratios n'a pas de sens statistique.

Exercice 5 Quand utiliser la méthode Chain Ladder classique et la méthode ALR?

Projection des sinistres futurs

Une fois les facteurs de développement calculés, on peut projeter les sinistres futurs pour les années où toutes les périodes de développement ne sont pas encore disponibles. Par exemple, pour estimer les sinistres à la période de développement 3 pour l'année 2021 :

 $$ C_{2021,3}=C_{2021,2}\times f_{2} $$ 

où  $ f_{2} $  est le facteur de développement de la période 2 à 3. Cette technique permet de compléter le triangle de développement et de calculer les provisions nécessaires pour les sinistres non réglés.

Exemple 7.3.1 (calcul des facteurs de développements des deux premières années) Les facteurs de développement sont calculés comme suit :

 $$ f_{1}=\frac{1500+1600}{1000+1100}=1.4545 $$ 

 $$ f_{2}=\frac{1800}{1500}=1.2 $$ 

Nous pouvons ensuite estimer les sinistres pour l'année 2020 au développement 3 :

 $$ C_{2020,3}=1600\times1.2=1920 $$ 

Et pour l'année 2021 aux développements 2 et 3 :

 $$ \begin{aligned}&C_{2021,2}=1200\times1.4545=1745.45\\&C_{2021,3}=1745.45\times1.2=2094.55\\ \end{aligned} $$ 

Pour projeter, on peut utiliser le code suivant :

def project_triangle(triangle, factors):
    triangle_projected = triangle.copy()
    for i, factor in enumerate(factors):
        # i indique l'année de développement
        for j in range(triangle_projected.shape[0]):
            # j indique l'année d'origine
            if pd.isna(triangle_projected.iloc[j, i + 1]):
                # pour toutes les valeurs manquantes, on prend le triangle projeté
                avec l'année de dev N-1 et on ajoute le facteur de développement
                triangle_projected.iloc[j, i + 1] = triangle_projected.iloc[j, i] * factor
    return triangle_projected

Listing 7.3 : Projection sur le triangle initial des facteurs choisis (factors)

Et on calcule pour obtenir un triangle complet!

# Projeter les valeurs pour obtenir un triangle complet
triangle_projected = triangle.copy()
for i in range(1, triangle.shape[1]):
    for j in range(i, triangle.shape[0]):
        if pd.isna(triangle_projected.iloc[j, i]):
            triangle_projected.iloc[j, i] = triangle_projected.iloc[j, i - 1] * factors_chain_ladder[i - 1]

# Calcul des résidus (différences entre valeurs observées et prédites)
reserve_IBNR = []
for i in range(1, triangle.shape[1]):
    observed = triangle.iloc[:, i]
    predicted = triangle.iloc[:, i - 1] * factors_chain_ladder[i - 1]
    reserve_IBNR.extend((observed - predicted).dropna())

print("Réserves IBNR calculées: ", )
display(pd.DataFrame(reserve_IBNR))

7.4 Comment calculer l'incertitude des réserve : la méthode de Mack

La méthode de Mack est une approche stochastique pour modéliser les triangles de développement des sinistres dans le cadre de la provisionnement en assurance. Elle étend la méthode déterministe du Chain-Ladder en y ajoutant une composante de variance, permettant ainsi d'estimer non seulement la meilleure prévision (best estimate), mais aussi l'incertitude associée aux prévisions.

Hypothèses du modèle de Mack

limits: La méthode de Mack repose sur des hypothèses fortes, notamment :

La méthode repose sur les hypothèses suivantes :

• L'indépendance des résidus entre les cellules du triangle.

• La proportionnalité entre la variance et les sinistres cumulés.

1. Les sinistres cumulés  $ C_{i,j} $  (pour l'année d'origine i et l'année de développement j) sont positifs et croissants avec j.

2. La relation entre deux colonnes consécutives s'exprime par un facteur de développement  $ f_{j} $ :

• La dépendance exclusive au triangle cumulé, sans prise en compte d'informations extérieures (inflation,...).

 $$ \mathbb{E}[C_{i,j+1}\mid C_{i,j}]=f_{j}\cdot C_{i,j}. $$ 

3. La variance conditionnelle des sinistres cumulés est proportionnelle aux sinistres de la période précédente :

 $$ \mathsf{V a r}(C_{i,j+1}\mid C_{i,j})=\sigma_{j}^{2}\cdot C_{i,j}. $$ 

4. Les erreurs  $ \varepsilon_{i,j} $  sont indépendantes entre elles et de moyenne nulle :

 $$ C_{i,j+1}=f_{j}\cdot C_{i,j}+\varepsilon_{i,j},\quad\mathsf{o}\grave{\mathsf{u}}\mathbb{E}[\varepsilon_{i,j}]=0. $$ 

Estimation des paramètres

Les paramètres  $ f_{j} $  (facteurs de développement) et  $ \sigma_{j}^{2} $  (variance) sont estimés à partir des données du triangle de développement :

 $$ f_{j}=\frac{\sum_{i}C_{i,j+1}}{\sum_{i}C_{i,j}}, $$ 

 $$ \sigma_{j}^{2}=\frac{\sum_{i}(C_{i,j+1}-f_{j}\cdot C_{i,j})^{2}}{\sum_{i}C_{i,j}}. $$ 

Listing 7.4 : Méthode de Mack

En python, cela donne :

def calculate_mack_variance(triangle, factors):
    variances = []
    for i in range(len(factors)):
        residuals = triangle.iloc[:, i + 1] - triangle.iloc[:, i] * factors[i]
        variance = np.nanmean(residuals ** 2)
        variances.append(variance)
    return variances

variances = calculate_mack_variance(triangle, factors)
print("Variances estimées: ")
display(pd.DataFrame(variances))

Ces hypothèses doivent être vérifiées ou adaptées selon les spécificités des données.

7.5 Le bootstrapping

Le bootstrapping est une méthode statistique de rééchantillonnage utilisée pour estimer des distributions d'échantillons, des variances ou des intervalles de confiance sans faire d'hypothèses strictes sur la distribution des données. En actuariat, le bootstrapping est particulièrement utilisé pour modéliser les incertitudes dans les triangles de développement des sinistres.

Principe du bootstrapping

Le principe du bootstrapping repose sur l'idée suivante :

1. Partir d'un échantillon de données observées, comme un triangle de développement.

2. Générer de nouveaux échantillons en rééchantillonnant avec remise les résidus ou les observations.

3. Calculer les statistiques ou les prévisions pour chaque nouvel échantillon.

4. Répéter le processus plusieurs fois pour obtenir une distribution empirique des résultats.

Étapes du bootstrapping appliqué à un triangle de développement

Voici les étapes typiques du bootstrapping pour un triangle de développement :

1. Modèle initial : Calibrer un modèle déterministe, comme la méthode Chain-Ladder, pour obtenir les facteurs de développement.

2. Calcul des résidus : Estimer les résidus, définis comme la différence entre les observations réelles et les prévisions du modèle.

3. Rééchantillonnage des résidus : Échantillonner avec remise les résidus pour générer des triangles simulés.

4. Projection des sinistres futurs : Appliquer les facteurs de développement au triangle simulé pour projeter les valeurs manquantes.

5. Répétition du processus : Répéter les étapes précédentes un grand nombre de fois (par exemple, 1 000 itérations).

6. Analyse des résultats : Calculer les statistiques d'intérêt (moyenne, écart-type, quantiles) à partir de la distribution des projections.

Voici une idée du code du boostrapping.

# Nombre de simulations
n_simulations = 1000
bootstrap_reserves = []

for _ in range(n_simulations):
    # Échantillonner les résidus avec remise
    sampled_residuals = np.random.choice(reserve_IBNR, size=len(reserve_IBNR), replace=True)

    # Créer un triangle bootstrap
    triangle_bootstrap = triangle.copy()
    k = 0
    for i in range(1, triangle.shape[1]):
        for j in range(i, triangle.shape[0]):
            if pd.isna(triangle_iloc[j, i]):
                predicted = triangle_bootstrap.iloc[j, i - 1] * factors_chain_ladder[i - 1]
                triangle_bootstrap.iloc[j, i] = predicted + sampled_residuals[k]
                k += 1

    # Calculer la réserve totale pour ce bootstrap
    reserve = triangle_bootstrap.iloc[:, 1:].sum().sum() - triangle.iloc[:, 1:].sum()
    (sum() .sum() .reserve.append(reserve))

Il faudrait ici ajouter à chaque tirage, un calcul par Chain ladder de la vision à l'ultime qui n'a pas été faite pour simplifier la lecture.

import plotly.graph_objects as go
import numpy as np

# Créer l'histogramme avec Plotly
fig = go.Figure()

# Ajouter l'histogramme des données
fig.add_trace(go.Histogram(
    x=bootstrap_reserves,
    nbinsx=30,
    marker_color='skyblue',
    opacity=0.7,
    name='Réserves bootstrapées')
)

# Ajouter une ligne verticale pour la moyenne
fig.add_trace(go.Scatter(
    x=[np.mean(bootstrap_reserves), np.mean(bootstrap_reserves)],
    y=[0, 100],
    mode='lines',
    line=dict(color='red', dash='dash'),
    name=f"Moyenne = {np.mean(bootstrap_reserves):.0f}"
)

# Mise en page du graphique
fig.update_layout(
    title="Distribution des Réserves Bootstrapées",
    xaxis_title="Réserves",
    yaxis_title="Fréquence",
    bargap=0.1,
    template="plotly_white"
)

Avantages et limites Avantages :

• Méthode non paramétrique : ne nécessite pas de connaître la distribution des données.

• Facilement applicable à une grande variété de modèles.

• Fournit des mesures d'incertitude utiles pour la gestion des risques.

Limits :

• Dépend fortement de la qualité des données de départ : les biais dans les données observées sont répliqués dans les échantillons bootstrappés.

• Le coût computationnel peut être élevé pour des échantillons très volumineux ou de nombreuses itéra-tions.

• Les hypothèses implicites, comme Listing 75d\penstaampidestrésigles, peuvent ne pas être satisfaites dans certains cas.

Listing 7.6 : graphique des résultats avec Plotly

# Afficher le graphique
fig.show()

On obtient avec les données RAA, le résultats suivants :

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl//44453948-142c-485c-8827-16372c660a37/markdown_63/imgs/img_in_chart_box_457_322_1036_691.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-01-11T18%3A43%3A02Z%2F-1%2F%2F2137265a058554d365c01f82e16b247d62cdefca22d0ae9646ba610151edc900" alt="Image" width="48%" /></div>


<div style="text-align: center;">Fig. 7.3: Résultat des réserves bootstrappées</div>


7.6 Exercices

Exercice 6 Je vous propose de travailler sur les donner Taylor-Ashe. Ces données sont très connus et sont souvent utilisés dans des articles.

import pandas as pd
import numpy as np

# Example de triangle de développement

# Création du DataFrame à partir des données extraites - Taylor Ashe 1978 ()
# https://www.researchgate.net/profile/Gary-Venter/publication/305980007/figure/tbl2/AS:666776325087251@1535983272841/Incremental-triangle-Taylor-Ashe-1983.png

data = {
    "Lag 0": [357848, 352118, 290507, 310608, 443160, 396132, 440832, 359480, 376686, 344014],
    "Lag 1": [766940, 884021, 1001799, 1108250, 693190, 937085, 847631, 1061648, 986608, None],
    "Lag 2": [610542, 933894, 926219, 776189, 991983, 847498, 1131398, 1443370, None],
    "Lag 3": [482940, 1183289, 1016654, 1562400, 769488, 805037, 1063269, None, None],
    "Lag 4": [527326, 445745, 750816, 272482, 504851, 705960, None, None, None, None],
    "Lag 5": [574398, 320996, 146923, 352053, 470639, None, None, None, None],
    "Lag 6": [146342, 527804, 495992, 286286, None, None, None, None, None],
    "Lag 7": [139950, 266172, 286405, None, None, None, None, None, None],
    "Lag 8": [227229, 425046, None, None, None, None, None, None],
    "Lag 9": [67948, None, None, None, None, None, None, None, None]
}

# Création du DataFrame

triangle = pd.DataFrame(data)

# Affichage du triangle

triangle

# Ajout de la variable "Année d'origine"

annees_origine = list(range(2000, 2000 + len(triangle)))
année de départ réelle

triangle["Année d'origine"] = annees_origine

# Réorganisation des colonnes pour placer "Année d'origine" en premier

triangle = triangle[["Année d'origine"] + [col for col in triangle.columns if col != "Année d'origine"]]

triangle.set_index("Année d'origine")

# mettre en cumulatif

display(triangle)

Powered by TCPDF (www.tcpdf.org)

8.1

Construction de tables par modèle de génération

qu'est ce qu'un modèle Age-période-Cohorte?

DÉFINITION 8.1.1 - MODÈLE PAR GÉNÉRATION - MODÈLE APC. Les modèles Âge-Période-Cohorte (APC) sont une classe de modèles de taux démographiques (mortalitémorbiditéfécondité ...) généralement observés sur une large tranche d'âge sur une période raisonnablement longue, et classés par âge et date de suivi ( période) et date de naissance (cohorte ou GÉNÉRATION).

PROPRIÉTÉ 1 On parle aussi de Modèles d'amélioration de la mortalité (Mortalité Improvement modelling [1, p.85]).

8.2 Diagramme de Lexis

Un diagramme de Lexis est un outil utilisé en démographie et en épidémiologie pour représenter des événements (naissances, décès, maladies, etc.) en fonction de l'âge (ordonnée) et du temps (abcisse). Le diagramme de Lexis permet de représenter une troisième information, l'année de naissance et de suivre les cohortes sur les diagonales.

Pour notre exemple, nous allons partir des données ___

Mise en forme

1 : Données insee 2024 avec par exemple les Survivants par sexe et âge – Séries depuis 1946 pour la France métropolitaine qui donne le nombre de survivantes pour 100 000 nées vivantes à chaque âge (âge en années révolues)

df = pd.read_excel(PATH + "/survivants.xlsx")

# renommer la colonne
df.rename(columns={"Unnamed: 0": "Année"}, inplace=True)

# Supprimer "an(s)" des noms de colonnes et convertir en entier
df.columns = [
    int(col.split([0])) if "an" in col.lower() and col.split([0].isdigit() else col for col in df.columns]
    # une ligne par année et âge
]

df_long = df.melt(id_vars="Année"),
    var_name="Âge",
    value_name="Population")

# Convertir les colonnes "Âge" et "Population" en numérique
df_long["Âge"] = pd.to_numeric(df_long["Âge"], errors='coerce')
df_long["Population"] = pd.to_numeric(df_long["Population"], errors='coerce')

# Filter les valeurs non nulles
df_long = df_long.dropna()

# Calcul de l'année (année de naissance + âge)
df_long["Année de naissance"] = df_long["Année"] - df_long["Âge"]
display(df_long)


<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Année</td><td style='text-align: center;'>0 an</td><td style='text-align: center;'>1 an</td><td style='text-align: center;'>2 ans</td><td style='text-align: center;'>3 ans</td><td style='text-align: center;'>4 ans</td><td style='text-align: center;'>5 ans</td></tr><tr><td style='text-align: center;'>1946</td><td style='text-align: center;'>100 000</td><td style='text-align: center;'>93 269</td><td style='text-align: center;'>92 350</td><td style='text-align: center;'>91 993</td><td style='text-align: center;'>91 761</td><td style='text-align: center;'>91 570</td></tr><tr><td style='text-align: center;'>1947</td><td style='text-align: center;'>100 000</td><td style='text-align: center;'>93 849</td><td style='text-align: center;'>93 152</td><td style='text-align: center;'>92 892</td><td style='text-align: center;'>92 733</td><td style='text-align: center;'>92 608</td></tr><tr><td style='text-align: center;'>1948</td><td style='text-align: center;'>100 000</td><td style='text-align: center;'>95 198</td><td style='text-align: center;'>94 698</td><td style='text-align: center;'>94 493</td><td style='text-align: center;'>94 360</td><td style='text-align: center;'>94 263</td></tr><tr><td style='text-align: center;'>1949</td><td style='text-align: center;'>100 000</td><td style='text-align: center;'>94 784</td><td style='text-align: center;'>94 183</td><td style='text-align: center;'>93 955</td><td style='text-align: center;'>93 816</td><td style='text-align: center;'>93 712</td></tr></table>

<div style="text-align: center;">Tab. 8.1: Table FM -Femme (extrait) des survivants à l'âge révolu.</div>


<div style="text-align: center;">Fig. 8.1 : Evolution d'une population fictive de 100 000 femmes - INSEE 2024</div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl//44453948-142c-485c-8827-16372c660a37/markdown_67/imgs/img_in_chart_box_448_165_1050_345.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-01-11T18%3A43%3A03Z%2F-1%2F%2Ff775f31bd24b5328541a8eb8f286078345106215b01cd07ddc6e981fdb5c05db" alt="Image" width="50%" /></div>


<div style="text-align: center;">Fig. 8.2 : Evolution du taux de mortalité des femmes - INSEE 2024</div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl//44453948-142c-485c-8827-16372c660a37/markdown_67/imgs/img_in_chart_box_448_390_1050_570.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-01-11T18%3A43%3A03Z%2F-1%2F%2F40b75450cb1aa6aa357dfc350d9bfbc7eea15ed1db7a16cfa12dc871e9935442" alt="Image" width="50%" /></div>


Graphique

On peut alors grapher l'évolution de la population :

import plotly.graph_objects as go
import numpy as np

fig = go.Figure(data=go.Contour(
    z=df_long["Population"],
    x=df_long["Année"], # horizontal axis
    y=(df_long["Âge"]), # vertical axis
    line_smoothing=0
))

#fig.add_trace(go.Heatmap(z=df_long["Population"], showscale=False, zsmooth='best'))
#fig.add_trace(go.Heatmap(z=df_long["Population"], line_smoothing=0))

fig.show()

Taux de mortalité

On calcule le taux de mortalité  $ q_{x} = \frac{mortalit}{survivants} $ . Pour le grapher, on passe au logarithme car le taux de mortalité peut varier fortement.

# Trier par Année et Âge

df_long = df_long.sort_values(by=["Année", "Âge"])

# Calculer le nombre de morts entre âges consécutifs
df_long["Morts"] = df_long.groupby("Année").[("Population").diff(-1)
df_long["Qx"] = df_long["Morts"] / df_long["Population"]

df_long

import plotly.graph_objects as go
import numpy as np

fig = go.Figure(data=
go.Contour(
z=np.log(df_long["Qx"]),
x=df_long["Année"], # horizontal axis
y=df_long["Âge"]), # vertical axis
line_smoothing=0
)

# Fig. add_trace(go.Heatmap(z=df_long["Population"], showscale=False, zsmooth='best'))
fig.add_trace(go.Heatmap(z=df_long["Population"], line_smoothing=0))

fig.show()

8.3 Modèle de Lee-Carter

Dans leur article $ ^{2} $ , Lee et Carter introduisent un modèle permettant de tenir compte à la fois de l'âge et de l'année calendaire.

DÉFINITION 8.3.1 - MODÈLE DE LEE-CARTER. (ex : TGH 05 et TGF 05) Il s'agit d'une méthode d'extrapolation des tendances passées. La modélisation retenue pour le taux instantané de mortalité est la suivante :

 $$ l n(\mu_{x,t})=\alpha_{x}+\beta_{x}\kappa_{t} $$ 

Pour le programmer, on utilise une décomposition en valeur singulière (comme par exemple pour l'Analyse en Composantes Principales -ACP).

La décomposition en valeurs singulières (SVD) est une technique mathématique utilisée pour décomposer une matrice rectangulaire en trois matrices spécifiques. Soit une matrice  $ A \in R^{m \times n} $ , la SVD permet de la décomposer comme suit :

 $$ A=U\Sigma V^{T} $$ 

où:

•  $ U \in R^{m \times m} $  est une matrice orthogonale (ses colonnes forment une base orthonormée dans  $ R^{m} $ ).

•  $ V \in R^{n \times n} $  est une matrice orthogonale (ses colonnes forment une base orthonormée dans  $ R^{n} $ ).

• Σ ∈ R^{m×n} est une matrice diagonale (ou semi-diagonale) contenant les valeurs singulières de A, notées σ₁, σ₂, … (telles que σ₁ ≥ σ₂ ≥ … ≥ 0).

La décomposition  $ A = U\Sigma V^{T} $  peut être interprétée comme une série de transformations linéaires :

1. $V^{T}$ applique une rotation dans l'espace source (les colonnes de $V$ sont les vecteurs propres de $A^{T}A$).

2. ∑ étire ou contracte selon des directions spécifiques données par les valeurs singulières.

3. U applique une rotation dans l'espace cible (les colonnes de U sont les vecteurs propres de  $ AA^{T} $ ).

• Les valeurs singulières de A sont les racines carrées des valeurs propres de  $ A^{T}A $  ou  $ AA^{T} $ .

• Si A est une matrice de rang r, alors  $ \Sigma $  contient exactement r valeurs singulières non nulles.

• La SVD est particulièrement utile pour la résolution de systèmes linéaires mal conditionnés comme Lee-Carter.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd
# Étape 1 : Calcul des taux de mortalité
df_long['Taux_mortalité'] = df_long['Morts'] / df_long['Population']
# Étape 2 : Transformation en matrice des taux de mortalité
# Pivot pour avoir une matrice âge x année
df_matrix = df_long.pivot(index="Âge", columns="Année", values="Taux_mortalité")
log_mortality = np.log(df_matrix)
# Étape 3 : Décomposition en valeurs singulières (SVD)
# log_mortality = alpha_x + beta_x * kappa_t
log_mortality = log_mortality.fillna() # Remplir les valeurs manquantes par 0
alpha_x = log_mortality.mean(axis=1) # Moyenne par âge (effet moyen)
# graphique

plt.figure(figsize=(12, 6))
plt.plot(alpha_x, label="alpha(x)")
plt.xlabel("Age")
plt.ylabel("Structure moyenne de mortalité par age (Lee-Carter)")
plt.show()

# Centrer la matrice pour retirer alpha_x
centered_log_mortality = log_mortality.sub(alpha_x, axis=0)

# Décomposer avec SVD
U, S, Vt = svd(centered_log_mortality, full_matrices=False)
beta_x = U[0]  # Premier vecteur singulier (par âge)
kappa_t = S[0] * Vt[0, :]  # Premier vecteur singulier (par temps)

# Étape 4 : Résultats et visualisation
plt.figure(figsize=(12, 6))
plt.plot(df_matrix.columns, kappa_t, label="Indice temporel (kappa_t)")
plt.xlabel("Année")
plt.ylabel("Kappa_t")
plt.title("Indice temporel de mortalité (Lee-Carter)")
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(beta_x, label="Indice d'impact par age (beta_x)")
plt.xlabel("Année")
plt.ylabel("beta_x")
plt.title("Structure par âge de l'amélioration (Lee-Carter)")

Résultat sur les données françaises

On peut ensuite grapher les trois paramètres de Lee-Carter.

<div style="text-align: center;">Fig. 8.3 : Modèle de Lee-Carter sur les données françaises 2024 - Femmes à partir des tables de mortalité INSEE. Pour  $ \alpha_{x} $  et  $ \beta_{x} $ , l'âge x varie de 0 à 99 ans, ce qui explique au dessus de 100 ans un saut pour  $ \alpha_{x} $ . Pour  $ k_{t} $ , 0 représente 1947 à 2024.</div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl//44453948-142c-485c-8827-16372c660a37/markdown_69/imgs/img_in_chart_box_468_813_1024_1119.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-01-11T18%3A43%3A04Z%2F-1%2F%2Fcea3642ab2e18334078c1f90318b1a69b78cdffc0e2a821db46f50a41d652c49" alt="Image" width="46%" /></div>


8.4 Introduction à SQL : langage d'accès aux données

SQL (Structured Query Language) est un langage standard pour interagir avec des bases de données relationnelles. En tant qu'actuaire, SQL vous permet d'extraire les données nécessaires à vos analyses : sinistres, primes, données client, etc.

Exemple d'utilisation de SQL en Python (avec pandas)

• On suppose que vous avez accès à une base de données via SQLite, PostgreSQL ou BigQuery.

• Voici un exemple simple avec SQLite :

import sqlite3
import pandas as pd

# Connexion à une base de données SQLite locale
conn = sqlite3.connect('assurance.db')

# Écriture 'dune requête SQL
query = '...
SELECT client_id, age, prime_annuelle, statut_contrat FROM contrats 
WHERE statut_contrat = 'actif'
AND age > 38
ORDER BY prime_annuelle DESC 
LIMIT 10
...

# Exécution de la requête et chargement dans un DataFrame
df = pd.read_sql_query(query, conn)
print(df.head())

Le code pour faire une demande SQL sur Google Big Query est similaire (les bibliothèques sont différentes mais le code est le même).

Principales commandes SQL à connaître

· SELECT : sélection de colonnes

FROM : table source

• WHERE : conditions

· GROUP BY, HAVING : agrégats

• JOIN : jointures entre tables

· ORDER BY, LIMIT

À retenir

• SQL est le langage incontournable pour extraire les données.

• Python permet d'exécuter des requêtes SQL avec des bibliothèques comme sqlite3, sqlalchemy, pandas-gbq ou bigquery.

• Vous pouvez écrire vos requêtes dans des chaînes multi-lignes " "

Les pipelines : organiser un traitement structuré des données

Un pipeline est une chaîne d'étapes successives appliquées aux données. En actuariat, on part souvent de données brutes qu'il faut transformer, nettoyer, normaliser, puis modéliser.

Pourquoi utiliser un pipeline?

• Reproductibilité : même traitement à chaque exécution.

• Lisibleité : chaque étape est explicitée.

• Maintenance : plus facile à tester et à faire évoluer.

Exemple avec scikit-learn

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
import pandas as pd

# Exempt de données
df = pd.DataFrame({
    'age': [25, 40, 35, None],
    'prime': [450.0, 900.0, None, 700.0]
})
X = df[['age']]
y = df['prime']

# Pipeline de traitement pour la variable "age"
pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'),
    ('scaler', StandardScaler)),
    ('regressor', LinearRegression()
])

# Entrainement du pipeline
pipeline.fit(X, y)

# Prédiction
print(pipeline.predict([[30]])) # Prédit la prime pour un client de 30 ans

Structure d'un pipeline typique

• Imputer : remplace les valeurs manquantes.

· Scaler : normalise les données.

• Modèle : applique une régression ou un modèle prédictif.

SYNTHÈSE 8.4.1 - PIPELINE. • Les pipelines sont essentiels pour enchaîner ne proprement plusieurs étapes de traitement.

• Ils s'intègrent parfaitement dans un projet de modélisation (tarification, scoring...).

• Ils peuvent inclure du prétraitement, de la sélection de variables et des modèles.

• Les pipelines sont essentiels pour enchaîner proprement plusieurs étapes de traitement.

• Ils s'intègrent parfaitement dans un projet de modélisation (tarification, scoring...).

• Ils peuvent inclure du prétraitement, de la sélection de variables et des modèles.

• un des avantages du pipeline est de définir des pipelines parallèles qui acceptent les mêmes données (ex : ACP et AutoEncoder) afin de modifier rapidement les résultats et de passer de l'un à l'autre.

Exercice 7 Faire un modèle de Pipeline sur Lee-Carter

# Modèles censurés

Application à la base Haberman Kaplan et Meier ont appliqué leur modèle sur la base Haberman $ ^{1} $  des cancers du sein pour diagnostiquer la survie en cas de ganglions lymphatiques.

:

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl//44453948-142c-485c-8827-16372c660a37/markdown_72/imgs/img_in_chart_box_256_383_628_647.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-01-11T18%3A43%3A05Z%2F-1%2F%2Fc200eb97025949c01c164a855820ed877ce5747005f12fea66d74a65c0f45944" alt="Image" width="31%" /></div>


Nous pouvons alors calculer cet estimateur sur deux populations de la base Haberman, celle ayant un ganglion lymphatique infecté et celle sans. $ ^{2} $ 

## creation d'un nouvel objet kmf
km_2 = KaplanMeierFitter()

## creation de deux cohortes : ceux avec au moins un ganglion infecté et ceux sans ganglion lymphatique infecté
groups = data['Nb_pos_detected']
i1 = (groups >= 1)
i2 = (groups < 1)

## fit the model for 1st cohort
km_2.fit(T[i1], E[i1], label='au moins un ganglion infecté')
a1 = km_2.plot()

## fit the model for 2nd cohort
km_2.fit(T[i2], E[i2], label='sans ganglion lymphatique infecté détecté')
km_2.plot(ax=a1)

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl//44453948-142c-485c-8827-16372c660a37/markdown_72/imgs/img_in_chart_box_256_1028_628_1291.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-01-11T18%3A43%3A05Z%2F-1%2F%2F55291b0bcc21431ec833306cd136b9dbcc3a275d91e12215370ecf8c2783716a" alt="Image" width="31%" /></div>


<div style="text-align: center;">Exercice 8 Un effectif de 510 assurés de même âge 60 ans est observé pendant 3 6 mois. Des censures ( sortie du périmètre assuré autre que par cause de décès) peuvent intervenir à la fin de chaque mois. Les décès sont enregistrés à la fin de chaque mois et on fait l'hypothèse qu'ils sont intervenus au milieu du mois considéré.</div>


• une censure intervient à la fin des mois 20 et 34

• deux censures interviennent à la fin des mois 4 et 24

• cinq censures interviennent à la fin des mois 11 et 16

• six censures interviennent à la fin des mois 8, 18 et 3 0

9

1: Cette base première utilisée pour les modèles censurées, contient les résultats d'une étude menée entre 1958 et 1970 à l'université de Chicago sur la survie des personnes ayant eu un cancer du sein.

Fig. 9.1 : Courbe de Kaplan-Meier appliquée à la totalité de la base Haberman. Sur le graphique, les petits traits verticaux indiquent ces censures (au niveau de 35 par exemple). Si aucune troncature ou censure n'intervient, la courbe de Kaplan-Meier est équivalente à la fonction de survie.

2 : Il existe une librairie Python, lifelines spécialisé dans les modèles de survie. https://lifelines.readthedocs.io/en/latest/. En utilisant la base Haberman en libre-accès source Kaggle; lien original de la base

Les variables de la base contiennent :

• Age du patient au moment de l'opération l'année où le patient a été opéré

• Le nombre de ganglions lymphatiques infectés (axillary nodes)

• Le statut du patient (1 = le patient a survécu au delà de 5 ans; 2 = le patient est mort dans les 5 ans)

<div style="text-align: center;">Fig. 9.2 : Courbe de Kaplan-Meier appliquée à deux populations de la base Haberman, celle avec au moins un ganglion infecté et celle sans ganglion infecté. En ajoutant l'intervalle de confiance, on peut voir que la population avec un ganglion lymphatique infecté a une espérance de vie significativement réduite à partir de 50 mois</div>


• 1 décès mensuel est enregistré aux mois 5, 7, 18, 24, 31 et 34ème mois.

• 2 décès mensuels sont enregistrés aux 3 ème et au 19 ème mois

• 3 décès sont enregistrés au cours des mois 11 et 15

Donner la valeur de l'estimateur de Kaplan Meier pour (produit des probabilités mensuelles de survie) $ _{m} $   $ P_{60} $  pour m = 12, 24 et 3 6 mois

• évaluer la variance des estimations calculées précédemment par la formule de Greenwood.

• En supposant que le nombre d'estimations est suffisant, construire des intervalles de confiance à 70 % en supposant que la variable  $ _{m}P_{60} $  est gaussienne.

La régression de Cox (modèle à risque proportionnel)

La régression de Cox conserve l'estimateur de Kaplan-Meier mais l'applique au cadre d'une régression afin de déterminer les facteurs explicatifs de survie. Le modèle de Cox est basé sur l'hypothèse des risques proportionnels, hypothèse forte qu'il convient de vérifier préalablement à l'utilisation du modèle

DÉFINITION 9.0.1 - HYPOTHÈSE DES RISQUES PROPORTIONNELS. Pour chaque variable (appelée aussi covariable)  $ b_{i}(x_{i}) $ , on teste alors si son effet est indépendant du temps. Cet effet doit être constant.

DÉFINITION 9.0.2 - MODÈLE DE COX. La formule du modèle de Cox est donnée comme suit :

 $$ h(t|x)=b_{0}(t)exp\sum_{i=1}^{n}b_{i}(x_{i}) $$ 

Le modèle fonctionne de telle sorte que le log-risque d'un sujet individuel est une fonction linéaire de ses variables statiques (équivalent à cela à un modèle GLM avec fonction lien log sur la fonction de risque  $ h(t) $ ) et une fonction de risque de base au niveau de la population qui change avec le temps. Le modèle est estimé par vraisemblance partielle.

• t représente le temps de survie, le risque peut varier dans le temps.

•  $ h(t) $  est la fonction de risque déterminée par un ensemble de n variables  $ (x_{1}, x_{2}, \ldots, x_{n}) $ .

•  $ b_{0}(t) $  est la fonction de risque de base et est définie comme la probabilité de vivre l'événement d'intérêt lorsque toutes les autres variables sont égales à zéro. C'est la seule composante dépendante du temps dans le modèle. Le modèle ne fait aucune hypothèse sur la fonction de risque de base et prend une forme paramétrique pour l'effet des variables sur le risque.

•  $ \sum_{i=1}^{n}b_{i}(x_{i}) $  est le risque partiel, un facteur scalaire invariant dans le temps qui ne fait qu'augmenter ou diminuer le risque de base. Il est similaire à l'ordonnée à l'origine dans la régression ordinaire. Les variables ou les coefficients de régression x donnent le changement proportionnel auquel on peut s'attendre dans le risque.

• les coefficients  $ (b_{1}, b_{2}, \ldots, b_{n}) $  mesurent l'effet des variables. si  $ b_{i} $  est positif, cela indique une dégradation du risque pour la patient.

Application à la base Haberman des cancers Le modèle de cox est un modèle que l'on trouve dans le package dédié au modèle vie, lifelines.

# Creation du Modele
import lifelines # package lifelines qui.contient les modèles vie
cph = CoxPHFitter()

# Estimation du modèle aux données
cph.fit(data, 'Age', event_col='Surv')

# Impression du résultat
cph.print_summary()


<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'></td><td style='text-align: center;'>coef</td><td style='text-align: center;'>exp(coef)</td><td style='text-align: center;'>se(coef)</td><td style='text-align: center;'>exp(coef) lower 95%</td><td style='text-align: center;'>exp(coef) upper 95%</td><td style='text-align: center;'>z</td><td style='text-align: center;'>p</td><td style='text-align: center;'>-log2(p)</td></tr><tr><td style='text-align: center;'>Operation year</td><td style='text-align: center;'>-0.02</td><td style='text-align: center;'>0.98</td><td style='text-align: center;'>0.02</td><td style='text-align: center;'>0.94</td><td style='text-align: center;'>1.01</td><td style='text-align: center;'>-1.39</td><td style='text-align: center;'>0.16</td><td style='text-align: center;'>2.61</td></tr><tr><td style='text-align: center;'>Nb pos detected</td><td style='text-align: center;'>0.01</td><td style='text-align: center;'>1.01</td><td style='text-align: center;'>0.01</td><td style='text-align: center;'>1.00</td><td style='text-align: center;'>1.03</td><td style='text-align: center;'>1.70</td><td style='text-align: center;'>0.09</td><td style='text-align: center;'>3.47</td></tr></table>

Les statistiques récapitulatives ci-dessus indiquent l'importance des cova-riables dans la prédiction du risque de survie.

• l'année de l'opération est négative, indiquant l'amélioration de l'espérance de survie récente (possiblement par un geste opératoire mieux maîtrisé)

• le nombre de ganglions infecté est positif, comme nous l'avions vu avec le modèle de Kaplan-Meier, indiquant une détérioration de l'espérance de vie.

L'intervalle de confiance est néanmoins assez large.

9.1 Modèle de Cox vs Modèle Linéaire Généralisé (GLM)

Avantages du modèle de Cox pour les durées de vie :

• Permet une analyse robuste des durées de vie ou de résiliation.

• Ne nécessite pas de spécification de la fonction de base  $  h_{0}(t)  $ 

Limits en assurance :

• Pas directement utilisable pour prédire un coût ou une prime.

• Mal adapté si la variable cible est un comptage ou une valeur monétaire.

• Difficulté à intégrer des variables catégorielles nombreuses ou à forte cardinalité.

Le modèle linéaire généralisé (GLM) est une généralisation de la régression linéaire, adaptée aux distributions de la famille exponentielle (Poisson, Gamma, Binomiale, etc.).

Avantages du GLM :

• Permet de modéliser une fréquence (nombre de sinistres, modèle Poisson) ou un coût moyen (modèle Gamma, Inverse-Gaussien).

• Facile à interpréter (comme Cox), mais directement applicable à la tarification.

• Compatible avec une structure additive des effets (tarification modulaire).

• Supporte des variables catégorielles nombreuses (via effets fixes ou dummy).

Exercice 9 Tester des modeles de survie basés sur les réseaux de Neurone (pyCox, DeepSurrv ou torchtuples par exemple). Cela peut correspondre à un devoir.

Powered by TCPDF (www.tcpdf.org)

10.1 qu'est ce qu'un modèle catastrophe

catastrophe modeling main steps

DÉFINITION 10.1.1 - MODÈLES DE CATASTROPHE. Les modèles de catastrophe combinent des représentations mathématiques des schémas d'occurrence et des caractéristiques physiques des périls (appelées Hazard Modeling), ainsi que des informations qualitatives et quantitatives sur le risque assuré (appelées Exposition) afin de générer des estimations de pertes permettant de calculer les Dommages.

Même si leur conception est fondamentalement différente, à un niveau très général, tous les modèles de catastrophe suivent les mêmes étapes.

PROPRIÉTÉ 5 - PRINCIPALES ÉTAPES DE MODÉLISATION. Les modèles de catastrophe sont structurés autour des étapes suivantes (voir fig. 10.1):

1. Le module d'aléa, dont l'objectif est de résumer la variabilité naturelle d'un péril donné dans un ensemble limité d'événements stochastiques. Il définit les caractéristiques dommageables de l'événement (magnitude, ...) ainsi que les conditions de site ayant un impact sur l'événement (type de sol, ...).

2. Le module de vulnérabilité, dont l'objectif est d'évaluer correctement les dommages causés par les phénomènes physiques, compte tenu des caractéristiques du risque analysé.

3. Le module financier, dont l'objectif est d'appliquer les différentes conditions d'assurance (franchises, limites d'indemnisation, ...) afin d'évaluer les dommages assurés et de fournir des résultats standardisés destinés à accompagner le processus de décision.

Cette architecture est commune à l'ensemble des modèles de catastrophe. Toutefois, la manière de construire ces modules repose sur des hypothèses qui peuvent varier, principalement selon le péril. De plus, ces hypothèses peuvent même différer entre les sociétés de modélisation pour un même péril. Ces trois modules sont détaillés dans les sections suivantes.

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl//44453948-142c-485c-8827-16372c660a37/markdown_76/imgs/img_in_image_box_139_1286_747_1523.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-01-11T18%3A43%3A06Z%2F-1%2F%2F8e24484d8b41a116e47eabaabd6a065283fba77bc518ab949217d92b65295b75" alt="Image" width="51%" /></div>


Fig. 10.1 : Structure générale des modèles de catastrophe

10.2 les modèles OpenSource

Beaucoup de logiciels catastrophe sont payantes (RMS, cybercube,...). Mais on peut aussi utiliser des plateformes opensource.

Option 1: Hurricanehist

hurricanehist, fichier sur CASACTDatasets.

Option 2 : OASIS

OASIS (Open Access Single Impact Simulation) est un environnement open-source pour la modélisation des risques catastrophe, souvent utilisé pour quantifier l'impact de phénomènes naturels (comme les ouragans, tremblements de terre, etc.) sur les portefeuilles d'assurance.

en cas de devoir sur Oasis, merci de me four-

nir le fichier ipynb car sinon je ne pourrai

voir le résultat

configuration d'OASIS Oasis ne fonctionne que sur MacOS et Linux mais son installation se révèle compliqué (personnellement je n'ai pas réussi à l'installer sur mon ordinateur).

pip install oasislmf

OASIS

# 1. Installation (dans un environnement Python)
# pip install oasislmf

import oasislmf
from oasislmf.cli import main as oasis_cli

# 2. Générer les fichiers 'dexposition (ex : transform exposure data vers format OED)
# Ici, on suppose que vous disposez 'dun dossier "input_data" avec vos fichiers exposure/raw
# et que vous souhaitez produire les fichiers OED.
# Note : la version CLI est utilisée. Vous pouvez aussi appeler directement les modules Python.

!oasislmf model generate-exposure-pre-analysis \
--config path/to/config.yaml \
--input-dir input_data \
--output-dir pre_analysis_out

# 3. Générer les clés (lookup keys) nécessaires au modèle
!oasislmf model generate-keys \
--model-settings path/to/model_settings.json \
--exposure-dir pre_analysis_out \
--output-dir keys_out

# 4. Générer les fichiers Oasis (input CSVs de type "oasis" files : locations, accounts, etc.)
!oasislmf model generate-oasis-files \
--model-settings path/to/model_settings.json \
--exposure-dir pre_analysis_out \
--keys-dir keys_out \
--output-dir oasis_files_out

# 5. Exécuter le modèle pour calculer les pertes (GroundUp Losses, etc.)
!oasislmf model run \
--model-settings path/to/model_settings.json \
--oasis-files-dir oasis_files_out \
--output-dir losses_out

# 6. Résumé ou postanalyse
# Par exemple, lire un fichier JSON "exposure_summary_report."json ou un fichier résultats .
import json
with open("losses_out/exposure_summary_report.json", "r") as f:
    summary = json.load(f)
print(summary)

• Le cheminement suit les étapes d'un modèle catastrophe : préparation exposition → lookup clés → génération fichiers d'entrée → exécution simulation → résultats.

• Le script CLI peut être complété / automatisé via des appels Python dans un notebook, ou via l'API oasislmf.api_client. $ ^{1} $ 

• Les fichiers de sortie peuvent ensuite être utilisés pour des analyses portefeuilles, des tables de retour (PML, AAL) ou intégrés dans une solution tarification.

Option 3 : CLIMADA

Climada est une plateforme moins adapté aux assureurs (moins d'information sur les portefeuilles) mais qui s'installe facilement et propose de nombreuses applications.

Pour installer Climada : Aller dans le terminal :

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl//44453948-142c-485c-8827-16372c660a37/markdown_78/imgs/img_in_image_box_215_585_671_875.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-01-11T18%3A43%3A07Z%2F-1%2F%2F14ca04dc60924c59548dee8abf05a89dc8d76c826c463edc45cf319266713c8d" alt="Image" width="38%" /></div>


sélectionner le noyau (ici noyau que j'appelle oasis, puis sur mamba (un équivalent de conda), j'installe climada :

conda create -n oasis python=3.9 -y
conda activate oasis
pip install mamba
mamba install -c conda-forge climada

Indispensable et bien fait : la formation en 10 mn de Climada : ici.

10.3 quelques mathématiques catastrophe

Cela ne remplace pas qui ne remplace pas le cours de mathématique des catastrophes, réassurance et Théorie des Valeurs Extrêmes.

Nous pouvons calculer des COURBES EP :

DÉFINITION 10.3.1 - COURBES DE PROBABILITÉ DE DÉPASSEMENT. Distributions cumulées qui indiquent la probabilité que les pertes annuelles dépassent un certain montant, qu'elles proviennent d'un ou de plusieurs événements :

• La courbe de PROBABILITÉ DE DÉPASSEMENT PAR OCCURRENCE (ou COURBE OEP) est la distribution cumulée du plus grand événement de l'année.

• La courbe de PROBABILITÉ DE DÉPASSEMENT AGRÉGÉE (ou COURBE AEP) est la distribution cumulée des pertes agrégées sur l'année.

La plupart des lois des OEP suivent des lois de Pareto. Pour simplifier, nous présentons la distribution de Pareto stricte, également appelée Pareto I :

DÉFINITION 10.3.1 - DISTRIBUTION DE PARETO I. Une distribution de Pareto de type I possède une fonction de densité et une fonction de répartition (CDF) données par

 $$ f(x)=\frac{\alpha,u^{\alpha}}{x^{\alpha+1}}\qquad\mathsf{et}\qquad F(x)=1-\left(\frac{x}{u}\right)^{-\alpha},\qquad\mathsf{pour}x\geq u $$ 

Le paramètre u est appelé seuil (threshold) et  $ \alpha $  détermine le comporte- ment de la queue de distribution; les deux paramètres sont strictement positifs.

Pourquoi le modèle de Pareto est-il si populaire parmi les réassureurs? La propriété la plus utile du modèle de queue de Pareto est l'invariance des paramètres lorsqu'on modélise les queues extrêmes [2]:

PROPRIÉTÉ 6 - INVARIANCE DU PARAMÈTRE. La moyenne conditionnelle au-dessus d'un seuil est proportionnelle à ce seuil et ne dépend pas du paramètre d'échelle u :

 $$ \mathbb{E}(X\mid X>u^{\prime})=\frac{\alpha,u^{\prime}}{\alpha-1},\qquad\alpha>1 $$ 

 $$ \dot{o}\ddot{u}\;u^{\prime}\geq u. $$ 

En d'autres termes, lorsque l'on modélise des pertes plus extrêmes, le modèle semble, d'une certaine manière, oublier le seuil initial u. Grâce à cette propriété, il n'est pas nécessaire de connaître exactement le point à partir duquel commence la queue : il suffit de se situer au-dessus. Évidemment, si l'on choisit un seuil trop élevé, il devient difficile de modéliser la queue en raison du nombre limité de sinistres.

Grâce à cette propriété, si deux portefeuilles reposent sur le même portefeuille sous-jacent, ils auront le même  $ \alpha $ . Ce type de comparaison des queues est extrêmement utile en pratique de réassurance : une valeur typique de  $ \alpha $  est associée à un potentiel de perte particulier, comme l'illustre le tableau ??.


<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Risque</td><td style='text-align: center;'>$ \alpha $</td></tr><tr><td style='text-align: center;'>Séisme - tempête</td><td style='text-align: center;'>1</td></tr><tr><td style='text-align: center;'>Incendie</td><td style='text-align: center;'>2</td></tr><tr><td style='text-align: center;'>Incendie industriel</td><td style='text-align: center;'>1.5</td></tr><tr><td style='text-align: center;'>Responsabilité automobile</td><td style='text-align: center;'>2.5</td></tr><tr><td style='text-align: center;'>Responsabilité générale</td><td style='text-align: center;'>1.8</td></tr><tr><td style='text-align: center;'>Accidents du travail</td><td style='text-align: center;'>2</td></tr></table>

10.4 simulation deux centenaires

Exercice 10 Je prends ma ville de naissance et je modélise un type de catastrophe qui la touche. Avec une modélisation de Pareto, je calcule la perte centenaire et le sinistre centenaire. Je les compare de façon pertinente.

La documentation OasisLMF précise :

• La préparation des données d'exposition (geocoding, agrégation, mapping vers les périls) peut s'avérer complexe, notamment quand les niveaux géographiques sont variés (commune, département, région).

• Le framework nécessite des composants système (ex. 'libspatialindex') pour certaines fonctionnalités de lookup géospatial.

- Même si open source, un usage industriel nécessite une configuration sérieuse (performance, mémoire, choix de modèle, validation).

10.5 Implémentation en Climada

# from 10 mn to learn how to use climada
from climada.hazard import Hazard
from climada.util import HAZ_DEMO_H5

haz = Hazard.from_hdf5(HAZ_DEMO_H5)

# to hide the warnings
import warnings

warnings.filterwarnings("ignore")

print(
    f"The hazard object contains {haz.intensity.shape[0]} events. \n"
    f"The maximal intensity contained in the Hazard object is {haz.intensity.max()}:.2f} {haz.units}.\n"
    f"The first event was observed in a time series of {int(1/haz.frequency[0])}{haz.frequency_unit[2:]}:s, \n"
    f"which is why CLIMADA estimates an annual probability of {haz.frequency[0]:.4f} for the occurrence of this event."
)

The hazard object contains 216 events.

The maximal intensity contained in the Hazard object is 72.75 m/s.

The first event was observed in a time series of 185 years,

which is why CLIMADA estimates an annual probability of 0.0054 for the occurrence of this event.

haz.plot_intensity(0, figsize=(6, 6));

Objects Python et fonctionnement de haz.plot_intensity()

En Python, la majorité des bibliothèques scientifiques (telles que NumPy, pandas, matplotlib ou climada) reposent sur une approche dite orien-

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl//44453948-142c-485c-8827-16372c660a37/markdown_80/imgs/img_in_image_box_218_1164_666_1518.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-01-11T18%3A43%3A08Z%2F-1%2F%2F38a351ee006c256b377266b9cfc5b9ad74f6f7fcd82dc11b1e84e24f0b258aef" alt="Image" width="37%" /></div>


82.5°W 81°W 79.5°W 78°W 76.5°W

<div style="text-align: center;">Fig. 10.3 : tracer les cyclones historiques</div>


tée objet. Cette manière de structurer le code permet de regrouper, au sein d'une même entité appelée objet, à la fois des données et les fonctions permettant d'agir sur ces données.

Définition : objet Python

DÉFINITION 10.5.1 - OBJET PYTHON. Un objet est une structure de données qui contient :

• des attributs : informations internes stockées dans l'objet (valeurs, tableaux, paramètres, métadonnées, etc.);

• des méthodes : fonctions associées à l'objet et permettant d'effectuer certaines opérations sur ces données.

Exemple 10.5.1 l'objet haz dans climada

Dans la bibliothèque climada, l'objet haz est une instance de la classe Hazard. Il représente un aléa naturel (tempête, cyclone, séisme, etc.). Cet objet regroupe de manière cohérente toutes les informations nécessaires à la description de cet aléa, notamment :

• la grille géographique utilisée ;

• l'intensité du phénomène en chaque point de la grille;

• les paramètres physiques décrivant l'aléa (vents, accélération au sol, pression, etc.);

• les éventuels ensembles d'événements stochastiques.

Autrement dit, haz n'est pas une simple variable, mais une structure complète permettant de manipuler un phénomène naturel de façon cohérente.

Méthode plot_intensity Lorsque l'on écrit :

haz.plot_intensity(0, figsize=(6, 6))

on demande à Python d'appeler la méthode plot_intensity associée à l'objet haz. Cette méthode :

• extrait l'intensité du premier événement (indice 0) contenu dans l'objet;

• prépare une figure selon la taille figsize=(6,6);

• affiche la carte d'intensité sur la zone géographique couverte par l'aléa.

Ce comportement est possible car :

1. l'objet haz contient les données d'intensité (attributes);

2. la fonction d'affichage plot_intensity sait comment utiliser ces données (méthode).

Pourquoi utiliser des objets? L'approche orientée objet permet :

• d'organiser le code de manière logique : chaque aléa est un objet;

• d'éviter les erreurs : toutes les informations nécessaires sont stockées ensemble;

• de disposer de méthodes spécialisées : par exemple afficher, interpoler ou filtrer des intensités.

Ainsi, l'objet haz encapsule toutes les informations relatives à l'aléa et offre une interface simple et cohérente pour travailler avec celui-ci dans un contexte de modélisation de catastrophes naturelles.

Exposition

from climada.entity import Exposures
from climada.util.constants import EXP_DEMO_H5

exp = Exposures.from_hdf5(EXP_DEMO_H5)

print(
    f"In the exposure object, a total amount of {exp.value_unit} {exp.gdf.value.sum}
    () / 1_000_000_000:.2f"
    f" is distributed among {exp.gdf.shape[0]} points."
)

# graphique
exp.plot_basemap(figsize=(6, 6));

fonction d'impact

from climada.entity import ImpactFuncSet, ImpfTropCyclone
# Emanuel is a widely used tropical cyclone impact function
impf_tc = ImpfTropCyclone.from_emmanuel_usa()
impf_set = ImpactFuncSet([impf_tc])
impf_set.plot();

concaténation des différentes fonctions pour obtenir le résultat

from climada.engine import ImpactCalc

imp = ImpactCalc(exp, impf_set, haz).impact(save_mat=True)
from climada.engine import ImpactCalc

imp = ImpactCalc(exp, impf_set, haz).impact(save_mat=True)
confidence_level = 0.01

# imp.at_event contient les impacts par événement
VaR_CI = np.percentile(imp.at_event, 100 * (1 - confidence_level))
print(VaR_CI)

Exercice 11 Quelle est la perte centenaire?

Loi de Pareto

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# __
# 1. Chargement des données OEP (ex : CLIMADA)
# __
# Le fichier doit contenir au moins deux colonnes :
# loss : niveau de perte
# ep : probabilité de dépassement (Exceedance Probability)
vector_loss = imp.at_event # <-- adapter ton fichier

# tri de vector_loss par perte croissante (vector_loss est un vecteur ndarray ici)
vector_loss = np.sort(vector_loss)

# on crée df DataFrame avec loss et ep
df = pd.DataFrame({"loss": vector_loss, "ep": np.arange(1, len(vector_loss) + 1) / (len(vector_loss) + 1)})

# __
# 2. Choisir un seuil u = début de la queue
# __
# Exemple : seuil = 95e percentile des pertes
u = df["loss"].quantile(0.97)
print(f"Quantile au Seuil u choisi (99e percentile) = {u:.4f}")
# Autre option : donner u explicitement

# u = 1e7

df_tail = df[df["loss"] >= u].sort_values("loss")

# 
# 3. Transformation log-log pour régression

df_tail["log_loss"] = np.log(df_tail["loss"])
df_tail["log_ep"] = np.log(df_tail["ep"])  # ép = survie
df_tail = df_tail.sort_values("loss")
X = df_tail[["log_loss"]].values
y = df_tail["log_ep"].values

print(df_tail)

# impression graphique de la queue
plt.figure(figsize=(8, 6))
plt.loglog(df_tail["loss"], df_tail["ep"], label="OEP CLIMADA", linewidth=3)
plt.scatter(df_tail["loss"], df_tail["ep"], color="red", label="Données de la queue utilisées pour la régression")
plt.xlabel("Perte (loss)")
plt.ylabel("Probabilité de dépassement (EP)")
plt.title("Données OEP CLIMADA avec sélection de la queue")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.show()

# 
# 4. Régression linéaire sur la queue
model = LinearRegression()
model.fit(X, y)

a = model.intercept_
b = model.coef_[0]

# 
# 5. Paramètres Pareto
# alpha_hat = -b
u_hat = np.exp(a / alpha_hat)

print("Pareto I estimé")
print("---")
print(f"alpha (queue) = {alpha_hat:.4f}")
print(f"u (seuil estimé) = {u_hat:.4f}")
print(f"u choisi (optionnel) = {u:.4f}")

# 
# 6. Courbe Pareto ajustée
# def pareto_survival(x, u, alpha):
#     return (x / u) ** (-alpha)

df_tail["pareto_ep"] = pareto_survival(df_tail["loss"], u_hat, alpha_hat)

# 
# 7. Vérification visuelle
#
plt.figure(figsize=(8, 6))
plt.loglog(df_tail["loss"], df_tail["ep"], label="OEP CLIMADA", linewidth=3)
plt.loglog(df_tail["loss"], df_tail["pareto_ep"], label="Pareto ajusté", linestyle="---", linewidth=2)
plt.xlabel("Perte (loss)")
plt.ylabel("Probabilité de dépassement (EP)")
plt.title("Ajustement Pareto sur la queue OEP CLIMADA")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.show()

On obtient le résultat suivant de la loi de Pareto :

___
Pareto I estimé
___
alpha (queue) = -0.0038
u (seuil estimé) = 262911673319.4620
u choisi (optionnel) = 24079711.5559

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl//44453948-142c-485c-8827-16372c660a37/markdown_84/imgs/img_in_chart_box_142_155_745_598.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-01-11T18%3A43%3A10Z%2F-1%2F%2Fe5a2e094fc4ceaf433886f391a62dd3402e1beae5a1fb420d5a15e7a6fbb1c82" alt="Image" width="50%" /></div>


<div style="text-align: center;">Fig. 10.4 : Estimation de la Loi de Pareto I</div>


<div style="text-align: center;">Pourquoi une régression pour obtenir les paramètres de la Loi de Pareto?</div>


PROPRIÉTÉ 7 Attention à bien choisir un seuil de sinistres qui correspond à une loi des extrêmes! Donc un quantile assez haut (mais pas trop haut pour avoir tout de même des sinistres)

pourquoi un α négatif?

Powered by TCPDF (www.tcpdf.org)

Nous cherchons à calculer la probabilité de ruine à 1 an, qui est le cadre de Solvabilité II.  $  r_{n}(W_{0}) = \mathbb{P}(W_{0} - S + E(S) < 0)  $  si on considère  $ \lambda = 0 $ .

def SimulationCapital(NbreAnneeSimulees = 10000, n = 10000, Qx = 0.00094, CapitalAssure = 200000):
    # Paramètres
    # indiquer le nombre de simulations que l'on veut NbreAnneeSimulees
    # nombre de clients NbreClients
    # Probabilité de mourir de King Fo Qx
    # Capital Assuré CapitalAssure
    # n : nombre de client
    Provision = CapitalAssure * Qx * NbreClients
    df = np.random.rand(NbreAnneeSimulees, n)
    df = np.where(df <= Qx, -CapitalAssure, 0)
    S = np.sum(df, axis=1)
    SituationFinAnnee = S + Provision
    sns.displot(SituationFinAnnee)
    print('Provision :', Provision)
    print('n :', Nn)
    print('Quantile 1% :', np.quantile(SituationFinAnnee, 0.01))
    print('r(K)', np.sum(np.where(SituationFinAnnee < 0.1 / NbreAnneeSimulees, 0)))
    SimulationCapital(n = 1)
    SimulationCapital(n = 1000)


<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'></td><td style='text-align: center;'>Provision</td><td style='text-align: center;'>Quantile 1%</td><td style='text-align: center;'>Proba Ruine</td></tr><tr><td style='text-align: center;'>1 client</td><td style='text-align: center;'>188</td><td style='text-align: center;'>188</td><td style='text-align: center;'>0.1%</td></tr><tr><td style='text-align: center;'>1000 clients</td><td style='text-align: center;'>188 000</td><td style='text-align: center;'>-612 000</td><td style='text-align: center;'>61.4%</td></tr></table>

Tab. 11.1 : Principaux résultats avec 1 et 1000 clients

Avec un seul client, le risque est très concentré dans la zone autour de 0, avec quelques années où le client meurt et la perte de la compagnie très importante (en proportion des provisions). A noter que le capital pour couvrir le risque centenaire est faible (et même négatif!), puisque le risque que King Fo ne meurt est inférieur à 1%.

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl//44453948-142c-485c-8827-16372c660a37/markdown_86/imgs/img_in_chart_box_198_946_441_1191.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-01-11T18%3A43%3A11Z%2F-1%2F%2Fa6c2edc45025e5b8b6602e25d91ae733689db2b6d9d780824bc6dd6ab5072183" alt="Image" width="20%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl//44453948-142c-485c-8827-16372c660a37/markdown_86/imgs/img_in_chart_box_448_944_690_1187.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-01-11T18%3A43%3A11Z%2F-1%2F%2Ff29d32befda8b9f80d87b5a087ddadb7af6669cba8a1e511eded3256baaf7b8b" alt="Image" width="20%" /></div>


On note qu'avec 1000 clients, la provision de 188 000 permet de payer à peu près 1 mort par an. Le quantile 1% est constitué de 4 morts dans la même année.

et si on augmente le nombre de personnes assurées (le secret de la mutualisation!) Augmentons à 10 000 le nombre de clients, puis en gardant le même nombre de clients, augmentons la probabilité de décès (probabilité de la table de Duvillard au lieu du taux de mortalité actuel).

SimulationCapital(NbreClients = 10000)
SimulationCapital(NbreClients = 10000, Qx = 0.014)

Tout d'abord avec la mortalité actuelle, alors qu'on a multiplié le nombre de clients par 10, on n'a besoin de multiplier le capital que par 2,5.

<div style="text-align: center;">Tab. 11.2 : Principaux résultats avec 10 000 clients, à gauche,  $ q_{x} = 0.094\% $  et à droite  $ q_{x} = 1.4\% $ </div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'></td><td style='text-align: center;'>q_{x}</td><td style='text-align: center;'>Provision</td><td style='text-align: center;'>Quantile 1%</td><td style='text-align: center;'>Proba Ruine</td></tr><tr><td style='text-align: center;'>10000 clients</td><td style='text-align: center;'>0.094%</td><td style='text-align: center;'>1 880 000</td><td style='text-align: center;'>-1 520 000</td><td style='text-align: center;'>46.4%</td></tr><tr><td style='text-align: center;'>10000 clients</td><td style='text-align: center;'>1.4%</td><td style='text-align: center;'>28 000 000</td><td style='text-align: center;'>-5 600 000</td><td style='text-align: center;'>47.4%</td></tr></table>

<div style="text-align: center;">FIG. 11.2: Distribution des résultats de la Centenaire avec 10 000 clients, à gauche,  $ q_{x} = 0.094\% $  et à droite  $ q_{x} = 1.4\% $ </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl//44453948-142c-485c-8827-16372c660a37/markdown_87/imgs/img_in_chart_box_502_1025_746_1271.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-01-11T18%3A43%3A11Z%2F-1%2F%2Fb0188959cc1fd592efebb63271d13b24b55e459b852729bb7629bc532ef02b70" alt="Image" width="20%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl//44453948-142c-485c-8827-16372c660a37/markdown_87/imgs/img_in_chart_box_752_1024_993_1271.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-01-11T18%3A43%3A11Z%2F-1%2F%2F503a95d25201d4ae3442d89566535780cf1e15020c39c870dc5e22d354cd20e5" alt="Image" width="20%" /></div>


SYNTHÈSE 12.0.1 - variables pour passer de test à "prod" - factorisation par les fonctions - SQL

Exemple 12.0.1 (Traduction de SAS) problème sur les PROC SQL (par défaut, fait sur les variables partagées dans les deux bases) mais non explicites.

Exemple 12.0.2 • Faire un premier modèle mathématiques

• et pas "vibeCode"

• puis demander à LLM de le coder

• ensuite interagir

• ne pas oublier documentation

Exercice 12 créer un prompt pour demander au LLM de restructurer le code pour le rendre plus fiable et respecter les normes les plus récentes de code.

Powered by TCPDF (www.tcpdf.org)

Objectif

Écrire un programme Python (extension .py) qui analyse une base de données parmi celles proposées. Vous devez montrer que vous savez :

• Charger des données avec pandas

• Manipuler et analyser les données

• Organiser le code en fonctions

• choisir la bonne méthode actuarielle

• Produire un résultat numérique clair

Format du rendu

• Un fichier principal .py : projet_nom.py

(exemple : projet_durand.py)

• un fichier résultat obtenu directement par py. Par exemple, si vous utilisez une fenêtre interactive, exportez cette fenêtre en pdf (cf fig 13.1 p. 87). sinon, exportez votre fichier python en Jupyter Notebook (cf p. 88, fig 13.2) puis réexecutez le.

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl//44453948-142c-485c-8827-16372c660a37/markdown_90/imgs/img_in_image_box_136_751_752_1303.jpg?authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-01-11T18%3A43%3A13Z%2F-1%2F%2F03fe6168ddbe108b599b8702fb748e06951dfe5f6cb656e035a11916f1a595e3" alt="Image" width="51%" /></div>


<div style="text-align: center;">Fig. 13.1 : Exportation de fenêtre interactive</div>


• Une fonction obligatoire : def_analyser, qui sera fournie par la suite.

def analyser(df):
    """
    Paramètre : df (pd.DataFrame)
    Retour : un résultat numérique
    """
    # Votre code ici
    return resultat

print("Ou lancez depuis le terminal: python app.py --help pour voir les options.")
# Exemple programmatique d'exécution (sécurisé): appeler run_pipeline_python
# Construire les paramètres depuis les variables d'environnement si désiré
def _env_list(key: str) -> Optional[list[str]]:
    val = os.getenv(key)
    if not val:
        return None
    return [x.strip() for x in val.split("",") if x.strip()]


<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>try:</td><td style='text-align: center;'></td><td style='text-align: center;'></td><td style='text-align: center;'></td></tr></table>

Fig. 13.2 : Exportation Jupyternotebook, menu obtenu via clic droit - contextuel -

Données

Critères de notation

• Fonctionnalité (6 pts) : code exécutable, résultat correct

• Style (4 pts): commentaires, fonctions, lisibilité, respect des consignes

• pertinence actuarielle (5 pt) du modèle choisi (ex : offset pour tenir compte de l'exposition GLM)

• implémentation statistique (5 pt) (ex : GLM avec régularisation pour éviter le surapprentissage)

Attention, merci à veiller à ce que votre devoir ait une partie statistique (ex : Climada : fit Pareto).

Exemple minimal

import pandas as pd

def analyser(df):
    """Retourne la moyenne de la colonne 'age'"""
    return df[“age”].mean()

# lire base de données
# mettre la base dans le même répertoire que projet_nom.py, en cas d'appel à une API,
    mettre le lien avec l'API.

# ex :

df = pd.read_parquet("mon_fichier.parquet")
analyser(df)

Conseils

13.1 Sujets possibles

Sujet standard

FrenchMotor Utilisez selon votre nom, la base a,b,c et modélisez une garantie ('TPL',...) en ayant le meilleur résultat possible (métrique à définir).

Lequel choisir?

• Votre nom commence par A-G : choisissez les fichiers a,

• votre nom commence par H-Q, choisissez le fichiers b,

• sinon, choisissez c.

Sujets difficiles

Modele de Cox avec réseau de Neurone voir p. 71

Travail sur le modèle de génération de catastrophe OASIS, sortir la distribution des évenements pour une adresse donnée (vous pouvez la choisir), puis modéliser avec une loi de Pareto et calculer la VAR à 0,05% (evenement deux-centenaire correspondant à solvabilité II). L'installation (qui ne marche que sur Linux et Mac) est particulièrement compliqué. lien vers OASIS.

Travail sur le modèle de génération de catastrophe Climada, sortir la distribution des évenements pour une adresse donnée (choisissez votre ville de naissance), puis modéliser avec une loi de Pareto et calculer la VAR à 0,05% (evenement deux-centenaire correspondant à solvabilité II). CLIMADA fonctionne sur Windows, macOS et Linux. Attention à bien choisir un péril qui touche votre ville de naissance. Climada Github Python.

Modèle d'actifs Modélisez un modèle de Vasicek ainsi qu'un modèle multifactoriel de Heath Jarrow Morton. Ce modèle est intéressant car il utilise une ACP pour réduire les différentes dimension des taux d'intérêt. Vous pourrez aussi réfléchir à un modèle avec auto-encoder si vous êtes motivés... Ce qui est important ici, c'est de travailler sur des vraies données, interpréter les axes,... Vous pouvez aussi utiliser des modèles de fondation.

Processus de subrogation . Un mémoire d'actuariat propose d'utiliser les modèles LLM pour identifier les processus de subrogation automatique. Si vous avez accès à ce type de données, ce sujet est complexe mais passionnant.

Powered by TCPDF (www.tcpdf.org)

Annexes

Powered by TCPDF (www.tcpdf.org)

Solutions

5 Quand utiliser ALR :

• Lorsque les données du triangle contiennent des valeurs extrêmes ou des anomalies susceptibles d'avoir un impact sur les sommes globales, la moyenne des ratios peut être plus robuste.

- SADF est souvent utilisée comme une méthode alternative ou de validation pour les projections basées sur Chain-Ladder.

Limites de ALR :

• Elle peut être plus sensible aux fluctuations des ratios individuels, surtout avec de petites tailles d'échantillons.

• Elle n'intègre pas le volume des sinistres dans le calcul des facteurs, ce qui peut biaiser les projections.

7 Voici ce que donne ChatGPT :

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# --- Etape 1 : Définir les transformateurs personnalisés ---
class LogMortalityTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return np.log(X)

class LeeCarterDecomposer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Step 1: compute alpha_x
        self.alpha_ = np.nanmean(X, axis=1)
        # Step 2: center matrix
        X_centered = X - self.alpha_[:, np.newaxis]
        # Step 3: SVD
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.beta_ = U[:, 0]
        self.kappa_ = S[0] * Vt[0, :]
        # Normalisation : contrainte de Lee-Carter
        self.beta_ / = np.sum(self.beta_)
        self.kappa_ += np.sum(self.beta_)
        return self

def transform(self, X):
    return {
        'alpha': self.alpha_,
        'beta': self.beta_,
        'kappa': self.kappa_
    }

class KappaForecaster(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.model = ARIMA(X['kappa'], order=(1, 1, 0)).fit()
        return self

def transform(self, X):
    return forecast = self.model.forecast(steps=10)

return forecast

# --- Etape 2 : Charger un exemple de données ---
# Exemple fictif de taux de mortalité mu_x_t
# Âges : 50 à 90 / Années : 1990 à 2019
ages = np.arange(50, 91)
years = np.arange(1990, 2020)

np.random.seed(42)
mu = np.exp(-10 + 0.08 * (ages[:, None]) - 0.01 * (years[None, :] - 1990)) * \
             np.random.lognormal(mean=0, sigma=0.05, size=(len(ages), len(years)))

# --- Etape 3 : Créer et exécuter le pipeline ---

pipeline = Pipeline(steps=[
('log_transform', LogMortalityTransformer)),
('lee_carter', LeeCarterDecomposer)),
('forecast_kappa', KappaForecaster))
])

forecast_kappa = pipeline.fit_transform(mu)

# --- Etape 4 : Visualisation de la projection de kappa ---

plt.plot(years, pipeline.named_steps['lee_carter'].kappa_, label="kappa observée")
plt.plot(np.arange(2020, 2030), forecast_kappa, label="kappa projetée", linestyle='--')
plt.xlabel("Année")
plt.ylabel("Indice de mortalité $(\kappa_{a} t_{s})$")
plt.title("Projection de $(\kappa_{a} t_{s})$ (modèle Lee-Carter)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

Bibliographie

Here are the references in citation order.

[1] D. C. M. DICKSON, Mary HARDY et H. R. WATERS. Actuarial mathematics for life contingent risks. Third edition. Cambridge; New York: Cambridge University Press, 2020 (cf. p. 63).

[2] Michael FACKLER. « Reinventing Pareto : Fits for both small and large losses ». In : (2013) (cf. p. 76).

