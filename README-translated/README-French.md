<p align="center">
  <img src="../assets/logo2.svg" alt="Marqo"/>
</p>

<h1 align="center">Marqo</h1>

<p align="center">
  <b>Recherche de tenseur pour les humains.</b>
</p>

<p align="center">
<a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
<a href="https://pypi.org/project/marqo/"><img src="https://img.shields.io/pypi/v/marqo?label=PyPI"></a>
<a href="https://github.com/marqo-ai/marqo/actions/workflows/CI.yml"><img src="https://img.shields.io/github/workflow/status/marqo-ai/marqo/CI?label=CI"></a>
<a href="https://pepy.tech/project/marqo"><img alt="PyPI - Downloads from pepy" src="https://static.pepy.tech/personalized-badge/marqo?period=month&units=international_system&left_color=grey&right_color=blue&left_text=downloads/month"></a>
<a align="center" href="https://join.slack.com/t/marqo-community/shared_invite/zt-1d737l76e-u~b3Rvey2IN2nGM4wyr44w"><img src="https://img.shields.io/badge/Slack-blueviolet?logo=slack&amp;logoColor=white"></a>
</p>


Un moteur de recherche de tenseur open-source qui s'intègre de manière transparente à vos applications, sites web et flux de travail. 

Marqo cloud ☁️ est actuellement en version bêta. Si vous êtes intéressé, postulez ici : https://q78175g1wwa.typeform.com/to/d0PEuRPC

## Qu'est-ce que la recherche tensorielle ?

La recherche tensorielle consiste à transformer des documents, des images et d'autres données en collections de vecteurs appelés "tenseurs". La représentation des données sous forme de tenseurs nous permet de faire correspondre des requêtes à des documents avec une compréhension de type humain du contenu de la requête et du document. La recherche tensorielle peut être utilisée dans de nombreux cas, tels que
- la recherche et les recommandations pour l'utilisateur final
- la recherche multimodale (image à image, texte à image, image à texte)
- les robots de chat et les systèmes de questions-réponses
- la classification de textes et d'images

<p align="center">
  <img src="../assets/output.gif"/>
</p>

<!-- end marqo-description -->

## Pour débuter

1. Marqo nécessite Docker. Pour installer Docker, allez sur le [site officiel de Docker] (https://docs.docker.com/get-docker/).
2. Utilisez Docker pour exécuter Marqo (les utilisateurs de Mac avec des puces M-series devront [aller ici](#m-series-mac-users)) :
```bash
docker rm -f marqo;
docker pull marqoai/marqo:latest;
docker run --name marqo -it --privileged -p 8882:8882 --add-host host.docker.internal:host-gateway marqoai/marqo:latest
```
3. Installez le client Marqo :
```bash
pip install marqo
```
4. Commencez l'indexation et la recherche ! Prenons un exemple simple ci-dessous :

```python
import marqo

mq = marqo.Client(url='http://localhost:8882')

mq.index("my-first-index").add_documents([
    {
        "Title": "The Travels of Marco Polo",
        "Description": "A 13th-century travelogue describing Polo's travels"
    }, 
    {
        "Title": "Extravehicular Mobility Unit (EMU)",
        "Description": "The EMU is a spacesuit that provides environmental protection, "
                       "mobility, life support, and communications for astronauts",
        "_id": "article_591"
    }]
)

results = mq.index("my-first-index").search(
    q="What is the best outfit to wear on the moon?"
)

```

- `mq` est le client qui enveloppe l'API `marqo`.
- `add_documents()` prend une liste de documents, représentés comme des dicts python, pour l'indexation
- `add_documents()` crée un index avec des paramètres par défaut, s'il n'en existe pas encore.
- Vous pouvez optionnellement définir l'ID d'un document avec le champ spécial `_id`. Sinon, Marqo va en générer un.
- Si l'index n'existe pas, Marqo le créera. S'il existe, Marqo ajoutera les documents à l'index.

Regardons les résultats :

```python
# let's print out the results:
import pprint
pprint.pprint(results)

{
    'hits': [
        {   
            'Title': 'Extravehicular Mobility Unit (EMU)',
            'Description': 'The EMU is a spacesuit that provides environmental protection, mobility, life support, and' 
                           'communications for astronauts',
            '_highlights': {
                'Description': 'The EMU is a spacesuit that provides environmental protection, '
                               'mobility, life support, and communications for astronauts'
            },
            '_id': 'article_591',
            '_score': 0.61938936
        }, 
        {   
            'Title': 'The Travels of Marco Polo',
            'Description': "A 13th-century travelogue describing Polo's travels",
            '_highlights': {'Title': 'The Travels of Marco Polo'},
            '_id': 'e00d1a8d-894c-41a1-8e3b-d8b2a8fce12a',
            '_score': 0.60237324
        }
    ],
    'limit': 10,
    'processingTimeMs': 49,
    'query': 'What is the best outfit to wear on the moon?'
}
```

- Chaque hit correspond à un document qui correspond à la requête de recherche.
- Ils sont classés du plus au moins correspondant
- `limit` est le nombre maximum de résultats à retourner. Il peut être défini comme un paramètre pendant la recherche.
- Chaque résultat a un champ `_highlights`. C'est la partie du document qui correspond le mieux à la requête.


## Autres opérations de base

### Obtenir un document
Récupère un document par ID.

```python
result = mq.index("my-first-index").get_document(document_id="article_591")
```

Notez qu'en ajoutant le document en utilisant ``add_documents`` à nouveau en utilisant le même ``_id``, un document sera mis à jour.

### Obtenir les statistiques d'un index
Obtenez des informations sur un index.

```python
results = mq.index("my-first-index").get_stats()
```

### Recherche lexicale
Effectuer une recherche par mot-clé.

```python
result =  mq.index("my-first-index").search('marco polo', search_method=marqo.SearchMethods.LEXICAL)
```

### Recherche de champs spécifiques
Utiliser la méthode de recherche tensorielle par défaut
```python
result = mq.index("my-first-index").search('adventure', searchable_attributes=['Title'])
```

### Supprimer des documents
Supprimer des documents.

```python
import marqo.tensor_search.tensor_search
import marqo.tensor_search.delete_docs

results = marqo.tensor_search.tensor_search.delete_documents(ids=["article_591", "article_602"])
```

### Supprimer un index
Supprime un index.

```python
results = mq.index("my-first-index").delete()
```

## Recherche multi modale et inter modale

Pour alimenter la recherche d'images et de textes, Marqo permet aux utilisateurs de brancher et de jouer avec les modèles CLIP de HuggingFace. **Pour commencer à indexer et à rechercher des images, créez d'abord un index avec une configuration CLIP, comme ci-dessous :

```python

settings = {
  "treat_urls_and_pointers_as_images":True,   # allows us to find an image file and index it 
  "model":"ViT-L/14"
}
response = mq.create_index("my-multimodal-index", **settings)
```

Les images peuvent ensuite être ajoutées dans les documents comme suit. Vous pouvez utiliser des urls provenant de l'internet (par exemple S3) ou du disque de la machine :

```python

response = mq.index("my-multimodal-index").add_documents([{
    "My Image": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f2/Portrait_Hippopotamus_in_the_water.jpg/440px-Portrait_Hippopotamus_in_the_water.jpg",
    "Description": "The hippopotamus, also called the common hippopotamus or river hippopotamus, is a large semiaquatic mammal native to sub-Saharan Africa",
    "_id": "hippo-facts"
}])

```

Vous pouvez ensuite effectuer une recherche par texte comme d'habitude. Les champs de texte et d'image seront recherchés :
```python

results = mq.index("my-multimodal-index").search('animal')

```
 Setting `searchable_attributes` to the image field `['My Image'] ` ensures only images are searched in this index:

```python

results = mq.index("my-multimodal-index").search('animal',  searchable_attributes=['My Image'])

```

### Recherche à l'aide d'une image
La recherche à l'aide d'une image peut être réalisée en fournissant le lien de l'image. 
```python
results = mq.index("my-multimodal-index").search('https://upload.wikimedia.org/wikipedia/commons/thumb/9/96/Standing_Hippopotamus_MET_DP248993.jpg/440px-Standing_Hippopotamus_MET_DP248993.jpg')
```

## Documentation
La documentation complète de Marqo se trouve ici [https://marqo.pages.dev/](https://marqo.pages.dev/).

## Avertissement

Notez que vous ne devez pas exécuter d'autres applications sur le cluster Opensearch de Marqo car Marqo modifie et adapte automatiquement les paramètres du cluster.

## Les utilisateurs de Mac série M
Marqo ne prend pas encore en charge la configuration du backend docker-in-docker pour l'architecture arm64. Cela signifie que si vous avez un Mac série M, vous devrez également exécuter le backend de Marqo, marqo-os, localement.

Pour exécuter Marqo sur un Mac série M, suivez les étapes suivantes.

1. Dans un terminal, exécutez la commande suivante pour lancer opensearch :

```shell
docker rm -f marqo-os; docker run -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" marqoai/marqo-os:0.0.3-arm
```

2. Dans un autre terminal, exécutez la commande suivante pour lancer Marqo :
```shell
docker rm -f marqo; docker run --name marqo --privileged \
    -p 8882:8882 --add-host host.docker.internal:host-gateway \
    -e "OPENSEARCH_URL=https://localhost:9200" \
    marqoai/marqo:latest
```

## Contributeurs
Marqo est un projet communautaire dont l'objectif est de rendre la recherche tensorielle accessible à l'ensemble de la communauté des développeurs. Nous sommes heureux que vous soyez intéressés à nous aider ! Veuillez lire [this](./CONTRIBUTING.md) pour commencer.

## Mise en place de l'environnement de développement
1. Créer un environnement virtuel ``python -m venv ./venv``.
2. Activez l'environnement virtuel ``source ./venv/bin/activate``
3. Installez les exigences à partir du fichier d'exigences : ``pip install -r requirements.txt``
4. Lancez les tests en exécutant le fichier tox. Placez le CD dans ce répertoire et exécutez "tox".
5. Si vous mettez à jour les dépendances, assurez-vous de supprimer le répertoire .tox et de réexécuter le fichier.

## Instructions de fusion :
1. Exécutez la suite de tests complète (en utilisant la commande `tox` dans ce répertoire).
2. Créez une demande de pull avec un problème github attaché.


## Soutenez

- Rejoignez notre [communauté Slack] (https://join.slack.com/t/marqo-community/shared_invite/zt-1d737l76e-u~b3Rvey2IN2nGM4wyr44w) et discutez de vos idées avec les autres membres de la communauté.
- Réunions de la communauté Marqo (à venir !)

### Stargazers
[!Liste de repo de Stargazers pour @marqo-ai/marqo](https://reporoster.com/stars/marqo-ai/marqo)](https://github.com/marqo-ai/marqo/stargazers)

### Forkers
[ ![Liste des repo des braconniers pour @marqo-ai/marqo](https://reporoster.com/forks/marqo-ai/marqo)](https://github.com/marqo-ai/marqo/network/members)


## Traductions

Ce fichier readme est disponible dans les traductions suivantes :

- [中文 Chinois](README-translated/README-Chinese.md)🇨🇳
- [Français](README-translated/README-French.md) 
