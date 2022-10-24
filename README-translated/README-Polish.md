<p align="center">
  <img src="assets/logo2.svg" alt="Marqo"/>
</p>

<h1 align="center">Marqo</h1>

<p align="center">
  <b>Tensor search for humans.</b>
</p>

<p align="center">
<a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
<a href="https://pypi.org/project/marqo/"><img src="https://img.shields.io/pypi/v/marqo?label=PyPI"></a>
<a href="https://github.com/marqo-ai/marqo/actions/workflows/CI.yml"><img src="https://img.shields.io/github/workflow/status/marqo-ai/marqo/CI?label=CI"></a>
<a href="https://pypistats.org/packages/marqo"><img alt="PyPI - Downloads from official pypistats" src="https://img.shields.io/pypi/dm/marqo?label=Downloads"></a>
<a align="center" href="https://join.slack.com/t/marqo-community/shared_invite/zt-1d737l76e-u~b3Rvey2IN2nGM4wyr44w"><img src="https://img.shields.io/badge/Slack-blueviolet?logo=slack&amp;logoColor=white"></a>
</p>


Wyszukiwarka tensorowa typu open source, która bezproblemowo integruje się z Twoimi aplikacjami, witrynami internetowymi i cyklami produkcji.

Chmura Marqo ☁️ jest obecnie w fazie beta. Jeśli jesteś zainteresowany, zgłoś się tutaj: https://q78175g1wwa.typeform.com/to/d0PEuRPC

## What is tensor search?

Wyszukiwanie tensorowe polega na przekształcaniu dokumentów, obrazów i innych danych w zbiory wektorów zwane „tensorami”. Reprezentowanie danych jako tensory pozwala nam dopasować zapytania do dokumentów z ludzkim zrozumieniem zapytania i treści dokumentu. Wyszukiwanie tensorowe może wspomagać różne przypadki użycia, takie jak:
- wyszukiwanie i rekomendacje użytkowników końcowych
- wyszukiwanie multimodalne (obraz na obraz, tekst na obraz, obraz na tekst)
- boty czatowe oraz systemy pytań i odpowiedzi
- klasyfikacja tekstu i obrazu

<p align="center">
  <img src="assets/output.gif"/>
</p>

<!-- end marqo-description -->

## Pierwsze kroki

1. Marqo wymaga dokera. Aby zainstalować Docker, przejdź do [Docker oficjalna strona.](https://docs.docker.com/get-docker/)
2. Użyj dokera, aby uruchomić Marqo (Mac users with M-series chips will need to [go here](#m-series-mac-users)):
```bash
docker rm -f marqo;
docker pull marqoai/marqo:0.0.5;
docker run --name marqo -it --privileged -p 8882:8882 --add-host host.docker.internal:host-gateway marqoai/marqo:0.0.5
```
3. Zainstaluj klienta Marqo:
```bash
pip install marqo
```
4. Rozpocznij indeksowanie i wyszukiwanie! Spójrzmy na prosty przykład poniżej:

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

- `mq` to klient, który otacza API `marqo`
- `add_documents()` pobiera do indeksowania listę dokumentów, reprezentowanych jako dykty Pythona
- `add_documents()` tworzy indeks z ustawieniami domyślnymi, jeśli jeszcze nie istnieje
- Możesz opcjonalnie ustawić identyfikator dokumentu za pomocą specjalnego pola `_id`. W przeciwnym razie Marqo wygeneruje jeden.
- Jeśli indeks nie istnieje, Marqo go stworzy. Jeśli istnieje, Marqo doda dokumenty do indeksu.

Przyjrzyjmy się wynikom:

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

– Każde trafienie odpowiada dokumentowi pasującemu do zapytania
– Są uporządkowane od najbardziej pasujących do najmniej pasujących
– `limit ` to maksymalna liczba trafień do zwrócenia. Można to ustawić jako parametr podczas wyszukiwania
– Każde trafienie ma pole `_highlights `. To była ta część dokumentu, która najlepiej pasowała do zapytania.


## Inne podstawowe operacje

### Pobierz dokument
Pobierz dokument według ID.

```python
result = mq.index("my-first-index").get_document(document_id="article_591")
```

Zauważ, że dodanie dokumentu za pomocą ```add_documents``` ponownie przy użyciu tego samego ```_id``` spowoduje aktualizację dokumentu.

### Uzyskaj statystyki indeksu
Uzyskaj informacje o indeksie.

```python
results = mq.index("my-first-index").get_stats()
```

### Wyszukiwanie leksykalne
Przeprowadź wyszukiwanie słów kluczowych.

```python
result =  mq.index("my-first-index").search('marco polo', search_method=marqo.SearchMethods.LEXICAL)
```

### Wyszukaj określone pola
Korzystanie z domyślnej metody wyszukiwania tensorów
```python
result = mq.index("my-first-index").search('adventure', searchable_attributes=['Title'])
```

### Usuń dokumenty
Usuń dokumenty.

```python
results = mq.index("my-first-index").delete_documents(ids=["article_591", "article_602"])
```

### Usuń indeks
Usuń indeks.

```python
results = mq.index("my-first-index").delete()
```

## Wyszukiwanie multimodalne i crossmodalne

Do obsługi wyszukiwania obrazów i tekstu, Marqo umożliwia użytkownikom podłączanie i odtwarzanie modeli CLIP firmy HuggingFace. **Pamiętaj, że jeśli nie skonfigurujesz wyszukiwania multimodalnego, adresy URL obrazów będą traktowane jako ciągi.** Aby rozpocząć indeksowanie i wyszukiwanie za pomocą obrazów, najpierw utwórz indeks z konfiguracją CLIP, jak poniżej:

```python

settings = {
  "treat_urls_and_pointers_as_images":True,   # allows us to find an image file and index it 
  "model":"ViT-L/14"
}
response = mq.create_index("my-multimodal-index", **settings)
```

Obrazy można następnie dodawać w dokumentach w następujący sposób. Możesz użyć adresów URL z Internetu (na przykład S3) lub z dysku maszyny:

```python

response = mq.index("my-multimodal-index").add_documents([{
    "My Image": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f2/Portrait_Hippopotamus_in_the_water.jpg/440px-Portrait_Hippopotamus_in_the_water.jpg",
    "Description": "The hippopotamus, also called the common hippopotamus or river hippopotamus, is a large semiaquatic mammal native to sub-Saharan Africa",
    "_id": "hippo-facts"
}])

```

Następnie możesz wyszukiwać jak zwykle, używając tekstu. Przeszukiwane będą zarówno pola tekstowe, jak i graficzne:
```python

results = mq.index("my-multimodal-index").search('animal')

```
 Ustawienie parametru `searchable_attributes` jako pole obrazu `['My Image'] ` gwarantuje, że w tym indeksie będą przeszukiwane tylko obrazy:

```python

results = mq.index("my-multimodal-index").search('animal',  searchable_attributes=['My Image'])

```

### Wyszukiwanie za pomocą obrazu
Wyszukiwanie za pomocą obrazu można osiągnąć, podając link do obrazu.
```python
results = mq.index("my-multimodal-index").search('https://upload.wikimedia.org/wikipedia/commons/thumb/9/96/Standing_Hippopotamus_MET_DP248993.jpg/440px-Standing_Hippopotamus_MET_DP248993.jpg')
```

## Dokumentacja
Pełna dokumentacja Marqo znajduje się tutaj: [https://marqo.pages.dev/](https://marqo.pages.dev/).

## Ostrzeżenie

Pamiętaj, że nie powinieneś uruchamiać innych aplikacji w klastrze Opensearch Marqo, ponieważ Marqo automatycznie zmienia i dostosowuje ustawienia w klastrze.

## M series Mac użytkownicy
Marqo nie obsługuje jeszcze konfiguracji zaplecza docker-in-docker dla architektury arm64. Oznacza to, że jeśli masz komputer Mac z serii M, będziesz musiał również uruchomić lokalnie backend marqo, marqo-os.

Aby uruchomić Marqo na komputerze Mac z serii M, wykonaj następujące kroki.

1. W jednym terminalu uruchom następujące polecenie, aby rozpocząć opensearch:

```shell
docker rm -f marqo-os; docker run -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" marqoai/marqo-os:0.0.2-arm
```

2. W innym terminalu uruchom następujące polecenie, aby uruchomić Marqo:
```shell
docker rm -f marqo; docker run --name marqo --privileged \
    -p 8882:8882 --add-host host.docker.internal:host-gateway \
    -e "OPENSEARCH_URL=https://localhost:9200" \
    marqoai/marqo:0.0.5
```

## Kontrybutorzy
Marqo to projekt społecznościowy, którego celem jest udostępnienie wyszukiwania tensorowego szerszej społeczności programistów. Cieszymy się, że jesteś zainteresowany pomocą! Aby rozpocząć, przeczytaj [to](./CONTRIBUTING.md)

## Dev set up
1. Utwórz wirtualne środowisko ```python -m venv ./venv```
2. Aktywuj środowisko wirtualne ```source ./venv/bin/activate```
3. Zainstaluj wymagania z pliku wymagań: ```pip install -r requirements.txt```
4. Uruchom testy, uruchamiając plik tox. CD do tego dir, a następnie uruchom "tox"
5. Jeśli aktualizujesz zależności, Upewnij się, że usunięto dir .tox i uruchom ponownie

## Merge instrukcje:
1. Uruchom pełny zestaw testowy (za pomocą komendy `tox` w tym dir).
2. Utwórz pull request z dołączonym rozwiązaniem.


## Support

- Dołączć do naszego [Slack community](https://join.slack.com/t/marqo-community/shared_invite/zt-1d737l76e-u~b3Rvey2IN2nGM4wyr44w) i rozmawiaj z innymi członkami społeczności o pomysłach.
- Spotkania społeczności Marqo (już wkrótce!)

### Stargazers
[![Stargazers repo roster for @marqo-ai/marqo](https://reporoster.com/stars/marqo-ai/marqo)](https://github.com/marqo-ai/marqo/stargazers)

### Forkers
[![Forkers repo roster for @marqo-ai/marqo](https://reporoster.com/forks/marqo-ai/marqo)](https://github.com/marqo-ai/marqo/network/members)


## Tłumaczenia

Ten plik readme jest dostępny w następujących tłumaczeniach:

- [English](../README.md)🇬🇧
- [中文 Chinese](README-Chinese.md)🇨🇳
- [Polski](README-Polish.md)🇵🇱
- [Українська](README-Ukrainian.md)🇺🇦
