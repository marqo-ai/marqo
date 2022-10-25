<p align="center">
  <img src="../assets/logo2.svg" alt="Marqo"/>
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


Пошукова система тензором з відкритим кодом, яка легко інтегрується з вашими програмами, веб-сайтами та робочим процесом.

Marqo хмара ☁️ на даний момент в бета-версії. Якщо ви зацікавлені, зверніться [сюди.](https://q78175g1wwa.typeform.com/to/d0PEuRPC)

## Що таке тензорний пошук (tensor search)?

Тензорний пошук (tensor search) передбачає перетворення документів, зображень та інших данних в набір векторів, що називаються "тензори". Представлення данних у вигляді тензорів дозволяє нам зіставити запити з документами з людиноподібним розумінням цього запиту і змісту документа. Тензорний пошук може бути сильним рішенням у багатьох випадках користування, наприклад:

- пошук і рекомендації кінцевого користувача
- мультимодальний пошук (картинка до картинки, текст до картинки, картинка до тексту)
- чат боти і системи запитань та відповідей
- класифікація тексту та зображення 

<p align="center">
  <img src="../assets/output.gif?raw=true"/>
</p>

<!-- end marqo-description -->

## Початок роботи

1. Marqo потребує docker. Для встановлення Docker перейдіть до [Docker Official website.](https://docs.docker.com/get-docker/)
2. Використовуйте docker для запуску Marqo (Користувачам Мас з чіпом серії М потрібно буде [перейти сюди](#m-series-mac-users)):
```bash
docker rm -f marqo;
docker pull marqoai/marqo:0.0.5;
docker run --name marqo -it --privileged -p 8882:8882 --add-host host.docker.internal:host-gateway marqoai/marqo:0.0.5
```
3. Встановіть клієнт Marqo:
```bash
pip install marqo
```
4. Почніть індексування та пошук! Розгляньмо простий приклад нижче:

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

- `mq` це клієнт, що обгортає `marqo` API
- `add_documents()` приймає список документів, представлених як словники python, для індексування
- `add_documents()` створює індекс зі стандартними налаштуваннями, якщо такого не існує
- За бажання, ви можете встановити ID документа за допомогою спеціального `_id` поля. Інакше, Marqo згенерує такий
- Якщо індекс не існує, Marqo створить його. Якщо він існує, то Marqo додасть документи до індексу

Погляньмо на результати:

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

- Кожен збіг відповідає документу, який збігається з пошуковим запитом
- Вони впорядковані від найбільш до найменш відповідних
- `limit` це максимальне число збігів, які слід повернути. Це може бути встановлено як параметр під час пошуку
- Кожен збіг має поле `_highlights`. Це частина документа, яка найкраще збігалася з запитом


## Інші базові операції

### Отримати документ
Отримати документ за ID.

```python
result = mq.index("my-first-index").get_document(document_id="article_591")
```

Зверніть увагу, що додавання документа за допомогою ```add_documents``` повторно використовуючи той самий ```_id``` призведе до того, що документ буде оновлено.

### Отримати статистику індексу
Отримати інформацію про індекс.

```python
results = mq.index("my-first-index").get_stats()
```

### Лексичний пошук
Виконати пошук за ключовим словом.

```python
result =  mq.index("my-first-index").search('marco polo', search_method=marqo.SearchMethods.LEXICAL)
```

### Знайти конкретні поля
Використовуючи стандартний метод тензорного пошуку
```python
result = mq.index("my-first-index").search('adventure', searchable_attributes=['Title'])
```

### Видалити документи
Видалити документи.

```python
results = mq.index("my-first-index").delete_documents(ids=["article_591", "article_602"])
```

### Видалити індекс
Видалити індекс.

```python
results = mq.index("my-first-index").delete()
```

## Мультимодальний та перехресно-модальний пошук

Для підсилення пошуку зображень та тексту, Marqo дозволяє користувачам підключати та працювати з CLIP моделями від HuggingFace. **Зверніть увагу, якщо ви не налаштуєте мультимодальний пошук, адреси зображень будуть розглядатися як рядки.** Для початку індексування та пошуку з зображеннями, спершу створіть індекс з конфігурацією CLIP, як показано нижче:

```python

settings = {
  "treat_urls_and_pointers_as_images":True,   # allows us to find an image file and index it 
  "model":"ViT-L/14"
}
response = mq.create_index("my-multimodal-index", **settings)
```

Зображення потім можуть бути додані до документу в такий спосіб. Ви можете використати посилання з інтернету (наприклад S3) або з диску комп'ютера:

```python

response = mq.index("my-multimodal-index").add_documents([{
    "My Image": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f2/Portrait_Hippopotamus_in_the_water.jpg/440px-Portrait_Hippopotamus_in_the_water.jpg",
    "Description": "The hippopotamus, also called the common hippopotamus or river hippopotamus, is a large semiaquatic mammal native to sub-Saharan Africa",
    "_id": "hippo-facts"
}])

```

Після цього ви можете виконувати пошук за текстом як зазвичай. Пошук буде здійснюватися як по тексту, так і по картинкам:
```python

results = mq.index("my-multimodal-index").search('animal')

```
 Встановлення `searchable_attributes` для поля зображення `['My Image'] ` гарантує пошук лише зображень у цьому індексі:

```python

results = mq.index("my-multimodal-index").search('animal',  searchable_attributes=['My Image'])

```

### Пошук за зображенням
Пошук за зображенням можна виконати, використовуючи посилання на зображення. 
```python
results = mq.index("my-multimodal-index").search('https://upload.wikimedia.org/wikipedia/commons/thumb/9/96/Standing_Hippopotamus_MET_DP248993.jpg/440px-Standing_Hippopotamus_MET_DP248993.jpg')
```

## Документація
Повну документацію Marqo можна знайти тут [https://marqo.pages.dev/](https://marqo.pages.dev/).

## Увага
Зверніть увагу, що вам не слід запускати інші програми на кластері Marqo's Opensearch, оскільки Marqo автоматично змінює та адаптує налаштування на кластері.

## Користувачі Mac серії M
Marqo ще не підтримує docker-in-docker бекенд конфігурацію для архітектури arm64. Це означає, якщо ви використовуєте Mac серії M, вам також знадобиться запустити marqo's бекенд, marqo-os, локально.

Для запуску Marqo на Mac серії M, виконайте наступні кроки.

1. В одному терміналі запустіть наступну команду для початку opensearch:

```shell
docker rm -f marqo-os; docker run -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" marqoai/marqo-os:0.0.2-arm
```

2. В іншому терміналі запустіть наступну команду для запуску Marqo:
```shell
docker rm -f marqo; docker run --name marqo --privileged \
    -p 8882:8882 --add-host host.docker.internal:host-gateway \
    -e "OPENSEARCH_URL=https://localhost:9200" \
    marqoai/marqo:0.0.5
```

## Зробити внесок
Marqo - це ком'юніті проект, метою якого є розповсюдити тензорний пошук для ширшої спільноти розробників. Ми раді, що ви зацікавлені у допомозі! Будь ласка, прочитайте [це](./CONTRIBUTING.md) для початку 

## Dev set up
1. Створити віртуальне середовище ```python -m venv ./venv```
2. Активувати віртуальне середовище ```source ./venv/bin/activate```
3. Встановити пакети з файлу: ```pip install -r requirements.txt```
4. Виконати тести запустивши файл tox. CD до цієї папки, потім запустити "tox" 
5. Якщо ви оновлюєте залежності, переконайтесь, що видалили папку .tox та перезапустіть

## Merge інструкції:
1. Запустити повний набір тестів (використовуючи команду `tox` в цій папці).
2. Створіть pull request з прикріпленим github issue.


## Підтримка

- Приєднайтесь до нашого [Slack community](https://join.slack.com/t/marqo-community/shared_invite/zt-1d737l76e-u~b3Rvey2IN2nGM4wyr44w) та спілкуйтесь з іншими учасниками спільноти щодо ідей.
- Зустрічі спільноти Marqo (скоро буде!)

### Stargazers
[![Stargazers repo roster for @marqo-ai/marqo](https://reporoster.com/stars/marqo-ai/marqo)](https://github.com/marqo-ai/marqo/stargazers)

### Forkers
[![Forkers repo roster for @marqo-ai/marqo](https://reporoster.com/forks/marqo-ai/marqo)](https://github.com/marqo-ai/marqo/network/members)


## Translations

Це readme доступно в наступних перекладах:

- [English](../README.md)🇬🇧
- [中文 Chinese](README-Chinese.md)🇨🇳
- [Polski](README-Polish.md)🇵🇱
- [Українська](README-Ukrainian.md)🇺🇦
