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
<a href="https://pepy.tech/project/marqo"><img alt="PyPI - Downloads from pepy" src="https://static.pepy.tech/personalized-badge/marqo?period=month&units=international_system&left_color=grey&right_color=blue&left_text=downloads/month"></a>
<a align="center" href="https://join.slack.com/t/marqo-community/shared_invite/zt-1d737l76e-u~b3Rvey2IN2nGM4wyr44w"><img src="https://img.shields.io/badge/Slack-blueviolet?logo=slack&amp;logoColor=white"></a>
</p>


这是一个开源的张量（tensor）搜索引擎。它可以无缝地整合到你的应用、网页、工作流程中。

Marqo云服务☁正在测试中。 如果你感兴趣的话，你可以在这里申请：https://q78175g1wwa.typeform.com/to/d0PEuRPC

## 什么是张量（tensor）搜索？

张量搜索是一个涉及到把文件、图片和其他数据转换成一个向量集合（我们称之为张量）的过程。通过张量来表示数据可以允许我们像人类那样去匹配搜索请求和文件。张量搜索可以助力众多的应用场景包括：
- 用户搜索和推荐系统
- 多模式搜索（图片搜图片，文字搜图片，图片搜文字）
- 聊天机器人和自动问答系统
- 文字和图片分类

<p align="center">
  <img src="../assets/output.gif"/>
</p>

<!-- end marqo-description -->

## 让我们开始吧

1. Marqo 需要 docker的支持. 安装docker请点击 [Docker Official website.](https://docs.docker.com/get-docker/)
2. 通过docker来运行Marqo (M系列芯片的Mac用户需要点击这里 [go here](#m-series-mac-users)):
```bash
docker rm -f marqo;
docker pull marqoai/marqo:0.0.5;
docker run --name marqo -it --privileged -p 8882:8882 --add-host host.docker.internal:host-gateway marqoai/marqo:0.0.5
```
3. 安装 Marqo client:
```bash
pip install marqo
```
4. 开始索引和搜索! 让我们用下面这个简单的例子来示范:

```python
import marqo

mq = marqo.Client(url='http://localhost:8882')

#添加索引文件
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
#进行第一次搜索
results = mq.index("my-first-index").search(
    q="What is the best outfit to wear on the moon?"
)

```

- `mq` 是一个把包含 `marqo` 客户端（Client）的接口（API）
- `add_documents()` 包括一系列的文件。文件通过Python的字典（Dict）表示，并进行索引
- `add_documents()` 会再索引不存在的前提下，用默认的设置生成索引
- 你可以选择要不要通过 `_id` 来设置一个文件编号（ID）. Marqo也可以帮你自动生成
- 如果索引不存在的话，Marqo就会生成一个。 如果索引存在的话，Marqo就会把文件添加到当前索引中。

我们一起来看看结果如何：

```python
# 输出结果:
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

- 每一个 `hit` 代表一个匹配搜索请求的文件
- 他们以最匹配到最不匹配的方式排序
- `limit` 是能够返回的最大结果数量。 这个可以在搜索中作为参数进行设置
- 每一个 `hit` 都有一个 `_highlights` 部分. 这是返回文件中和搜索请求最匹配的部分


## 其他基本操作

### 获取文件
通过ID获取文件.

```python
result = mq.index("my-first-index").get_document(document_id="article_591")
```

注意： 通过 ```add_documents``` 添加文件并且使用同一个```_id``` 会导致文件的覆盖且更新。

### 获取索引状态
获取索引的信息。

```python
results = mq.index("my-first-index").get_stats()
```

### 词汇搜索（Lexical search）
进行一起关键搜索.

```python
result =  mq.index("my-first-index").search('marco polo', search_method=marqo.SearchMethods.LEXICAL)
```

### 搜索特定的域（field）
使用默认的张量搜索方法
```python
result = mq.index("my-first-index").search('adventure', searchable_attributes=['Title'])
```

### 删除文件
删除一些文件。

```python
results = mq.index("my-first-index").delete_documents(ids=["article_591", "article_602"])
```

### 删除索引
删除一些索引。

```python
results = mq.index("my-first-index").delete()
```

## 多模型、跨模型搜索

为了支持文字和图片搜索, Marqo允许用户接入并且调试来自于HuggingFace的CLIP模型. **注意：如果你没有在参数中设置多模型搜索, 图片urls将会被当做字符串（string）处理.** 在开始图片索引和搜索执勤啊, 我们首先要设置一个 CLIP 模型的配置, 方法如下:

```python

settings = {
  "treat_urls_and_pointers_as_images":True,   # allows us to find an image file and index it 
  "model":"ViT-L/14"
}
response = mq.create_index("my-multimodal-index", **settings)
```

图片能够以如下方式添加到文件中. 你可以使用来自互联网的 urls (例如百度图) 或者自己硬盘上的图片：

```python

response = mq.index("my-multimodal-index").add_documents([{
    "My Image": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f2/Portrait_Hippopotamus_in_the_water.jpg/440px-Portrait_Hippopotamus_in_the_water.jpg",
    "Description": "The hippopotamus, also called the common hippopotamus or river hippopotamus, is a large semiaquatic mammal native to sub-Saharan Africa",
    "_id": "hippo-facts"
}])

```

然后你就可以和之前一样搜索文字. 该搜索会同时搜索文字和图片域:
```python

results = mq.index("my-multimodal-index").search('animal')

```
 把 `searchable_attributes` 设置成图片域 `['My Image'] ` 会让你只在图片域中进行这次索引:

```python

results = mq.index("my-multimodal-index").search('animal',  searchable_attributes=['My Image'])

```

### 用图片搜索
我们可以通过提供图片链接的方式来进行图片搜索
```python
results = mq.index("my-multimodal-index").search('https://upload.wikimedia.org/wikipedia/commons/thumb/9/96/Standing_Hippopotamus_MET_DP248993.jpg/440px-Standing_Hippopotamus_MET_DP248993.jpg')
```

## 文档
关于Marqo的完整文档可以在这里获取[https://marqo.pages.dev/](https://marqo.pages.dev/).

## 注意
注意请不要在Marqo的开源搜索引擎中运行其他的应用。Marqo会自动更新、改变引擎上的设置。

## M系列芯片的Mac用户
对于arm64架构，Marqo 暂未支持 docker-in-docker 后端配置. 这意味着如果你使用M系列Mac, 你需在在本地运行marqo的后端, marqo-os.

如果你想在M系列Mac上运行Marco, 请按照如下步骤：

1. 在一个终端（terminal） 运行如下命令来启动引擎：

```shell
docker rm -f marqo-os; docker run -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" marqoai/marqo-os:0.0.2-arm
```

2. 在另一个终端（terminal）运行如下命令来启动Marqo:
```shell
docker rm -f marqo; docker run --name marqo --privileged \
    -p 8882:8882 --add-host host.docker.internal:host-gateway \
    -e "OPENSEARCH_URL=https://localhost:9200" \
    marqoai/marqo:0.0.5
```

## 贡献者
Marqo是一个社区项目。我们的目标是让张量搜索能够被更多的开发者群体使用。
如果你愿意提供帮助我们会非常开心！请阅读[这里](./CONTRIBUTING.md)来开始你的贡献。

## 开发者设置
1. 创建一个新的 virtual env ```python -m venv ./venv```
2. 激活 virtual environment ```source ./venv/bin/activate```
3. 通过需求列表安装需要的文件: ```pip install -r requirements.txt```
4. 运行 tox 文件进行测试。
5. 如果你更新了环境, 确保你删除了.tox 文件夹并再洗运行。

## 如何合并:
1. 运行所有的测试 (通过运行文件夹下的 `tox` 文件).
2. 创建一个 pull request 并附带一个issue.

<!-- start support-pitch -->


## 支持

- 加入我们的 [Slack社区](https://join.slack.com/t/marqo-community/shared_invite/zt-1d737l76e-u~b3Rvey2IN2nGM4wyr44w) 并和其他社区成员分享你的看法
- Marqo 社区面对面 (正在筹备中!)

<!-- end support-pitch -->
