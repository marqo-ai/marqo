Original article [here](https://www.marqo.ai/blog/from-iron-manual-to-ironman-augmenting-gpt-with-marqo-for-fast-editable-memory-to-enable-context-aware-question-answering).

# Prerequisites
1. You will need an OpenAI API key.
```
export OPENAI_API_KEY="..."
```

2. Install marqo
```bash
docker pull marqoai/marqo:0.0.12;
docker rm -f marqo;
docker run --name marqo -it -p 8882:8882 --add-host host.docker.internal:host-gateway marqoai/marqo:0.0.12
pip install marqo
```

3. Install other dependencies
```
pip install -r requirements.txt
```

# 1. Product question and answering
```python
python product_q_n_a.py
```

# 2. Chat agent with history/NPC 
```python
python ironman.py
```
