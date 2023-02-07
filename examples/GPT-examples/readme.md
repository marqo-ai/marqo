# Prerequisites
1. You will need an OpenAI API key.
```
export OPENAI_API_KEY="..."
```

2. Install marqo
```bash
docker pull marqoai/marqo:0.0.12;
docker rm -f marqo;
docker run --name marqo -it --privileged -p 8882:8882 --add-host host.docker.internal:host-gateway marqoai/marqo:0.0.12
pip install marqo
```

3. Install other dependencies
```
pip install transformers
pip install torch
pip install langchain
pip install --user -U nltk
pip install pandas
pip install numpy
```

# 1. Product question and answering
```python
python product_q_n_a.py
```

# 2. Chat agent with history/NPC 
```python
python ironman.py
```
