# Resolution Neural Architecture Search

Search the image resolution and model architecture simultaneously.

## pretrain the supernet
```python
python pretrain.py
```

## evolutionary search the image resolution and model architecture in the pretrained supernet
```python
python search.py
```

## retrain the searched model by searched image resolution
```python
python retrain.py
```