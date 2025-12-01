The datasets we used were open and publicly available, and we correctly added references.

Train the model:

install the requirements

```
pip install -r requirements.txt
```

generate the `.h5` files

```
python ./NACD/utils/tools.py
```

generate the redundant annotations

```
python ./noise_label/generate.py
```

train the model

```
python ./NACD/train.py --Lambda=0.2 --alpha=0.3 --beta=0.7
```

