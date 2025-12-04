Before training the model, you need to generate the .h5 file and the redundant annotations. To do this, run tools.py and generate.py:

```
python ./NACD/utils/tools.py
python ./noise_label/generate.py
```

After the .h5 file and the redundant annotations are generated, you can train the model as follows:

```
python ./NACD/train.py --Lambda=0.2 --alpha=0.3 --beta=0.7
```

