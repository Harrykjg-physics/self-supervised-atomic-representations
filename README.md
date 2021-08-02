# Self-supervised atomic representations for material discovery

## Train your GNN model with self-supervised learning 

The simplest way to train a new model:

```bash
python ssl_pred.py  root_dir  --epochs 100  --output_model_file cgcnn_pretrained
```

Where the structure of `root_dir` is as follows:

```
├──raw
 ├── id_prop.csv
 ├── atom_init.json
 ├── id0.cif
 ├── id1.cif
 ├── ...
├──processed
```

The `processed` directory contains processed structure file to reduce loading time

You can find more settings by:

 ```bash
python ssl_pred.py -h
```

Once trained, you will get `loss_avg.npy` and `acc_avg.npy` files, which record loss and accuracy vesus training epochs