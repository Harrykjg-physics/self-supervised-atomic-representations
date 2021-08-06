# Self-supervised atomic representations for material discovery

## Train your GNN model with self-supervised learning 

The simplest way to train a new model:

predictive:

```bash
cd ssl
python ssl_pred.py  root_dir  --epochs 100  --output_model_file cgcnn_pretrained
```

contrastive:

```bash
cd ssl
python ssl_contra.py  root_dir --epochs 100 --cell_size1 77 --cell_size2 77 
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
python ssl_contra.py -h
```

Once trained, you will get `loss_avg.npy` and `acc_avg.npy` files, which record loss and accuracy vesus training epochs

The trained model is saved as `cgcnn_pretrained.pth` file

## Generate multiscale atomic representations using trained GNN

you can generate atomic representations of a specified dataset using a self-supervised trained GNN:

```bash
cd ssl
python get_sl_emb.py  root  --lth_emb "01"  --input_model_file pretrained_model_dir
```

The generated atomic representations are saved as `sl_01_embedding_dict.npy`, you can read the npy file as python dictionary `emb_dict` thruogh:

```
emb_dict = np.load("sl_01_embedding_dict.npy", allow_pickle=True).item()
```

the key of the dictionary , i.e. the atom of material,  are named as : `cif_id + elemental_type + atomic_number + idx`, for example, 9011998_Si_14_4

Here the argumment `lth_emb` specifies the scale of your atomic representations, possible choices are ["0", "1", ...,"5","01", "012", ..., "012345"]

"0" stands for initial embeddings, "1" stands for the first embedding and "012" stands for the concatenation of initial, first and second embedding

`root` dir saves the machine learning dataset you are interested in, the structure of root dir is:

```
├── id_prop.csv
├── atom_init.json
├── id0.cif
├── id1.cif
├── ...
```

`input_model_file` specifies the path of your pretrained GNN

if you only want to generate some atomic representations, such as those with no-zero magnetic moments, you can specify `partial_csv` arguments:

```bash
python get_sl_emb.py  root  --lth_emb "01"  --input_model_file pretrained_model_dir --partial_csv`  xx.csv
```

where `csv` file contains the atoms you are interested in

## Generate multiscale material representations using trained GNN

Similarly, you can generate multi-scale material representations using:

```bash
python get_sl_emb_mat.py  root  --lth_emb "01"  --input_model_file pretrained_model_dir
```

The generated atomic representations are saved as `sl_cgcnn_mat_01_embedding_dict.npy`

## Machine learning pipeline

We can perform a kernel ridge regression on specified property by:

 ```bash
cd ml
python cg_gs.py  --root1 root1_dir --root2 roo2_dir
```

Here `root1_dir` is the path to your `id_prop.csv` file, while `root2_dir` is the path to your atomic representations, that is , `.npy file`

You will get 3 files after training finished, `cv_results` which records the results of your k-fold grid search; `y_results` which records the results of your test set; 
`best_dict` which records best hyperparameters of your grid search

By default, we use 10-fold cross-validation to choose the best hyperparameters.

