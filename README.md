# DSB_GNN
Requirements:
  Python >= 3.6 /
  PyTorch>=1.6
  Numpy>=1.16.0
  torch-geometric>=1.6.0
  pandas>=0.24.0
  scikit-learn>=0.20.0

# GraphMaker
Save the NHEK and K562 datasets in the data directory, run the graphmaker.py file, and the running instructions are:
  python Graphmaker.py
The generated file is automatically saved in data/NHEK_dgl_data and data/K562_dgl_data folder.

# Model training
The operation instructions are:
  python K562_training.py --test_chr=X --device=device_num
  python NHEK_training.py --test_chr=X --device=device_num
Where, X represents the test chromosome number, ranging from 1 to 23, device_num is the GPU number. The trained model will be saved in the model folder.

# Model test
The operation instructions are:
  python K562_test.py --test_chr=X --device=device_num
  python NHEK_test.py --test_chr=X --device=device_num
The prediction results are saved in the prediction folder.

# Model explain
The operation instructions are:
  python K562_feature_explain.py --test_chr=X --device=device_num
  python NHEK_feature_explain.py --test_chr=X --device=device_num
  python K562_edge_explain.py --test_chr=X --device=device_num
  python NHEK_edge_explain.py --test_chr=X --device=device_num
The feature importance and edge importance corresponding to the feature number will be generated in the current folder.
