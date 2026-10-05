Directory structure :

AIDS
 |- AIDS
     |- AIDS_A.txt
     |- AIDS_edge_labels.txt
     |- AIDS_graph_indices.txt
     |- AIDS_graph_labels.txt
     |- AIDS_label_readme.txt
     |- AIDS_node_attributes.txt
     |- AIDS_node_labels.txt

ARI_NMI.ipynb : Consists of ARI and NMI calculations for justifications of Q1,Q2 and Q3
main.py : Original code, modified to return labels_main.npy file used for ARI 
main2.py : Same as above, labels_main2.npy returned upon exeution
main_Q2.py : Node label is taken as betweeness centrality , PyG execution, returns labels_pyg_bc.npy
main2_Q2.py: Same as above, manual, returns labels_raw_bc.npy
main_Q3.py: One-hot encoding for node labels,PyG, returns labels_pyg_onehot.npy
main2_Q3.py: Same as above, manual , returns labels_raw_onehot.py
main_Q3_v2.py: One hot for node id's, PyG, returns labels_pyg_onehot_v2.npy
main2_Q3_v2.py: Same as above, manual , returns labels_raw_onehot_v2.npy

To run these , in the terminal write:

python main_x.py / main2_x.py 

Ensure that the directory structure remains same, paths are taken care of 

Run all the main_x.py and main2_x.py files and according .npy files are saved and then execute cells in the jupyter notebook 
