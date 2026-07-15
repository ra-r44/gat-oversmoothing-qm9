GAT Over-smoothing Explorer
A quick tool I put together to look at over-smoothing in Graph Attention Networks (GATs) using QM9 molecular graphs.

Basically, when you stack too many layers in a GNN, all the node embeddings start looking exactly the same. For molecules, this means the network forgets how to tell different atoms apart.

This repo has a small Gradio app where you can type in a molecule (as a SMILES string) and watch it happen. It runs the molecule through three different GAT setups (plain, with residual connections, and with LayerNorm) up to 16 layers deep, and plots the node diversity at each step. The plain one collapses, the other two don't.

Results
I trained the models on QM9 using Kaggle. Here's what it looks like (validation loss on the left, embedding diversity vs depth on the right). You can see the baseline model's embeddings totally collapse as it gets deeper, while the fixes hold up pretty well.

Kaggle Results

Running it locally
You'll need Python 3.8+ and a virtual environment (highly recommended, torch and rdkit can get messy otherwise).

pip install -r requirements.txtpython app.py
Then just open http://127.0.0.1:7860 in your browser.

Note: It only takes H, C, N, O, and F atoms since that's what QM9 covers, and I capped it at 60 atoms so it runs fast