
### 2. `app.py`
```python
# app.py - Gradio demo for the over-smoothing project
# run with: python app.py

import os
import torch
import gradio as gr
import matplotlib.pyplot as plt

from featurize import smiles_to_graph
from models import MODEL_REGISTRY
from metrics import mad_per_layer

torch.manual_seed(0)

# just some quick examples to populate the UI
EXAMPLE_MOLECULES = [
    ["Ethanol", "CCO"],
    ["Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"],
    ["Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"],
    ["Benzene", "c1ccccc1"],
    ["Formaldehyde", "C=O"],
    ["Furan", "c1ccoc1"],
]

# drop trained .pt files in here (e.g. Baseline_L8.pt) to use real weights instead of random ones
CHECKPOINT_DIR = "checkpoints"

def _has_trained_weights(arch_name, depth):
    fname = f"{arch_name.split()[0]}_L{depth}.pt"
    path = os.path.join(CHECKPOINT_DIR, fname)
    return path if os.path.exists(path) else None

def build_model(arch_name, depth, in_channels):
    cls = MODEL_REGISTRY[arch_name]
    model = cls(in_channels=in_channels, hidden_channels=16, out_channels=1,
                heads=4, dropout=0.0, num_layers=depth)
    model.eval()

    ckpt_path = _has_trained_weights(arch_name, depth)
    if ckpt_path:
        try:
            state = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state)
            return model, True
        except Exception:
            pass  # just use random weights if the checkpoint shape doesn't match
    return model, False

def run_demo(smiles, max_depth):
    x, edge_index, error = smiles_to_graph(smiles)
    if error:
        return None, f"Error: {error}", ""

    # figure out which depths to plot so the graph looks decent
    depths_to_plot = sorted(set([2, 4, 8, min(max_depth, 16)] + list(range(2, max_depth + 1, max(1, max_depth // 6)))))
    depths_to_plot = [d for d in depths_to_plot if d <= max_depth and d >= 2]

    results = {}
    any_trained = False
    for arch_name in MODEL_REGISTRY:
        mads_by_depth = []
        for d in depths_to_plot:
            model, is_trained = build_model(arch_name, d, in_channels=x.shape[1])
            any_trained = any_trained or is_trained
            with torch.no_grad():
                _, layer_embeddings = model(x, edge_index, return_all_layers=True)
            mads = mad_per_layer(layer_embeddings)
            mads_by_depth.append(mads[-1])
        results[arch_name] = mads_by_depth

    # plot it
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"Baseline (no mitigation)": "#d62728", "Residual connections": "#2ca02c", "LayerNorm": "#1f77b4"}
    for arch_name, mads in results.items():
        ax.plot(depths_to_plot, mads, marker="o", label=arch_name, color=colors.get(arch_name))
    ax.set_xlabel("Number of GAT layers (depth)")
    ax.set_ylabel("MAD - node embedding diversity")
    ax.set_title(f"Over-smoothing vs. depth\n({x.shape[0]} atoms, {edge_index.shape[1]//2} bonds)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if any_trained:
        mode_note = "Using trained weights loaded from checkpoints/."
    else:
        mode_note = (
            "Note: running on randomly initialized weights right now. "
            "I did this so the demo works instantly without a training run. "
            "The baseline collapse is a structural issue with stacking GAT layers, "
            "so it happens even before training. If you want the real numbers, "
            "drop the trained .pt files into the checkpoints/ folder."
        )

    # build the markdown table
    table_md = "| Depth | Baseline | Residual | LayerNorm |\n|---|---|---|---|\n"
    for i, d in enumerate(depths_to_plot):
        row = [f"{results[name][i]:.3f}" for name in MODEL_REGISTRY]
        table_md += f"| {d} | {row[0]} | {row[1]} | {row[2]} |\n"

    return fig, mode_note, table_md

with gr.Blocks(title="GAT Over-smoothing Explorer") as demo:
    gr.Markdown("""
    # Over-smoothing in Graph Attention Networks

    This is a small tool for a project on why GATs get worse, not better, as you stack more layers onto them.

    Type in a molecule as a SMILES string below. The app turns it into a graph, runs it through three versions of a GAT (plain, residual connections, and LayerNorm) at depths from 2 to 16 layers, and plots how different the node embeddings still are from each other at each depth. When that line drops toward zero, the network can't tell the atoms apart anymore - that's over-smoothing.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            smiles_input = gr.Textbox(label="SMILES string", value="CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
                                       placeholder="e.g. CCO for ethanol")
            depth_slider = gr.Slider(minimum=2, maximum=16, step=1, value=16,
                                      label="Maximum depth to test")
            run_btn = gr.Button("Run over-smoothing analysis", variant="primary")
            gr.Examples(examples=EXAMPLE_MOLECULES, inputs=[gr.Textbox(visible=False), smiles_input],
                        label="Try an example molecule")
            gr.Markdown("Only H, C, N, O, F atoms are supported (QM9's coverage).")

        with gr.Column(scale=2):
            plot_output = gr.Plot(label="Embedding diversity vs. depth")
            mode_note_output = gr.Markdown()
            table_output = gr.Markdown()

    run_btn.click(fn=run_demo, inputs=[smiles_input, depth_slider],
                   outputs=[plot_output, mode_note_output, table_output])

if __name__ == "__main__":
    demo.launch()