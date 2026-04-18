import gradio as gr
import numpy as np
from pathlib import Path
import rasterio
import matplotlib.pyplot as plt
import io
from PIL import Image

# Dummy function to load precomputed predictions/data instead of live inference for speed
def load_tile_data(tile_id):
    DATA_ROOT = Path("data/makeathon-challenge/")
    s2_dir = DATA_ROOT / "sentinel-2" / "train" / f"{tile_id}__s2_l2a"
    files = sorted(list(s2_dir.glob("*.tif")))
    
    if not files:
        return None, None
        
    # Read the first available S2 true color (RGB typically bands 4,3,2 or similar. Assume indexing 3,2,1 for raw L2A if needed)
    with rasterio.open(files[-1]) as src:
        # Just grab first 3 bands for visualization dummy
        data = src.read((1,2,3)).transpose(1, 2, 0)
        # Normalize for display
        data = np.clip(data / 3000.0, 0, 1)
        
    # Dummy mock masks
    H, W = data.shape[:2]
    mock_prediction = np.zeros((H, W))
    mock_prediction[H//2:H//2+50, W//2:W//2+50] = 1 # Fake deforestation box
    
    mock_uncertainty = np.random.rand(H, W) * 0.5
    
    return data, mock_prediction, mock_uncertainty

def render_visualization(tile_id):
    s2_image, pred, uncertainty = load_tile_data(tile_id)
    if s2_image is None:
        return "Tile not found.", None, None
        
    # Plotting code
    fig1, ax1 = plt.subplots()
    ax1.imshow(s2_image)
    ax1.imshow(pred, cmap='Reds', alpha=0.4)
    ax1.set_title(f"S2 with Deforestation Prediction - Tile {tile_id}")
    ax1.axis('off')
    buf1 = io.BytesIO()
    plt.savefig(buf1, format='png', bbox_inches='tight')
    buf1.seek(0)
    
    fig2, ax2 = plt.subplots()
    im = ax2.imshow(uncertainty, cmap='viridis')
    ax2.set_title("MC Dropout Uncertainty")
    fig2.colorbar(im, ax=ax2)
    ax2.axis('off')
    buf2 = io.BytesIO()
    plt.savefig(buf2, format='png', bbox_inches='tight')
    buf2.seek(0)
    
    return Image.open(buf1), Image.open(buf2)

with gr.Blocks(title="Deforestation Detection Dashboard") as demo:
    gr.Markdown("# 🌳 osapiens Deforestation Detection Dashboard")
    gr.Markdown("Visualize Multimodal Model Predictions & MC Dropout Uncertainty")
    
    with gr.Row():
        with gr.Column(scale=1):
            tile_input = gr.Textbox(label="Enter Tile ID (e.g. tile_18NVJ)")
            btn = gr.Button("Analyze")
            
        with gr.Column(scale=3):
            output_pred = gr.Image(label="Prediction Overlay")
            output_unc = gr.Image(label="Uncertainty Map")
            
    btn.click(fn=render_visualization, inputs=tile_input, outputs=[output_pred, output_unc])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
