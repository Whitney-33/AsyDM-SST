import torch
import matplotlib.pyplot as plt
import numpy as np

def compute_anomaly_mask(x1_com, x3, sigma=0.2):
    """Compute anomaly"""
    diff = torch.abs(x1_com - x3)
    anomaly_mask = torch.sigmoid(diff / sigma)
    return anomaly_mask

# def visualize_anomaly_mask(anomaly_mask, i=0, title="Anomaly Mask", save_path=None):
#     plt.figure(figsize=(10, 8))

#     if isinstance(anomaly_mask, torch.Tensor):
#         anomaly_np = anomaly_mask.squeeze().detach().cpu().numpy()
#     else:
#         anomaly_np = anomaly_mask.squeeze()

#     im = plt.imshow(anomaly_np, cmap='YlOrRd', vmin=0, vmax=1)
    
#     cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
#     cbar.set_label('Anomaly Score', rotation=270, labelpad=15)
    
#     plt.title(f"{title} - Batch {i}", fontsize=14, fontweight='bold')
#     plt.axis('off')
    
#     height, width = anomaly_np.shape
#     if height <= 20 and width <= 20:
#         for i in range(height):
#             for j in range(width):
#                 plt.text(j, i, f'{anomaly_np[i, j]:.2f}', 
#                         ha='center', va='center', 
#                         color='white' if anomaly_np[i, j] > 0.5 else 'black',
#                         fontsize=8)
    
#     plt.tight_layout()
    
#     if save_path is None:
#         save_path = f"batch{i}_anomaly_mask.png"
#     else:
#         if not save_path.startswith(f"batch{i}_"):
#             save_path = f"batch{i}_{save_path}"
    
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     # print(f"save: {save_path}")
    