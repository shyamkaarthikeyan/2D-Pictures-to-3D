"""
Fallback implementation for torchmcubes using PyMCubes
"""
import torch
import numpy as np
import mcubes

def marching_cubes(volume, isolevel=0.0):
    """
    Fallback implementation of marching cubes using PyMCubes
    """
    # Convert torch tensor to numpy if needed
    if isinstance(volume, torch.Tensor):
        volume_np = volume.detach().cpu().numpy().astype(np.float32)
    else:
        volume_np = np.array(volume, dtype=np.float32)
    
    # Ensure isolevel is float
    isolevel = float(isolevel)
    
    # Use PyMCubes to extract mesh
    try:
        vertices, faces = mcubes.marching_cubes(volume_np, isolevel)
        
        # Convert back to torch tensors with proper device handling
        if isinstance(volume, torch.Tensor):
            device = volume.device
        else:
            device = torch.device('cpu')
            
        vertices = torch.from_numpy(vertices.astype(np.float32)).to(device)
        faces = torch.from_numpy(faces.astype(np.int64)).to(device)
        
        return vertices, faces
    except Exception as e:
        print(f"Error in marching cubes: {e}")
        # Return empty mesh as fallback
        if isinstance(volume, torch.Tensor):
            device = volume.device
        else:
            device = torch.device('cpu')
        vertices = torch.zeros((0, 3), dtype=torch.float32, device=device)
        faces = torch.zeros((0, 3), dtype=torch.int64, device=device)
        return vertices, faces
