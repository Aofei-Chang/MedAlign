import numpy as np

def get_patch_indices_from_box_xywh(box, image_size, patch_grid_size):
    """
    box: [x_min, y_min, width, height] in pixel coordinates
    image_size: (H, W) — actual image size
    patch_grid_size: (gh, gw) — number of patches (e.g., 24×24 for ViT-L)
    
    Returns: list of flat patch indices that fall inside the box
    """
    x_min, y_min, box_w, box_h = box
    x_max = x_min + box_w
    y_max = y_min + box_h
    
    H, W = image_size
    gh, gw = patch_grid_size

    # Compute patch size in pixels
    patch_h = H / gh
    patch_w = W / gw

    # Convert box coordinates to patch grid indices
    row_start = int(y_min // patch_h)
    row_end = int(np.ceil(y_max / patch_h))
    col_start = int(x_min // patch_w)
    col_end = int(np.ceil(x_max / patch_w))

    patch_indices = []
    for row in range(row_start, row_end):
        for col in range(col_start, col_end):
            if 0 <= row < gh and 0 <= col < gw:
                idx = row * gw + col  # flatten to 1D
                patch_indices.append(idx)
    
    return patch_indices

