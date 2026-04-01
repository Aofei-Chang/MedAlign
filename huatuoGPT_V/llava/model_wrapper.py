import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import orthogonal


class LayerWrapper(nn.Module):
    def __init__(
        self,
        block,
        W,
        epsilon = 1.0,  #! hyperparameters
        use_adapters=True,
    ):
        super(LayerWrapper, self).__init__()
        self.block = block
        if use_adapters:
            self.steer_W = W  #* (hid_dim,)
        self.epsilon = epsilon
        self.apply_vector = False
        self.visual_steered_states = None
        self.use_adapters = use_adapters
        
    def forward(self, *args, **kwargs): 
        outputs = self.block(*args, **kwargs)
        hidden_states = outputs[0]  #* (batch_size, seq_len, hid_dim)
        dim = hidden_states.shape[-1]
        sequence_length = hidden_states.shape[1]
        if sequence_length == 1:
            return outputs
        
        image_start_index = kwargs.get('image_token_start_index', 34)
        image_start_index = image_start_index[0] if isinstance(image_start_index, (list, tuple)) else image_start_index
        if image_start_index is None:
            image_start_index = 34

        if not self.use_adapters:
            #* If not using adapters, just return the block output
            self.visual_steered_states = hidden_states[:, image_start_index:image_start_index+576, :]
            return outputs
        E_v = hidden_states[:, image_start_index:image_start_index+576, :]  # shape: [B, 576, D]
        I_epsW = self.epsilon * self.steer_W       # shape: [D, D]
        E_v_steered = E_v @ I_epsW.T          # shape: [B, 576, D]

        # Reconstruct hidden_states without in-place operation
        hidden_states = torch.cat([
            hidden_states[:, :image_start_index, :],
            E_v + E_v_steered,
            hidden_states[:, image_start_index+576:, :]
        ], dim=1)

        self.visual_steered_states = hidden_states[:, image_start_index:image_start_index+576, :]
        return (hidden_states, *outputs[1:])
    
    def orthogonality_loss(self):
        """
        Computes the orthogonality regularization loss: ||WWᵀ - I||²_F
        """
        W = self.steer_W
        D = W.shape[0]
        WT_W = W @ W.T
        I = torch.eye(D, device=W.device, dtype=W.dtype)
        frobenius_norm = torch.norm(WT_W - I, p='fro') ** 2
        return frobenius_norm / (D * D)
    
    def calculate_spatial_align_loss(self, gt_matrix, image_token_start_index=[34], attn_weight=None):
        """
        Calculate the spatial alignment loss between selected LLM hidden states and ground truth image patch similarity.

        Args:
            hidden_states: list of hidden states for each layer [num_layers][batch, seq_len, dim]
            gt_matrix: [576, 576] cosine similarity target matrix
            layers: list of layer indices to use
            epsilon: scaling for the loss (optional)
            image_token_start_index: where image tokens start
            question_token_end_index: (optional) where image tokens end; default is all remaining tokens after image start

        Returns:
            scalar loss value (averaged across layers)
        """
        hs = self.visual_steered_states  # shape: [1, seq_len, dim]
        if hs is None:
            return 0.0
        # print(hs.shape, "hidden states shape")
        image_tokens = hs.squeeze(0)  # [576, D]
        # Normalize patch embeddings
        image_tokens = F.normalize(image_tokens, dim=-1)  # [576, D]

        # Compute cosine similarity matrix
        pred_sim = image_tokens @ image_tokens.T  # [576, 576]


        # MSE loss with target matrix
        if attn_weight is not None:
            weights = attn_weight.to(pred_sim.device)  # shape [L]
            weight_matrix = torch.ger(weights, weights)  # shape [L, L], outer product
            mse = (pred_sim - gt_matrix) ** 2
            loss = (mse * weight_matrix).mean()
        else:
            loss = F.mse_loss(pred_sim, gt_matrix.to(pred_sim.device))
            

        return loss



class AttnWrapper(nn.Module):
    def __init__(
        self,
        block: nn.Module,
    ):
        super().__init__()
        self.block = block

    def forward(self, *args, **kwargs):
        
        hidden_states = kwargs.get('hidden_states', None)
        B, L, D = hidden_states.shape
        if L == 1:
            outputs = self.block(*args, **kwargs)
            return outputs
        
        outputs = self.block(*args, **kwargs)
        self.layer_attn_weights = outputs[1]  # [B, num_heads, seq_len, seq_len]

        return outputs
    
    def get_attention_map_loss(self, attention_map_label, image_token_start_index=[34], image_length=576, question_token_end_index=[-1]):
        image_start_index = image_token_start_index[0]
        question_token_end_index = question_token_end_index[0]
        attention_map_layer = self.layer_attn_weights  # [num_heads, seq_len, seq_len]
        L = attention_map_layer.shape[-1]
        attention_map = attention_map_layer.squeeze(0).mean(0)  # [seq_len, seq_len]
        if question_token_end_index >= L:
            question_token_end_index = -1
        
        # use average context token to image token attention, instead of the last context token to image token
        pred_attention_map = attention_map[image_start_index + image_length:question_token_end_index, image_start_index:image_start_index + image_length]
        pred_attention_map = pred_attention_map.mean(0)
        pred_attention_map = F.normalize(pred_attention_map, dim=-1)  # [576]
        # KL loss between pred and gt
        attention_map_label = attention_map_label.squeeze(0)  # [576]
        attention_map_label = F.normalize(attention_map_label, dim=-1)
        # KL divergence loss
        loss = F.kl_div(pred_attention_map.log(), attention_map_label, reduction='batchmean')

        return loss

