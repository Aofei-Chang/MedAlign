import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import orthogonal


class LayerWrapper(nn.Module):
    def __init__(
        self,
        block,
        W,
        hidden_dim=4096,
        epsilon = 1.0,  #! hyperparameters
        orthogonal_map='householder',
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
        # Initialize steer_W with orthogonal parametrization
        # base_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # self.steer_W = orthogonal(base_linear, orthogonal_map=orthogonal_map)
        
    def forward(self, *args, **kwargs): 
        outputs = self.block(*args, **kwargs)
        hidden_states = outputs[0]  #* (batch_size, seq_len, hid_dim)
        dim = hidden_states.shape[-1]
        sequence_length = hidden_states.shape[1]
        if sequence_length == 1:
            #* During generation, skip steering
            return outputs
        
        # print(f"hidden_states: {hidden_states.shape}, dim: {dim}")
        image_start_index = kwargs.get('image_token_start_index', 34)
        # print(image_start_index, "align, image_start_index")
        image_start_index = image_start_index[0] if isinstance(image_start_index, (list, tuple)) else image_start_index
        if image_start_index is None:
            image_start_index = 34

        if not self.use_adapters:
            #* If not using adapters, just return the block output
            self.visual_steered_states = hidden_states[:, image_start_index:image_start_index+576, :]
            return outputs
        E_v = hidden_states[:, image_start_index:image_start_index+576, :]  # shape: [B, 576, D]
        I_epsW = self.epsilon * self.steer_W       # shape: [D, D]
        # I_epsW = self.epsilon * self.steer_W.weight       # shape: [D, D]
        E_v_steered = E_v @ I_epsW.T          # shape: [B, 576, D]

        # Reconstruct hidden_states without in-place operation
        hidden_states = torch.cat([
            hidden_states[:, :image_start_index, :],
            E_v + E_v_steered,
            hidden_states[:, image_start_index+576:, :]
        ], dim=1)

        self.visual_steered_states = hidden_states[:, image_start_index:image_start_index+576, :]
        # print(self.visual_steered_states.shape, "visual_steered_states", hidden_states.shape)
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
            # print("attn_weight is not none", attn_weight.shape)
            # Apply attention weight to the predicted similarity matrix
            weights = attn_weight.to(pred_sim.device)  # shape [L]
            weight_matrix = torch.ger(weights, weights)  # shape [L, L], outer product
            mse = (pred_sim - gt_matrix) ** 2
            loss = (mse * weight_matrix).mean()
            # print("loss with weight", loss)
        else:
            loss = F.mse_loss(pred_sim, gt_matrix.to(pred_sim.device))
            

        return loss


class MoELayerWrapper(nn.Module):
    def __init__(
        self,
        block,
        bottleneck_dim: int = 256,
        num_adapters: int = 16,
        hidden_dim: int = 4096,
        epsilon = 1.0,  #! hyperparameters
        orthogonal_map='householder',
        use_adapters=True,
    ):
        super(MoELayerWrapper, self).__init__()
        self.block = block
        self.epsilon = epsilon
        self.apply_vector = False
        self.visual_steered_states = None
        self.use_adapters = use_adapters
        if self.use_adapters:
            self.steer_W_down = nn.Parameter(
                torch.randn(num_adapters, bottleneck_dim, hidden_dim) * 1e-3
            )
            self.steer_W_up = nn.Parameter(
                torch.randn(num_adapters, hidden_dim, bottleneck_dim) * 1e-3
            )

            # Gating head from query to adapter weights
            self.query_gate_proj = nn.Linear(hidden_dim, num_adapters)

        # Caching for generation
        self.cached_query_end_index = None
        self.steering_applied = False
        self.dropout = nn.Dropout(p=0.05)
        
    def forward(self, *args, **kwargs): 
        outputs = self.block(*args, **kwargs)
        hidden_states = outputs[0]  #* (batch_size, seq_len, hid_dim)
        # dim = hidden_states.shape[-1]
        B, L, D = hidden_states.shape
        sequence_length = hidden_states.shape[1]
        if sequence_length == 1:
            #* During generation, skip steering
            return outputs
        
        # print(f"hidden_states: {hidden_states.shape}, dim: {dim}")
        image_start_index = kwargs.get('image_token_start_index', 34)
        # print(image_start_index, "align, image_start_index")
        image_start_index = image_start_index[0] if isinstance(image_start_index, (list, tuple)) else image_start_index
        if image_start_index is None:
            image_start_index = 34
        query_end_index = kwargs.get('question_token_end_index', None)
        # print(f"query_end_index: {query_end_index}")
        query_end_index = query_end_index[0] if isinstance(query_end_index, (list, tuple)) else query_end_index
        if query_end_index is None or query_end_index >= L or query_end_index < 576 + image_start_index:
            query_end_index = -1  # Default to last token
        self.cached_query_end_index = query_end_index

        if not self.use_adapters:
            #* If not using adapters, just return the block output
            self.visual_steered_states = hidden_states[:, image_start_index:image_start_index+576, :]
            return outputs
        
        E_v = hidden_states[:, image_start_index:image_start_index+576, :]  # shape: [B, 576, D]
        E_q = hidden_states[:, query_end_index, :]  # [B, D]
        query_logits = self.query_gate_proj(E_q)  # [B, num_adapters]
        query_gates = torch.softmax(query_logits, dim=-1)  # [B, num_adapters]
        # print(query_gates.shape, query_gates, "query_gates")
        # Z_v_list = []
        # for i in range(self.steer_W_down.shape[0]):  # num_adapters
        #     Z_v = F.linear(E_v, self.epsilon * self.steer_W_down[i])  # [B, 576, bottleneck_dim]
        #     Z_v_list.append(Z_v)
        # Z_v_all = torch.stack(Z_v_list, dim=2)
        Z_v_all = torch.einsum('bsd,ahd->bsah', E_v, self.epsilon * self.steer_W_down)  # shape: [B, S, A, d]
        # print(Z_v_all.shape, "Z_v_all")
        # E_v_steered_all = torch.einsum('bsad,adh->bsah', Z_v_all, self.epsilon * self.steer_W_up)
        E_v_steered_all = torch.einsum('bsad,ahd->bsah', Z_v_all, self.epsilon * self.steer_W_up)
        # print(E_v_steered_all.shape, "E_v_steered_all", query_gates.shape)
        query_gates_expanded = query_gates.unsqueeze(-1).unsqueeze(1)
        E_v_steered = (query_gates_expanded * E_v_steered_all).sum(dim=2)
        
        E_v_final = E_v + E_v_steered

        # Reconstruct hidden_states without in-place operation
        hidden_states = torch.cat([
            hidden_states[:, :image_start_index, :],
            E_v_final,
            hidden_states[:, image_start_index+576:, :]
        ], dim=1)

        self.visual_steered_states = hidden_states[:, image_start_index:image_start_index+576, :]
        # print(self.visual_steered_states.shape, "visual_steered_states", hidden_states.shape)
        return (hidden_states, *outputs[1:])

    def calculate_spatial_align_loss(self, gt_matrix, image_token_start_index=[34]):
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
        # print(f"attn hidden_states: {hidden_states.shape}")
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
        # print(attention_map_layer.shape, "attention map layer shape", question_token_end_index)
        # print(attention_map_layer.shape, "attention map layer shape")
        attention_map = attention_map_layer.squeeze(0).mean(0)  # [seq_len, seq_len]
        # print(attention_map.shape, "attention map shape")
        if question_token_end_index >= L:
            question_token_end_index = -1
        pred_attention_map = attention_map[image_start_index + image_length:question_token_end_index, image_start_index:image_start_index + image_length]
        pred_attention_map = pred_attention_map.mean(0)  # [576]
        # pred_attention_map = attention_map[question_token_end_index, image_start_index:image_start_index + image_length] # [1, 576], take the last token's (of query) attention
        # pred_attention_map = pred_attention_map.squeeze(0)  # [576]
        #normalize the attention map
        pred_attention_map = F.normalize(pred_attention_map, dim=-1)  # [576]
        # KL loss between pred and gt
        attention_map_label = attention_map_label.squeeze(0)  # [576]
        attention_map_label = F.normalize(attention_map_label, dim=-1)
        # KL divergence loss
        loss = F.kl_div(pred_attention_map.log(), attention_map_label, reduction='batchmean')

        # Convert both to proper probability distributions
        # temperature = 0.5  # lower → sharper
        # pred_attention_map = F.log_softmax(pred_attention_map / temperature, dim=-1)
        # attention_map_label = F.softmax(attention_map_label.squeeze(0) / temperature, dim=-1)

        # KL divergence loss
        
        # loss = F.kl_div(pred_attention_map, attention_map_label, reduction='batchmean')

        return loss



class QueryGatedAdapter(nn.Module):
    def __init__(
        self,
        block: nn.Module,
        bottleneck_dim: int = 16,
        hidden_dim: int = 4096,
        num_adapters: int = 8,
        epsilon: float = 1.0,
        residual: bool = True,
        inference: bool = False,
        use_adapters: bool = True,
    ):
        super().__init__()
        self.block = block
        self.epsilon = epsilon
        self.residual = residual
        self.inference = inference
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.use_adapters = use_adapters

        # Mixture of adapters: each has its own down and up
        if self.use_adapters:
            self.steer_W_down = nn.Parameter(
                torch.randn(num_adapters, bottleneck_dim, hidden_dim) * 1e-3
            )
            self.steer_W_up = nn.Parameter(
                torch.randn(num_adapters, hidden_dim, bottleneck_dim) * 1e-3
            )

            # Gating head from query to adapter weights
            self.query_gate_proj = nn.Linear(hidden_dim, num_adapters)

        # Caching for generation
        self.cached_query_end_index = None
        self.steering_applied = False
        

    def forward(self, *args, **kwargs):
        
        hidden_states = kwargs.get('hidden_states', None)
        # print(f"attn hidden_states: {hidden_states.shape}")
        B, L, D = hidden_states.shape
        # print(f"hidden_states: {hidden_states.shape}, B:{B} L: {L}, D: {D}")
        # === Only apply steering once during full forward ===
        

        if L == 1 and self.steering_applied:
            outputs = self.block(*args, **kwargs)
            return outputs
        elif not self.use_adapters:
            # If not using adapters, just return the block output, but keep the attention weights
            outputs = self.block(*args, **kwargs)
            self.layer_attn_weights = outputs[1]
            return outputs
        # === Locate image and query tokens ===
        image_start_index = kwargs.get('image_token_start_index', 34)
        image_start_index = image_start_index[0] if isinstance(image_start_index, (list, tuple)) else image_start_index
        if image_start_index is None:
            image_start_index = 34

        query_end_index = kwargs.get('question_token_end_index', None)
        # print(f"query_end_index: {query_end_index}")
        query_end_index = query_end_index[0] if isinstance(query_end_index, (list, tuple)) else query_end_index
        if query_end_index is None or query_end_index >= L or query_end_index < 576 + image_start_index:
            query_end_index = -1  # Default to last token
        self.cached_query_end_index = query_end_index
        # print(f"image_start_index: {image_start_index}, query_end_index: {query_end_index}", L)
        # Extract visual tokens and query token
        E_v = hidden_states[:, image_start_index:image_start_index+576, :]  # [B, 576, D]
        # try:
        E_q = hidden_states[:, query_end_index, :]  # [B, D]
        # except:
        #     # If query_end_index is -1, use the last token
        #     E_q = hidden_states[:, -1, :]

        # === Project query to get gating over adapters ===
        query_logits = self.query_gate_proj(E_q)  # [B, num_adapters]
        query_gates = torch.softmax(query_logits, dim=-1)  # [B, num_adapters]

        # === Project visual tokens through each adapter ===
        Z_v_list = []
        for i in range(self.steer_W_down.shape[0]):  # num_adapters
            Z_v = F.linear(E_v, self.epsilon * self.steer_W_down[i])  # [B, 576, bottleneck_dim]
            Z_v_list.append(Z_v)

        Z_v_all = torch.stack(Z_v_list, dim=2)  # [B, 576, num_adapters, bottleneck_dim]
        # print(Z_v_all.shape, "Z_v_all")

        # === Correct up projection ===
        # W_up: [num_adapters, hidden_dim, bottleneck_dim]
        # We need: [num_adapters, bottleneck_dim, hidden_dim]
        W_up = self.steer_W_up.transpose(1, 2)  # [num_adapters, bottleneck_dim, hidden_dim]

        # Each adapter: project from bottleneck_dim to hidden_dim
        E_v_steered_all = torch.einsum('bsad,adh->bsah', Z_v_all, self.epsilon * W_up)  # [B, 576, num_adapters, hidden_dim]
        # print(E_v_steered_all.shape, "E_v_steered_all")

        # === Query-gated mixture ===
        # query_gates: [B, num_adapters]
        # Expand to [B, 1, num_adapters, 1]
        # print(query_gates.shape, "query_gates", E_v_steered_all.shape)
        query_gates_expanded = query_gates.unsqueeze(-1).unsqueeze(1)  # [B, 1, num_adapters, 1]

        # Elementwise multiply and sum over num_adapters
        E_v_steered = (query_gates_expanded * E_v_steered_all).sum(dim=2)  # [B, 576, hidden_dim]
        # print(E_v_steered.shape, "E_v_steered")


        # === Residual addition ===
        E_v_final = E_v + E_v_steered

        # === Update hidden states ===
        hidden_states = torch.cat([
            hidden_states[:, :image_start_index, :],
            E_v_final,
            hidden_states[:, image_start_index+576:, :]
        ], dim=1)

        self.steering_applied = True
        # Update kwargs with new hidden states
        kwargs['hidden_states'] = hidden_states

        outputs = self.block(*args, **kwargs)
        self.layer_attn_weights = outputs[1]  # [B, num_heads, seq_len, seq_len]

        return outputs
    
    def get_attention_map_loss(self, attention_map_label, image_token_start_index=[34], image_length=576, question_token_end_index=[-1]):
        image_start_index = image_token_start_index[0]
        question_token_end_index = question_token_end_index[0]
        attention_map_layer = self.layer_attn_weights  # [num_heads, seq_len, seq_len]
        # print(attention_map_layer.shape, "attention map layer shape")
        attention_map = attention_map_layer.squeeze(0).mean(0)  # [seq_len, seq_len]
        # print(attention_map.shape, "attention map shape")
        # pred_attention_map = attention_map[image_start_index + image_length:, image_start_index:image_start_index + image_length]
        pred_attention_map = attention_map[question_token_end_index, image_start_index:image_start_index + image_length] # [1, 576], take the last token's (of query) attention
        pred_attention_map = pred_attention_map.squeeze(0)  # [576]
        #normalize the attention map
        pred_attention_map = F.normalize(pred_attention_map, dim=-1)  # [576]
        # KL loss between pred and gt
        attention_map_label = attention_map_label.squeeze(0)  # [576]
        attention_map_label = F.normalize(attention_map_label, dim=-1)
        # KL divergence loss
        loss = F.kl_div(pred_attention_map.log(), attention_map_label, reduction='batchmean')

        # Convert both to proper probability distributions
        # temperature = 0.5  # lower → sharper
        # pred_attention_map = F.log_softmax(pred_attention_map / temperature, dim=-1)
        # attention_map_label = F.softmax(attention_map_label.squeeze(0) / temperature, dim=-1)

        # KL divergence loss
        
        # loss = F.kl_div(pred_attention_map, attention_map_label, reduction='batchmean')

        return loss


def calculate_spatial_align_loss(hidden_states, gt_matrix, layers="0,13", image_token_start_index=34, question_token_end_index=-1):
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
    layers = [int(layer) for layer in layers.split(",")]
    loss_total = 0.0
    n_layers = len(layers)

    for layer in layers:
        # print(layer, "layer")
        if layer < 0:
            continue
        # if layer == 0:
        #     n_layers -= 1
        #     continue
        hs = hidden_states[layer]  # shape: [1, seq_len, dim]
        # print(hs.shape, "hidden states shape")
        image_tokens = hs[0, image_token_start_index[0]:image_token_start_index[0]+576, :]  # [576, D]

        # Normalize patch embeddings
        image_tokens = F.normalize(image_tokens, dim=-1)  # [576, D]

        # Compute cosine similarity matrix
        pred_sim = image_tokens @ image_tokens.T  # [576, 576]

        # MSE loss with target matrix
        loss = F.mse_loss(pred_sim, gt_matrix.to(pred_sim.device))
        loss_total += loss

    return loss_total / n_layers if n_layers > 0 else 0.0

def get_attention_map_loss(attention_map_label, attention_map_pred, image_start_index, layers="23", image_length=576):
    layers = [int(layer)-1 for layer in layers.split(",")]
    image_start_index = image_start_index[0]
    attention_map_layer_selected = []
    for layer in layers:
        attention_map_layer = attention_map_pred[layer]
        attention_map_layer_selected.append(attention_map_layer)
    attention_map_layer = torch.stack(attention_map_layer_selected, dim=0).mean(0)  # [num_heads, seq_len, seq_len]
    attention_map = attention_map_layer.squeeze(0).mean(0)  # [seq_len, seq_len]
    # print(attention_map.shape, "attention map shape")
    # pred_attention_map = attention_map[image_start_index + image_length:, image_start_index:image_start_index + image_length]
    pred_attention_map = attention_map[-1, image_start_index:image_start_index + image_length] # [1, 576], take the last token's attention
    pred_attention_map = pred_attention_map.squeeze(0)  # [576]
    #normalize the attention map
    pred_attention_map = F.normalize(pred_attention_map, dim=-1)  # [576]
    # KL loss between pred and gt
    attention_map_label = attention_map_label.squeeze(0)  # [576]
    attention_map_label = F.normalize(attention_map_label, dim=-1)
    # KL divergence loss
    loss = F.kl_div(pred_attention_map.log(), attention_map_label, reduction='batchmean')

    return loss



# class BottleNeckLayerWrapper(nn.Module):
#     def __init__(
#         self,
#         block: nn.Module,
#         bottleneck_dim: int,
#         hidden_dim: int,
#         epsilon: float = 1.0,
#         residual: bool = True,
#         lambda_sparse: float = 1e-2,
#         lambda_recons: float = 0.05,
#         use_reconstruction: bool = True,
#         inference: bool = False,
#     ):
#         super().__init__()
#         self.block = block
#         self.epsilon = epsilon
#         self.residual = residual

#         self.lambda_sparse = lambda_sparse
#         self.lambda_recons = lambda_recons
#         self.use_reconstruction = use_reconstruction

#         # Shared down projection
#         self.steer_W_down = nn.Parameter(
#             torch.randn(bottleneck_dim, hidden_dim) * 1e-3
#         )

#         # Shared up projection
#         self.steer_W_up = nn.Parameter(
#             torch.zeros(hidden_dim, bottleneck_dim)
#         )

#         # Loss placeholders
#         self.loss_sparse = 0.0
#         self.loss_recons = 0.0

#         # Inference settings
#         self.inference = inference
#         self.cached_query_end_index = None  # Save query end index after first forward
#         self.steering_applied = False        # Flag to indicate if steering was already applied

#     def forward(self, *args, **kwargs):
#         outputs = self.block(*args, **kwargs)
#         hidden_states = outputs[0]  # [B, seq_len, hidden_dim]
#         B, L, D = hidden_states.shape

#         # === Detect First Full Forward or Generation, considering the inference time ===
#         if L == 1 and self.steering_applied:
#             # During generation (after first step), or already applied
#             return outputs

#         # === First Full Forward: Apply Steering ===

#         # Get image tokens start index
#         image_start_index = kwargs.get('image_token_start_index', None)
#         if image_start_index is None:
#             image_start_index = 34
#         else:
#             image_start_index = image_start_index[0]

#         # Get query end index (record and reuse)
#         query_end_index = kwargs.get('question_token_end_index', None)
#         if query_end_index is None:
#             query_end_index = -1  # Default to last token
#         self.cached_query_end_index = query_end_index

#         # Extract visual tokens
#         E_v = hidden_states[:, image_start_index:image_start_index+576, :]  # [B, 576, D]
#         # Extract query token
#         E_q = hidden_states[:, query_end_index, :]  # [B, D]

#         # === Shared Down Projection ===
#         Z_v = F.linear(E_v, self.epsilon * self.steer_W_down)  # [B, 576, bottleneck_dim]
#         Z_q = F.linear(E_q, self.epsilon * self.steer_W_down)  # [B, bottleneck_dim]

#         # === Query-Guided Sparse Activation ===
#         query_gate = torch.sigmoid(Z_q).unsqueeze(1)  # [B, 1, bottleneck_dim]
#         Z_v_modulated = Z_v * query_gate  # [B, 576, bottleneck_dim]
#         # print(f"Z_v_modulated: {Z_v_modulated}, Z_q: {Z_q}, query_gate: {query_gate}")
#         # === Shared Up Projection ===
#         E_v_steered = F.linear(Z_v_modulated, self.epsilon * self.steer_W_up)  # [B, 576, D]
#         E_v_steered = E_v_steered.squeeze(0)  # [B, 576, D]
#         # === Residual Addition or Replacement ===
#         if self.residual:
#             E_v_final = E_v + E_v_steered
#         else:
#             E_v_final = E_v_steered
#         # print(f"steer_w: {self.steer_W_up.shape}, E_v_final: {E_v_final.shape}, hidden_states: {hidden_states.shape}")
#         # === Rebuild hidden_states ===
#         hidden_states = torch.cat([
#             hidden_states[:, :image_start_index, :],
#             E_v_final,
#             hidden_states[:, image_start_index+576:, :]
#         ], dim=1)

#         # === Loss Computation ===
#         if not self.inference:
#             self.loss_sparse = self.lambda_sparse * (Z_v.abs().mean() + Z_q.abs().mean())
#             if self.use_reconstruction:
#                 self.loss_recons = self.lambda_recons * F.mse_loss(E_v_steered, E_v)

#         # === Mark that steering was applied ===
#         self.steering_applied = True

#         # === Return updated hidden states ===
#         return (hidden_states, *outputs[1:])



# class BottleNeckLayerWrapper(nn.Module):
#     def __init__(
#         self,
#         block: nn.Module,
#         bottleneck_dim: int,
#         hidden_dim: int,
#         epsilon: float = 1.0,
#         residual: bool = True,
#         lambda_sparse: float = 1e-2,
#         lambda_recons: float = 0.05,
#         use_reconstruction: bool = True,
#         inference: bool = False,
#     ):
#         super().__init__()
#         self.block = block
#         self.epsilon = epsilon
#         self.residual = residual

#         self.lambda_sparse = lambda_sparse
#         self.lambda_recons = lambda_recons
#         self.use_reconstruction = use_reconstruction

#         # Shared down projection
#         self.steer_W_down = nn.Parameter(
#             torch.randn(bottleneck_dim, hidden_dim) * 1e-3
#         )

#         # Shared up projection
#         self.steer_W_up = nn.Parameter(
#             torch.zeros(hidden_dim, bottleneck_dim)
#         )

#         # Loss placeholders
#         self.loss_sparse = 0.0
#         self.loss_recons = 0.0

#         # Inference settings
#         self.inference = inference
#         self.cached_query_end_index = None  # Save query end index after first forward
#         self.steering_applied = False        # Flag to indicate if steering was already applied

#     def forward(self, *args, **kwargs):
#         outputs = self.block(*args, **kwargs)
#         hidden_states = outputs[0]  # [B, seq_len, hidden_dim]
#         B, L, D = hidden_states.shape

#         # === Detect First Full Forward or Generation, considering the inference time ===
#         if L == 1 and self.steering_applied:
#             # During generation (after first step), or already applied
#             return outputs

#         # === First Full Forward: Apply Steering ===

#         # Get image tokens start index
#         image_start_index = kwargs.get('image_token_start_index', None)
#         if image_start_index is None:
#             image_start_index = 34
#         else:
#             image_start_index = image_start_index[0]

#         # Get query end index (record and reuse)
#         query_end_index = kwargs.get('question_token_end_index', None)
#         if query_end_index is None:
#             query_end_index = -1  # Default to last token
#         self.cached_query_end_index = query_end_index

#         # Extract visual tokens
#         E_v = hidden_states[:, image_start_index:image_start_index+576, :]  # [B, 576, D]
#         # Extract query token
#         E_q = hidden_states[:, query_end_index, :]  # [B, D]

#         # === Shared Down Projection ===
#         Z_v = F.linear(E_v, self.epsilon * self.steer_W_down)  # [B, 576, bottleneck_dim]
#         Z_q = F.linear(E_q, self.epsilon * self.steer_W_down)  # [B, bottleneck_dim]

#         # === Query-Guided Sparse Activation ===
#         query_gate = torch.sigmoid(Z_q).unsqueeze(1)  # [B, 1, bottleneck_dim]
#         Z_v_modulated = Z_v * query_gate  # [B, 576, bottleneck_dim]
#         # print(f"Z_v_modulated: {Z_v_modulated}, Z_q: {Z_q}, query_gate: {query_gate}")
#         # === Shared Up Projection ===
#         E_v_steered = F.linear(Z_v_modulated, self.epsilon * self.steer_W_up)  # [B, 576, D]
#         E_v_steered = E_v_steered.squeeze(0)  # [B, 576, D]
#         # === Residual Addition or Replacement ===
#         if self.residual:
#             E_v_final = E_v + E_v_steered
#         else:
#             E_v_final = E_v_steered
#         # print(f"steer_w: {self.steer_W_up.shape}, E_v_final: {E_v_final.shape}, hidden_states: {hidden_states.shape}")
#         # === Rebuild hidden_states ===
#         hidden_states = torch.cat([
#             hidden_states[:, :image_start_index, :],
#             E_v_final,
#             hidden_states[:, image_start_index+576:, :]
#         ], dim=1)

#         # === Loss Computation ===
#         if not self.inference:
#             self.loss_sparse = self.lambda_sparse * (Z_v.abs().mean() + Z_q.abs().mean())
#             if self.use_reconstruction:
#                 self.loss_recons = self.lambda_recons * F.mse_loss(E_v_steered, E_v)

#         # === Mark that steering was applied ===
#         self.steering_applied = True

#         # === Return updated hidden states ===
#         return (hidden_states, *outputs[1:])


def compute_sparse_mi_losses(Z_v, Z_q, lambda_sparse=1e-3, lambda_mi=1.0):
    """
    Args:
        Z_v: [B, 576, K]
        Z_q: [B, K]
    Returns:
        loss_sparse, loss_mi
    """
    # Sparsity Loss (L1 norm)
    loss_sparse = lambda_sparse * (Z_v.abs().mean() + Z_q.abs().mean())

    # Mutual Information Loss (maximize cosine similarity)
    Z_v_mean = Z_v.mean(dim=1)  # [B, K]
    cos_sim = F.cosine_similarity(Z_q, Z_v_mean, dim=-1)  # [B]
    loss_mi = -lambda_mi * cos_sim.mean()

    return loss_sparse, loss_mi
