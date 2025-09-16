import torch
import torch.nn.functional as F
from torch import nn, Tensor


def balance_loss(router_probs, expert_assignments):
    num_experts = expert_assignments.size(-1)
    assignment_density = torch.mean(expert_assignments, dim=0)
    routing_density = torch.mean(router_probs, dim=0)
    loss = torch.mean(routing_density * assignment_density)
    
    return loss


class Router(nn.Module):

    def __init__(self, in_dim, num_experts: int, capacity_factor: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.in_dim = in_dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.eps = eps
        self.proj = nn.Linear(in_dim, num_experts)

    def forward(self, inputs: Tensor, use_aux_loss=True):
        
        routing_probs = F.softmax(self.proj(inputs), dim=-1)
        raw_probs = routing_probs.clone()
        
        capacity = int(self.capacity_factor * inputs.size(0))
    
        topk_vals, topk_idx = routing_probs.topk(3, dim=-1)
        expert_mask = torch.zeros_like(routing_probs).scatter_(1, topk_idx, 1)
 
        norm_factor = masked_probs.sum(0, keepdim=True) + self.eps
        routing_weights = (masked_probs / norm_factor) * capacity

        if use_aux_loss:
            aux_loss = balance_loss(raw_probs, expert_mask)
            return routing_weights, aux_loss
        return routing_weights, None


class MoELayer(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_experts: int, capacity_factor: float = 1.0, mult: int = 4, use_aux_loss=True):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.mult = mult
        self.use_aux_loss = use_aux_loss

        self.experts = nn.ModuleList([ExpertVAE(in_dim, in_dim) for _ in range(num_experts)])
        self.router = Router(in_dim, num_experts, capacity_factor)

    def forward(self, inputs: Tensor):
        original_shape = inputs.shape
        
        if inputs.dim() == 3:
            B, T, D = inputs.shape
            outputs, total_loss = [], 0
            for t in range(T):
                out_t, loss_t = self._forward_step(inputs[:, t, :])
                outputs.append(out_t)
                total_loss += loss_t / T
            stacked = torch.stack(outputs, dim=1)
            assert stacked.shape == original_shape
            return stacked, total_loss
        out, loss = self._forward_step(inputs)
        assert out.shape == original_shape
        return out, loss

    def _forward_step(self, inputs):
        routing_weights, aux_loss = self.router(inputs, use_aux_loss=self.use_aux_loss)
        
        expert_outputs, kl_losses, variances = [], [], []
        for expert in self.experts:
            out, kl_loss, sigma = expert(inputs)
            expert_outputs.append(out)
            kl_losses.append(kl_loss)
            variances.append(sigma)
   
        stacked_outs = torch.stack(expert_outputs, dim=-1)
        moe_out = torch.sum(routing_weights.unsqueeze(-2) * stacked_outs, dim=-1)
        return moe_out


class MoE_block(nn.Module):
    
    def __init__(self, in_dim: int, num_experts: int = 4, mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.moe = MoELayer(in_dim, in_dim * mult, in_dim, num_experts)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, inputs: Tensor, residual=False):
        if residual:
            res = inputs
        out, loss = self.moe(inputs)
        if residual:
            out = self.norm(out + res)
        return out, loss


class DialogueMoE(nn.Module):
    def __init__(
        self,
        dim: int,  
        num_experts: int = 4, 
        capacity_factor: float = 1.0,  
        dropout: float = 0.1,  
        *args,
        **kwargs,
    ):
        super().__init__()
        self.moe_block = MoE_block(
            dim=dim,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
            dropout=dropout,
            *args,
            **kwargs
        )
    
    def forward(self, x):
        return self.moe_block(x)