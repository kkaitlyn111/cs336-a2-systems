import torch
from einops import rearrange
import numpy as np
from einops import einsum

class FlashAttentionTorch(torch.autograd.Function):
    @staticmethod
    def forward(context, Q, K, V, is_causal = True):
        Bq = 16
        Bk = 16

        if len(Q.shape) == 2: Q = Q.unsqueeze(0)
        if len(K.shape) == 2: K = K.unsqueeze(0)
        if len(V.shape) == 2: V = V.unsqueeze(0)

        # now [batch, sequence_length, d]

        d = Q.shape[-1]

        Q = rearrange(Q, "batch (Tq Bq) d -> Tq batch Bq d", Bq = Bq) # [Tq, batch, Bq, d]
        K = rearrange(K, "batch (Tk Bk) d -> Tk batch Bk d", Bk = Bk) # [Tk, batch, Bk, d]
        V = rearrange(V, "batch (Tk Bk) d -> Tk batch Bk d", Bk = Bk)

        d = Q.shape[-1]
        Tq = Q.shape[0]
        Tk = K.shape[0]

        assert K.shape == V.shape 
        assert Q.shape[1] == K.shape[1]

        batch = Q.shape[1]

        O = torch.zeros(Tq, batch, Bq, d)
        L = torch.zeros(Tq, batch, Bq)
        Mi = torch.full((batch, Bq), float('-inf'))

        for i in range(Tq):
            Qi = Q[i]

            Oi_prev = torch.zeros(batch, Bq, d)
            Li_prev = torch.zeros(batch, Bq)
            Mi_prev = torch.full((batch, Bq), float('-inf'))

            for j in range(Tk):
                Kj = K[j]
                Vj = V[j]

                Sij = einsum(Qi, Kj, "batch Bq d, batch Bk d -> batch Bq Bk") / np.sqrt(d)
                rowmax = torch.max(Sij, dim=2, keepdim=True)
                Mij = torch.max(Mi_prev, rowmax.squeeze(-1))
                Pij = torch.exp(Sij - Mij)
                rowsum_Pij = torch.sum(Pij, dim=2, keepdim=True)
                Lij = torch.exp(Mi_prev - Mij) * Li_prev + rowsum_Pij
                
                # formula: diag(exp(Mi[j-1] - Mi[j])) * Oi[j-1] + Pij * Vj
                diag_scale = torch.diag_embed(torch.exp(Mi_prev - Mij))  # [batch, Bq, Bq]
                Oij = einsum(diag_scale, Oi_prev, "batch Bq Bq, batch Bq d -> batch Bq d") + \
                       einsum(Pij, Vj, "batch Bq Bk, batch Bk d -> batch Bq d")

                Oi_prev = Oij
                Li_prev = Lij
                Mi_prev = Mij

            inv_Li_diag = torch.diag_embed(1.0 / Li_prev)  # [batch, Bq, Bq]    
            O[i] = einsum(inv_Li_diag, Oi_prev, "batch Bq Bq, batch Bq d -> batch Bq d")
            L[i] = Mi_prev + torch.log(Li_prev)
        
        O = rearrange(O, 'Tq batch Bq d -> batch (Tq Bq) d')
        L = rearrange(L, 'Tq batch Bq -> batch (Tq Bq)')
        
        context.save_for_backward(Q, K, V, O, L)
        
        return O, L

    @staticmethod
    def backward(context, dO):
        print(f"dO: {dO.shape}")
        return dO, None, None, None
    