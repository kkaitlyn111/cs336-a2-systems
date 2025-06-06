import torch
from einops import rearrange
import numpy as np
from einops import einsum
import triton
import triton.language as tl

class FlashAttentionTorch(torch.autograd.Function):
    @staticmethod
    def forward(context, Q, K, V, is_causal = False):
        Bq = 16
        Bk = 16
        if len(Q.shape) == 2: Q = Q.unsqueeze(0)
        if len(K.shape) == 2: K = K.unsqueeze(0)
        if len(V.shape) == 2: V = V.unsqueeze(0)    

        # now [batch, sequence_length, d]

        d = Q.shape[-1]
        device = Q.device

        # assume divisible, all powers of 2
        Q = rearrange(Q, "batch (Tq Bq) d -> Tq batch Bq d", Bq = Bq) # [Tq, batch, Bq, d]
        K = rearrange(K, "batch (Tk Bk) d -> Tk batch Bk d", Bk = Bk) # [Tk, batch, Bk, d]
        V = rearrange(V, "batch (Tk Bk) d -> Tk batch Bk d", Bk = Bk)

        d = Q.shape[-1]
        Tq = Q.shape[0]
        Tk = K.shape[0]

        batch = Q.shape[1]

        O = torch.zeros(Tq, batch, Bq, d, device=device)
        L = torch.zeros(Tq, batch, Bq, device=device)
        Mi = torch.full((batch, Bq), float('-inf'), device=device)

        for i in range(Tq):
            Qi = Q[i]

            Oi_prev = torch.zeros(batch, Bq, d, device=device)
            Li_prev = torch.zeros(batch, Bq, device=device)
            Mi_prev = torch.full((batch, Bq), float('-inf'), device=device)

            for j in range(Tk):
                Kj = K[j]
                Vj = V[j]

                Sij = einsum(Qi, Kj, "batch Bq d, batch Bk d -> batch Bq Bk") / np.sqrt(d)
                rowmax = torch.max(Sij, dim=2, keepdim=False)[0]
                Mij = torch.maximum(Mi_prev, rowmax)
                Pij = torch.exp(Sij - Mij.unsqueeze(-1))
                rowsum_Pij = torch.sum(Pij, dim=2, keepdim=False)

                Lij = (torch.exp(Mi_prev - Mij) * Li_prev) + rowsum_Pij
                
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
        Q = rearrange(Q, 'Tq batch Bq d -> batch (Tq Bq) d')
        K = rearrange(K, 'Tk batch Bk d -> batch (Tk Bk) d')
        V = rearrange(V, 'Tk batch Bk d -> batch (Tk Bk) d')
        
        context.save_for_backward(Q, K, V, O, L)
        context.is_causal = is_causal
        return O

    @staticmethod
    def backward(context, dO):
        Q, K, V, O, L = context.saved_tensors  
        is_causal = context.is_causal
        return compiled_backward(Q, K, V, O, L, dO, is_causal)
    
class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(context, Q, K, V, is_causal = False):

        Bq = 16
        Bk = 16

        device = torch.device("cuda")
        Q = Q.to(device)
        V = V.to(device)
        K =K.to(device)

        if len(Q.shape) == 2: Q = Q.unsqueeze(0)
        if len(K.shape) == 2: K = K.unsqueeze(0)
        if len(V.shape) == 2: V = V.unsqueeze(0)

        d = Q.shape[-1]
        
        N_QUERIES = Q.shape[-2]
        N_KEYS = K.shape[-2]
        T_q = N_QUERIES // Bq

        batch= Q.shape[0]
        O = torch.empty((batch, N_QUERIES, d), dtype = torch.float32).to(device)
        L = torch.empty((batch, N_QUERIES), dtype = torch.float32).to(device)
        
        launch_grid = (T_q, batch)
        scale = 1.0 / np.sqrt(d)

        flash_attention_kernel[launch_grid](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES=N_QUERIES, N_KEYS=N_KEYS,
            scale=scale,
            D=d, Q_TILE_SIZE=Bq, K_TILE_SIZE=Bk, is_causal=is_causal)

        context.save_for_backward(Q, K, V, O, L)
        context.is_causal = is_causal
        return O
    
    @staticmethod
    def backward(context, dO):
        Q, K, V, O, L = context.saved_tensors
        is_causal = context.is_causal
        return compiled_backward(Q, K, V, O, L, dO, is_causal)


@triton.jit
def flash_attention_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale, # 1 / sqrt(d)
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr):

    query_tile_index = tl.program_id(0)
    # n_query_tiles = N_QUERIES // Q_TILE_SIZE
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(Q_ptr + batch_index * stride_qb,
                                    shape = (N_QUERIES, D),
                                    strides = (stride_qq, stride_qd),
                                    offsets = (query_tile_index * Q_TILE_SIZE, 0),
                                    block_shape = (Q_TILE_SIZE, D),
                                    order = (1, 0),)
    K_block_ptr = tl.make_block_ptr(K_ptr + batch_index * stride_kb,
                                    shape = (N_KEYS, D),
                                    strides = (stride_kk, stride_kd),
                                    offsets = (0, 0),
                                    block_shape = (K_TILE_SIZE, D),
                                    order = (1, 0),)

    V_block_ptr = tl.make_block_ptr(V_ptr + batch_index * stride_vb,
                                    shape = (N_KEYS, D),
                                    strides = (stride_vk, stride_vd),
                                    offsets = (0, 0),
                                    block_shape = (K_TILE_SIZE, D),
                                    order = (1, 0),)
    
    O_block_ptr = tl.make_block_ptr(O_ptr + batch_index * stride_ob,
                                    shape = (N_QUERIES, D),
                                    strides = (stride_oq, stride_od),
                                    offsets = (query_tile_index * Q_TILE_SIZE, 0),
                                    block_shape = (Q_TILE_SIZE, D),
                                    order = (1, 0),)

    L_block_ptr = tl.make_block_ptr(L_ptr + batch_index * stride_lb,
                                    shape = (N_QUERIES,),
                                    strides = (stride_lq,),
                                    offsets = (query_tile_index * Q_TILE_SIZE,),
                                    block_shape = (Q_TILE_SIZE,),
                                    order = (0,))

    Q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    Tk = tl.cdiv(N_KEYS, K_TILE_SIZE)

    oi_prev = tl.zeros((Q_TILE_SIZE,D), dtype=tl.float32)
    mi_prev = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)
    li_prev = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)

    # Sij = tl.zeros((Q_TILE_SIZE, K_TILE_SIZE), dtype=tl.float32)
    # rowmax = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)

    # for j in range(Tk):
    #     Kj = tl.load(K_block_ptr)
    #     Vj = tl.load(V_block_ptr)
        
    #     Sij = tl.dot(Q, tl.trans(Kj)) * scale

    #     sij_max = tl.max(Sij, axis=1)
    #     pij = tl.exp(Sij - sij_max[:, None])
    #     mij = tl.maximum(mi_prev, sij_max)

    #     lij = li_prev * tl.exp(mi_prev - mij) + tl.sum(pij, axis=-1)
    #     diag = tl.exp(mi_prev - mij)
    #     scaled_oi_prev = oi_prev * diag[:, None]
    #     # for an accumulation, make sure to set high precision
    #     tmp = tl.dot(pij.to(tl.float32), Vj.to(tl.float32))
    #     oij = tmp + scaled_oi_prev

    #     K_block_ptr = K_block_ptr.advance((K_TILE_SIZE,0))
    #     V_block_ptr = V_block_ptr.advance((K_TILE_SIZE,0))

    #     oi_prev = oij
    #     mi_prev = mij
    #     li_prev = lij

    for j in range(Tk):
        Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        Sij = tl.dot(Q, tl.trans(Kj)) * scale

        if is_causal:
            query_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            key_offsets = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            mask = query_offsets[:, None] < key_offsets[None, :]
            Sij = tl.where(mask, float('-inf'), Sij)

        sij_max = tl.max(Sij, axis=1)
        mij = tl.maximum(mi_prev, sij_max)
        pij = tl.exp(Sij - mij[:, None])
        

        lij = li_prev * tl.exp(mi_prev - mij) + tl.sum(pij, axis=-1)
        diag = tl.exp(mi_prev - mij)
        scaled_oi_prev = oi_prev * diag[:, None]

        tmp = tl.dot(pij.to(tl.float32), Vj.to(tl.float32))
        oij = tmp + scaled_oi_prev

        oi_prev = oij
        mi_prev = mij
        li_prev = lij

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE,0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE,0))

    inv_Li_diag = 1.0 / li_prev[:, None]
    oi = inv_Li_diag * oi_prev
    li = mi_prev + tl.log(li_prev)

    # cast oi to black pointer's element type before storing
    # same with li
    oi = oi.to(O_block_ptr.type.element_ty)
    li = li.to(L_block_ptr.type.element_ty)

    tl.store(O_block_ptr, oi, boundary_check=(0, 1))
    tl.store(L_block_ptr, li, boundary_check=(0,))
    
     

def backward_pass_recomp(Q, K, V, O, L, dO, is_causal = False):
    Nq, Nk, d = Q.shape[-2], K.shape[-2], Q.shape[-1]
    if is_causal:
        mask = torch.triu(torch.ones(Nq, Nk, device=Q.device, dtype=torch.bool), diagonal=1)
    D = torch.sum(O * dO, dim=-1)
    S = einsum(Q, K, "... Nq d, ... Nk d -> ... Nq Nk") / np.sqrt(d)
    if is_causal:
        S = S.masked_fill(mask, float('-inf'))
    if L.shape != S.shape[:-1]:
        # Try to reshape L to match S's batch dims
        L = L.view(*S.shape[:-1])
    P = torch.exp(S - L.unsqueeze(-1))
    dV = einsum(P, dO, "... Nq Nk, ... Nq d -> ... Nk d")
    dP = einsum(dO, V, "... Nq d, ... Nk d -> ... Nq Nk")
    dS = P * (dP - D[..., None])
    dQ = einsum(dS, K, "... Nq Nk, ... Nk d -> ... Nq d") / np.sqrt(d)
    dK = einsum(dS, Q, "... Nq Nk, ... Nq d -> ... Nk d") / np.sqrt(d)
    return dQ, dK, dV, None
    
compiled_backward = torch.compile(backward_pass_recomp)
 

    




