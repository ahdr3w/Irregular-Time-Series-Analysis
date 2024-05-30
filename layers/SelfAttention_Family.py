import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from einops import rearrange, repeat
from torch.autograd import Function
import torch.nn.functional as F

class DSAttention(nn.Module):
    '''De-stationary Attention'''

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(DSAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        tau = 1.0 if tau is None else tau.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x 1
        delta = 0.0 if delta is None else delta.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x S

        # De-stationary Attention, rescaling pre-softmax score with learned de-stationary factors
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # real U = U_part(factor*ln(L_k))*L_q
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(
            L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H,
                                                L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) /
                     L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[
                                                  None, :, None], index, :] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * \
                 np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * \
            np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(
            queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class ReformerLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, causal=False, bucket_size=4, n_hashes=4):
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal
        )

    def fit_length(self, queries):
        # inside reformer: assert N % (bucket_size * 2) == 0
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            # fill the time series
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

    def forward(self, queries, keys, values, attn_mask, tau, delta):
        # in Reformer: defalut queries=keys
        B, N, C = queries.shape
        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None


class TwoStageAttentionLayer(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''

    def __init__(self, configs,
                 seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                           output_attention=configs.output_attention), d_model, n_heads)
        self.dim_sender = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                       output_attention=configs.output_attention), d_model, n_heads)
        self.dim_receiver = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                         output_attention=configs.output_attention), d_model, n_heads)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc, attn = self.time_attention(
            time_in, time_in, time_in, attn_mask=None, tau=None, delta=None
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
        dim_buffer, attn = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=None, tau=None, delta=None)
        dim_receive, attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer, attn_mask=None, tau=None, delta=None)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)

        return final_out


class CustomTopK(Function):
    @staticmethod
    def forward(ctx, inputs, k):
        _, indices = inputs.topk(k, dim=-1, sorted=False)
        seq_len = torch.tensor(inputs.shape[-1]).to(indices.device)
        indices = indices
        ctx.save_for_backward(seq_len, indices)
        return indices.float()
    @staticmethod
    def backward(ctx, grad_score):
        seq_len, indices = ctx.saved_tensors
        mask = F.one_hot(indices, seq_len).sum(dim=-2).bool()
        gradient = torch.zeros((mask.shape)).to(mask.device)
        gradient[mask] = grad_score.view(-1)
        return gradient, None

class SkipConnection(Function):
    @staticmethod
    def forward(ctx, index, K, Q):
        ctx.save_for_backward(torch.tensor([index.shape[-1]], dtype=int).to(K.device))
        return index, K, Q
    @staticmethod
    def backward(ctx, grad_index, grad_K, grad_Q):
        n_topks, = ctx.saved_tensors
        skip_grad = (grad_K+grad_Q).sum(dim=-2).sum(dim=-1).unsqueeze(-1).unsqueeze(-1)
        skip_grad = skip_grad.repeat(1, 1, 1, n_topks)
        return skip_grad, grad_K, grad_Q

skip_connection = SkipConnection.apply 
custom_topk = CustomTopK.apply

class LogitParametrization(torch.nn.Module):
    def forward(self, x):
        return torch.sigmoid(x)
class NegParametrization(torch.nn.Module):
    def forward(self, x):
        return torch.abs(x)

class CorrAttention(nn.Module):
    def __init__(self,
                 beta=0.0,
                 tau=1.0,
                 l=0.0,
                 factor=1,
                 dropout=0.0):
        super().__init__()
        self.factor = factor
        self.beta = nn.Parameter(torch.tensor(beta))
        self.tau = nn.Parameter(torch.tensor(tau))
        self.l = nn.Parameter(torch.tensor(l))
        nn.utils.parametrize.register_parametrization(self,
                                                      'beta',
                                                      LogitParametrization())
        nn.utils.parametrize.register_parametrization(self,
                                                      'l',
                                                      LogitParametrization())
        nn.utils.parametrize.register_parametrization(self,
                                                      'tau',
                                                      NegParametrization())
        self.dropout = nn.Dropout(dropout)

    def _lagged_cross_correlation_filtering(self, K, Q):
        _, _, L, d = K.shape
        fft_K = torch.fft.fft(K, n=L, dim=2)
        fft_Q_conj = torch.fft.fft(Q, n=L, dim=2).conj()
        cc = torch.fft.ifft(torch.einsum('...ia,...ib->...iab', fft_K, fft_Q_conj), dim=-3)
        cc = torch.abs(cc.real)
        diagonal_sum = torch.einsum('...aii->...a', cc)
        non_diagonal_sum = torch.einsum('...aij->...a', cc) - diagonal_sum
        score = self.l * diagonal_sum + (1 - self.l) * non_diagonal_sum
        topK = self.factor * int(np.ceil(np.log(L)))
        if topK > L:
            topK = int(np.ceil(np.log(L)))
        _, top_lags = score.topk(topK, dim=-1, sorted=False)
        # try:
        #     top_lags = custom_topk(score, topK)
        # except:
        #     print(topK, K.shape)
        return top_lags, topK

    def forward(self, Q, K, V):
        Q = Q.transpose(2, 1)
        K = K.transpose(2, 1)
        V = V.transpose(2, 1)
        Q, K = F.normalize(Q, dim=2), F.normalize(K, dim=2)

        top_lags, topk = self._lagged_cross_correlation_filtering(K, Q)
        score_instant = torch.einsum('...ji,...jk->...ik', K, Q)
        softmax_instant = F.softmax(score_instant / self.tau, dim=-1)
        softmax_instant = self.dropout(softmax_instant)
        V_instant = torch.einsum('...ij,...jk->...ik', V, softmax_instant)

        _, _, L, d = K.shape

        top_lags = top_lags.unsqueeze(-1).transpose(-1, -2)
        
        # top_lags, K, Q = skip_connection(top_lags, K, Q)

        top_lags = top_lags.int()
        idxs = torch.arange(L).reshape(1, 1, -1, 1).to(top_lags.device)
        index = (top_lags + idxs) % L
        index = index.unsqueeze(-1).expand(-1, -1, -1, -1, d).transpose(-1, -2)

        K = K.unsqueeze(-1).expand(-1, -1, -1, -1, topk)
        V = V.unsqueeze(-1).expand(-1, -1, -1, -1, topk)
        Q = Q.unsqueeze(-1).expand(-1, -1, -1, -1, topk)

        K_rolled = torch.gather(K, 2, index)
        V_rolled = torch.gather(V, 2, index)
        score_lagged = torch.einsum('...jia,...jka->...ika', K_rolled, Q)
        softmax_lagged = F.softmax(score_lagged / self.tau, dim=-2)
        softmax_lagged = self.dropout(softmax_lagged)
        V_lagged = torch.einsum('...ija,...jka->...ika', V_rolled, softmax_lagged)

        V_lagged_total = V_lagged.sum(-1)

        final_output = (1 - self.beta) * V_instant + self.beta * V_lagged_total

        return final_output.transpose(2, 1).contiguous() 

class CorrAttentionLayer(AttentionLayer):
    def __init__(self, 
                 attention, 
                 d_model, 
                 n_heads, 
                 corr_attention=None, 
                 n_corr_heads=0, 
                 d_keys=None, 
                 d_values=None
    ):
        super(CorrAttentionLayer, self).__init__(attention, d_model, n_heads, d_keys=None, d_values=None)

        self.corr_attention = corr_attention
        self.n_corr_heads = n_corr_heads if n_heads >= n_corr_heads else n_heads
        
    def corr_forward(self, queries, keys, values):
        Ht = self.n_heads - self.n_corr_heads
        
        Qc, Kc, Vc = queries[:,:,Ht:,:], keys[:,:,Ht:,:], values[:,:,Ht:,:]
        corr_out = self.corr_attention(Qc, Kc, Vc)
        
        if Ht > 0:
            Qt, Kt, Vt = queries[:,:,:Ht,:], keys[:,:,:Ht,:], values[:,:,:Ht,:]
            temp_out, _ = self.inner_attention(Qt, Kt, Vt, None, None, None)
            out = torch.cat((temp_out, corr_out), dim=2)
        else:
            out = corr_out
        attn = None
        
        return out, attn

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        if self.corr_attention is not None and self.n_corr_heads > 0:
            out, attn = self.corr_forward(queries, keys, values)
        else:
            out, attn = self.inner_attention(
                queries,
                keys,
                values,
                attn_mask,
                tau=tau,
                delta=delta
            )
            
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
