import abc
import functools
import math
from typing import (
    Any,
    Callable,
    List,
    Mapping,
    NotRequired,
    Optional,
    Tuple,
    TypedDict,
    Unpack,
)

import torch
import torch.distributed
from torch import Tensor, nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch.nn import functional as F

from fms import distributed
from fms.distributed.tensorparallel import (
    copy_to_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
)
from fms.modules.linear import (
    LinearModuleShardingInfo,
    get_all_linear_type_to_sharding_maps,
    get_linear,
    get_linear_type,
)
from fms.modules.positions import PositionEncoder
from fms.modules.tp import TPModule

from torch.autograd import Function

class UniversalAttention(Function):
    @staticmethod
    def forward(kc, vc, xq, static_src, static_dest):
        b,h,r,l,d = xq.shape
        _,_,n,c,_ = kc.shape
        mask = torch.ones(c,l, dtype=torch.bool, device=xq.device)
        out = torch.empty(b,h,r,l,d,n, dtype=xq.dtype, device=xq.device)
        denom = torch.empty(b,h,r,l,n, dtype=xq.dtype, device=xq.device)
        affs = torch.empty(b,h,l, dtype=torch.float, device=xq.device)
        static_src = static_src.pow(1/3)
        static_dest = static_dest.pow(1/3)
        kt = kc.view(b,h,l,d).transpose(-2,-1)  # b h d l
        for i in range(n):
            k_ = kc[:,:,i]  # b h c d
            v_ = vc[:,:,i]  # b h c d
            static_src_ = static_src[:,:,i]  # b h c

            # Calculate decay matrix
            affinity = k_.matmul(kt).relu().pow(2/3).float()  # deltanet style decay
            affinity = affinity * static_src_.unsqueeze(-1) * static_dest.unsqueeze(-2)  # incorporate mamba-style and per-token decay
            affinity = torch.log1p(affinity.clamp(min=0, max=1-1e-6).neg())  # b h c l
            affinity = affinity.triu(i*c+1).cumsum(3)  # Accumulate decay with causal masking
            affinity = affinity.masked_fill(mask.tril(i*c-1), -1e12)  # Re-mask, with 1s on diagonal

            # Perform actual attention operation
            score = k_.unsqueeze(2).matmul(xq.transpose(-1,-2)).add(affinity.unsqueeze(2))  # b h r c l
            denom_ = score.logsumexp(dim=-2)  # b h r l
            out_ = score.transpose(-1,-2).softmax(dim=-1).to(dtype=xq.dtype).matmul(v_.unsqueeze(2))  # b h r l d

            out[...,i] = out_
            denom[...,i] = denom_
            affs[...,i*c:(i+1)*c] = affinity[...,-1]
        return out, denom, affs

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        kc,vc,xq,ss,sd = inputs
        # out,denom = outputs
        ctx.save_for_backward(kc,vc,xq,ss,sd)

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        kc,vc,xq,ss,sd = inputs
        # out,denom = outputs
        ctx.save_for_backward(kc,vc,xq,ss,sd)

    @staticmethod
    def backward(ctx, g_out, g_denom, g_affs):
        # Note: when using mixed precision, g_out is downcast but g_denom is always fp32
        kc,vc,xq,static_src,static_dest = ctx.saved_tensors
        dkc,dvc,dxq,dstat_src,dstat_dest = [torch.zeros_like(x) for x in [kc,vc,xq,static_src,static_dest]]
        b,h,r,l,d = xq.shape
        _,_,n,c,_ = kc.shape
        mask = torch.ones(c,l, dtype=torch.bool, device=xq.device)
        static_src = static_src.pow(1/3)
        static_dest = static_dest.pow(1/3)
        kt = kc.view(b,h,l,d).transpose(-2,-1)  # b h d l

        for i in range(n):
            k_ = kc[:,:,i]  # b h c d
            v_ = vc[:,:,i]  # b h c d
            static_src_ = static_src[:,:,i]  # b h c
            dout_ = g_out[...,i]
            ddenom_ = g_denom[...,i]

            # Rerun forward pass
            aff1 = k_.matmul(kt)
            aff2 = aff1.relu().pow(2/3).float()
            aff3 = aff2 * static_src_.unsqueeze(-1) * static_dest.unsqueeze(-2)
            score = torch.log1p(aff3.clamp(min=0,max=1-1e-6).neg()).triu(i*c+1).cumsum(3).masked_fill(mask.tril(i*c-1), -1e12)
            score = k_.unsqueeze(2).matmul(xq.transpose(-1,-2)).add(score.unsqueeze(2))  # b h r c l
            sscore = score.softmax(dim=-2)

            # Backward pass
            dvc[:,:,i] += sscore.to(dtype=dvc.dtype).matmul(dout_).sum(2)  # bhrcl,bhrld -> bhcd
            
            dscore = v_.unsqueeze(2).matmul(dout_.transpose(-1,-2))  # bhcd,bhrld -> bhrcl   <-- from out
            dscore = dscore.sub(dscore.mul(sscore).sum(-2,True)).mul(sscore)  # <-- from softmax
            dscore += sscore * ddenom_.unsqueeze(-2)  # b h r c l   <-- from denom

            dxq += dscore.to(dtype=dxq.dtype).transpose(-1,-2).matmul(k_.unsqueeze(2))  # bhrcl, bhcd -> bhrld
            dkc[:,:,i] += dscore.to(dtype=dkc.dtype).transpose(2,3).flatten(3,4).matmul(xq.flatten(2,3))  # bhrcl, bhrld -> bhcd

            daff = dscore.sum(2)  # b h c l
            daff = daff.flip([3]).cumsum(3).flip([3]).triu(i*c+1)  # <-- from cumsum
            daff /= aff3.clamp(min=1e-6, max=1-1e-6)-1  # <-- from ln(1-x)
            daff *= aff3.ge(0)
            daff *= aff3.le(1-1e-6)
            dstat = daff.mul(aff2).to(dtype=static_src.dtype)  # b h c l

            dstat_src[:,:,i] += dstat.mul(static_dest.unsqueeze(-2)).sum(-1).div(static_src_.pow(2).mul(3))  # bhcl, bhl -> bhc
            dstat_dest += dstat.mul(static_src_.unsqueeze(-1)).sum(-2).div(static_dest.pow(2).mul(3))  # bhcl, bhc -> bhl

            daff = daff.mul(static_src_.unsqueeze(-1)*static_dest.unsqueeze(-2))  # <-- from prod with statics
            daff = daff.to(dtype=xq.dtype) * aff1.abs().add(1e-9).pow(-1/3).mul(2/3).mul(aff1.gt(0))  # <-- from relu + pow

            dkc += daff.transpose(-1,-2).matmul(k_).view(b,h,n,c,d)  # bhcl, bhcd -> bhld, <-- grad via kt
            dkc[:,:,i] += daff.matmul(kt.transpose(-1,-2))  # bhcl, bhdl -> bhcd, <-- grad via k_

        return dkc,dvc,dxq,dstat_src,dstat_dest


class SMVecMatMul(Function):
    @staticmethod
    def forward(mat, vec):
        # mat: ... d n
        # vec: ... n
        return mat.mul(vec.softmax(dim=-1).unsqueeze(-2)).sum(-1)

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        mat, vec = inputs
        ctx.save_for_backward(mat, vec)

    @ staticmethod
    def backward(ctx, g):
        mat, vec = ctx.saved_tensors
        vec = vec.softmax(dim=-1)
        d_mat = g.unsqueeze(-1).mul(vec.unsqueeze(-2))  # ... d n
        d_vec = g.unsqueeze(-1).mul(mat).sum(-2)  # ... n
        d_vec = d_vec.sub(d_vec.mul(vec).sum(-1,True)).mul(vec)
        return d_mat, d_vec


class AttentionKwargs(TypedDict, total=False):
    """
    The attention kwargs to be passed to fms model forward.

    attn_name: str
        this is the name corresponding to the attention op registered in register_attention_op
    """

    attn_name: str


class SDPAAttentionKwargs(AttentionKwargs):
    mask: NotRequired[torch.Tensor]
    attn_algorithm: NotRequired[str]
    is_causal_mask: bool


def _sdpa_update_attn_kwargs(**attn_kwargs):
    # this is updating the mask for decoding
    mask = attn_kwargs.get("mask", None)
    if mask is not None:
        # get the last row of the 3d mask
        mask = mask[:, -1:, :]
        # extend the mask one slot
        mask = torch.cat(
            (
                mask,
                torch.zeros(mask.size(0), 1, 1, device=mask.device),
            ),
            dim=2,
        )
        if torch._dynamo.config.dynamic_shapes:
            torch._dynamo.mark_dynamic(mask, 2)

        attn_kwargs["mask"] = mask
    return attn_kwargs


class QKV(nn.Module, metaclass=abc.ABCMeta):
    """Simple module for applying qkv in attention"""

    def __init__(
        self,
        emb_dim: int,
        nheads: int,
        kvheads: int,
        emb_kq_per_head: int,
        emb_v_per_head: int,
        use_bias: bool,
        linear_config: Optional[Mapping[str, Any]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.emb_dim = emb_dim
        self.nheads = nheads
        self.kvheads = kvheads
        self.emb_kq_per_head = emb_kq_per_head
        self.emb_v_per_head = emb_v_per_head
        self.use_bias = use_bias
        self.linear_config = linear_config

    @abc.abstractmethod
    def forward(
        self, q: torch.Tensor, k: Optional[torch.Tensor], v: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """applies query/key/value transformations on q, k, v inputs respectively and returns the resulting values

        Args:
            q: torch.Tensor
                the query tensor
            k: Optional[torch.Tensor]
                the optional key tensor
            v: Optional[torch.Tensor]
                the optional value tensor

        Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            the query, key, and value computed
        """
        pass

    @abc.abstractmethod
    def reset_parameters(self):
        """resets the query, key, and value weights for training

        Args:
            gain: int
                gain for std in norm (default is 1)
        """
        pass


class UnfusedQKV(QKV):
    """
    Unfused Weights implementation of QKV
    """

    def __init__(
        self,
        emb_dim: int,
        nheads: int,
        kvheads: int,
        emb_kq_per_head: int,
        emb_v_per_head: int,
        use_bias: bool,
        linear_config: Optional[Mapping[str, Any]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            emb_dim,
            nheads,
            kvheads,
            emb_kq_per_head,
            emb_v_per_head,
            use_bias,
            linear_config,
            *args,
            **kwargs,
        )

        self.query = get_linear(
            self.emb_dim,
            self.nheads * self.emb_kq_per_head,
            bias=use_bias,
            linear_config=linear_config,
        )
        self.key = get_linear(
            self.emb_dim,
            self.kvheads * self.emb_kq_per_head,
            bias=use_bias,
            linear_config=linear_config,
        )
        self.value = get_linear(
            self.emb_dim,
            self.kvheads * self.emb_v_per_head,
            bias=use_bias,
            linear_config=linear_config,
        )

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
                if self.use_bias:
                    m.bias.data.zero_()

    def forward(
        self, q: torch.Tensor, k: Optional[torch.Tensor], v: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if k is None and v is None:
            k = q
            v = q
        elif k is None or v is None:
            raise ValueError(
                "both k and v must either be given as tensors or both None"
            )

        # b x h x qlen x ds
        queries = self.query(q)
        keys = self.key(k)
        values = self.value(v)
        return queries, keys, values


class FusedQKV(QKV):
    """
    Fused Weights implementation of QKV
    """

    def __init__(
        self,
        emb_dim: int,
        nheads: int,
        kvheads: int,
        emb_kq_per_head: int,
        emb_v_per_head: int,
        use_bias: bool,
        linear_config: Optional[Mapping[str, Any]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            emb_dim,
            nheads,
            kvheads,
            emb_kq_per_head,
            emb_v_per_head,
            use_bias,
            linear_config,
            *args,
            **kwargs,
        )
        self.splits = [
            self.nheads * self.emb_kq_per_head,
            self.kvheads * self.emb_kq_per_head,
            self.kvheads * self.emb_v_per_head,
        ]

        self.qkv_fused = get_linear(
            self.emb_dim,
            sum(self.splits),
            bias=self.use_bias,
            linear_config=linear_config,
        )

    def unfuse_weights(self):
        with torch.device("meta"):
            result = UnfusedQKV(
                self.emb_dim,
                self.nheads,
                self.kvheads,
                self.emb_kq_per_head,
                self.emb_v_per_head,
                self.use_bias,
            )
        query, key, value = torch.split(self.qkv_fused.weight, self.splits, dim=0)
        result.query.weight = torch.nn.Parameter(query)
        result.key.weight = torch.nn.Parameter(key)
        result.value.weight = torch.nn.Parameter(value)
        if self.use_bias:
            query_bias, key_bias, value_bias = torch.split(
                self.qkv_fused.bias, self.splits, dim=0
            )
            result.query.bias = torch.nn.Parameter(query_bias)
            result.key.bias = torch.nn.Parameter(key_bias)
            result.value.bias = torch.nn.Parameter(value_bias)
        return result

    def reset_parameters(self):
        nn.init.trunc_normal_(self.qkv_fused.weight, mean=0.0, std=0.02)
        if self.use_bias:
            self.qkv_fused.bias.data.zero_()

    def forward(
        self, q: torch.Tensor, k: Optional[torch.Tensor], v: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (k is None and v is None) or (k is q and v is q):
            qkv = q
        else:
            raise ValueError("q, k, and v must be the same or k and v must be None")
        return self.qkv_fused(qkv).split(self.splits, dim=-1)


class MultiHeadAttention(nn.Module):
    """
    Performs multi-headed self- or cross-attention, with optional attention masking.
    ...
    Args
    ----
    emb_dim : int
        Latent dimensionality of input and output tensors.
    emb_kq : int
        Latent dimensionality of each head in key and query projections (attention dimension).
    emb_v : int
        Latent dimensionality of each head in value projection (mixing dimension).
    nheads : int
        Number of attention heads.
    p_dropout : float|None
        Dropout probability. Must be in range [0,1]. If 0 or None, dropout will not be used.
    use_bias : bool
        Include bias terms in fully-connected sublayers?
    fused : bool
        If True, qkv weights will be fused, otherwise qkv weights will be unfused.
    linear_config : Mapping[str, Any] | None
        Configuration for selection of linear modules (QKV, dense).
        Pass as {"linear_type": [str | callable], <other kwargs>}.
        "linear_type" should provide the string identifier of a registered type
        (e.g., "torch_linear", "gptq", ...) or a callable for module selection depending
        on module name. Additional config options should be provided as kwargs in
        linear_config.
    """

    def __init__(
        self,
        emb_dim,
        emb_kq,
        emb_v,
        nheads,
        kvheads,
        p_dropout=None,
        use_bias=False,
        position_encoder: Optional[PositionEncoder] = None,
        fused: bool = True,
        linear_config: Optional[Mapping[str, Any]] = None,
        scale_factor: Optional[float] = None,
    ):
        super(MultiHeadAttention, self).__init__()
        self.nheads = nheads
        self.kvheads = kvheads
        self.emb_dim = emb_dim
        self.emb_kq_per_head = emb_kq
        self.emb_v_per_head = emb_v
        self.p_dropout = p_dropout if p_dropout is not None else 0.0
        self.use_bias = use_bias
        self.fused = fused
        self.linear_config = linear_config
        self.scale_factor = scale_factor

        self.in_proj: QKV = (FusedQKV if self.fused else UnfusedQKV)(
            self.emb_dim,
            self.nheads,
            self.kvheads,
            self.emb_kq_per_head,
            self.emb_v_per_head,
            self.use_bias,
            linear_config=linear_config,
        )

        self.dense = nn.Linear(
            self.nheads * self.emb_v_per_head,
            self.emb_dim,
            bias=use_bias,
            # linear_config=linear_config,
        )

        if self.p_dropout:
            self.attn_dropout = nn.Dropout(self.p_dropout)
        self.position_encoder = position_encoder

        self.wstatic = nn.Linear(self.emb_dim, self.kvheads*2, bias=True)
        # self.register_buffer("staticb", torch.empty(self.kvheads*2))

        self.UA = UniversalAttention.apply
        self.SMVMM = SMVecMatMul.apply

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
                if self.use_bias:
                    m.bias.data.zero_()
            elif isinstance(m, QKV):
                m.reset_parameters()
        static_max = math.log(.1)
        static_min = math.log(.001)
        # nn.init.uniform_(self.wstatic.bias)
        self.wstatic.bias.data.zero_()
        # self.staticb = torch.rand_like(self.staticb) * (static_max - static_min) + static_min

    # def to_tp(self, group: ProcessGroup) -> "TPMultiHeadAttention":
    #     return TPMultiHeadAttention.import_module(self, group)

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        position_ids=None,
        past_key_value_state: Optional[Tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]] = None,
        use_cache=False,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        """
        past_key_value_state: tuple
            the cache to be used in attention of the form (<self/cross>_key, <self/cross>_value)
        position_ids: Optional[torch.LongTensor]
            The position of each of the tokens encoded in q and k. Used for RoPE embeddings
        use_cache: bool
            if True, the kv states for self/cross attention will be saved, otherwise they will not be saved

        Returns
        -------
        tensor or tuple
            If use_cache=False, only the hidden state will be returned as a tensor. If use_cache=True, a tuple will be
            returned in the form (hidden_state, cache) where hidden_state is a tensor and cache is of the form specified
            in past_key_value_state
        """
        # q, k, v: batch_size x seq_len x emb_dim
        # mask: batch_size x seq_len x seq_len
        batch_size, q_len, _ = q.size()

        # if this is self attention, we always recompute
        # cross attention only gets computed when a cache does not exist
        # if we dont have the cache yet, we need to compute
        # d x (h x ds)
        # b x kvlen x d
        # b x kvlen x h x ds
        # b x h x kvlen x ds
        # todo: Cross attention (This always is true for now)
        q_out, k_out, v_out = self.in_proj(q, k, v)
        static = self.wstatic(q)
        # static = F.linear(q, self.wstatic.weight, self.staticb + self.wstatic.bias * math.sqrt(self.emb_dim))
        static = static.sigmoid().view(batch_size, q_len, 2, self.kvheads).permute(2,0,3,1)  # 2 b h l
        static_src = static[0]  # b h l
        static_dest = static[1]  # b h l

        # note: transposes will be moved in a later PR to fix dis-contiguous tensor issues
        queries = q_out.view(batch_size, q_len, self.nheads, self.emb_kq_per_head)
        keys = k_out.view(batch_size, q_len, self.kvheads, self.emb_kq_per_head)
        values = v_out.view(batch_size, q_len, self.kvheads, self.emb_v_per_head)

        # Normalize keys
        keys = keys / keys.pow(2).sum(-1, True).sqrt().add(1e-6)

        # You want to apply rotary embeddings pre-cache
        if self.position_encoder is not None:
            queries, keys = self.position_encoder.adjusted_qk(
                queries, keys, position_ids, past_key_value_state, use_cache
            )

        if past_key_value_state is not None:
            # Iterative universal attention
            assert q_len == 1, "UA decoding not currently supported for more than 1 token"
            # thresh = math.log(1e-4)
            (k, v, r, a) = past_key_value_state  # bhld, bhld, bhl, bhl
            k_ = keys.squeeze(1)  # b h d
            v_ = values.squeeze(1)  # b h d
            q = queries.view(batch_size, self.kvheads, -1, self.emb_kq_per_head)  # b h r d
            static_src = static_src.squeeze(2)  # b h
            static_dest = static_dest.squeeze(2)  # b h

            # Update k/v cache
            k = torch.cat((k, k_.unsqueeze(2)), dim=2)
            v = torch.cat((v, v_.unsqueeze(2)), dim=2)

            # q/k/k products
            qkkk = k.matmul(torch.cat([q,k_.unsqueeze(2)], dim=2).transpose(-1,-2))  # b h l+1 r+1
            qk = qkkk[...,:-1]  # b h l+1 r
            kk = qkkk[:,:,:-1,-1]  # b h l

            # Calculate decays
            decay = kk.relu().float().pow(2)
            decay = (decay * r * static_dest.unsqueeze(-1)).pow(1/3)
            decay = torch.log1p(decay.clamp(min=0, max=1-1e-6).neg())
            a = a + decay

            # Update r/a cache
            r = torch.cat((r, static_src.unsqueeze(2)), dim=2)
            a = torch.cat((a, torch.zeros(batch_size, self.kvheads, 1, device=a.device, dtype=a.dtype)), dim=2)

            # Perform scaled attention
            attn = qk.float().add(a.unsqueeze(-1)).softmax(dim=2).to(dtype=v.dtype).transpose(-1,-2).matmul(v)  # b h r d
            (keys, values, rates, affs) = k,v,r,a
            
        else:
            # Blockwise universal attention
            queries = queries.transpose(1,2).view(batch_size, self.kvheads, -1, q_len, self.emb_kq_per_head)  # b h r l d
            keys = keys.transpose(1,2)  # b h l d
            values = values.transpose(1,2)  # b h l d
            rates = static_src

            c = 512
            b = batch_size
            # Right-pad k,v,src if len not divisible by chunksize
            if q_len % c != 0:
                slack = c - q_len % c
                queries = torch.cat([queries, torch.zeros(b, self.kvheads, self.nheads//self.kvheads, slack, self.emb_kq_per_head, 
                                                          device=queries.device, dtype=queries.dtype)], dim=-2)
                keys = torch.cat([keys, torch.zeros(b, self.kvheads, slack, self.emb_kq_per_head, 
                                                          device=keys.device, dtype=keys.dtype)], dim=-2)
                values = torch.cat([values, torch.zeros(b, self.kvheads, slack, self.emb_v_per_head, 
                                                          device=values.device, dtype=values.dtype)], dim=-2)
                static_src = torch.cat([static_src, torch.zeros(b, self.kvheads, slack,
                                                               device=static_src.device, dtype=static_src.dtype)], dim=-1)
                static_dest = torch.cat([static_dest, torch.zeros(b, self.kvheads, slack,
                                                               device=static_dest.device, dtype=static_dest.dtype)], dim=-1)

            # Chunk inputs
            l = static_src.size(2)
            n = l//c
            s = [b, self.kvheads, n, c, -1]
            kc = keys.view(*s)  # b h n c d
            vc = values.view(*s)
            static_src = static_src.view(b, self.kvheads, n, c)  # b h n c

            # Perform UA
            output, denom, affs = self.UA(kc, vc, queries, static_src, static_dest)

            # Weighted avg for final softmax
            output = self.SMVMM(output, denom)  # b h r l d
            attn = output.permute(0,3,1,2,4).reshape(b,l,-1)

            # Prune any right-padding
            keys = keys[:,:,:q_len]
            values = values[:,:,:q_len]
            affs = affs[:,:,:q_len]
            attn = attn[:,:q_len]

        attn = attn.view(batch_size, q_len, self.nheads * self.emb_v_per_head)
        out = self.dense(attn)

        # if use_cache=True, we return the hidden_state as well as the kv cache
        if use_cache:
            return out, (keys, values, rates, affs)
        else:
            return out


class TPMultiHeadAttention(MultiHeadAttention, TPModule):
    """
    Performs multi-headed self- or cross-attention, with optional attention masking.
    This subclass adds support for Tensor Parallel
    ...
    Args
    ----
    Check MultiHeadAttention for up-to-date docs

    world_size: int
        the number of processes running this model in TP
    rank: int
        the index of this process wrt to the rest running the model in TP
    """

    def __init__(
        self,
        emb_dim,
        emb_kq,
        emb_v,
        nheads,
        kvheads,
        p_dropout=None,
        use_bias=False,
        position_encoder: Optional[PositionEncoder] = None,
        fused: bool = True,
        group: Optional[ProcessGroup] = None,
        linear_config: Optional[Mapping[str, Any]] = None,
        scale_factor: Optional[float] = None,
    ):
        assert torch.distributed.is_initialized()

        rank, world_size = distributed.rank_and_world(group)
        assert nheads % world_size == 0, (
            "The number of heads must be divisible by world size"
        )
        assert (kvheads >= world_size and kvheads % world_size == 0) or (
            kvheads < world_size and world_size % kvheads == 0
        ), (
            "the kv heads must be divisible by the world size or the world size must be divisible by kv heads"
        )
        MultiHeadAttention.__init__(
            self,
            emb_dim,
            emb_kq,
            emb_v,
            nheads // world_size,
            (kvheads // world_size) if kvheads >= world_size else 1,
            p_dropout,
            use_bias,
            position_encoder,
            fused,
            linear_config,
            scale_factor,
        )
        self.pre_tp_nheads = nheads
        self.pre_tp_kvheads = kvheads
        self.setup_tp(rank, group)

        # linear_type must handle module_name = None to support TP of MHA
        self.linear_type = get_linear_type(self.linear_config)

    def load_weights(
        self,
        tensor_values: dict[str, torch.Tensor],
    ) -> Optional[set]:
        """Define sharding info of MHA module as:
        {'module_name': (module_obj, sharding_dim, max_partition)}
        Then, call the pre-registered sharding function associated with
        self.linear_type.

        `sharding_dim` is sharding dimension of the `weights` parameter
        of nn.Linear. It may differ for other types of linear or other
        parameters.

        The numbers in `max_partition` signify the largest world size
        till we need to duplicate. For instance if we have nheads=16 and
        world_size=32, then first 2 ranks will get first 1/16th of query
        """

        if self.fused:
            module_sharding_info = {
                "qkv_fused": LinearModuleShardingInfo(
                    self.in_proj.get_submodule("qkv_fused"),
                    0,
                    [self.pre_tp_nheads, self.pre_tp_kvheads, self.pre_tp_kvheads],
                ),
                "dense": LinearModuleShardingInfo(self.dense, 1, [self.world_size]),
            }
        else:
            module_sharding_info = {
                "query": LinearModuleShardingInfo(
                    self.in_proj.get_submodule("query"), 0, [self.pre_tp_nheads]
                ),
                "key": LinearModuleShardingInfo(
                    self.in_proj.get_submodule("key"), 0, [self.pre_tp_kvheads]
                ),
                "value": LinearModuleShardingInfo(
                    self.in_proj.get_submodule("value"), 0, [self.pre_tp_kvheads]
                ),
                "dense": LinearModuleShardingInfo(self.dense, 1, [self.world_size]),
            }

        type_sharding_map = get_all_linear_type_to_sharding_maps()
        unused_keys = type_sharding_map[self.linear_type](
            tensor_values,
            self,
            module_sharding_info,
        )
        return unused_keys

    @staticmethod
    def import_module(
        mha: MultiHeadAttention, group: ProcessGroup
    ) -> "TPMultiHeadAttention":
        tp_mha = TPMultiHeadAttention(
            emb_dim=mha.emb_dim,
            emb_kq=mha.emb_kq_per_head,
            emb_v=mha.emb_v_per_head,
            nheads=mha.nheads,
            kvheads=mha.kvheads,
            p_dropout=mha.p_dropout,
            use_bias=mha.use_bias,
            position_encoder=mha.position_encoder,
            group=group,
            fused=mha.fused,
            linear_config=mha.linear_config,
            scale_factor=mha.scale_factor,
        )
        return tp_mha

    def _copy_to_tp_region(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
    ):
        if (k is None and v is None) or (k is q and v is q):
            q_par = copy_to_tensor_model_parallel_region(q, self.group)
            if self.fused:
                k_par = None
                v_par = None
            else:
                k_par = copy_to_tensor_model_parallel_region(k, self.group)
                v_par = copy_to_tensor_model_parallel_region(v, self.group)
        else:
            raise ValueError(
                "both k and v must either be given as tensors or both None"
            )

        return q_par, k_par, v_par

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        position_ids=None,
        past_key_value_state: Optional[Tuple[Tensor | None, Tensor | None]] = None,
        use_cache=False,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        """
        Check MultiHeadAttention for up-to-date arguments and docs
        """

        q_par, k_par, v_par = self._copy_to_tp_region(q, k, v)

        out_par = MultiHeadAttention.forward(
            self,
            q_par,
            k_par,
            v_par,
            position_ids,
            past_key_value_state,
            use_cache,
            **attn_kwargs,
        )

        # if use_cache=True, we return the hidden_state as well as the kv cache.
        # We only reduce the output, and keep the cache thread-local
        if use_cache:
            out = reduce_from_tensor_model_parallel_region(out_par[0], self.group)
            return out, out_par[1]
        else:
            out = reduce_from_tensor_model_parallel_region(out_par, self.group)
            return out
