# Modified by Rodrigo Marcuzzi from https://github.com/facebookresearch/Mask2Former
import mask_pls.models.blocks as blocks
import torch
from mask_pls.models.positional_encoder import PositionalEncoder
from torch import nn


class MaskedTransformerDecoder(nn.Module):
    def __init__(self, cfg, bb_cfg, data_cfg):
        super().__init__()
        hidden_dim = cfg.HIDDEN_DIM

        cfg.POS_ENC.FEAT_SIZE = cfg.HIDDEN_DIM

        self.pe_layer = PositionalEncoder(cfg.POS_ENC)

        self.num_layers = cfg.FEATURE_LEVELS * cfg.DEC_BLOCKS
        self.nheads = cfg.NHEADS

        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                blocks.SelfAttentionLayer(
                    d_model=hidden_dim, nhead=self.nheads, dropout=0.0
                )
            )
            self.transformer_cross_attention_layers.append(
                blocks.CrossAttentionLayer(
                    d_model=hidden_dim, nhead=self.nheads, dropout=0.0
                )
            )
            self.transformer_ffn_layers.append(
                blocks.FFNLayer(
                    d_model=hidden_dim, dim_feedforward=cfg.DIM_FFN, dropout=0.0
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = cfg.NUM_QUERIES
        self.num_feature_levels = cfg.FEATURE_LEVELS

        self.query_feat = nn.Embedding(cfg.NUM_QUERIES, hidden_dim)
        self.query_embed = nn.Embedding(cfg.NUM_QUERIES, hidden_dim)
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)

        self.mask_feat_proj = nn.Sequential()
        in_channels = bb_cfg.CHANNELS
        if in_channels[-1] != hidden_dim:
            self.mask_feat_proj = nn.Linear(in_channels[-1], hidden_dim)

        in_channels = in_channels[:-1][-self.num_feature_levels :]
        self.input_proj = nn.ModuleList()
        for ch in in_channels:
            if ch != hidden_dim:  # linear projection to hidden_dim
                self.input_proj.append(nn.Linear(ch, hidden_dim))
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        self.class_embed = nn.Linear(hidden_dim, data_cfg.NUM_CLASSES + 1)
        self.mask_embed = blocks.MLP(hidden_dim, hidden_dim, cfg.HIDDEN_DIM, 3)

    def forward(self, feats, coors, pad_masks):
        last_coors = coors.pop()
        mask_features = self.mask_feat_proj(feats.pop()) + self.pe_layer(last_coors)
        last_pad = pad_masks.pop()
        src = []
        pos = []
        size_list = []

        for i in range(self.num_feature_levels):
            size_list.append(feats[i].shape[1])
            pos.append(self.pe_layer(coors[i]))
            feat = self.input_proj[i](feats[i])
            src.append(feat)

        bs = src[0].shape[0]
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        output = self.query_feat.weight.unsqueeze(0).repeat(bs, 1, 1)

        predictions_class = []
        predictions_mask = []

        # predictions on learnable query features, first attn_mask
        outputs_class, outputs_mask, attn_mask = self.pred_heads(
            output,
            mask_features,
            pad_mask=last_pad,
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            if attn_mask is not None:
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output,
                src[level_index],
                attn_mask=attn_mask,
                padding_mask=pad_masks[level_index],
                pos=pos[level_index],
                query_pos=query_embed,
            )
            output = self.transformer_self_attention_layers[i](
                output, attn_mask=None, padding_mask=None, query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](output)

            # get predictions and attn mask for next feature level
            outputs_class, outputs_mask, attn_mask = self.pred_heads(
                output,
                mask_features,
                pad_mask=last_pad,
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1

        out = {"pred_logits": predictions_class[-1], "pred_masks": predictions_mask[-1]}

        out["aux_outputs"] = self.set_aux(predictions_class, predictions_mask)

        return out, last_pad

    def pred_heads(
        self,
        output,
        mask_features,
        pad_mask=None,
    ):
        decoder_output = self.decoder_norm(output)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bpc->bpq", mask_embed, mask_features)

        attn_mask = (outputs_mask.sigmoid() < 0.5).detach().bool()
        attn_mask[pad_mask] = True
        attn_mask = (
            attn_mask.unsqueeze(1)
            .repeat(1, self.nheads, 1, 1)
            .flatten(0, 1)
            .permute(0, 2, 1)
        )

        return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    def set_aux(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_masks": b}
            for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
        ]
