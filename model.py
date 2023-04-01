import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_biased_mask(n_head, max_seq_len, period):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2 ** (-2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]

    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.div(torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1, period).view(-1),
                     period, rounding_mode='trunc')
    bias = - torch.flip(bias, dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i + 1] = bias[-(i + 1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask


# Alignment Bias
def enc_dec_mask(device, T, S, wid):
    mask = torch.ones(T, S)
    for i in range(T):
        mask[i, max(0, i * 2 - (wid - 2)):i * 2 + 2] = 0
    return (mask == 1).to(device=device)


# Periodic Positional Encoding, Adapted from https://github.com/EvelynFan/FaceFormer
class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=25, max_seq_len=600):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        repeat_num = (max_seq_len // period) + 1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Audio2Landmark(nn.Module):
    def __init__(self, args):
        super(Audio2Landmark, self).__init__()
        self.rep = args.repeat
        self.audio_feature_map = nn.Linear(768, args.feature_dim)
        self.align_wid = args.align_window
        self.feature_dim = args.feature_dim
        self.vertice_dim = args.vertice_dim
        self.vertice_map = nn.Linear(args.vertice_dim, args.feature_dim)
        self.PPE = PeriodicPositionalEncoding(args.feature_dim, period=args.period, max_seq_len=args.audio_len * 25)
        self.biased_mask = init_biased_mask(n_head=4, max_seq_len=args.audio_len * 25, period=args.period)
        decoder_layer = nn.TransformerDecoderLayer(d_model=args.feature_dim, nhead=4, dim_feedforward=2 * args.feature_dim, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.vertice_map_r = nn.Linear(args.feature_dim, args.vertice_dim)
        self.device = args.device
        nn.init.constant_(self.vertice_map_r.weight, 0)
        nn.init.constant_(self.vertice_map_r.bias, 0)

    def forward(self, audio_embedding, vertice):
        """
        audio_embedding: B x T x D
        vertice: B x T x 3 x 468
        """
        frame_num = vertice.shape[1]
        bs = vertice.shape[0]
        hidden_states = self.audio_feature_map(audio_embedding)

        for i in range(frame_num):
            if i == 0:
                vertice_emb = self.vertice_map(
                    torch.zeros(bs, 1, self.vertice_dim, device=self.device)) 
                vertice_input = self.PPE(vertice_emb)
            else:
                vertice_input = self.PPE(vertice_emb)
            tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(
                device=self.device)

            if self.rep == 'repeat':
                tgt_mask = tgt_mask.repeat(bs, 1, 1)
            else:
                tgt_mask = torch.repeat_interleave(tgt_mask, bs, dim=0)
            memory_mask = enc_dec_mask(self.device, vertice_input.shape[1], hidden_states.shape[1], self.align_wid)

            vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
            vertice_out = self.vertice_map_r(vertice_out)
            new_output = self.vertice_map(vertice_out[:, -1, :]).unsqueeze(1)
            vertice_emb = torch.cat((vertice_emb, new_output), 1)

        loss = F.mse_loss(vertice_out, vertice.flatten(start_dim=2)) 
        return loss

    def inference(self, audio_embedding, frame_num):
        hidden_states = self.audio_feature_map(audio_embedding)
        bs = audio_embedding.shape[0]
        for i in range(frame_num):
            if i == 0:
                vertice_emb = self.vertice_map(
                    torch.zeros(bs, 1, self.vertice_dim, device=self.device)) 
                vertice_input = self.PPE(vertice_emb)
            else:
                vertice_input = self.PPE(vertice_emb)
            tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(
                device=self.device)
            tgt_mask = tgt_mask[None, :].repeat(bs, 1, 1, 1)
            memory_mask = enc_dec_mask(self.device, vertice_input.shape[1], hidden_states.shape[1], self.align_wid)

            vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask,
                                                   memory_mask=memory_mask)
            vertice_out = self.vertice_map_r(vertice_out)

            new_output = self.vertice_map(vertice_out[:, -1, :]).unsqueeze(1)
            vertice_emb = torch.cat((vertice_emb, new_output), 1)
        return vertice_out
