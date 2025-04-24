import math
from collections import OrderedDict
from math import sqrt

import numpy as np
import pywt
import torch
import torch.nn as nn
from torch.nn import Flatten
from torch.nn.init import trunc_normal_

from layers.AutoCorrelation import AutoCorrelationLayer, AutoCorrelation
from layers.Autoformer_EncDec import Decoder, DecoderLayer, my_Layernorm
from layers.StandardNorm import Normalize
from model.Autoformer_Embed import DataEmbedding_wo_pos
from model.DDGCRNCell import DDGCRNCell
import torch.nn.functional as F

from model.LIFT import LeadRefiner


class DGCRM(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(DGCRM, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.DGCRM_cells = nn.ModuleList()
        self.DGCRM_cells.append(DDGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.DGCRM_cells.append(DDGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings):

        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.DGCRM_cells[i](current_inputs[:, t, :, :], state, [node_embeddings[0][:, t, :, :],
                                                                                node_embeddings[
                                                                                    1]])
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)

        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.DGCRM_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)  # (num_layers, B, N, hidden_dim)

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, q, k):
        B, N, C = q.shape
        q = self.proj_q(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        kv = self.kv(k).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, horizon, down_sampling_layers, down_sampling_window):
        super(MultiScaleSeasonMixing, self).__init__()
        self.horizon = horizon
        self.down_sampling_layers = down_sampling_layers
        self.down_sampling_window = down_sampling_window
        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        self.horizon // (self.down_sampling_window ** i),
                        self.horizon // (self.down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        self.horizon // (self.down_sampling_window ** (i + 1)),
                        self.horizon // (self.down_sampling_window ** (i + 1)),
                    ),

                )
                for i in range(self.down_sampling_layers)
            ]
        )

    def forward(self, season_list):

        # mixing high->low
        out_high = season_list[0].permute(0, 2, 1)
        out_low = season_list[1].permute(0, 2, 1)
        out_season_list = [season_list[0]]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MFSFFE(nn.Module):
    def __init__(self, args):
        super(MFSFFE, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.lag = args.lag
        self.batch_size = args.batch_size
        self.horizon = args.horizon
        self.num_layers = args.num_layers
        self.use_D = args.use_day
        self.use_W = args.use_week
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.default_graph = args.default_graph
        self.node_embeddings1 = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)
        self.device = args.device
        self.T_i_D_emb = nn.Parameter(torch.empty(288, args.embed_dim))
        self.D_i_W_emb = nn.Parameter(torch.empty(7, args.embed_dim))

        self.encoder1 = DGCRM(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                              args.embed_dim, args.num_layers)
        self.encoder2 = DGCRM(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                              args.embed_dim, args.num_layers)
        # predictor
        self.end_conv1 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        self.end_conv2 = nn.Conv2d(1, args.lag * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        self.end_conv3 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

        self.sigmoid = torch.nn.Sigmoid()

        # todo 4 Autoformer
        self.decomp = series_decomp(args.moving_avg)
        self.dec_embed_model_dim = args.dec_embed_model_dim
        self.tod_model_dim = args.tod_model_dim
        self.dow_model_dim = args.dow_model_dim

        self.model_dim = (
                self.dec_embed_model_dim
                + self.tod_model_dim
                + self.dow_model_dim
            # + self.adaptive_embedding_dim
        )
        self.model_heads = 4

        self.dec_embedding = DataEmbedding_wo_pos(args.num_nodes, self.dec_embed_model_dim, 0.1)
        self.T_i_D_trend_emb = nn.Parameter(torch.empty(288, self.tod_model_dim))
        self.D_i_W_trend_emb = nn.Parameter(torch.empty(7, self.dow_model_dim))
        self.T_i_D_trend_en_emb = nn.Parameter(torch.empty(288, self.model_dim))
        self.D_i_W_trend_en_emb = nn.Parameter(torch.empty(7, self.model_dim))
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, 3, attention_dropout=0.1,
                                        output_attention=False),
                        self.model_dim, self.model_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, 3, attention_dropout=0.1,
                                        output_attention=False),
                        self.model_dim, self.model_heads),
                    self.model_dim,
                    self.num_node,
                    args.d_ff,
                    moving_avg=args.moving_avg,
                    dropout=0.1,
                    activation=args.activation,
                )
                for _ in range(1)
            ],
            norm_layer=my_Layernorm(self.model_dim),
            projection=nn.Linear(self.model_dim, self.num_node, bias=True)
        )

        # todo 5 记忆网络
        self.mem_num = args.mem_num
        self.mem_dim = self.model_dim
        self.memory = self.construct_memory()

        # todo 6 agent_att
        self.attn = Attention(
            self.hidden_dim,
            num_heads=8, qkv_bias=True, qk_scale=None,
            attn_drop=0, proj_drop=0.1, sr_ratio=1)

        # todo 7 Lift
        self.lead_refiner = LeadRefiner(args.lag, self.horizon, self.num_node, 4,
                                        16, temperature=0.1)
        self.fc_dim = 20
        self.middle_dim = 2
        self.embed_poj = nn.ModuleList([nn.Linear(self.horizon // (i + 1), self.fc_dim) for i in range(2)])
        self.embed_fc = nn.Sequential(
            OrderedDict([
                ('sigmoid1', nn.Sigmoid()),
                ('fc2', nn.Linear(self.fc_dim, self.middle_dim)),
                ('sigmoid2', nn.Sigmoid()),
                ('fc3', nn.Linear(self.middle_dim, args.embed_dim))]))

        self.weights_pool = nn.init.xavier_normal_(
            nn.Parameter(
                torch.FloatTensor(args.embed_dim, self.hidden_dim, self.hidden_dim)
            )
        )
        self.bias_pool = nn.init.xavier_normal_(
            nn.Parameter(torch.FloatTensor(args.embed_dim, self.hidden_dim))
        )

        self.down_sampling_layers = 1
        self.down_sampling_window = 2
        self.MultiScaleSeasonMixing = MultiScaleSeasonMixing(self.horizon,self.down_sampling_layers, self.down_sampling_window)

        self.predict_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    (self.horizon) // (self.down_sampling_window ** i),
                    self.horizon,
                )
                for i in range(self.down_sampling_layers + 1)
            ]
        )

    def FFT_for_Period(self, x, k=2):
        # [B, T, C]
        xf = torch.fft.fft(x, dim=1)
        # find period by amplitudes
        frequency_list = abs(xf).mean(0).mean(-1)
        frequency_list[0] = 0
        _, top_list = torch.topk(frequency_list, k)
        top_list = top_list.detach().cpu().numpy()
        # period = x.shape[1] // top_list
        for i in top_list:
            xf[:, i, :] = 0
        xf = torch.fft.ifft(xf, x.shape[1], dim=1)
        return xf

    def construct_memory(self):
        memory_dict = nn.ParameterDict()
        memory_dict['Memory'] = nn.Parameter(torch.randn(self.mem_num, self.mem_dim), requires_grad=True)  # (M, d)
        memory_dict['Wq'] = nn.Parameter(torch.randn(self.num_node, self.mem_dim),
                                         requires_grad=True)  # project to query
        for param in memory_dict.values():
            nn.init.xavier_normal_(param)
        return memory_dict

    def query_memory(self, h_t: torch.Tensor):
        query = torch.matmul(h_t, self.memory['Wq'])  # (B, N, d)

        att_score = torch.softmax(torch.matmul(query, self.memory['Memory'].t()), dim=-1)  # alpha: (B, N, M)
        value = torch.matmul(att_score, self.memory['Memory'])  # (B, N, d)

        return value

    def __multi_scale_process_inputs(self, x_enc):
        # down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)

        down_pool = torch.nn.AvgPool1d(2)
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc

        x_enc_sampling_list = []
        x_enc_sampling_list.append(x_enc)

        x_enc_sampling = down_pool(x_enc_ori)

        x_enc_sampling_list.append(x_enc_sampling)

        x_enc = x_enc_sampling_list

        return x_enc

    def forward(self, source, mark, i=2):

        node_embedding1 = self.node_embeddings1

        if self.use_D:
            t_i_d_data = source[..., 1]
            t_i_d_en_data = source[:, :, 1, 1]
            feature_time_1 = mark[:, :, 1, 0]
            t_i_d_trend_data = torch.cat([source[:, -6:, 1, 1], feature_time_1], dim=1)

            T_i_D_emb = self.T_i_D_emb[(t_i_d_data * 288).type(torch.LongTensor)]
            T_i_D_trend_emb = self.T_i_D_trend_emb[(t_i_d_trend_data * 288).type(torch.LongTensor)]
            T_i_D_en_trend_emb = self.T_i_D_trend_en_emb[(t_i_d_en_data * 288).type(torch.LongTensor)]

            node_embedding1 = torch.mul(node_embedding1, T_i_D_emb)

        if self.use_W:
            d_i_w_data = source[..., 2]
            d_i_w_en_data = source[:, :, 1, 2]
            feature_time_2 = mark[:, :, 1, 1]
            d_i_w_trend_data = torch.cat([source[:, -6:, 1, 2], feature_time_2], dim=1)

            D_i_W_emb = self.D_i_W_emb[(d_i_w_data).type(torch.LongTensor)]
            D_i_W_trend_emb = self.D_i_W_trend_emb[(d_i_w_trend_data).type(torch.LongTensor)]
            D_i_W_en_trend_emb = self.D_i_W_trend_en_emb[(d_i_w_en_data).type(torch.LongTensor)]

            node_embedding1 = torch.mul(node_embedding1, D_i_W_emb)

        node_embeddings = [node_embedding1, self.node_embeddings1]

        source = source[..., 0].unsqueeze(-1)
        init_state1 = self.encoder1.init_hidden(source.shape[0])  # [2,64,307,64] 前面是2是因为有两层GRU
        HS, _ = self.encoder1(source, init_state1, node_embeddings)  # B, T, N, hidden
        output1 = HS[:, -1:, :, :]  # B,1,N,hideen
        # todo 记忆网络
        output1 = self.dropout1(output1)
        HS_1 = self.end_conv2(output1)
        source1 = self.end_conv2(output1)
        source_fix = self.lead_refiner(source.squeeze(-1).permute(0, 2, 1), source1.squeeze(-1).permute(0, 2, 1))
        source2 = source - source1
        # todo itransformer
        # 重新加gcn
        HS_2 = HS_1[..., 0]
        origin_source_2 = source_fix.permute(0, 2, 1)
        # 频率阈值
        HS_2_high = self.FFT_for_Period(origin_source_2)
        souce_2_high = HS_2_high.type(torch.float32)
        #
        xh = self.__multi_scale_process_inputs(souce_2_high)
        init_state2 = HS[:, -1, :, :]
        for i, x in zip(range(len(xh)), xh, ):
            x_embed = self.embed_poj[i](x)
            x_embed = self.embed_fc(x_embed)
            x_embed = torch.mul(x_embed, self.node_embeddings1)

            weights = torch.einsum(
                "bnd,dio->bnio", x_embed, self.weights_pool
            )  # B, cheb_k*in_dim, out_dim
            bias = torch.matmul(x_embed, self.bias_pool)  # B, out_dim

            re = torch.einsum("bni,bnio->bno", init_state2, weights) + bias
            re = torch.tanh(re)
            init_state2 = torch.mul(init_state2, re)

        init_state2 = self.attn(init_state2, init_state2)
        init_state2 = [init_state2] * self.num_layers


        output2_state, _ = self.encoder2(source2, init_state2, node_embeddings)  # B, T, N, hidden
        output2 = self.dropout2(output2_state[:, -1:, :, :])
        output2 = self.end_conv3(output2)

        # todo Autoformer

        seasonal_HS_init, _ = self.decomp(HS_2)
        seasonal_source_init, trend_init = self.decomp(source[..., 0])
        # decoder input
        seasonal_init = torch.cat([seasonal_source_init[:, -6:, :], seasonal_HS_init], dim=1)
        # enc
        dec_out = self.dec_embedding(seasonal_init)
        feature = [dec_out]

        if self.use_D:
            feature.append(T_i_D_trend_emb)
            cross = T_i_D_en_trend_emb

        if self.use_W:
            feature.append(D_i_W_trend_emb)
            cross = torch.mul(cross, D_i_W_en_trend_emb)
        dec_out = torch.cat(feature, dim=-1)

        source2 = source2[..., 0]
        x_s_down = self.__multi_scale_process_inputs(source2)
        enc_out_list = []

        for i, x in zip(range(len(x_s_down)), x_s_down, ):
            # x = self.normalize_layers[i](x, 'norm')
            x = x.permute(0, 2, 1)
            self.decomp(x)
            enc_out = self.query_memory(x)
            enc_out_list.append(enc_out)
        enc_out_list = self.MultiScaleSeasonMixing(enc_out_list)
        dow_enc_out = []
        for i, enc_out in zip(range(len(enc_out_list)), enc_out_list):
            enc_re = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)  # align temporal dimension
            if i > 0:
                enc_re = torch.mul(enc_re, dow_enc_out[i - 1])
            dow_enc_out.append(enc_re)

        cross = torch.mul(enc_re, cross)

        dec_out = self.decoder(dec_out, cross, trend=trend_init)
        # B N E -> B N S -> B S N
        dec_out = dec_out[:, -self.horizon:, :]  # filter the covariates
        dec_out = dec_out.unsqueeze(-1)

        z = self.sigmoid(torch.add(output2, dec_out))
        output2 = torch.add((z * output2), ((1 - z) * dec_out))

        return HS_1 + output2
