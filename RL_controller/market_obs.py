#！/usr/bin/python
# -*- coding: utf-8 -*-#

'''
---------------------------------
 Name:         market_obs.py
 Description:  Implement the market observer of the proposed MASA framework.
 Author:       MASA
---------------------------------
'''
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict, Callable, Any
th.autograd.set_detect_anomaly(True)
_mkt_obs_model_entrypoints: Dict[str, Callable[..., Any]] = {}  

def register_mkt_obs_model(fn: Callable[..., Any]) -> Callable[..., Any]:
    net_name = fn.__name__ # e.g., 'deeptradernet_v1'
    _mkt_obs_model_entrypoints[net_name] = fn
    return fn


class MarketObserver_Algorithmic:
    def __init__(self, config, action_dim):
        self.config = config
        self.action_dim = action_dim # The dim of the hidden vector.
        input_kwargs = {'action_dim': self.action_dim}
        self.mkt_obs_model = create_mkt_obs_model(config=self.config, **input_kwargs)
    def train(self, **label_kwargs):
        pass
    def reset(self):
        pass
    def predict(self, finemkt_feat, finestock_feat, **kwargs):
        cur_hidden_vector_ay, lambda_val_ay, sigma_val_ay = self.mkt_obs_model(**kwargs)
        return cur_hidden_vector_ay, lambda_val_ay, sigma_val_ay
    def update_hidden_vec_reward(self, mode, rate_of_price_change, mkt_direction):
        pass

class MarketObserver:
    def __init__(self, config, action_dim):
        self.config = config
        self.action_dim = action_dim # The dim of the hidden vector.
        input_kwargs = {'action_dim': self.action_dim}
        self.mkt_obs_model = create_mkt_obs_model(config=self.config, **input_kwargs)

        if th.cuda.is_available():
            cuda_status = 'cuda:0'
        else:
            cuda_status = 'cpu'
        self.device = th.device(cuda_status)
        isParallel = False
        self.mkt_obs_model = self.mkt_obs_model.to(self.device)
        if isParallel:
            self.optimizer = optim.Adam(self.mkt_obs_model.module.parameters(), lr=self.config.po_lr, weight_decay=self.config.po_weight_decay)
        else:
            self.optimizer = optim.Adam(self.mkt_obs_model.parameters(), lr=self.config.po_lr, weight_decay=self.config.po_weight_decay)

        decay_steps = self.config.num_epochs // 3

        self.exp_lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=decay_steps, gamma=0.1)

        self.cur_hidden_vector_lst = []
        self.hidden_vector_reward_lst = []
        self.lambda_log_p_lst = []
        self.sigma_log_p_lst = []

        self.mkt_direction_lst = []

        self.mkt_direction_loss_sigma = th.nn.CrossEntropyLoss()
        self.mkt_direction_loss_lambda = th.nn.CrossEntropyLoss()

    def train(self, **label_kwargs):
  
        self.mkt_obs_model.train()
        # obs_weight_tensor = th.cat(self.cur_hidden_vector_lst, dim=0) 
        # Calculate Loss
        hidden_vector_reward_tensor = th.cat(self.hidden_vector_reward_lst, dim=0) # (num_of_batch, batch_size) -> (all_samples, )
        loss_hidden = -th.mean(hidden_vector_reward_tensor) # (1, )
        loss_val = self.config.hidden_vec_loss_weight * loss_hidden
        disp_str = 'Loss(Hidden): {} |'.format(self.config.hidden_vec_loss_weight * loss_hidden.detach().cpu().item())

        if self.config.is_enable_dynamic_risk_bound:
            # sigma
            sigma_log_p_tensor = th.cat(self.sigma_log_p_lst, dim=0) # (num_of_batch, batch_size) -> (all_samples, )
            mkt_direction_tensor = th.cat(self.mkt_direction_lst, dim=0) # (num_of_batch, batch_size) -> (all_samples, )
            loss_sigma = self.mkt_direction_loss_sigma(sigma_log_p_tensor[:-1], mkt_direction_tensor.long())
            loss_val = loss_val + loss_sigma
            disp_str = disp_str +  'Loss(Sigma): {} |'.format(self.config.sigma_loss_weight * loss_sigma.detach().cpu().item())

        self.optimizer.zero_grad()
        loss_val.backward()
        th.cuda.synchronize()
        self.optimizer.step()
        self.exp_lr_scheduler.step()

        disp_str = '{} | Loss(Total): {} |'.format(label_kwargs['mode'], loss_val.detach().cpu().item()) + disp_str
        print(disp_str)

        self.reset()
        th.cuda.empty_cache()

    def reset(self):
        self.cur_hidden_vector_lst = []
        self.hidden_vector_reward_lst = []
        self.lambda_log_p_lst = []
        self.sigma_log_p_lst = []
        self.mkt_direction_lst = []

    def predict(self, finemkt_feat, finestock_feat, **kwargs):
        # fine market data: (batch, features, window_size) 
        # fine stock data: (batch, features, num_of_stocks, window_size)
        finemkt_feat = th.from_numpy(finemkt_feat).to(th.float32)
        finestock_feat = th.from_numpy(finestock_feat).to(th.float32)

        finemkt_feat = finemkt_feat.to(self.device)
        finestock_feat = finestock_feat.to(self.device)
        input_kwargs = {'market': finemkt_feat, 'device': self.device}
        if kwargs['mode'] == 'train':
            self.mkt_obs_model.train()
            input_kwargs['deterministic'] = False
            cur_hidden_vector, lambda_val, sigma_val, lambda_log_p, sigma_log_p = self.mkt_obs_model(x=finestock_feat, **input_kwargs)

            self.cur_hidden_vector_lst.append(cur_hidden_vector) # cur_hidden_vector: (batch, num_of_stocks), device loc: cuda
            self.lambda_log_p_lst.append(lambda_log_p) # lambda_log_p: (batch,), device loc: cuda
            self.sigma_log_p_lst.append(sigma_log_p) # sigma_log_p: (batch,), device loc: cuda

        elif kwargs['mode'] in ['valid', 'test']:
            self.mkt_obs_model.eval()
            with th.no_grad():
                input_kwargs['deterministic'] = True
                cur_hidden_vector, lambda_val, sigma_val, lambda_log_p, sigma_log_p = self.mkt_obs_model(x=finestock_feat, **input_kwargs)
            
        else:
            raise ValueError("Unknown mode: {}".format(kwargs['mode']))

        cur_hidden_vector_ay = cur_hidden_vector.detach().cpu().numpy() # (batch, num_of_stocks)
        lambda_val_ay = lambda_val.detach().cpu().numpy() # (batch, )
        sigma_val_ay = sigma_val.detach().cpu().numpy() # (batch, )
        return cur_hidden_vector_ay, lambda_val_ay, sigma_val_ay

    def update_hidden_vec_reward(self, mode, rate_of_price_change, mkt_direction):
        # Input rate_of_price_change: (batch, num_of_stocks)
        self.mkt_obs_model.train()
        rate_of_price_change = th.from_numpy(rate_of_price_change).to(th.float32).to(self.device)
        last_hidden_vec = self.cur_hidden_vector_lst[-1] # (batch, num_of_stocks)
        curday_reward = th.log(th.sum((rate_of_price_change-1.0) * last_hidden_vec, dim=-1) + 1.0) # (batch, )
        self.hidden_vector_reward_lst.append(curday_reward) # 
        self.mkt_direction_lst.append(th.from_numpy(mkt_direction).to(self.device)) # mkt_direction_lst: (num_of_batch, batch_size)


@register_mkt_obs_model
def mlp_1(config, **kwargs):
    model = MLP_1(config, **kwargs)
    return model

class MLP_1(nn.Module):
    def __init__(self, config, **kwargs):
        super(MLP_1, self).__init__()
        self.name = 'mlp_1'
        self.config = config
        self.output_action_dim = kwargs['action_dim'] # for hidden vector to RL agent

        # stock
        feat_s_length = self.config.topK * len(self.config.use_features) * self.config.fine_window_size
        self.flatten_s = nn.Flatten() 
        self.fc1_s = nn.Linear(feat_s_length, 256, bias=True)
        self.relu1_s = nn.ReLU()
        self.fc2_s = nn.Linear(256, 64, bias=True)
        self.shortcut_s = nn.Linear(feat_s_length, 64, bias=True)
        self.relu2_s = nn.ReLU()
        # market
        feat_m_length = len(self.config.use_features) * self.config.fine_window_size
        self.flatten_m = nn.Flatten()
        self.fc1_m = nn.Linear(feat_m_length, 16, bias=True)
        self.relu1_m = nn.ReLU()
        self.fc2_m = nn.Linear(16, 16, bias=True)
        self.relu2_m = nn.ReLU()

        # 64+16
        self.fc_merge = nn.Linear(80, self.output_action_dim, bias=True) 
        self.sm = nn.Softmax(dim=1)

        # self.fc_lambda = nn.Linear(80, 2, bias=True)
        self.fc_lambda = nn.Linear(80, 3, bias=True)

        self.gen_lambda = GenScore()

        # self.fc_sigma = nn.Linear(80, 2, bias=True)
        self.fc_sigma = nn.Linear(80, 3, bias=True)

        self.gen_sigma = GenScore()

    def forward(self, x, **kwargs):
        # fine stock data (x): (batch, features, num_of_stocks, window_size)
        # fine market data: (batch, features, window_size) 
        # kwargs: market, deterministic, device

        # stock
        s0 = self.flatten_s(x)
        s1 = self.fc1_s(s0)
        s1 = self.relu1_s(s1)
        s1 = self.fc2_s(s1)
        s2 = self.shortcut_s(s0) # shortcut
        s3 = s1 + s2 # add
        s3 = self.relu2_s(s3)

        # market
        m = kwargs['market']
        m0 = self.flatten_m(m)
        m1 = self.fc1_m(m0)
        m1 = self.relu1_m(m1)
        m1 = self.fc2_m(m1)
        m2 = m1 + m0
        m2 = self.relu2_m(m2)

        # merge
        k1 = th.cat((s3, m2), dim=1)
        k2 = self.fc_merge(k1) # (batch, num_of_stocks) when not considering cash, (batch, num_of_stocks+1) when considering cash
        hidden_vec = self.sm(k2) # hidden vectors
        lambda_vec = self.fc_lambda(k1)
        lambda_kwargs = {'name': 'lambda', 'deterministic': kwargs['deterministic'], 'device': kwargs['device'], 'score_min': self.config.lambda_min, 'score_max': self.config.lambda_max}
        lambda_val, lambda_log_p = self.gen_lambda(lambda_vec, **lambda_kwargs)
        sigma_vec = self.fc_sigma(k1)
        sigma_kwargs = {'name': 'sigma', 'deterministic': kwargs['deterministic'], 'device': kwargs['device'], 'score_min': self.config.sigma_min, 'score_max': self.config.sigma_max}
        sigma_val, sigma_log_p = self.gen_sigma(sigma_vec, **sigma_kwargs)

        return hidden_vec, lambda_val, sigma_val, lambda_log_p, sigma_log_p


@register_mkt_obs_model
def lstm_1(config, **kwargs):
    model = LSTM_1(config, **kwargs)
    return model

class LSTM_1(nn.Module):
    def __init__(self, config, **kwargs):
        super(LSTM_1, self).__init__()
        self.name = 'lstm_1'
        self.config = config
        self.output_action_dim = kwargs['action_dim'] # for hidden vector to RL agent

        self.in_features = len(self.config.use_features) * (self.config.topK + 1) # +1 for market data
        self.window_len = self.config.fine_window_size
        hidden_dim = 128

        self.flatten_s = nn.Flatten(start_dim=1, end_dim=2)

        self.lstm = nn.LSTM(input_size=self.in_features, hidden_size=hidden_dim)
        self.attn1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.attn2 = nn.Linear(hidden_dim, 1)

        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.fc_merge = nn.Linear(hidden_dim, self.output_action_dim, bias=True)
        self.sm = nn.Softmax(dim=1)

        # self.fc_lambda = nn.Linear(hidden_dim, 2, bias=True)
        self.fc_lambda = nn.Linear(hidden_dim, 3, bias=True)

        self.gen_lambda = GenScore()

        # self.fc_sigma = nn.Linear(hidden_dim, 2, bias=True)
        self.fc_sigma = nn.Linear(hidden_dim, 3, bias=True)

        self.gen_sigma = GenScore()

    def forward(self, x, **kwargs):
        # fine stock data: (batch, features, num_of_stocks, window_size)
        # fine market data: (batch, features, window_size) 
        # kwargs: market, deterministic, device

        # stock
        # (batch, features, num_of_stocks, window_size) -> (batch, features*num_of_stocks, window_size)
        s0 = self.flatten_s(x) 
        m = kwargs['market']
        x1 = th.cat((s0, m), dim=1) # -> (batch, mkt_feat+stock_feat, window_size)
        x1 = x1.permute(2, 0, 1) # -> (window_size, batch, mkt_feat+stock_feat)
        outputs, (h_n, c_n) = self.lstm(x1) # outputs: (window_size, batch, 1*hidden_size), hn:(1*1, batch, hidden_size)
        H_n = h_n.repeat((self.window_len, 1, 1)) # (window_len, batch, hidden_size)
        scores = self.attn2(th.tanh(self.attn1(th.cat([outputs, H_n], dim=2))))  # (win_len, batch, 1), [L, B*N, 1]
        scores = scores.squeeze(2).transpose(1, 0)  # (batch, win_len), [B*N, L]
        attn_weights = th.softmax(scores, dim=1) # (batch, win_len), [B*N, L]

        outputs = outputs.permute(1, 0, 2)  #(batch, win_len, hidden_size), [B*N, L, H]
        attn_embed = th.bmm(attn_weights.unsqueeze(1), outputs).squeeze(1) # (batch, hidden_size)
        if attn_embed.size(0) == 1:
            embed = th.relu(self.linear1(attn_embed)) # (batch, hidden_size)
        else:
            embed = th.relu(self.bn1(self.linear1(attn_embed))) # (batch, hidden_size)
        
        hidden_vec = self.fc_merge(embed) # (batch, num_of_stocks) when not considering cash, (batch, num_of_stocks+1) when considering cash
        hidden_vec = self.sm(hidden_vec) # softmax

        lambda_vec = self.fc_lambda(embed) # -> (batch, 2)
        lambda_kwargs = {'name': 'lambda', 'deterministic': kwargs['deterministic'], 'device': kwargs['device'], 'score_min': self.config.lambda_min, 'score_max': self.config.lambda_max}
        lambda_val, lambda_log_p = self.gen_lambda(lambda_vec, **lambda_kwargs) # -> (batch, )
        sigma_vec = self.fc_sigma(embed) # -> (batch, 2)
        sigma_kwargs = {'name': 'sigma', 'deterministic': kwargs['deterministic'], 'device': kwargs['device'], 'score_min': self.config.sigma_min, 'score_max': self.config.sigma_max}
        sigma_val, sigma_log_p = self.gen_sigma(sigma_vec, **sigma_kwargs) # -> (batch, )

        return hidden_vec, lambda_val, sigma_val, lambda_log_p, sigma_log_p

@register_mkt_obs_model
def stf_1(config, **kwargs):
    model = STF_1(config, **kwargs)
    return model
class STF_1(nn.Module):
    def __init__(self, config, **kwargs):
        super(STF_1, self).__init__()
        self.name = 'stf_1'
        self.config = config
    def forward(self, x):
        pass
        return x

class GenScore(nn.Module):
    def __init__(self, **kwargs):
        super(GenScore, self).__init__()
        self.name = 'gen_score'
        # self.sigmoid_fn = nn.Sigmoid()
    
    def forward(self, x, **kwargs):   
        score_log_p = x
        score = th.argmax(x.detach(), dim=1)
        return score, score_log_p


@register_mkt_obs_model
def ma_1(config, **kwargs):
    model = MA_1(config, **kwargs)
    return model
class MA_1:
    def __init__(self, config, **kwargs):
        super(MA_1, self).__init__()
        self.name = 'ma_1'
        self.config = config
        self.output_action_dim = kwargs['action_dim'] # for hidden vector to RL agent

    def __call__(self, **kwargs):
        # Input: (batch, num_of_stocks)
        up_num = np.sum(kwargs['stock_cur_close_price'] > kwargs['stock_ma_price'], axis=1)
        hold_num = np.sum(kwargs['stock_cur_close_price'] == kwargs['stock_ma_price'], axis=1)
        down_num = np.sum(kwargs['stock_cur_close_price'] < kwargs['stock_ma_price'], axis=1)
        direction = np.argmax(np.array([up_num, hold_num, down_num]), axis=0)
        sigma_val_ay = direction
        lambda_val_ay = direction


        up_idx = np.argwhere(np.array(kwargs['stock_cur_close_price'] > kwargs['stock_ma_price']))
        cur_hidden_vector_ay = np.zeros((kwargs['stock_cur_close_price'].shape[0], self.output_action_dim)) # (batch, action_dim)
        zidx = np.argwhere(up_num==0).flatten()
        up_num_norm = np.divide(np.ones_like(up_num), up_num.astype(np.float32), out=np.zeros_like(up_num)*1.0, where=up_num!=0)

        if self.output_action_dim == self.config.topK:
            cur_hidden_vector_ay[up_idx[:,0], up_idx[:, 1]] = up_num_norm[up_idx[:, 0]]
        elif self.output_action_dim == self.config.topK + 1:
            cur_hidden_vector_ay[up_idx[:,0], up_idx[:, 1]+1] = up_num_norm[up_idx[:, 0]]
        else:
            raise ValueError("Unmatch action_dim: {}, stock_num: {}".format(self.output_action_dim, self.config.topK))
        cur_hidden_vector_ay[zidx] = 1.0 / self.output_action_dim
        # hidden_vec: (batch, num_of_stocks) is the hidden vector sent to RL agents.
        # sigma_val_ay: (batch, )
        # lambda_val_ay: (batch, )
        return cur_hidden_vector_ay, lambda_val_ay, sigma_val_ay

@register_mkt_obs_model
def dc_1(config, **kwargs):
    model = DC_1(config, **kwargs)
    return model
class DC_1:
    def __init__(self, config, **kwargs):
        super(DC_1, self).__init__()
        self.name = 'dc_1'
        self.config = config
        self.output_action_dim = kwargs['action_dim'] # for hidden vector to RL agent

    def __call__(self, **kwargs):
        up_events_num = np.sum(kwargs['dc_events'], axis=1)
        fth = kwargs['dc_events'].shape[-1] / 2
        lambda_val_ay = np.ones(kwargs['dc_events'].shape[0])
        sigma_val_ay = np.ones(kwargs['dc_events'].shape[0])
        upidx = np.argwhere(up_events_num > fth).flatten()
        downidx = np.argwhere(up_events_num < fth).flatten()
        lambda_val_ay[upidx] = 0 # up
        lambda_val_ay[downidx] = 2 # down
        sigma_val_ay[upidx] = 0 # up
        sigma_val_ay[downidx] = 2 # down
        
        up_idx = np.argwhere(kwargs['dc_events'])
        cur_hidden_vector_ay = np.zeros((kwargs['dc_events'].shape[0], self.output_action_dim)) # (batch, action_dim)
        
        zidx = np.argwhere(up_events_num==0).flatten()
        up_num_norm = np.divide(np.ones_like(up_events_num), up_events_num.astype(np.float32), out=np.zeros_like(up_events_num)*1.0, where=up_events_num!=0)
        if self.output_action_dim == self.config.topK:
            cur_hidden_vector_ay[up_idx[:,0], up_idx[:, 1]] = up_num_norm[up_idx[:, 0]]
        elif self.output_action_dim == self.config.topK + 1:
            # The first dim is cash.
            cur_hidden_vector_ay[up_idx[:,0], up_idx[:, 1]+1] = up_num_norm[up_idx[:, 0]]
        else:
            raise ValueError("Unmatch action_dim: {}, stock_num: {}".format(self.output_action_dim, self.config.topK))
        cur_hidden_vector_ay[zidx] = 1.0 / self.output_action_dim
        # hidden_vec: (batch, num_of_stocks) is the hidden vector sent to RL agents.
        # sigma_val_ay: (batch, )
        # lambda_val_ay: (batch, )
        return cur_hidden_vector_ay, lambda_val_ay, sigma_val_ay

def is_model(net_name: str) -> bool:
    return net_name in _mkt_obs_model_entrypoints

def mkt_obs_model_entrypoint(net_name):
    return _mkt_obs_model_entrypoints[net_name]

def create_mkt_obs_model(config, **kwargs):
    """
    model net name: 'mlp_1', {arch_name}-{rc_version}
    """
    if not is_model(net_name=config.mktobs_algo):
        algo_arch, algo_verison = config.mktobs_algo.split('_', 1)
        raise ValueError("Unknown market observer model/architecture: {}, tag: {}".format(algo_arch, algo_verison))
    create_fn = mkt_obs_model_entrypoint(net_name=config.mktobs_algo)
    model = create_fn(
        config=config,
        **kwargs,
    )
    # Load checkpoint
    return model

@register_mkt_obs_model
def transformer_1(config, **kwargs):
    model = Transformer_1(config, **kwargs)
    return model

class PositionalEncoding(nn.Module):
    """位置编码模块，为Transformer添加位置信息"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = th.zeros(max_len, d_model)
        position = th.arange(0, max_len, dtype=th.float).unsqueeze(1)
        div_term = th.exp(th.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Transformer_1(nn.Module):
    def __init__(self, config, **kwargs):
        super(Transformer_1, self).__init__()
        self.name = 'transformer_1'
        self.config = config
        self.output_action_dim = kwargs['action_dim'] # for hidden vector to RL agent
        
        # 输入特征维度计算
        self.num_stocks = self.config.topK
        self.num_features = len(self.config.use_features)
        self.window_len = self.config.fine_window_size
        
        # Transformer配置参数（从config获取，如果不存在则使用默认值）
        self.d_model = getattr(self.config, 'transformer_d_model', 256)
        self.nhead = getattr(self.config, 'transformer_nhead', 8)
        self.num_encoder_layers = getattr(self.config, 'transformer_num_layers', 4)
        self.dim_feedforward = getattr(self.config, 'transformer_dim_feedforward', 512)
        self.dropout = getattr(self.config, 'transformer_dropout', 0.1)
        
        # 输入维度
        self.stock_input_dim = self.num_stocks * self.num_features  # 每个时间步的股票特征维度
        self.market_input_dim = self.num_features                   # 每个时间步的市场特征维度
        
        # 输入投影层
        self.stock_projection = nn.Linear(self.stock_input_dim, self.d_model // 2)
        self.market_projection = nn.Linear(self.market_input_dim, self.d_model // 2)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=self.window_len, dropout=self.dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation='gelu',
            batch_first=False  # 使用(seq_len, batch, features)格式
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_encoder_layers)
        
        # 全局池化层
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 输出层
        self.fc_merge = nn.Linear(self.d_model, self.output_action_dim, bias=True)
        self.sm = nn.Softmax(dim=1)
        
        # Lambda和Sigma预测头
        self.fc_lambda = nn.Linear(self.d_model, 3, bias=True)
        self.fc_sigma = nn.Linear(self.d_model, 3, bias=True)
        
        self.gen_lambda = GenScore()
        self.gen_sigma = GenScore()
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(self.d_model)
        
    def forward(self, x, **kwargs):
        # fine stock data (x): (batch, features, num_of_stocks, window_size)
        # fine market data: (batch, features, window_size) 
        # kwargs: market, deterministic, device
        
        batch_size = x.size(0)
        market = kwargs['market']
        
        # 重新组织数据维度
        # x: (batch, features, stocks, window) -> (batch, window, features*stocks)
        x_reshaped = x.permute(0, 3, 1, 2).contiguous()  # (batch, window, features, stocks)
        x_reshaped = x_reshaped.view(batch_size, self.window_len, -1)  # (batch, window, features*stocks)
        
        # market: (batch, features, window) -> (batch, window, features)
        market_reshaped = market.permute(0, 2, 1).contiguous()  # (batch, window, features)
        
        # 投影到模型维度
        stock_embedded = self.stock_projection(x_reshaped)      # (batch, window, d_model//2)
        market_embedded = self.market_projection(market_reshaped)  # (batch, window, d_model//2)
        
        # 融合股票和市场特征
        combined_embedded = th.cat([stock_embedded, market_embedded], dim=2)  # (batch, window, d_model)
        
        # 转换为Transformer期望的格式: (seq_len, batch, d_model)
        transformer_input = combined_embedded.permute(1, 0, 2)  # (window, batch, d_model)
        
        # 添加位置编码
        transformer_input = self.pos_encoder(transformer_input)
        
        # Transformer编码
        transformer_output = self.transformer_encoder(transformer_input)  # (window, batch, d_model)
        
        # 转回 (batch, window, d_model) 格式
        transformer_output = transformer_output.permute(1, 0, 2)  # (batch, window, d_model)
        
        # 全局平均池化获取序列级表示
        # (batch, window, d_model) -> (batch, d_model, window) -> (batch, d_model, 1) -> (batch, d_model)
        pooled_output = self.global_pool(transformer_output.transpose(1, 2)).squeeze(-1)
        
        # 层归一化
        final_representation = self.layer_norm(pooled_output)  # (batch, d_model)
        
        # 生成输出
        hidden_vec = self.fc_merge(final_representation)  # (batch, num_of_stocks)
        hidden_vec = self.sm(hidden_vec)  # softmax归一化
        
        # Lambda预测
        lambda_vec = self.fc_lambda(final_representation)  # (batch, 3)
        lambda_kwargs = {
            'name': 'lambda', 
            'deterministic': kwargs['deterministic'], 
            'device': kwargs['device'], 
            'score_min': self.config.lambda_min, 
            'score_max': self.config.lambda_max
        }
        lambda_val, lambda_log_p = self.gen_lambda(lambda_vec, **lambda_kwargs)
        
        # Sigma预测
        sigma_vec = self.fc_sigma(final_representation)  # (batch, 3)
        sigma_kwargs = {
            'name': 'sigma', 
            'deterministic': kwargs['deterministic'], 
            'device': kwargs['device'], 
            'score_min': self.config.sigma_min, 
            'score_max': self.config.sigma_max
        }
        sigma_val, sigma_log_p = self.gen_sigma(sigma_vec, **sigma_kwargs)
        
        return hidden_vec, lambda_val, sigma_val, lambda_log_p, sigma_log_p