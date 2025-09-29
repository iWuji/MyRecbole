import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from logging import getLogger


class STULayer(nn.Module):
    """STU注意力层（仅保留因果掩码，不处理padding掩码）"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, causal=True):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.causal = causal
        
        # 确保前馈网络维度可被头数整除
        assert dim_feedforward % (2 * nhead) == 0, \
            f"dim_feedforward({dim_feedforward}) must be divisible by 2*nhead({2*nhead})"
        self.hidden_dim = dim_feedforward // (2 * nhead)
        self.attention_dim = self.hidden_dim
        
        # 层定义
        self.input_norm = nn.LayerNorm(d_model)
        self.uvqk_proj = nn.Linear(d_model, (self.hidden_dim * 2 + self.attention_dim * 2) * nhead)
        self.output_proj = nn.Linear(self.hidden_dim * nhead + d_model, d_model)
        self.output_norm = nn.LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_alpha = 1.0 / math.sqrt(self.attention_dim)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """参数初始化（确保训练稳定性）"""
        nn.init.xavier_uniform_(self.uvqk_proj.weight)
        nn.init.zeros_(self.uvqk_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, src, src_mask=None):
        T, B, d_model = src.shape
        src_norm = self.input_norm(src)
        
        # 投影为U/V/Q/K
        uvqk = self.uvqk_proj(src_norm)
        h, a, H = self.hidden_dim, self.attention_dim, self.nhead
        U = uvqk[:, :, :H*h].view(T, B, H, h).permute(2, 1, 0, 3).contiguous()
        V = uvqk[:, :, H*h:2*H*h].view(T, B, H, h).permute(2, 1, 0, 3).contiguous()
        Q = uvqk[:, :, 2*H*h:2*H*h+H*a].view(T, B, H, a).permute(2, 1, 0, 3).contiguous()
        K = uvqk[:, :, 2*H*h+H*a:].view(T, B, H, a).permute(2, 1, 0, 3).contiguous()
        
        # 因果注意力计算
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.attn_alpha
        if self.causal:
            causal_mask = torch.triu(torch.ones(T, T, device=src.device), diagonal=1).bool()
            attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # 注意力权重与输出
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.permute(2, 1, 0, 3).contiguous().view(T, B, H*h)
        
        # 残差连接与归一化
        U_flat = U.permute(2, 1, 0, 3).contiguous().view(T, B, H*h)
        residual = U_flat + attn_output
        residual = torch.cat([residual, src_norm], dim=-1)
        output = self.output_proj(residual)
        output = self.output_norm(output)
        output = F.dropout(output, p=self.dropout, training=self.training)
        output = output + src_norm
        
        return output


class Mymodel(SequentialRecommender):
    r"""统一索引逻辑的语义增强模型（1-based语义token适配）"""

    def __init__(self, config, dataset):
        super(Mymodel, self).__init__(config, dataset)
        self.logger = getLogger()

        # -------------------------- 1. 基础参数（对齐SASRec） --------------------------
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.loss_type = config["loss_type"]
        self.max_seq_length = config["MAX_ITEM_LIST_LENGTH"] * 2  # 适配混合序列（Item+Sem）
        self.ITEM_LIST_LENGTH_FIELD = config['ITEM_LIST_LENGTH_FIELD']
        
        # 统一Pad值（0号为Pad，所有字段一致）
        self.pad_token = 0
        self.item_pad = self.pad_token
        self.sem_pad = self.pad_token

        # -------------------------- 2. 语义参数（统一索引逻辑） --------------------------
        # 语义配置：1-based语义token总数（1~num_sem）
        quantizer_config = config["quantizer"]
        self.clusters_per_layer = quantizer_config["clusters_per_layer"]
        self.num_sem = self.clusters_per_layer[0]  # 有效语义类别数（1~num_sem）
        self.num_interest = self.num_sem  # 多兴趣数与语义类别数一致
        assert self.num_interest == self.num_sem, \
            f"兴趣数({self.num_interest})必须与语义类别数({self.num_sem})相等"
        
        # Token范围定义（核心：索引无重叠）
        self.sem_token_start = self.n_items  # Sem Token ID起始（=n_items，避免与Item重叠）
        self.sem_token_end = self.sem_token_start + self.num_sem  # Sem Token ID范围：n_items~n_items+num_sem
        # - Sem Token ID = n_items → 对应Sem嵌入层索引0（Sem Pad）
        # - Sem Token ID = n_items+1 ~ n_items+num_sem → 对应Sem嵌入层索引1~num_sem（有效语义）

        # -------------------------- 3. 模型层（统一索引维度） --------------------------
        # Item嵌入：维度=n_items，索引0~n_items-1（0号=Item Pad）
        self.item_embedding = nn.Embedding(
            num_embeddings=self.n_items,
            embedding_dim=self.hidden_size,
            padding_idx=self.item_pad
        )
        
        # Sem嵌入：维度=num_sem+1，索引0~num_sem（0号=Sem Pad，1~num_sem=有效语义）
        self.sem_embedding = nn.Embedding(
            num_embeddings=self.num_sem + 1,
            embedding_dim=self.hidden_size,
            padding_idx=self.sem_pad
        )
        # 初始化Sem Pad向量为全0（冻结，避免噪声）
        self.sem_embedding.weight.data[self.sem_pad] = torch.zeros(self.hidden_size)
        self.sem_embedding.weight.data[self.sem_pad].requires_grad = False
        
        # 类型嵌入：区分Pad（0）、Sem（1）、Item（2）
        self.type_emb = nn.Embedding(
            num_embeddings=3,
            embedding_dim=self.hidden_size,
            padding_idx=self.pad_token
        )
        self.type_emb.weight.data[self.pad_token] = torch.zeros(self.hidden_size)
        self.type_emb.weight.data[self.pad_token].requires_grad = False
        
        # 位置嵌入：适配混合序列长度
        self.pos_emb = nn.Embedding(
            num_embeddings=self.max_seq_length,
            embedding_dim=self.hidden_size
        )
        
        # 编码器与Dropout
        self.encoder_layers = nn.ModuleList([
            STULayer(
                d_model=self.hidden_size,
                nhead=self.n_heads,
                dim_feedforward=self.inner_size,
                dropout=self.hidden_dropout_prob,
                causal=True
            ) for _ in range(self.n_layers)
        ])
        self.emb_dropout = nn.Dropout(self.hidden_dropout_prob)
        self.interest_dropout = nn.Dropout(self.hidden_dropout_prob)

        # -------------------------- 4. 损失函数 --------------------------
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss(ignore_index=self.item_pad)
        else:
            raise NotImplementedError("loss_type仅支持['BPR', 'CE']")

        # 初始化参数
        self.apply(self._init_weights)
        self.logger.info(
            f"模型初始化完成：\n"
            f"- Item索引范围：0~{self.n_items-1}（嵌入维度{self.n_items}）\n"
            f"- Sem Token ID范围：{self.sem_token_start}~{self.sem_token_end}（嵌入维度{self.num_sem+1}）\n"
            f"- 有效语义类别数：{self.num_sem}（嵌入索引1~{self.num_sem}）\n"
            f"- 混合序列最大长度：{self.max_seq_length}"
        )

    def _init_weights(self, module):
        """参数初始化（确保各层权重合理）"""
        if isinstance(module, nn.Embedding):
            if module is not self.item_embedding and module is not self.sem_embedding and module is not self.type_emb:
                module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def _get_token_type_mask(self, item_seq):
        """生成Token类型掩码：0=Pad，1=Sem，2=Item"""
        type_mask = torch.full_like(item_seq, fill_value=self.pad_token, device=self.device)
        
        # Item掩码：0~n_items-1 且 非Pad
        item_mask = (item_seq >= 0) & (item_seq < self.n_items) & (item_seq != self.item_pad)
        type_mask[item_mask] = 2
        
        # Sem掩码：sem_token_start~sem_token_end
        sem_mask = (item_seq >= self.sem_token_start) & (item_seq <= self.sem_token_end)
        type_mask[sem_mask] = 1
        
        return type_mask

    def _generate_causal_mask(self, seq_len):
        """生成因果掩码（防止未来信息泄露）"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=self.device), diagonal=1).bool()
        return mask.masked_fill(mask, float('-inf'))

    def _encode(self, x):
        """STU编码器前向传播"""
        T, B, _ = x.shape
        causal_mask = self._generate_causal_mask(T)
        current_x = x
        for layer in self.encoder_layers:
            current_x = layer(src=current_x, src_mask=causal_mask)
        return current_x  # [T, B, H]

    def gather_indexes(self, output, gather_index):
        """提取每个样本的最后有效位置特征（对齐序列长度）"""
        B, T, H = output.shape
        # 扩展索引维度以匹配output（[B] → [B, 1, H]）
        gather_index = gather_index.unsqueeze(1).unsqueeze(1).expand(-1, -1, H)
        # 提取最后有效位置特征
        output_tensor = output.gather(dim=1, index=gather_index)  # [B, 1, H]
        return output_tensor.squeeze(1)  # [B, H]

    def _generate_multi_interest(self, last_valid_emb):
        """基于最后有效特征生成多兴趣（仅用有效语义向量）"""
        B, H = last_valid_emb.shape
        
        # 取有效语义向量（索引1~num_sem，排除Pad索引0）
        valid_sem_indices = torch.arange(1, self.num_sem + 1, device=self.device)  # [1, 2, ..., num_sem]
        sem_emb = self.sem_embedding(valid_sem_indices)  # [num_sem, H]
        sem_emb_norm = F.normalize(sem_emb, p=2, dim=-1)
        
        # 计算注意力权重
        last_valid_norm = F.normalize(last_valid_emb, p=2, dim=-1)  # [B, H]
        att_scores = torch.matmul(last_valid_norm.unsqueeze(1), sem_emb_norm.T).squeeze(1)  # [B, num_sem]
        att_scores_balanced = torch.pow(torch.abs(att_scores) + 1e-8, 0.5) * torch.sign(att_scores)  # 平衡权重
        att_weights = F.softmax(att_scores_balanced, dim=1).unsqueeze(2)  # [B, num_sem, 1]
        
        # 生成多兴趣（加权求和）
        multi_interest = last_valid_emb.unsqueeze(1) * att_weights  # [B, num_sem, H]
        multi_interest = self.interest_dropout(multi_interest)
        
        return multi_interest

    def _get_best_interest(self, multi_interest, target_emb):
        """选择与目标Item最匹配的兴趣向量"""
        # 计算每个兴趣与目标Item的相似度
        sim = torch.bmm(multi_interest, target_emb.unsqueeze(-1)).squeeze(-1)  # [B, num_sem]
        # 选择相似度最高的兴趣
        best_idx = torch.argmax(sim, dim=1, keepdim=True).unsqueeze(-1)  # [B, 1, 1]
        best_idx = best_idx.expand(-1, -1, multi_interest.size(-1))  # [B, 1, H]
        best_interest = torch.gather(multi_interest, dim=1, index=best_idx).squeeze(1)  # [B, H]
        
        return best_interest

    def forward(self, item_seq, item_seq_len):
        """前向传播（统一索引转换，无越界风险）"""
        B, T = item_seq.shape
        device = self.device

        # -------------------------- 1. 基础掩码与编码 --------------------------
        # 类型掩码（区分Pad/Sem/Item）
        type_mask = self._get_token_type_mask(item_seq)  # [B, T]
        # 位置编码（0~T-1）
        pos_ids = torch.arange(T, dtype=torch.long, device=device).unsqueeze(0).expand(B, T)  # [B, T]

        # -------------------------- 2. 嵌入层计算（索引严格对齐） --------------------------
        # 2.1 Item嵌入（仅Item区域有效，Sem区域置0）
        item_mask = (item_seq >= 0) & (item_seq < self.n_items)  # 过滤Sem区域
        item_emb = self.item_embedding(item_seq * item_mask.long())  # [B, T, H]
        
        # 2.2 Sem嵌入（仅Sem区域有效，Item区域置0）
        sem_mask = (item_seq >= self.sem_token_start) & (item_seq <= self.sem_token_end)  # 过滤Item区域
        sem_idx = (item_seq - self.sem_token_start) * sem_mask.long()  # Sem Token ID → 嵌入层索引（0~num_sem）
        # 调试校验：确保sem_idx不超出嵌入层维度（仅训练时开启）
        if self.training and sem_mask.any():
            valid_sem_idx = sem_idx[sem_mask]
            if valid_sem_idx.max() > self.num_sem:
                self.logger.warning(
                    f"发现超出范围的Sem索引：max={valid_sem_idx.max()}, 嵌入层最大索引={self.num_sem}"
                )
        sem_emb = self.sem_embedding(sem_idx) * sem_mask.unsqueeze(-1).float()  # [B, T, H]

        # -------------------------- 3. 嵌入合并与编码 --------------------------
        # 合并Item+Sem+Type+Pos嵌入
        token_emb = item_emb + sem_emb  # [B, T, H]
        type_emb = self.type_emb(type_mask)  # [B, T, H]
        pos_emb = self.pos_emb(pos_ids)  # [B, T, H]
        input_emb = token_emb + type_emb + pos_emb  # [B, T, H]
        
        # Dropout与编码
        input_emb = self.emb_dropout(input_emb)
        encoded = self._encode(input_emb.transpose(0, 1))  # [T, B, H] → 编码器要求T在前
        encoded = encoded.transpose(0, 1)  # [B, T, H]

        # -------------------------- 4. 多兴趣生成 --------------------------
        # 提取最后有效位置特征（item_seq_len-1：序列长度对应最后一个有效Token）
        last_valid_index = item_seq_len - 1  # [B]
        last_valid_emb = self.gather_indexes(encoded, last_valid_index)  # [B, H]
        # 生成多兴趣
        multi_interest = self._generate_multi_interest(last_valid_emb)  # [B, num_sem, H]

        return multi_interest

    def calculate_loss(self, interaction):
        """计算损失（适配BPR/CE两种损失类型）"""
        # 提取输入数据
        item_seq = interaction[self.ITEM_SEQ]  # [B, T]：混合序列（Item+Sem）
        item_seq_len = interaction[self.ITEM_LIST_LENGTH_FIELD]  # [B]：序列实际长度
        pos_items = interaction[self.POS_ITEM_ID]  # [B]：目标正样本Item

        # 前向传播获取多兴趣
        multi_interest = self.forward(item_seq, item_seq_len)  # [B, num_sem, H]
        pos_emb = self.item_embedding(pos_items)  # [B, H]：正样本Item嵌入

        # 计算损失
        if self.loss_type == "BPR":
            # BPR损失：需要负样本
            if self.NEG_ITEM_ID not in interaction:
                raise ValueError("BPR损失需要负样本字段（NEG_ITEM_ID）")
            neg_items = interaction[self.NEG_ITEM_ID]  # [B]
            neg_emb = self.item_embedding(neg_items)  # [B, H]
            
            # 计算多兴趣与正负样本的相似度
            pos_sim = torch.bmm(multi_interest, pos_emb.unsqueeze(-1)).squeeze(-1)  # [B, num_sem]
            neg_sim = torch.bmm(multi_interest, neg_emb.unsqueeze(-1)).squeeze(-1)  # [B, num_sem]
            
            # 取每个样本相似度最高的兴趣计算损失
            pos_score = pos_sim.max(dim=1)[0]  # [B]
            neg_score = neg_sim.max(dim=1)[0]  # [B]
            loss = self.loss_fct(pos_score, neg_score)
        
        else:
            # CE损失：选择最佳兴趣后计算分类损失
            best_interest = self._get_best_interest(multi_interest, pos_emb)  # [B, H]
            all_item_emb = self.item_embedding.weight  # [n_items, H]：所有Item嵌入
            logits = torch.matmul(best_interest, all_item_emb.transpose(0, 1))  # [B, n_items]
            loss = self.loss_fct(logits, pos_items)  # 分类损失（目标：预测正样本Item）

        return loss

    def predict(self, interaction):
        """预测单个Item的得分（用于评估）"""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_LIST_LENGTH_FIELD]
        test_item = interaction[self.ITEM_ID]  # [B]：待预测Item

        # 前向传播获取多兴趣
        multi_interest = self.forward(item_seq, item_seq_len)  # [B, num_sem, H]
        test_item_emb = self.item_embedding(test_item)  # [B, H]

        # 计算相似度（取最高兴趣的相似度作为最终得分）
        sim = torch.bmm(multi_interest, test_item_emb.unsqueeze(-1)).squeeze(-1)  # [B, num_sem]
        scores = sim.max(dim=1)[0]  # [B]：每个样本的最高相似度得分

        return scores

    def full_sort_predict(self, interaction):
        """全量Item排序预测（用于评估）"""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_LIST_LENGTH_FIELD]

        # 前向传播获取多兴趣
        multi_interest = self.forward(item_seq, item_seq_len)  # [B, num_sem, H]
        all_item_emb = self.item_embedding.weight  # [n_items, H]

        # 计算多兴趣与所有Item的相似度（取最高兴趣的相似度）
        sim = torch.matmul(multi_interest, all_item_emb.transpose(0, 1))  # [B, num_sem, n_items]
        scores = sim.max(dim=1)[0]  # [B, n_items]：每个样本对所有Item的得分

        return scores