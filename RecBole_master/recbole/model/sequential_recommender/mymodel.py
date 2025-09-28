import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss


class STULayer(nn.Module):
    """修复维度不匹配问题的STU注意力层，确保与输入掩码形状兼容"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, causal=True):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.causal = causal  # 因果注意力，确保时序合理性
        
        # 计算STU内部维度（确保dim_feedforward能被2*nhead整除）
        assert dim_feedforward % (2 * nhead) == 0, \
            f"dim_feedforward({dim_feedforward}) must be divisible by 2*nhead({2*nhead})"
        self.hidden_dim = dim_feedforward // (2 * nhead)  # 每个头的hidden_dim
        self.attention_dim = self.hidden_dim  # Q/K维度与V保持一致
        
        # 输入归一化（预归一化策略，适配多层堆叠）
        self.input_norm = nn.LayerNorm(d_model)
        
        # Q/K/V/U投影层（一次线性变换得到所有需要的向量）
        self.uvqk_proj = nn.Linear(
            d_model, 
            (self.hidden_dim * 2 + self.attention_dim * 2) * nhead
        )
        
        # 输出投影层（将注意力输出映射回d_model维度）
        self.output_proj = nn.Linear(
            self.hidden_dim * nhead + d_model,  # U+attn_output拼接归一化输入
            d_model
        )
        
        # 输出归一化+增强正则化（缓解多层过拟合）
        self.output_norm = nn.LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(dropout * 1.2)  # 增强注意力权重正则化
        
        # 注意力缩放系数（与标准Transformer一致）
        self.attn_alpha = 1.0 / math.sqrt(self.attention_dim)
        
        # 初始化权重（遵循Transformer标准初始化）
        self._reset_parameters()
    
    def _reset_parameters(self):
        """初始化权重，确保与原Transformer初始化策略一致"""
        nn.init.xavier_uniform_(self.uvqk_proj.weight)
        nn.init.zeros_(self.uvqk_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        T, B, d_model = src.shape  # T:序列长度, B:批次大小, d_model:特征维度
        
        # 1. 输入归一化（预归一化，避免多层堆叠时数值膨胀）
        src_norm = self.input_norm(src)  # [T, B, d_model]
        
        # 2. 生成Q/K/V/U（通过一次线性变换后拆分，确保维度匹配）
        uvqk = self.uvqk_proj(src_norm)  # [T, B, (2h+2a)*H]
        h, a, H = self.hidden_dim, self.attention_dim, self.nhead
        
        # 拆分U/V/Q/K（调整维度顺序为 [H, B, T, dim]，便于多头注意力计算）
        U = uvqk[:, :, :H*h].view(T, B, H, h).permute(2, 1, 0, 3).contiguous()  # [H, B, T, h]
        V = uvqk[:, :, H*h:2*H*h].view(T, B, H, h).permute(2, 1, 0, 3).contiguous()  # [H, B, T, h]
        Q = uvqk[:, :, 2*H*h:2*H*h+H*a].view(T, B, H, a).permute(2, 1, 0, 3).contiguous()  # [H, B, T, a]
        K = uvqk[:, :, 2*H*h+H*a:].view(T, B, H, a).permute(2, 1, 0, 3).contiguous()  # [H, B, T, a]
        
        # 3. 计算注意力分数（添加数值稳定机制）
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.attn_alpha  # [H, B, T, T]
        
        # 4. 应用因果掩码（确保时序合理性，避免未来信息泄露）
        if self.causal:
            causal_mask = torch.triu(torch.ones(T, T, device=src.device), diagonal=1).bool()
            attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # 5. 应用padding掩码（忽略pad token的影响）
        if src_key_padding_mask is not None:
            padding_mask = src_key_padding_mask.unsqueeze(0).unsqueeze(2)  # [1, B, 1, T]
            attn_scores = attn_scores.masked_fill(padding_mask, float('-inf'))
        
        # 6. 注意力权重计算与增强dropout（缓解过拟合）
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)  # 增强正则化
        
        # 7. 注意力输出（确保维度连续，避免计算错误）
        attn_output = torch.matmul(attn_weights, V)  # [H, B, T, h]
        attn_output = attn_output.permute(2, 1, 0, 3).contiguous().view(T, B, H*h)  # [T, B, H*h]
        
        # 8. 残差连接与输出投影（预归一化+残差，适配多层堆叠）
        U_flat = U.permute(2, 1, 0, 3).contiguous().view(T, B, H*h)  # [T, B, H*h]
        residual = U_flat + attn_output
        residual = torch.cat([residual, src_norm], dim=-1)  # [T, B, H*h + d_model]
        
        output = self.output_proj(residual)  # [T, B, d_model]
        output = self.output_norm(output)
        output = F.dropout(output, p=self.dropout, training=self.training)
        
        # 9. 最终残差连接（用归一化输入，避免多层尺度不一致）
        output = output + src_norm
        return output


class Mymodel(SequentialRecommender):
    r"""
    融合语义增强STU层的SASRec模型（Item从0开始索引，与原生SASRec一致）
    核心特性：仅用第一层SEM生成多兴趣，基于目标item动态筛选最佳兴趣
    """

    def __init__(self, config, dataset):
        super(Mymodel, self).__init__(config, dataset)

        # -------------------------- 1. 基础参数初始化（对齐SASRec） --------------------------
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # 与d_model一致
        self.inner_size = config["inner_size"]  # STU的dim_feedforward
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]

        # -------------------------- 2. SEM专属参数（从config提取，不使用get方法） --------------------------
        quantizer_config = config["quantizer"]
        self.clusters_per_layer = quantizer_config["clusters_per_layer"]
        self.sem_layers_count = len(self.clusters_per_layer)
        
        # 核心逻辑：仅用第一层SEM生成多兴趣
        self.num_interest = self.clusters_per_layer[0]  # 兴趣数=第一层SEM聚类数
        self.num_sem = self.num_interest  # SEM数=兴趣数（1:1匹配）
        assert self.num_interest == self.num_sem, \
            f"兴趣数({self.num_interest})必须与第一层SEM数({self.num_sem})相等"
        
        # -------------------------- 关键修改：Item从0开始，SEM Token避开Item范围 --------------------------
        self.pad_token = 0  # Pad token固定为0
        self.item_start = 0  # Item从0开始索引（与原生SASRec一致）
        self.item_end = self.n_items - 1  # Item结束索引（0-based，共n_items个Item）
        self.sem_start = self.n_items  # SEM Token放在Item之后，避免重叠（Item:0~n_items-1，SEM:n_items~n_items+num_sem-1）
        self.sem_end = self.sem_start + self.num_sem - 1  # SEM结束索引
        # ------------------------------------------------------------------------------------------
        
        # SEM Token索引（放在Item之后，不与Item重叠）
        self.sem_token_indices = torch.arange(
            self.sem_start, self.sem_end + 1, device=self.device
        )  # [num_interest]

        # 其他SEM参数（直接赋值，不依赖config.model）
        self.add_pos = True  # 固定启用位置编码（与您当前代码一致）

        # 校验Token范围有效性（确保Item与SEM不重叠）
        self._validate_token_ranges()

        # -------------------------- 3. 模型层定义 --------------------------
        # 1. 嵌入层（包含Item+SEM+Type+Position，覆盖所有Token）
        self.token_emb = nn.Embedding(
            num_embeddings=self.sem_end + 1,  # 覆盖Pad(0)+Item(0~n_items-1)+SEM(n_items~sem_end)
            embedding_dim=self.hidden_size,
            padding_idx=self.pad_token
        )
        self.type_emb = nn.Embedding(
            num_embeddings=3,  # 0:pad, 1:SEM, 2:Item（区分Token类型）
            embedding_dim=self.hidden_size
        )
        self.type_emb.weight.data[0] = torch.zeros(self.hidden_size)  # Pad类型emb不更新
        self.type_emb.weight.data[0].requires_grad = False

        # 位置嵌入（适配max_seq_length）
        self.pos_emb = nn.Embedding(
            num_embeddings=self.max_seq_length,
            embedding_dim=self.hidden_size
        )

        # 2. STU编码器（完全保留原逻辑）
        self.encoder_layers = nn.ModuleList([
            STULayer(
                d_model=self.hidden_size,
                nhead=self.n_heads,
                dim_feedforward=self.inner_size,
                dropout=self.hidden_dropout_prob,
                causal=True
            ) for _ in range(self.n_layers)
        ])

        # 3. 正则化层（保留原逻辑）
        self.emb_dropout = nn.Dropout(self.hidden_dropout_prob * 1.5)
        self.interest_dropout = nn.Dropout(self.hidden_dropout_prob * 1.2)

        # 4. 损失函数（对齐SASRec）
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_token)
        else:
            raise NotImplementedError("loss_type必须为['BPR', 'CE']")

        # 权重初始化（保留原逻辑）
        self.apply(self._init_weights)

    def _validate_token_ranges(self):
        """校验Token范围：Pad、Item、SEM互不重叠"""
        assert self.pad_token == 0, "Pad Token必须为0（与SASRec一致）"
        assert self.item_start == 0, "Item必须从0开始索引（与SASRec一致）"
        assert self.item_end == self.n_items - 1, f"Item结束索引应为{self.n_items-1}（0-based）"
        assert self.sem_start == self.n_items, "SEM应从Item结束后开始（避免重叠）"
        assert self.sem_end == self.sem_start + self.num_sem - 1, "SEM范围计算错误"
        # 校验无重叠：Pad(0)、Item(0~item_end)、SEM(sem_start~sem_end)
        assert self.item_end < self.sem_start, "Item与SEM Token范围重叠！"
        if self.sem_layers_count > 1:
            print(f"⚠️  仅第一层SEM（{self.num_sem}个，索引{self.n_items}~{self.sem_end}）生效，其他层SEM配置已忽略")

    def _init_weights(self, module):
        """整合初始化逻辑（完全保留原逻辑）"""
        if isinstance(module, nn.Embedding):
            if module is self.token_emb:
                module.weight.data.normal_(mean=0.0, std=0.02)
                module.weight.data[self.pad_token] = 0.0  # Pad嵌入置零
            elif module is self.type_emb:
                module.weight.data[1:].normal_(mean=0.0, std=0.02)  # 仅初始化SEM和Item类型
            elif module is self.pos_emb:
                module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, (nn.Linear, STULayer)):
            if hasattr(module, "uvqk_proj"):  # STULayer的投影层
                nn.init.xavier_uniform_(module.uvqk_proj.weight)
                nn.init.zeros_(module.uvqk_proj.bias)
                nn.init.xavier_uniform_(module.output_proj.weight)
                nn.init.zeros_(module.output_proj.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def _get_token_type(self, item_seq):
        """获取Token类型掩码（区分Pad/Item/SEM，适配0-based Item）"""
        type_mask = torch.zeros_like(item_seq, dtype=torch.long, device=self.device)
        # Item：0~item_end（0-based，与SASRec一致）
        type_mask[(item_seq >= self.item_start) & (item_seq <= self.item_end)] = 2
        # SEM：sem_start~sem_end（在Item之后）
        type_mask[(item_seq >= self.sem_start) & (item_seq <= self.sem_end)] = 1
        # Pad：默认0，无需额外赋值
        return type_mask

    def _generate_causal_mask(self, seq_len):
        """生成因果掩码（完全保留原逻辑）"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=self.device), diagonal=1).bool()
        return mask.masked_fill(mask, float('-inf'))

    def _encode(self, x, padding_mask):
        """STU编码器：多层堆叠编码（完全保留原逻辑）"""
        T, B, _ = x.shape
        causal_mask = self._generate_causal_mask(T)
        current_x = x
        for layer in self.encoder_layers:
            current_x = layer(
                src=current_x,
                src_mask=causal_mask,
                src_key_padding_mask=padding_mask
            )
        return current_x  # [T, B, hidden_size]

    def _generate_multi_interest(self, seq_emb):
        """核心：基于第一层SEM生成多兴趣嵌入（完全保留原逻辑）"""
        B, T, H = seq_emb.shape

        # 1. 第一层SEM嵌入（使用Item之后的SEM Token）
        sem_emb = self.token_emb(self.sem_token_indices)  # [num_interest, H]
        sem_emb_norm = F.normalize(sem_emb, p=2, dim=-1)  # 归一化确保相似度稳定

        # 2. 计算序列与SEM的相似度（均衡化避免权重集中）
        seq_emb_norm = F.normalize(seq_emb, p=2, dim=-1)  # [B, T, H]
        att_scores = torch.matmul(seq_emb_norm, sem_emb_norm.T)  # [B, T, num_interest]
        att_scores_balanced = torch.pow(torch.abs(att_scores) + 1e-8, 0.5) * torch.sign(att_scores)

        # 3. 兴趣权重归一化+多兴趣生成
        att_weights = F.softmax(att_scores_balanced, dim=2)  # [B, T, num_interest]
        att_weights = F.dropout(att_weights, p=0.1, training=self.training)
        pos_interest_emb = seq_emb.unsqueeze(2) * att_weights.unsqueeze(3)  # [B, T, num_interest, H]
        pos_interest_emb = self.interest_dropout(pos_interest_emb)

        return pos_interest_emb

    def _get_last_valid_interest(self, pos_interest_emb, item_seq_len):
        """根据item_seq_len获取最后一个有效位置的多兴趣（完全保留原逻辑）"""
        B, T, num_interest, H = pos_interest_emb.shape
        
        # 获取每个序列的最后有效位置索引
        last_idx = (item_seq_len - 1).clamp(min=0)  # [B]
        
        # 为每个样本选择最后有效位置的兴趣（避免维度扩展错误）
        last_interest = []
        for i in range(B):
            last_interest.append(pos_interest_emb[i, last_idx[i], :, :])
        
        # 堆叠结果形成批次张量
        return torch.stack(last_interest, dim=0)  # [B, num_interest, H]

    def _get_best_interest(self, multi_interest, target_emb):
        """根据目标item嵌入筛选最佳兴趣（完全保留原逻辑）"""
        # 计算每个兴趣与目标的相似度 [B, num_interest]
        sim = torch.bmm(
            multi_interest, 
            target_emb.unsqueeze(-1)  # [B, H, 1]
        ).squeeze(-1)  # [B, num_interest]
        
        # 选择最相似的兴趣（argmax获取索引）
        best_idx = torch.argmax(sim, dim=1, keepdim=True)  # [B, 1]
        best_idx = best_idx.unsqueeze(-1).expand(-1, -1, multi_interest.size(-1))  # [B, 1, H]
        
        # 提取最佳兴趣
        best_interest = torch.gather(multi_interest, dim=1, index=best_idx).squeeze(1)  # [B, H]
        return best_interest

    def forward(self, item_seq, item_seq_len):
        """前向传播：输出最后有效位置的多兴趣嵌入（仅修改Token类型判断）"""
        B, T = item_seq.shape
        device = item_seq.device

        # 1. 获取Token类型与位置编码（适配0-based Item）
        type_mask = self._get_token_type(item_seq)  # [B, T]（区分Pad/Item/SEM）
        pos_ids = torch.arange(T, dtype=torch.long, device=device).unsqueeze(0).expand(B, T)  # [B, T]

        # 2. 嵌入层计算（TokenEmb + TypeEmb + PosEmb，保留原逻辑）
        token_emb = self.token_emb(item_seq)  # [B, T, H]
        type_emb = self.type_emb(type_mask)  # [B, T, H]（区分Token类型）
        pos_emb = self.pos_emb(pos_ids)  # [B, T, H]
        
        # 位置编码启用（固定True，与您当前代码一致）
        input_emb = token_emb + type_emb + pos_emb
        input_emb = self.emb_dropout(input_emb)  # [B, T, H]

        # 3. STU编码（完全保留原逻辑）
        padding_mask = (item_seq == self.pad_token)  # [B, T]（标记Pad Token）
        encoded = self._encode(input_emb.transpose(0, 1), padding_mask)  # [T, B, H]
        encoded = encoded.transpose(0, 1)  # [B, T, H]

        # 4. 生成多兴趣 + 提取最后有效位置（完全保留原逻辑）
        pos_interest_emb = self._generate_multi_interest(encoded)  # [B, T, num_interest, H]
        last_interest = self._get_last_valid_interest(pos_interest_emb, item_seq_len)  # [B, num_interest, H]

        return last_interest

    def calculate_loss(self, interaction):
        """计算损失：删除Item索引偏移，与原生SASRec一致"""
        # 1. 前向计算获取最后有效位置的多兴趣（保留原逻辑）
        item_seq = interaction[self.ITEM_SEQ]  # [B, T]（0-based Item）
        item_seq_len = interaction[self.ITEM_SEQ_LEN]  # [B]
        last_interest = self.forward(item_seq, item_seq_len)  # [B, num_interest, H]
        B, num_interest, H = last_interest.shape

        # 2. 获取正样本嵌入（目标Item，0-based，无需偏移）
        pos_items = interaction[self.POS_ITEM_ID]  # [B]（0-based，与SASRec一致）
        pos_emb = self.token_emb(pos_items)  # [B, H]（目标Item嵌入）

        if self.loss_type == "BPR":
            # BPR损失：多兴趣取最大相似度（完全保留原逻辑）
            neg_items = interaction[self.NEG_ITEM_ID]  # [B]（0-based）
            neg_emb = self.token_emb(neg_items)  # [B, H]
            
            # 计算多兴趣与正负样本的相似度
            pos_sim = torch.bmm(last_interest, pos_emb.unsqueeze(-1)).squeeze(-1)  # [B, num_interest]
            neg_sim = torch.bmm(last_interest, neg_emb.unsqueeze(-1)).squeeze(-1)  # [B, num_interest]
            
            # 多兴趣取最大相似度
            pos_score = pos_sim.max(dim=1)[0]  # [B]
            neg_score = neg_sim.max(dim=1)[0]  # [B]
            
            loss = self.loss_fct(pos_score, neg_score)

        else:  # CE损失：删除Item索引偏移，直接使用0-based标签
            # 核心：根据目标Item嵌入筛选最佳兴趣（保留原逻辑）
            best_interest = self._get_best_interest(last_interest, pos_emb)  # [B, H]
            
            # 计算所有Item的Logits（0-based Item，无需切片偏移）
            all_item_emb = self.token_emb.weight[self.item_start:self.item_end+1]  # [n_items, H]（0~n_items-1）
            logits = torch.matmul(best_interest, all_item_emb.transpose(0, 1))  # [B, n_items]（对应0-based Item）
            
            # 直接使用0-based pos_items作为标签（与SASRec一致，无需调整）
            loss = self.loss_fct(logits, pos_items)

        return loss

    def predict(self, interaction):
        """预测单个Item的分数（适配0-based Item，无需偏移）"""
        item_seq = interaction[self.ITEM_SEQ]  # [B, T]（0-based）
        item_seq_len = interaction[self.ITEM_SEQ_LEN]  # [B]
        last_interest = self.forward(item_seq, item_seq_len)  # [B, num_interest, H]
        
        test_item = interaction[self.ITEM_ID]  # [B]（0-based，与SASRec一致）
        test_item_emb = self.token_emb(test_item)  # [B, H]
        
        # 计算每个兴趣与测试Item的相似度，取最大（保留原逻辑）
        sim = torch.bmm(last_interest, test_item_emb.unsqueeze(-1)).squeeze(-1)  # [B, num_interest]
        scores = sim.max(dim=1)[0]  # [B]

        return scores

    def full_sort_predict(self, interaction):
        """预测所有Item的分数（适配0-based Item，无需偏移）"""
        item_seq = interaction[self.ITEM_SEQ]  # [B, T]（0-based）
        item_seq_len = interaction[self.ITEM_SEQ_LEN]  # [B]
        last_interest = self.forward(item_seq, item_seq_len)  # [B, num_interest, H]
        
        # 所有Item的嵌入（0-based，无需切片偏移）
        all_item_emb = self.token_emb.weight[self.item_start:self.item_end+1]  # [n_items, H]（0~n_items-1）
        
        # 多兴趣计算相似度后取最大（保留原逻辑）
        sim = torch.matmul(last_interest, all_item_emb.transpose(0, 1))  # [B, num_interest, n_items]
        scores = sim.max(dim=1)[0]  # [B, n_items]

        return scores