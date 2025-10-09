# @Time   : 2025/09/28
# @Author : Your Name
# @Email  : your_email@example.com

"""
SemEnhancedSequentialDataset - 核心特性：
1. 所有Pad值统一为0（定死，无需配置）
2. 加载嵌套字典格式的item_semantic_dict.pkl
3. 简化混合序列生成流程（边遍历边构建）
4. 保持与原始SequentialDataset兼容
"""

import numpy as np
import torch
import pickle
import os
from recbole.data.dataset import Dataset
from recbole.data.interaction import Interaction
from recbole.utils.enum_type import FeatureType, FeatureSource
from logging import getLogger


class SemEnhancedSequentialDataset(Dataset):
    def __init__(self, config):
        # -------------------------- 1. 基础配置  --------------------------
        self.max_item_list_len = config["MAX_ITEM_LIST_LENGTH"]
        self.actual_seq_len = self.max_item_list_len * 2  # 混合序列最大长度（Item数×2）
        self.item_list_length_field = config["ITEM_LIST_LENGTH_FIELD"]
        self.pad_token = 0  # 所有字段的Pad值统一为0
        self.sem_pad_token = self.pad_token  # Sem Pad=0，与其他Pad统一

        self.item_sem_mask_field = "item_sem_mask"  # mask字段名称（存储1=Sem，0=Item）
        
        # -------------------------- 2. Sem映射配置 --------------------------
        # Sem字典路径（默认值：item_semantic_dict.pkl）
        if "ITEM_SEMANTIC_DICT_PATH" in config:
            self.sem_dict_path = config["ITEM_SEMANTIC_DICT_PATH"]
        else:
            self.sem_dict_path = "item_semantic_dict.pkl"
        
        self.item_to_sem = {}  # 存储{item_id: 0-based sem_id}
        self.n_items = 0  # 总Item数（从父类获取，用于后续Sem ID范围计算）
        self.sem_start = 0  # Sem Token起始ID（=n_items，避免与Item ID冲突）
        
        # -------------------------- 3. 父类初始化（优先执行） --------------------------
        super().__init__(config)
        self.n_items = self.item_num  # 从父类获取总Item数（Item ID范围：0~n_items-1，含Pad=0）
        
        # -------------------------- 4. 核心步骤：加载Sem映射+初始化Sem范围 --------------------------
        self._load_item_semantic_dict()
        self.sem_start = self.n_items  # Sem ID从n_items开始（确保与Item ID不冲突）
        
        # -------------------------- 5. 基准数据集处理 --------------------------
        if "benchmark_filename" in config and config["benchmark_filename"] is not None:
            self._benchmark_presets()

        self.logger = getLogger()
        self.logger.info(
            f"初始化完成：Pad=0，n_items={self.n_items}，sem_start={self.sem_start}，"
            f"有效Item-Sem映射数={len(self.item_to_sem)}"
        )

    def _load_item_semantic_dict(self):
        """加载嵌套字典格式的映射文件，提取{item_id: 0-based sem_id}"""
        candidate_paths = [self.sem_dict_path]
        if hasattr(self, "dataset_path"):
            candidate_paths.append(os.path.join(self.dataset_path, "item_semantic_dict.pkl"))
        
        for path in candidate_paths:
            if os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        item_to_sem_nested = pickle.load(f, encoding="utf-8")
                    
                    valid_map = {}
                    invalid_count = 0
                    for item_id, sem_info in item_to_sem_nested.items():
                        # 1. 验证Item ID为整数
                        if not isinstance(item_id, (int, np.integer)):
                            invalid_count += 1
                            continue
                        # 2. 提取layer_0的sem_id（0-based）
                        if not (isinstance(sem_info, dict) and "layer_0" in sem_info):
                            invalid_count += 1
                            continue
                        sem_id = sem_info["layer_0"]
                        # 3. 验证sem_id为非负整数
                        if isinstance(sem_id, (int, np.integer)) and sem_id >= 0:
                            valid_map[item_id] = sem_id
                        else:
                            invalid_count += 1
                    
                    self.logger.info(
                        f"加载Item-Sem映射：{path} "
                        f"（有效{len(valid_map)}/总{len(item_to_sem_nested)}，无效{invalid_count}）"
                    )
                    self.item_to_sem = valid_map
                    return
                except Exception as e:
                    self.logger.error(f"加载{path}失败：{str(e)}", exc_info=True)
        
        raise FileNotFoundError(f"未找到映射文件，尝试路径：{candidate_paths}")

    def _change_feat_format(self):
        """转换格式并触发增强（无调试代码）"""
        super()._change_feat_format()
        if "benchmark_filename" in self.config and self.config["benchmark_filename"] is not None:
            return
        self.logger.debug("执行数据增强（混合序列生成）")
        self.data_augmentation()

    def _aug_presets(self):
        """预设字段属性（Item序列长度适配混合序列）"""
        list_suffix = self.config["LIST_SUFFIX"]
        for field in self.inter_feat:
            if field != self.uid_field:
                list_field = field + list_suffix
                setattr(self, f"{field}_list_field", list_field)
                ftype = self.field2type[field]

                # 序列类型：Token序列/数值序列
                list_ftype = FeatureType.TOKEN_SEQ if ftype in [FeatureType.TOKEN, FeatureType.TOKEN_SEQ] else FeatureType.FLOAT_SEQ
                # 长度配置：Item字段翻倍（含Sem），其他字段按原始长度
                if field == self.iid_field:
                    list_len = self.actual_seq_len if ftype not in [FeatureType.TOKEN_SEQ, FeatureType.FLOAT_SEQ] else (self.actual_seq_len, self.field2seqlen[field])
                else:
                    list_len = self.max_item_list_len if ftype not in [FeatureType.TOKEN_SEQ, FeatureType.FLOAT_SEQ] else (self.max_item_list_len, self.field2seqlen[field])

                self.set_field_property(list_field, list_ftype, FeatureSource.INTERACTION, list_len)

        # 注册序列长度字段
        self.set_field_property(self.item_list_length_field, FeatureType.TOKEN, FeatureSource.INTERACTION, 1)
        # 注册Sem掩码字段（与混合序列长度相同）
        self.set_field_property(
            self.item_sem_mask_field, 
            FeatureType.TOKEN_SEQ, 
            FeatureSource.INTERACTION, 
            self.actual_seq_len  # 与混合序列长度相同
        )

    def data_augmentation(self):
        """简化版：遍历用户时直接生成混合序列（Item+Sem交替，Pad=0）"""
        self.logger.debug("开始生成混合序列")
        self._aug_presets()
        self._check_field("uid_field", "time_field", "iid_field")

        # 1. 按用户+时间排序
        self.sort(by=[self.uid_field, self.time_field], ascending=True)
        last_uid = None
        target_index = []  # 目标Item索引
        mixed_sequences = []  # 混合序列（Item+Sem）
        mask_sequences = []  # 新增：对应的mask序列
        seq_lengths = []  # 混合序列实际长度
        seq_start = 0
        all_items = self.inter_feat[self.iid_field].numpy()  # 预取所有Item

        # 2. 核心遍历：边遍历边生成混合序列
        for i, uid in enumerate(self.inter_feat[self.uid_field].numpy()):
            if last_uid != uid:
                # 切换用户：重置起始位置
                last_uid = uid
                seq_start = i
            else:
                # 控制Item序列最大长度（不超过max_item_list_len）
                current_item_count = i - seq_start
                if current_item_count > self.max_item_list_len:
                    seq_start += 1

                # 提取当前用户的Item序列（从seq_start到i-1）
                current_items = all_items[seq_start:i].tolist()
                # 构建混合序列（Item + Sem，无映射用Pad=0）
                mixed_seq = []
                mask_seq = []  # 新增：当前序列的mask
                for item in current_items:
                    mixed_seq.append(item)  # 添加Item
                    mask_seq.append(0)  # Item位置为0
                    # 获取Sem ID（转换为模型用的ID：sem_start + sem_id）
                    sem_id_0based = self.item_to_sem.get(item, self.sem_pad_token)
                    sem_token_id = self.sem_start + sem_id_0based if sem_id_0based != self.sem_pad_token else self.sem_pad_token
                    mixed_seq.append(sem_token_id)  # 添加Sem/Pad
                    mask_seq.append(1 if sem_token_id != self.sem_pad_token else 0)  # Sem有效时为1，Pad为0

                # 记录关键信息
                target_index.append(i)  # 目标Item是当前i位置的Item
                mixed_sequences.append(mixed_seq)
                mask_sequences.append(mask_seq)  # 新增：保存mask
                seq_lengths.append(len(mixed_seq))  # 混合序列实际长度

        # 3. 初始化新数据结构
        new_length = len(mixed_sequences)
        new_data = self.inter_feat[target_index]
        new_dict = {self.item_list_length_field: torch.tensor(seq_lengths, dtype=torch.long)}

        # 4. 填充Item混合序列（Pad=0）
        item_list_field = getattr(self, f"{self.iid_field}_list_field")
        item_list_len = self.field2seqlen[item_list_field] if isinstance(self.field2seqlen[item_list_field], int) else self.field2seqlen[item_list_field][0]
        # 初始化张量（统一用Pad=0）
        new_dict[item_list_field] = torch.full(
            (new_length, item_list_len), fill_value=self.pad_token, dtype=torch.long
        )
        # 填充混合序列（超出长度自动截断）
        for i, seq in enumerate(mixed_sequences):
            new_dict[item_list_field][i, :len(seq)] = torch.tensor(seq, dtype=torch.long)

        new_dict[self.item_sem_mask_field] = torch.full(
            (new_length, item_list_len), fill_value=0, dtype=torch.long  # mask的Pad为0
        )
        for i, mask in enumerate(mask_sequences):
            new_dict[self.item_sem_mask_field][i, :len(mask)] = torch.tensor(mask, dtype=torch.long)
        # 5. 填充其他字段（如timestamp/rating，Pad=0）
        for field in self.inter_feat:
            if field == self.uid_field or field == self.iid_field:
                continue  # 已单独处理

            list_field = getattr(self, f"{field}_list_field")
            list_len = self.field2seqlen[list_field] if isinstance(self.field2seqlen[list_field], int) else self.field2seqlen[list_field][0]
            # 初始化张量（Pad=0）
            new_dict[list_field] = torch.full(
                (new_length, list_len), fill_value=self.pad_token, dtype=self.inter_feat[field].dtype
            )
            # 填充原始值（仅取Item对应的位置）
            field_values = self.inter_feat[field].numpy()
            for i in range(new_length):
                start = seq_start - (i - seq_start)  # 匹配原始Item序列范围
                end = target_index[i]
                original_values = field_values[start:end].tolist()
                new_dict[list_field][i, :len(original_values)] = torch.tensor(
                    original_values, dtype=self.inter_feat[field].dtype
                )

        # 6. 更新交互数据
        new_data.update(Interaction(new_dict))
        self.inter_feat = new_data
        self.logger.debug(f"混合序列生成完成：{new_length}条，平均长度{np.mean(seq_lengths):.1f}")

    def _benchmark_presets(self):
        """基准数据集处理（新增mask生成）"""
        list_suffix = self.config["LIST_SUFFIX"]
        for field in self.inter_feat:
            if field + list_suffix in self.inter_feat:
                list_field = field + list_suffix
                setattr(self, f"{field}_list_field", list_field)
        
        # 注册序列长度和mask字段（新增mask）
        self.set_field_property(self.item_list_length_field, FeatureType.TOKEN, FeatureSource.INTERACTION, 1)
        self.set_field_property(
            self.item_sem_mask_field, 
            FeatureType.TOKEN_SEQ, 
            FeatureSource.INTERACTION, 
            self.actual_seq_len
        )

        # 处理Item混合序列（原有）
        item_list_field = getattr(self, f"{self.iid_field}_list_field")
        if item_list_field not in self.inter_feat:
            self.logger.warning(f"基准数据集缺少{item_list_field}，跳过Sem处理")
            return

        item_seqs = self.inter_feat[item_list_field].numpy()
        user_ids = self.inter_feat[self.uid_field].numpy()
        mixed_seqs = []
        mask_seqs = []  # 新增：基准数据集的mask
        seq_lengths = []
        last_uid = None
        seq_len_fixed = item_seqs.shape[1]

        # 逐用户生成混合序列和mask
        for uid, item_seq in zip(user_ids, item_seqs):
            if last_uid != uid:
                last_uid = uid
            
            mixed_seq = []
            mask_seq = []  # 新增：当前序列的mask
            for item in item_seq:
                if item == self.pad_token:
                    break
                # 添加Item（mask=0）
                mixed_seq.append(item)
                mask_seq.append(0)
                # 添加Sem（mask=1，Pad=0）
                sem_id_0based = self.item_to_sem.get(item, self.sem_pad_token)
                sem_token_id = self.sem_start + sem_id_0based if sem_id_0based != self.sem_pad_token else self.sem_pad_token
                mixed_seq.append(sem_token_id)
                mask_seq.append(1 if sem_token_id != self.sem_pad_token else 0)
            
            # 补Pad（混合序列补0，mask也补0）
            mixed_seq += [self.pad_token] * (seq_len_fixed - len(mixed_seq))
            mask_seq += [0] * (seq_len_fixed - len(mask_seq))  # mask的Pad为0
            mixed_seqs.append(mixed_seq)
            mask_seqs.append(mask_seq)  # 新增：保存mask
            actual_len = len([x for x in mixed_seq if x != self.pad_token])
            seq_lengths.append(actual_len)

        # 更新基准数据（新增mask）
        self.inter_feat[item_list_field] = torch.tensor(mixed_seqs, dtype=torch.long)
        self.inter_feat[self.item_sem_mask_field] = torch.tensor(mask_seqs, dtype=torch.long)  # 新增：添加mask字段
        self.inter_feat[self.item_list_length_field] = torch.tensor(seq_lengths, dtype=torch.long)
        self.logger.debug(f"基准数据集处理完成：{len(mixed_seqs)}条混合序列及mask")

    def inter_matrix(self, form="coo", value_field=None):
        """生成交互矩阵（忽略Sem，仅保留Item）"""
        if not self.uid_field or not self.iid_field:
            raise ValueError("缺少uid/iid字段，无法生成矩阵")

        # 筛选仅含Item的序列（长度=1，无Sem）
        l1_idx = self.inter_feat[self.item_list_length_field] == 1
        if l1_idx.sum() == 0:
            self.logger.warning("无有效序列，返回空矩阵")
            return torch.sparse_coo_tensor(size=(self.user_num, self.item_num))

        l1_inter_dict = self.inter_feat[l1_idx].interaction
        new_dict = {}
        list_suffix = self.config["LIST_SUFFIX"]

        # 提取Item（仅取序列第0位）
        if self.iid_field + list_suffix in l1_inter_dict:
            new_dict[self.iid_field] = torch.cat([
                self.inter_feat[self.iid_field],
                l1_inter_dict[self.iid_field + list_suffix][:, 0]
            ])
        # 提取用户和其他字段
        if self.uid_field in l1_inter_dict:
            new_dict[self.uid_field] = torch.cat([
                self.inter_feat[self.uid_field],
                l1_inter_dict[self.uid_field]
            ])
        for field in l1_inter_dict:
            if field not in [self.uid_field, self.iid_field] and not field.endswith(list_suffix) and field != self.item_list_length_field:
                new_dict[field] = torch.cat([self.inter_feat[field], l1_inter_dict[field]])

        local_inter_feat = Interaction(new_dict)
        return self._create_sparse_matrix(local_inter_feat, self.uid_field, self.iid_field, form, value_field)

    def build(self):
        """仅支持时序排序（TO）"""
        ordering_args = self.config["eval_args"]["order"]
        if ordering_args != "TO":
            raise ValueError(f"序列推荐需排序参数为'TO'，当前为{ordering_args}")
        return super().build()