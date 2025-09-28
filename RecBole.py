from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import SASRec, DIN, SRGNN, DIEN, S3Rec, FEARec, CORE
from recbole.model.general_recommender import LightGCN
from recbole.model.context_aware_recommender import KD_DAGFM
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
from logging import getLogger
import argparse
import pdb

# 模型配置字典
MODEL_CONFIG = {
    'SASRec': 'SASRec.yaml',
    'DIN': 'DIN.yaml',
    'SRGNN': 'SRGNN.yaml',
    'DIEN': 'DIEN.yaml',
    'S3Rec': 'S3Rec.yaml',
    'FEARec': 'FEARec.yaml',
    'CORE': 'CORE.yaml',
    'NCL': 'NCL.yaml',
    'LightGCN': 'LightGCN.yaml',
    'KD_DAGFM': 'KD_DAGFM.yaml',
}

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Run a RecBole sequential model with specified parameters.')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the sequential model to run')
    args = parser.parse_args()

    # 检查模型名称是否有效
    if args.model_name not in MODEL_CONFIG:
        print(f"Error: Model {args.model_name} is not supported.")
        return

    # 使用配置文件创建配置对象
    config = Config(model=args.model_name, dataset='mydata', config_file_list=[MODEL_CONFIG[args.model_name]])

    # 初始化日志记录和随机种子
    init_logger(config)
    logger = getLogger()
    logger.info(config)
    init_seed(config['seed'], reproducibility=False)

    # 创建数据集和数据准备
    dataset = create_dataset(config)
    logger.info(dataset)
    pdb.set_trace()
    train_data, valid_data, test_data = data_preparation(config, dataset)
    print('_'*10)
    print(len(train_data))
    print(len(valid_data))
    print(len(test_data))
    print('_'*10)

    # 初始化模型和训练器
    device = 'cuda'
    model = globals()[args.model_name](config, train_data.dataset).to(device)
    logger.info(model)
    trainer = Trainer(config, model)

    # 开始训练
    trainer.fit(train_data, valid_data)

    # 评估模型
    test_result = trainer.evaluate(test_data)
    print(test_result)

if __name__ == '__main__':
    main()
