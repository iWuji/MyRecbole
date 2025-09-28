from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import SASRec, DIN, SRGNN, DIEN, S3Rec, FEARec, CORE, Mymodel
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
from logging import getLogger
import argparse
import pdb

if __name__ == '__main__':
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Run a RecBole sequential model with specified parameters.')
    parser.add_argument('--model', type=str, required=True, help='Name of the sequential model to run')
    args = parser.parse_args()

    # 使用配置文件
    if args.model == 'SASRec':
        config = Config(model=args.model, dataset='mydata', config_file_list=['SASRec.yaml'])
    elif args.model == 'DIN':
        config = Config(model=args.model, dataset='mydata', config_file_list=['DIN.yaml'])
    elif args.model == 'SRGNN':
        config = Config(model=args.model, dataset='mydata', config_file_list=['SRGNN.yaml'])
    elif args.model == 'DIEN':
        config = Config(model=args.model, dataset='mydata', config_file_list=['DIEN.yaml'])
    elif args.model == 'S3Rec':
        config = Config(model=args.model, dataset='mydata', config_file_list=['S3Rec.yaml'])
    elif args.model == 'FEARec':
        config = Config(model=args.model, dataset='mydata', config_file_list=['FEARec.yaml'])
    elif args.model == 'CORE':
        config = Config(model=args.model, dataset='mydata', config_file_list=['CORE.yaml'])
    elif args.model == 'Mymodel':
        config = Config(model=args.model, dataset='mydata', config_file_list=['Mymodel.yaml'])

    # 初始化随机种子
    init_seed(config['seed'], config['reproducibility'])

    # 初始化日志记录器
    init_logger(config)
    logger = getLogger()

    # 记录配置
    logger.info(config)

    # 数据集创建与过滤
    dataset = create_dataset(config)
    logger.info(dataset)

    # 数据集分割
    train_data, valid_data, test_data = data_preparation(config, dataset)
    print('_'*10)
    print(len(train_data))
    print(len(valid_data))
    print(len(test_data))
    

    # 模型加载与初始化
    if args.model == 'SASRec':
        model = SASRec(config, train_data.dataset).to(config['device'])
    elif args.model == 'DIN':
        model = DIN(config, train_data.dataset).to(config['device'])
    elif args.model == 'SRGNN':
        model = SRGNN(config, train_data.dataset).to(config['device'])
    elif args.model == 'DIEN':
        model = DIEN(config, train_data.dataset).to(config['device'])
    elif args.model == 'S3Rec':
        model = S3Rec(config, train_data.dataset).to(config['device'])
    elif args.model == 'FEARec':
        model = FEARec(config, train_data.dataset).to(config['device'])
    elif args.model == 'CORE':
        model = CORE(config, train_data.dataset).to(config['device'])
    elif args.model == 'Mymodel':
        model = Mymodel(config, train_data.dataset).to(config['device'])

    logger.info(model)

    # 训练器加载与初始化
    trainer = Trainer(config, model)
        
    # 模型训练
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    # 模型评估
    test_result = trainer.evaluate(test_data)
    logger.info('best valid result: {}'.format(best_valid_result))
    logger.info('test result: {}'.format(test_result))