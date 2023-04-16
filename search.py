import os
import torch
from engine.searcher import Searcher
from configs.search_config import obtain_search_args
from model.supernet import SuperNet

os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"
# 参数设置
args = obtain_search_args()
population_size = 10
individual_params = {'layers': args.layers, 'depth': args.depth, 'target_layers': args.layers}
crossover_rate = 0.8
mutation_rate = 0.05
generations = 100

# 初始化模型
model = SuperNet(args.layers, args.depth, args.input_channel, args.num_classes, args.stem_multiplier, args.base_multiplier)
if torch.cuda.is_available():
    device = torch.device("cuda:0,1")
else:
    device = torch.device("cpu")
    
# 创建 Searcher 实例
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
searcher = Searcher(population_size, individual_params, crossover_rate, mutation_rate, generations, model, device, args, loss_fn=loss_fn)

# 运行遗传算法
best_individual = searcher.genetic_algorithm_iterator()
print("最优个体:", best_individual)
print("适应度:", searcher.fitness_function(best_individual))