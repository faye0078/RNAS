import random
import torch
import time
from engine.baseEngine import BaseEngine
from model.encoder import modelEncoder
from dataloaders import make_search_data_loader
class Searcher(BaseEngine):
    def __init__(self, population_size, individual_params, crossover_rate, mutation_rate, generations, model, device, args, optimizer=None, loss_fn=None):
        model = model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)
        model.load_state_dict(torch.load('./pretrain_model_60ep.pth'))
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        super(Searcher, self).__init__(model, optimizer, loss_fn, device)
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.generations = generations
        
        self.fitness_values = None
        self.model_encoder = modelEncoder(individual_params['layers'], individual_params['depth'], target_layers=individual_params['target_layers'], is_fixed=True)
    
        self.proxy_train_loader, self.val_loader, _ = make_search_data_loader(args, image_size=512)
    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            model_encode = self.model_encoder.generate()
            image_size = random.randint(256, 900)
            
            chromosome = {
                "model_encode": model_encode,
                "image_size": image_size,
            }
            population.append(chromosome)
        self.update_fitness_values(population, generation=0)
        return population

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            child1 = {}
            child2 = {}
            model_encode1, model_encode2 = self.model_encoder.crossover(parent1["model_encode"], parent2["model_encode"])
            image_size1 = random.randint(min(parent1["image_size"], parent2["image_size"]), max(parent1["image_size"], parent2["image_size"]))
            image_size2 = random.randint(min(parent1["image_size"], parent2["image_size"]), max(parent1["image_size"], parent2["image_size"]))
            
            child1["model_encode"] = model_encode1
            child1["image_size"] = image_size1
            child2["model_encode"] = model_encode2
            child2["image_size"] = image_size2
            return child1, child2
        return parent1, parent2

    def mutation(self, chromosome):
        mutated_chromosome = {}
        mutated_chromosome["model_encode"] = self.model_encoder.mutation(chromosome["model_encode"], self.mutation_rate)
        mutated_chromosome["image_size"] = chromosome["image_size"] + random.randint(-64, 64)
        return mutated_chromosome

    def fitness_function(self, chromosome, generation):
        if torch.cuda.device_count() > 1:
            self.model.module.update_active_encode(chromosome["model_encode"])
            self.model.module.load_state_dict(torch.load('./pretrain_model_60ep.pth'))
        else:
            self.model.update_active_encode(chromosome["model_encode"])
            self.model.load_state_dict(torch.load('./pretrain_model_60ep.pth'))
        
        self.proxy_train_loader.dataset.update_image_size(chromosome["image_size"])
        self.val_loader.dataset.update_image_size(chromosome["image_size"])
        # self.train(self.proxy_train_loader, 5)
        fitness = self.test(self.val_loader, generation)
        return fitness
    
    def update_fitness_values(self, population, generation=0):
        self.fitness_values = [self.fitness_function(chromosome, generation) for chromosome in population]

    def selection(self, population):
        total_fitness = sum(self.fitness_values)
        probabilities = [fitness / total_fitness for fitness in self.fitness_values]
        selected_index = random.choices(range(len(population)), probabilities)[0]
        return population[selected_index]

    def genetic_algorithm_iterator(self):
        population = self.initialize_population()
        
        start_time = time.time()
        for generation in range(self.generations):
            generation_start_time = time.time()
            print("Generation: ", generation)
            self.save_population_fitness(population, self.fitness_values, generation, "./population_{}.txt".format(generation))
            children = []
            for _ in range(self.population_size // 2):
                parent1 = self.selection(population)
                parent2 = self.selection(population)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                children.extend([child1, child2])
                
            children_fitness = [self.fitness_function(child, generation+1) for child in children]
            population.extend(children)
            self.fitness_values.extend(children_fitness)
            
            sorted_indices = sorted(range(len(self.fitness_values)), key=lambda i: self.fitness_values[i], reverse=True)
            population = [population[i] for i in sorted_indices[:self.population_size]]
            self.fitness_values = [self.fitness_values[i] for i in sorted_indices[:self.population_size]]
             # 输出计时信息
            generation_end_time = time.time()
            generation_duration = generation_end_time - generation_start_time
            elapsed_time = generation_end_time - start_time
            remaining_time = (self.generations - generation - 1) * generation_duration

            print(f"Generation duration: {generation_duration:.2f} seconds")
            print(f"Elapsed time: {elapsed_time:.2f} seconds")
            print(f"Estimated remaining time: {remaining_time:.2f} seconds")
            print("===" * 10)
            
            
        best_individual = population[0]
        return best_individual
    
    def save_population_fitness(self, population, fitness, generation, save_path):
        with open(save_path, 'a+') as f:
            f.write("Generation: " + str(generation) + "\n")
            for i in range(len(population)):
                f.write(str(population[i]) + " " + str(fitness[i]) + "\n")
            f.write("\n")