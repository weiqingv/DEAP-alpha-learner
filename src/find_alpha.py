import random
import itertools
import operator
import numpy as np
import pandas
import pandas as pd
from numpy import ndarray
from deap import base
from deap import creator
from deap import tools
from deap import gp
from deap import algorithms
from user_functions import rank, delay, correlation, covariance, delta, ts_rank, ts_min, ts_max, \
    ts_argmax, ts_argmin, ts_sum, ts_prod, ts_stddev, sma, get5, get15, get30, get60
import preparation
import pickle

stock_pool, y_list = preparation.get()
data = stock_pool.transpose((2, 0, 1))  # 6x5x201

pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(ndarray, 6), ndarray)
pset.addPrimitive(np.add, [ndarray, ndarray], ndarray)
pset.addPrimitive(np.subtract, [ndarray, ndarray], ndarray)
pset.addPrimitive(np.multiply, [ndarray, ndarray], ndarray)
# pset.addPrimitive(np.true_divide, [ndarray, ndarray], ndarray)
pset.addPrimitive(np.fabs, [ndarray], ndarray)
pset.addPrimitive(rank, [ndarray], ndarray)
pset.addPrimitive(delay, [ndarray, int], ndarray)
pset.addPrimitive(correlation, [ndarray, ndarray, int], ndarray)
pset.addPrimitive(covariance, [ndarray, ndarray, int], ndarray)
pset.addPrimitive(delta, [ndarray, int], ndarray)
pset.addPrimitive(ts_min, [ndarray, int], ndarray)
pset.addPrimitive(ts_max, [ndarray, int], ndarray)
pset.addPrimitive(ts_argmin, [ndarray, int], ndarray)
pset.addPrimitive(ts_argmax, [ndarray, int], ndarray)
pset.addPrimitive(ts_rank, [ndarray, int], ndarray)
pset.addPrimitive(ts_sum, [ndarray, int], ndarray)
# pset.addPrimitive(ts_prod, [ndarray, int], ndarray)
pset.addPrimitive(ts_stddev, [ndarray, int], ndarray)
pset.addPrimitive(sma, [ndarray, int], ndarray)
pset.addPrimitive(get5, [], int)
pset.addPrimitive(get15, [], int)
pset.addPrimitive(get30, [], int)
pset.addPrimitive(get60, [], int)

periods = [5, 15, 30, 60]
for item in periods:
    pset.addTerminal(item, int)
pset.renameArguments(ARG0='open', ARG1='close', ARG2='low', ARG3='high', ARG4='volume', ARG5='vwap')

creator.create('Fitness', base.Fitness, weights=(1.0,))
creator.create('Individual', gp.PrimitiveTree, fitness=creator.Fitness)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def eval_ic(individual):
    func = toolbox.compile(expr=individual)
    ys = y_list.transpose((1, 0))  # 201x5
    res = func(*data).transpose((1, 0))  # 201x5

    ic_list = []
    for i in range(np.size(res, 0)):
        ic = pd.Series(res[i].flatten()).corr(pd.Series(ys[i].flatten()), method='spearman')
        ic_list.append(ic)
    ic_mean = pandas.Series(ic_list).fillna(0).mean()
    return ic_mean,


toolbox.register("evaluate", eval_ic)
toolbox.register("select", tools.selTournament, tournsize=20)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

random.seed(1021)
pop = toolbox.population(n=100)
hof = tools.HallOfFame(6)

stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("avg", np.mean)
mstats.register("std", np.std)
mstats.register("min", np.min)
mstats.register("max", np.max)

pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 10, stats=mstats, halloffame=hof, verbose=True)

for i in range(6):
    print(hof.items[i])

cp = dict(halloffame=hof, logbook=log)
with open("rank_100_10.pkl", "wb") as cp_file:
    pickle.dump(cp, cp_file)
