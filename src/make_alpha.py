import itertools
import math
import random
import operator
import numpy as np
import pandas as pd
from numpy import ndarray
from deap import base
from deap import creator
from deap import tools
from deap import gp
from deap import algorithms
from user_functions import rank, delay, correlation, covariance, delta, ts_rank, ts_min, ts_max, \
    ts_argmax, ts_argmin, ts_sum, ts_prod, ts_stddev, sma, get5, get15, get30, get60

fields = ['open', 'high', 'low', 'avg', 'pre_close', 'high_limit', 'low_limit', 'close']
stock_price = pd.read_csv('./test.csv')
source = stock_price[fields].values
target = stock_price['pct'].values
periods = [5, 15, 30, 60]

test_size = 0.2
test_num = int(len(source) * test_size)
x_train = source[:-test_num]
x_test = source[-test_num:]
y_train = np.nan_to_num(target[:-test_num])
y_test = np.nan_to_num(target[-test_num:])

pset = gp.PrimitiveSetTyped("main", itertools.repeat(ndarray, 8), ndarray)
pset.addPrimitive(np.add, [ndarray, ndarray], ndarray)
pset.addPrimitive(np.subtract, [ndarray, ndarray], ndarray)
pset.addPrimitive(np.multiply, [ndarray, ndarray], ndarray)
pset.addPrimitive(np.true_divide, [ndarray, ndarray], ndarray)
pset.addPrimitive(np.fabs, [ndarray], ndarray)
pset.addPrimitive(rank, [ndarray], ndarray)
pset.addPrimitive(delay, [ndarray, int], ndarray)
# pset.addPrimitive(delay, [ndarray], ndarray)
pset.addPrimitive(correlation, [ndarray, ndarray, int], ndarray)
pset.addPrimitive(covariance, [ndarray, ndarray, int], ndarray)
# pset.addPrimitive(correlation, [ndarray, ndarray], ndarray)
# pset.addPrimitive(covariance, [ndarray, ndarray], ndarray)
pset.addPrimitive(delta, [ndarray, int], ndarray)
# pset.addPrimitive(delta, [ndarray], ndarray)
pset.addPrimitive(ts_min, [ndarray, int], ndarray)
pset.addPrimitive(ts_max, [ndarray, int], ndarray)
pset.addPrimitive(ts_argmin, [ndarray, int], ndarray)
pset.addPrimitive(ts_argmax, [ndarray, int], ndarray)
pset.addPrimitive(ts_rank, [ndarray, int], ndarray)
pset.addPrimitive(ts_sum, [ndarray, int], ndarray)
pset.addPrimitive(ts_prod, [ndarray, int], ndarray)
pset.addPrimitive(ts_stddev, [ndarray, int], ndarray)
pset.addPrimitive(sma, [ndarray, int], ndarray)
pset.addPrimitive(get5, [], int)
pset.addPrimitive(get15, [], int)
pset.addPrimitive(get30, [], int)
pset.addPrimitive(get60, [], int)

for item in periods:
    pset.addTerminal(item, int)
# pset.addEphemeralConstant('rand100', lambda: random.sample(periods, 1), int)
pset.renameArguments(ARG0='open', ARG1='high', ARG2='low', ARG3='avg', ARG4='pre_close', ARG5='high_limit',
                     ARG6='low_limit', ARG7='close')

creator.create('Fitness', base.Fitness, weights=(1.0,))
creator.create('Individual', gp.PrimitiveTree, fitness=creator.Fitness)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def eval_rank_ic(individual, data):
    func = toolbox.compile(expr=individual)
    # print(tree)
    y = target[1:]
    data = data.transpose(1, 0)
    ic = pd.Series(func(*data)).corr(pd.Series(y), method='pearson')
    return ic,


toolbox.register("evaluate", eval_rank_ic, data=source[:-1])
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

random.seed(21)
pop = toolbox.population(n=300)
hof = tools.HallOfFame(3)

stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("avg", np.mean)
mstats.register("std", np.std)
mstats.register("min", np.min)
mstats.register("max", np.max)

pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, stats=mstats,
                               halloffame=hof, verbose=True)

print(hof.items[0])
print(hof.items[1])
print(hof.items[2])

# if __name__ == '__main__':
#     data = source[:-1]
#     tar = target[1:]
#     print(data)
