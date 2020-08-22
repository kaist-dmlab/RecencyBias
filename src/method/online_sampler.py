import numpy as np
import math, operator
from structure.minibatch import *
from structure.sample import *

class ProbTable(object):
    def __init__(self, size_of_data, num_of_classes, s_es, epochs):

        # Local Variables
        self.size_of_data = size_of_data
        self.num_of_classes = num_of_classes
        self.max_uncertainty= 1.0
        self.min_uncertainty= 0.0
        self.epochs = epochs
        self.s_es = s_es
        self.fixed_term = math.exp(math.log(s_es[0]) / self.size_of_data)
        self.table = np.ones(self.size_of_data, dtype=float)


        # Initialize Table: equal probability being selected
        for i in range(self.size_of_data):
            self.table[i] = math.pow(self.fixed_term, 1)

    # cur_epoch start from 1
    def compute_s_e(self, cur_epoch):
        return self.s_es[0] * math.pow( math.exp((math.log(self.s_es[1] / self.s_es[0])) / (self.epochs[1]-self.epochs[0])), (cur_epoch-self.epochs[0]))

    def get_sampling_probability(self, quantization_index):
        return 1.0 / math.pow(self.fixed_term, quantization_index)

    def update_p_table(self, sorted_map, cur_epoch):

        # compute_s_e
        s_e = self.compute_s_e(cur_epoch)
        print("updated s_e: ", s_e)
        # update fixed term
        self.fixed_term = math.exp(math.log(s_e) / self.size_of_data)

        cur_order = 1
        for key in sorted_map.keys():
            self.table[key] = self.get_sampling_probability(cur_order)
            cur_order += 1

class Sampler(object):
    def __init__(self, size_of_data, num_of_classes, s_es, epochs):
        self.size_of_data = size_of_data
        self.num_of_classes = num_of_classes
        self.prob_table = ProbTable(self.size_of_data, self.num_of_classes, s_es, epochs)

        # prediction histories of samples
        self.all_losses = np.zeros(self.size_of_data, dtype=float)

    def async_update_prediction_matrix(self, ids, xentropy):
        for i in range(len(ids)):
            id = ids[i]
            self.all_losses[id] = xentropy[i]



    def update_sampling_probability(self, cur_epoch):

        sorted_map = {}
        for i in range(self.size_of_data):
            sorted_map[i] = self.all_losses[i]
        sorted_map = dict(sorted(sorted_map.items(), key=operator.itemgetter(1), reverse=True))

        self.prob_table.update_p_table(sorted_map, cur_epoch)

