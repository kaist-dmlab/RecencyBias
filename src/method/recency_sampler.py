import numpy as np
import math
from structure.minibatch import *
from structure.sample import *

class Quantizer(object):
    def __init__(self, num_steps, min_uncertainty, max_uncertainty):
        self.num_steps = num_steps
        self.min_uncertainty = min_uncertainty
        self.max_uncertainty = max_uncertainty
        self.step_size = self.max_uncertainty / float(num_steps)

    def quantizer_func_for_boudnary(self, uncertainty):
        return int(math.ceil(uncertainty / self.step_size))

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

        # Quantizer Module
        self.quantizer = Quantizer(self.size_of_data, self.min_uncertainty, self.max_uncertainty)
        self.quantizer_func = self.quantizer.quantizer_func_for_boudnary

        # Initialize Table: equal probability being selectedupdate_sampling_probability
        for i in range(self.size_of_data):
            self.table[i] = math.pow(self.fixed_term, 1)

    # cur_epoch start from 1
    def compute_s_e(self, cur_epoch):
        return self.s_es[0] * math.pow( math.exp((math.log(self.s_es[1] / self.s_es[0])) / (self.epochs[1]-self.epochs[0])), (cur_epoch-self.epochs[0]))

    def get_sampling_probability(self, quantization_index):
        return 1.0 / math.pow(self.fixed_term, quantization_index)

    def update_p_table(self, distances, cur_epoch, normalize=False):
        # compute_s_e
        s_e = self.compute_s_e(cur_epoch)
        print("updated s_e: ", s_e)
        # update fixed term
        self.fixed_term = math.exp(math.log(s_e) / self.size_of_data)
        for i in range(self.size_of_data):
            self.table[i] = self.get_sampling_probability(self.quantizer_func(np.fabs(distances[i])))

        if normalize:
            total_sum = np.sum(self.table)
            self.table = self.table / total_sum


class Sampler(object):
    def __init__(self, size_of_data, num_of_classes, queue_size, s_es, epochs):
        self.size_of_data = size_of_data
        self.num_of_classes = num_of_classes
        self.queue_size = queue_size
        self.prob_table = ProbTable(self.size_of_data, self.num_of_classes, s_es, epochs)

        # prediction histories of samples
        self.all_predictions = {}
        for i in range(size_of_data):
            self.all_predictions[i] = np.zeros(queue_size, dtype=int)

        self.max_certainty = -np.log(1.0 / float(self.num_of_classes))
        self.update_counters = np.zeros(size_of_data, dtype=int)

        # distances
        self.distances = np.zeros(self.size_of_data, dtype=float)

    def async_update_prediction_matrix(self, ids, softmax_matrix):
        for i in range(len(ids)):
            id = ids[i]
            predicted_label = np.argmax(softmax_matrix[i])
            # append the predicted label to the prediction matrix
            cur_index = self.update_counters[id] % self.queue_size
            self.all_predictions[id][cur_index] = predicted_label
            self.update_counters[id] += 1

    def update_all_uncertainties(self, loaded_data):
        accumulator = {}
        for i in range(self.size_of_data):
            predictions = self.all_predictions[i]
            accumulator.clear()

            for prediction in predictions:
                if prediction not in accumulator:
                    accumulator[prediction] = 1
                else:
                    accumulator[prediction] = accumulator[prediction] + 1

            p_dict = np.zeros(self.num_of_classes, dtype=float)
            for key, value in accumulator.items():
                p_dict[key] = float(value) / float(self.queue_size)

            # based on entropy
            negative_entropy = 0.0
            for j in range(len(p_dict)):
                if p_dict[j] == 0:
                    negative_entropy += 0.0
                else:
                    negative_entropy += p_dict[j] * np.log(p_dict[j])
            uncertainty = - negative_entropy / self.max_certainty

            #distance is sign(S(y,hat{y})(1.0 - uncertainty)
            if np.argmax(p_dict) == loaded_data[i].label:
                self.distances[i] = (1.0 - uncertainty)
            else:
                self.distances[i] = -(1.0 - uncertainty)

    def predictions_clear(self):
        self.all_predictions.clear()
        for i in range(self.size_of_data):
            self.all_predictions[i] = np.zeros(self.queue_size, dtype=int)

    def update_sampling_probability(self, cur_epoch, loaded_data, normalize=False):
        self.update_all_uncertainties(loaded_data)
        self.prob_table.update_p_table(self.distances, cur_epoch, normalize=normalize)

