from multiprocessing import Pool

class ThresholdTask:
    def __init__(self, f1_best, th_best, num_worker=16):
        self.f1_best = 0
        self.th_best = 0

        self.pool = Pool(num_worker)

    def get_best_threshold(self, pr_values):
        f1_best, th_best = -1, 0
        for precision, recall, threshold in zip(*pr_values):
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 1
            if f1 > f1_best and f1 != 1:
                f1_best = f1
                th_best = threshold

        return f1_best, th_best

    def get