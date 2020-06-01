import os
import sys
import csv

try:
    sys.path.append(os.environ["CARLA_PYTHON"])
    from carla import Image
except:
    raise Exception('No CARLA module found.')

class collectData():
    def __init__(self, path, isPerception):
        self.path = path
        self.isPerception = isPerception
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        csv_file = os.path.join(self.path, "label.csv")
        self.f = open(csv_file, 'w')
        self.csv_writer = csv.writer(self.f)

    def __call__(self, p_list, gt_distance, velocity, brake, attacked, regression_distance=0):
        csv_line = []
        for i in range(len(p_list)):
            csv_line.append(p_list[i])
        if self.isPerception:
            csv_line.append(round(gt_distance, 4))
            csv_line.append(round(regression_distance, 4))
            csv_line.append(round(velocity, 4))
            csv_line.append(round(brake, 4))
            csv_line.append(attacked)
            self.csv_writer.writerow(csv_line)
        else:
            self.csv_writer.writerow([file_path, round(gt_distance, 4), round(velocity, 4), round(brake, 4)])
    
    def close_csv(self):
        self.f.close()