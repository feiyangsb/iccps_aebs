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

    def __call__(self, image, gt_distance, velocity, brake, timestamp, regression_distance=0):
        file_path = os.path.join(self.path, str(timestamp))
        image.save_to_disk(file_path)
        if self.isPerception:
            self.csv_writer.writerow([file_path, round(gt_distance, 4), round(regression_distance, 4), round(velocity, 4), round(brake, 4)])
        else:
            self.csv_writer.writerow([file_path, round(gt_distance, 4), round(velocity, 4), round(brake, 4)])
    
    def close_csv(self):
        self.f.close()