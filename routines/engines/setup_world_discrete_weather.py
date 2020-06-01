#!/usr/bin/python3
import sys
import logging
import os
import queue
import random
from routines.engines.viewer import pygameViewer
from routines.engines.distance_calculation_pytorch import DistanceCalculation
from routines.engines.pid import PID
from routines.engines.collect_data import collectData
from routines.engines.get_weather_presents import findWeatherPresets
import time
import math
import numpy as np
import cv2


try:
    sys.path.append(os.environ["CARLA_PYTHON"])
    import carla
    from carla import Transform, Location, Rotation
except:
    raise Exception('No CARLA module found.')

class SetupWorld():
    def __init__(self, host='127.0.0.1', weather=None, port=2000, town=1, gui=False, collect=None, perception=None):
        self.episode = 0
        self.collect_path = collect
        self.perception = perception
        self.gui = gui
        # try to import carla python API
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world('Town0{}'.format(town))
        self.weather = weather
        self.episode_count = 0
        self.random_weather_list = [0,1,2,3,4]
        """
        if not hasattr(carla.WeatherParameters, weather):
            print('ERROR: weather preset %r not found.' % weather)
        else:
            self.world.set_weather(getattr(carla.WeatherParameters, weather))
        """
        #self.world = self.client.get_world()
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 0.05
        self.frame = None
        settings.synchronous_mode = True
        self.world.apply_settings(settings)
        self.actor = []
        self.display = None
        #self.spawn_vehicles()
    
    def reset(self, initial_distance, initial_speed):
        if self.gui and self.display:
            self.display.stop()

        if self.weather == None:
            weather_present = findWeatherPresets()[self.random_weather_list[0]][1]
        else:
            weather_present = self.weather
        if not hasattr(carla.WeatherParameters, weather_present):
            print('ERROR: weather preset %r not found.' % weather_present)
        else:
            self.world.set_weather(getattr(carla.WeatherParameters, weather_present))

        # Clean all vehicles before reset the environment
        for i, _ in enumerate(self.actor):
            if self.actor[i] is not None:
                self.actor[i].destroy()
                self.actor[i] = None
        self.actor = []

        self.spawn_vehicles()
        self.dist_calc = DistanceCalculation(self.ego_vehicle, self.leading_vehicle, self.perception)
        self.pid_controller = PID(P=3.0, I=0.0003, D=0.0)

        if self.gui:
            self.display = pygameViewer()
        
        if self.collect_path is not None:
            self.collect_data = collectData(os.path.join(self.collect_path, str(self.episode)), self.episode)
        
        self.step_count = 0
        self.episode+=1
        S0 = self.reset_episode(initial_distance, initial_speed)
        self.episode_count += 1
        return S0
    
    def reset_episode(self, initial_distance, initial_speed):
        velocity = 0.0
        while True:
            action = self.pid_controller.update(initial_speed, velocity)
            control = carla.VehicleControl(
                        throttle = action + 0.4,
                        brake = 0.0,
                        steer = 0.0
            )
            self.ego_vehicle.apply_control(control)
            self.world.tick()
            image = self.image_queue.get()
            if not self.collision_queue.empty():
                _ = self.collision_queue.get()
                print("collision occurred during reset...")
            distance = self.dist_calc.getTrueDistance()
            velocity = self.ego_vehicle.get_velocity().y
            """
            if self.collect_path is not None and distance<110.0:
                self.collect_data(image, distance, velocity, -1, self.step_count)
                self.step_count += 1
            """
            if self.gui:
                self.display.updateViewer(image)
            if (distance<=initial_distance):
                break
        #print("The car reset to initial distance: {} and initial speed: {}".format(distance, velocity))
        return [distance, velocity]


    def spawn_vehicles(self):
        self.bp_lib = self.world.get_blueprint_library()
        ego_bp = self.bp_lib.filter('vehicle.tesla.model3')[0]
        spawn_point = Transform(Location(x=392.1, y=10.0, z=0.02), Rotation(yaw=89.6))
        self.ego_vehicle = self.world.spawn_actor(ego_bp, spawn_point)
        self.setup_sensors(self.ego_vehicle)
        self.actor.append(self.ego_vehicle)
        self.ego_vehicle.get_world()
        #print(self.bp_lib.filter('vehicle.audi.*'))
        leading_bp = self.bp_lib.filter('vehicle.audi.a2')[0]
        #leading_bp = self.bp_lib.filter('vehicle.audi.etron')[0]
        #leading_bp = self.bp_lib.filter('vehicle.audi.tt')[0]
        #leading_bp = random.choice(self.bp_lib.filter('vehicle.audi.*'))
        spawn_point_leading = Transform(Location(x=392.1, y=320.0, z=0.0), Rotation(yaw=90))
        self.leading_vehicle = self.world.try_spawn_actor(leading_bp, spawn_point_leading)
        self.leading_vehicle.get_world()
        self.actor.append(self.leading_vehicle)
    
    def setup_sensors(self, ego_vehicle):
        camera_bp = self.bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '100')
        self.camera = self.world.try_spawn_actor(
                        camera_bp,
                        Transform(Location(x=0.8,y=0.0,z=1.7), Rotation(yaw=0.0)), 
                        attach_to=ego_vehicle)
        self.actor.append(self.camera)
        collision_bp = self.bp_lib.find('sensor.other.collision')
        self.collision = self.world.try_spawn_actor(
                        collision_bp,
                        Transform(),
                        attach_to=self.ego_vehicle)
        self.actor.append(self.collision)
        self.image_queue = queue.Queue()
        self.camera.listen(self.image_queue.put)
        self.collision_queue = queue.Queue()
        self.collision.listen(self.collision_queue.put)
    
    def step(self, action):
        control=carla.VehicleControl(
            throttle = 0.0,
            brake = action
        )
        self.ego_vehicle.apply_control(control)
        self.world.tick()
        image = self.image_queue.get()
        if not self.collision_queue.empty():
            collision = self.collision_queue.get().normal_impulse
            isCollision =True
        else:
            isCollision = False

        groundtruth_distance = self.dist_calc.getTrueDistance()
        distance = groundtruth_distance
        if groundtruth_distance < 110.0 and self.perception is not None:
            regression_distance = self.dist_calc.getRegressionDistance(image)
            attack_distance = self.dist_calc.getAttackDistance(image)
            #distance = regression_distance
            if self.step_count > 20:
                distance = attack_distance
            print("Groundtruth distance: {}, Regression distance: {}, Attack Distance: {}".format(groundtruth_distance, regression_distance, attack_distance))
        velocity = self.ego_vehicle.get_velocity().y
        if self.collect_path is not None:
            self.collect_data(image, groundtruth_distance, velocity, action, self.step_count)
        """
        if self.step_count==50:
            self.world.set_weather(getattr(carla.WeatherParameters, 'HardRainSunset'))
        """
        self.step_count += 1

        if self.gui:
            self.display.updateViewer(image)

        isStop = velocity <= 0.0
        done = isStop or isCollision

        self.image = image
            
        if done:
            if (isCollision):
                #reward = -math.sqrt(collision.x**2+collision.y**2+collision.z**2)/100.0
                reward = -200.0 - math.sqrt(collision.x**2+collision.y**2+collision.z**2)/100.0
                print("Collision: {}".format(reward))
            elif (isStop):
                #too_far_reward = -(distance>5.0)*(distance-5)#-((distance-5)/250.0*400 + 20) * (distance>5.0)
                too_far_reward = -((groundtruth_distance-3.0)/250.0*400+20) * (groundtruth_distance>3.0) 
                #too_close_reward = 0.0#-(20.0)*(distance<1.0)
                too_close_reward = -(20.0)*(groundtruth_distance<1.0)
                reward = too_far_reward + too_close_reward
                print("Stop: {}, Distance: {}".format(reward, groundtruth_distance))
        else:
            reward = 0

        return [[distance, velocity], reward, done] 

    def get_image(self):
        img = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
        img = np.reshape(img, (self.image.height, self.image.width, 4))
        img = img[:, :, :3]
        img = cv2.resize(img, (224,224))
        img = np.asarray(img) / 255.
        return img
