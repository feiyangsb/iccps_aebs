import os
os.environ["CARLA_SERVER"] = os.path.abspath("../CarlaUE4.sh")

import argparse
from routines.engines.server_manager import ServerManagerBinary
from routines.engines.setup_world import SetupWorld
from routines.rl_agent.ddpg_agent import ddpgAgent
from routines.rl_agent.input_preprocessor import InputPreprocessor
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Assurance Monitoring for RL-based emergency braking system.')
    parser.add_argument("-g", "--gui", help="set gui mode.", action="store_true")
    parser.add_argument("-t", "--testing", help="set testing mode", action="store_true", default=False)
    parser.add_argument("-c", "--collect", help="collect the image data")
    parser.add_argument("-p", "--perception", help="set the path of perception neural network")

    args = parser.parse_args()

    EPISODE = 2

    try:
        carla_server = ServerManagerBinary({'CARLA_SERVER': os.environ["CARLA_SERVER"]})
        carla_server.reset()
        carla_server.wait_until_ready()
        env = SetupWorld(town=1, gui=args.gui, collect=args.collect, perception=args.perception)
        agent = ddpgAgent(Testing=args.testing)
        input_preprocessor = InputPreprocessor()
        for episode in range(EPISODE):
            initial_distance = np.random.normal(100, 1) # 1: 100.3963583313768 # 2: 99.0 # 3: 101.0
            initial_speed = np.random.uniform(26,30) #1: 26.491814391729154 # 2:30.0 # 3: 27.0

            p0 = 0
            rate = 0.1
            t0 = 10
            t1 = 90
            s = env.reset(initial_distance, initial_speed, p0, rate, t0, t1)
            print("Episode {} is started, target distance: {}, target speed: {}, initial distance: {}, initial speed: {}".format(episode, initial_distance, initial_speed, s[0], s[1]))
            s = input_preprocessor(s)
            epsilon = 1.0 - (episode+1)/(EPISODE)
            time_step = 0
            while True:
                a = agent.getAction(s, epsilon)
                s_, r, done, _, _, _= env.step(a[0][0])
                s_ = input_preprocessor(s_)
                if args.testing is False:
                    agent.storeTrajectory(s, a, r, s_, done)
                    agent.learn()
                s = s_
                if done:
                    print("Episode {} is done, the reward is {}".format(episode,r))
                    break
            if args.testing is False:
                if np.mod(episode, 10) == 0:
                    agent.save_model()
        carla_server.stop()
    
    except Exception as error:
        print('Caught this error: ' + repr(error))
        carla_server.stop()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        carla_server.stop()