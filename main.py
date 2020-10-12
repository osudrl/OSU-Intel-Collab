import torch
import sys
import time
import pickle
import numpy as np

from cassie.cassie import CassieEnv

if __name__ == '__main__':
    with torch.no_grad():
        run_args = pickle.load(open('generic_policy/experiment.pkl', 'rb'))

        policy = torch.load('generic_policy/actor.pt')
        policy.eval()

        env = CassieEnv()

        while True:
            policy.init_hidden_state()
            state = env.reset()
            done = False

            env.phase = 0
            for t in range(300):
                # set command inputs here
                #env.speed = 0
                #env.ratio = [0, 1]
                #env.ratio = [0.5, 0.5]
                #env.period_shift = [0, 0]

                start_time = time.time()
                action = policy(state)
                state, _, _, _ = env.step(action)
                env.render()

                while time.time() - start_time < env.simrate / 2000:
                    time.sleep(0.001)
