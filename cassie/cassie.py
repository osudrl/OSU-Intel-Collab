# Consolidated Cassie environment.
try:
    from .cassiemujoco import pd_in_t, state_out_t, CassieSim, CassieVis
except ImportError:
    from cassiemujoco import pd_in_t, state_out_t, CassieSim, CassieVis

try:
    from .udp import euler2quat, quaternion_product, inverse_quaternion, quaternion2euler, rotate_by_quaternion
except ImportError:
    from udp import euler2quat, quaternion_product, inverse_quaternion, quaternion2euler, rotate_by_quaternion

from math import floor

import numpy as np
import os
import random

import pickle
import time
import copy

class CassieEnv:
    def __init__(self, **kwargs):
        self.sim = CassieSim("./cassie/cassiemujoco/cassie.xml")
        self.vis = None
        self.state_est = True

        # fundamental inputs : state est and clock
        state_est_size = 35
        clock_size     = 2  # [sin(t+p0), sin(t+p1)]

        # command inputs
        speed_size     = 2  # [x speed, y speed]

        # periodic behavior function
        ratio_size     = 2  # [swing ratio, stance ratio]

        obs_size = state_est_size + speed_size + clock_size + ratio_size

        self.ratio_size = ratio_size

        self.observation_space = np.zeros(obs_size)

        self.action_space = np.zeros(10)

        self.phase_len = 1500

        self.P = np.array([100,  100,  88,  96,  50])
        self.D = np.array([10.0, 10.0, 8.0, 9.6, 5.0])

        self.u = pd_in_t()

        self.simrate = 50

        # TODO: should probably initialize this to current state
        self.cassie_state = state_out_t()
        self.phase             = 0  # portion of the phase the robot is in

        self.offset = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])

    def step_simulation(self, action):
        target = action[:10] + self.offset

        self.u = pd_in_t()
        for i in range(5):
            # TODO: move setting gains out of the loop?
            # maybe write a wrapper for pd_in_t ?
            self.u.leftLeg.motorPd.pGain[i]  = self.P[i]
            self.u.rightLeg.motorPd.pGain[i] = self.P[i]

            self.u.leftLeg.motorPd.dGain[i]  = self.D[i]
            self.u.rightLeg.motorPd.dGain[i] = self.D[i]

            self.u.leftLeg.motorPd.torque[i]  = 0  # Feedforward torque
            self.u.rightLeg.motorPd.torque[i] = 0

            self.u.leftLeg.motorPd.pTarget[i]  = target[i]
            self.u.rightLeg.motorPd.pTarget[i] = target[i + 5]

            self.u.leftLeg.motorPd.dTarget[i]  = 0
            self.u.rightLeg.motorPd.dTarget[i] = 0

        self.cassie_state = self.sim.step_pd(self.u)

    def step(self, action):
        for _ in range(self.simrate):
            self.step_simulation(action)

        self.phase += self.phase_add
        self.time  += 1

        if self.phase >= self.phase_len:
            self.phase = self.phase % self.phase_len - 1

        done = False

        state = self.get_full_state()
        return state, 1, False, {}

    def rotate_to_orient(self, vec):
        quaternion  = euler2quat(z=self.orient_add, y=0, x=0)
        iquaternion = inverse_quaternion(quaternion)

        if len(vec) == 3:
            return rotate_by_quaternion(vec, iquaternion)

        elif len(vec) == 4:
            new_orient = quaternion_product(iquaternion, vec)
            if new_orient[0] < 0:
                new_orient = -new_orient
            return new_orient

    def reset(self):
        self.phase = random.randint(0, self.phase_len)
        self.time = 0

        self.sim.set_const()
        self.cassie_state = self.sim.step_pd(self.u)

        # Command Input: speed and side_speed (you can change this to get the policy to move forwards, backwards, sideways)
        self.speed       = np.random.uniform(-0.2, 2.5)
        self.side_speed  = np.random.uniform(-.3, .3)
        
        # Command Input: you can change this to get policies to speed up their stepping frequency
        self.phase_add   = int(self.simrate * np.random.uniform(0.9, 1.25))

        # Command Input: swing-stance ratio (you can change this to go from walking to running)
        r = np.random.uniform(0.4, 0.6)
        self.ratio = [r, 1-r]

        # Command Input: period shifts (you can change this to go from walking to hopping)
        self.period_shift = random.choice(([0, 0], [0, 0.5], [0.5, 0]))

        # Command Input: orient_add (you can change this to get the policy to turn to face a new heading)
        self.orient_add = 0

        return self.get_full_state()

    def get_clock(self):
        return  [np.sin(2 * np.pi *  (self.phase / self.phase_len + self.period_shift[0])),
                 np.sin(2 * np.pi *  (self.phase / self.phase_len + self.period_shift[1]))]

    def get_full_state(self):
        clock = self.get_clock()

        ext_state = np.concatenate(([*self.ratio, self.speed, self.side_speed], clock))

        motor_pos = self.cassie_state.motor.position[:]
        joint_pos = self.cassie_state.joint.position[:]

        motor_vel = self.cassie_state.motor.velocity[:]
        joint_vel = self.cassie_state.joint.velocity[:]

        # remove double-counted joint/motor positions
        joint_pos = np.concatenate([joint_pos[:2], joint_pos[3:5]])
        joint_vel = np.concatenate([joint_vel[:2], joint_vel[3:5]])

        robot_state = np.concatenate([
            self.rotate_to_orient(self.cassie_state.pelvis.orientation), # pelvis orientation
            self.cassie_state.pelvis.rotationalVelocity[:],              # pelvis rotational velocity 
            motor_pos,                                                   # actuated joint positions
            motor_vel,                                                   # actuated joint velocities
            joint_pos,                                                   # unactuated joint positions
            joint_vel                                                    # unactuated joint velocities
        ])

        return np.concatenate([robot_state, ext_state])

    def render(self):
        if self.vis is None:
            self.vis = CassieVis(self.sim, "./cassie/cassiemujoco/cassie.xml")

        return self.vis.draw(self.sim)
