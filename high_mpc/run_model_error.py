import os
import time
import datetime
import argparse
import numpy as np
from functools import partial
#
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# A Gym style environment
from high_mpc.simulation.dynamic_gap import DynamicGap
from high_mpc.mpc.mpc import MPC
from high_mpc.mpc.high_mpc import High_MPC
from high_mpc.simulation.animation import SimVisual2
from high_mpc.common import logger
from high_mpc.common import util as U
from high_mpc.policy import high_policy
from high_mpc.policy import deep_high_policy
#

def run_mpc(env):
    tracking_error = []
    goal_error = []    
    #
    for i in range(1000):    
        env.reset()
        t, n = 0, 0
        t0 = time.time()
        diff_min = 1e+10
        x_diff_min = np.zeros(3)
        goal = np.array([4.0, 0.0, 2.0])
        while t < env.sim_T:
            t = env.sim_dt * n
            _, _, _, info = env.step()
            t_now = time.time()
            #print(t_now - t0)
            #
            t0 = time.time()
            #
            n += 1
            update = False
            if t>= env.sim_T:
                update = True
            state_q = info["quad_obs"]
            state_p = info["pend_obs"]
            x_diff = np.abs(state_q[0:3] - state_p[0:3])
            if x_diff[0] < diff_min:
                diff_min = x_diff[0]
                x_diff_min = x_diff
            x_last = state_q[0:3]
            #yield [info, t, update]
        tracking_error.append(np.linalg.norm(x_diff_min))
        goal_error.append(np.linalg.norm(x_last - goal))
    error = np.vstack((np.array(tracking_error),np.array(goal_error)))
    np.save("./mpc_error.npy",error)

def run_hmpc(env,mu):
    #
    tracking_error = []
    goal_error = []
    for i in range(1000):
        env.reset()
        t, n = 0, 0
        t0 = time.time()
        diff_min = 1e+10
        x_diff_min = np.zeros(3)
        goal = np.array([4.0, 0.0, 2.0])    
        while t < env.sim_T:
            t = env.sim_dt * n
            _, _, _, info = env.step(mu)
            t_now = time.time()
            #print(t_now - t0)
            #
            t0 = time.time()
            #
            n += 1
            update = False
            if t>= env.sim_T:
                update = True
            state_q = info["quad_obs"]
            state_p = info["pend_obs"]
            x_diff = np.abs(state_q[0:3] - state_p[0:3])
            if x_diff[0] < diff_min:
                diff_min = x_diff[0]
                x_diff_min = x_diff
            x_last = state_q[0:3]
            #yield [info, t, update]
        tracking_error.append(np.linalg.norm(x_diff_min))
        goal_error.append(np.linalg.norm(x_last - goal))
    error = np.vstack((np.array(tracking_error),np.array(goal_error)))
    np.save("./hmpc_error.npy",error)
    # print("minimum x diff:",diff_min)
    # print("tracking_error:",tracking_error)
    # print("goal_error:",goal_error)
    # print("error:",error)

def run_deep_high_mpc(env, actor_params, load_dir):
    tracking_error = []
    goal_error = []
    for i in range(1000):
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        #
        actor = deep_high_policy.Actor(obs_dim, act_dim)
        actor.load_weights(load_dir)
        #
        ep_len, ep_reward =  0, 0
        obs = env.reset()
        t = 0
        diff_min = 1e+10
        x_diff_min = np.zeros(3)
        goal = np.array([4.0, 0.0, 2.0]) 
        while t < env.sim_T:
            t += env.sim_dt
            #
            obs_tmp = np.reshape(obs, (1, -1)) # to please tensorflow
            act = actor(obs_tmp).numpy()[0]

            # execute action
            next_obs, reward, _, info = env.step(act)

            #
            obs = next_obs
            ep_reward += reward

            #
            ep_len += 1

            #
            update = False
            if t >= env.sim_T:
                update = True
            state_q = info["quad_obs"]
            state_p = info["pend_obs"]
            x_diff = np.abs(state_q[0:3] - state_p[0:3])
            if x_diff[0] < diff_min:
                diff_min = x_diff[0]
                x_diff_min = x_diff
            x_last = state_q[0:3]
            #yield [info, t, update]
        tracking_error.append(np.linalg.norm(x_diff_min))
        goal_error.append(np.linalg.norm(x_last - goal))
    error = np.vstack((np.array(tracking_error),np.array(goal_error)))
    np.save("./dmpc_error.npy",error)
    #error = tracking_error + goal_error
    #print("minimum x diff:",diff_min)
    #print("tracking_error:",tracking_error)
    #print("goal_error:",goal_error)
    #print("error:",error)

def main():
    
    plan_T = 2.0
    plan_dt = 0.04

    so_path = "./mpc/saved/high_mpc.so"
    high_mpc = High_MPC(T=plan_T, dt=plan_dt, so_path=so_path)
    env = DynamicGap(high_mpc, plan_T, plan_dt)
    
    #U.set_global_seed(2347)
    
    wml_params = dict(
        sigma0=100,
        max_iter=20,
        n_samples=20,
        beta0=3.0,
    )

    save_dir = U.get_dir(os.path.dirname(os.path.realpath(__file__)) + "/saved_policy")
    save_dir = os.path.join(save_dir, datetime.datetime.now().strftime("highmpc-%m-%d-%H-%M-%S"))

    #
    logger.configure(dir=save_dir)
    logger.log("***********************Log & Store Hyper-parameters***********************")
    logger.log("weighted maximum likelihood params")
    logger.log(wml_params)
    logger.log("***************************************************************************")
    #mu = high_policy.run_wml(env=env, logger=logger, save_dir=save_dir, **wml_params)
    mu = 1.35
    run_hmpc(env,mu)
    #run_frame = partial(run_hmpc_wi, env, logger, save_dir, wml_params)
    #ani = animation.FuncAnimation(sim_visual.fig, sim_visual.update, frames=run_frame,
    #        init_func=sim_visual.init_animate, interval=100, blit=True, repeat=False)
    
    #high_mpc = High_MPC(T=plan_T, dt=plan_dt, so_path=so_path)
    #env = DynamicGap(high_mpc, plan_T, plan_dt)
    actor_params = dict(
        hidden_units=[32, 32],
        learning_rate=1e-4,
        activation='relu',
        train_epoch=1000,
        batch_size=128
    )
    load_dir = os.path.dirname(os.path.realpath(__file__)) + "/Dataset/act_net/weights_999.h5"
    run_deep_high_mpc(env, actor_params, load_dir)    

    plan_T = 2.0
    plan_dt = 0.1
    so_path = "./mpc/saved/mpc_v1.so"
    mpc = MPC(T=plan_T, dt=plan_dt, so_path=so_path)
    env = DynamicGap(mpc, plan_T, plan_dt)
    run_mpc(env)

if __name__ == "__main__":
    main()