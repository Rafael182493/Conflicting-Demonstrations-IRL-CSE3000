import irlmaxentstuff.gridworld as W
import irlmaxentstuff.maxent as M
import irlmaxentstuff.plot as P
import irlmaxentstuff.trajectory as T
import irlmaxentstuff.solver as S
import irlmaxentstuff.optimizer as O

import numpy as np
import matplotlib.pyplot as plt
import random

from scipy.spatial.distance import euclidean

from fastdtw import fastdtw

import time
from multiprocessing import Pool


# reward_pairs form: [(index i, reward i), ...]
# terminal_states form: [index i, ...]
def setup_mdp(reward_pairs, terminal_states):
    """
    Set up our MDP/GridWorld
    """
    # create our world
    world = W.IcyGridWorld(size=7, p_slip=0.2)

    # set up the reward function
    reward = np.zeros(world.n_states)
    for i in reward_pairs:
        reward[i[0]] = i[1]

    # set up terminal states
    terminal = []
    for i in terminal_states:
        terminal.append(i)

    return world, reward, terminal


def generate_trajectories(world, reward, terminal, ntraj):
    """
    Generate some "expert" trajectories.
    """
    # parameters
    n_trajectories = ntraj
    discount = 0.7
    weighting = lambda x: x**5

    # set up initial probabilities for trajectory generation
    initial = np.zeros(world.n_states)
    #indexes = [0,1,2,3,5,6,7,8,10,11,12,13,15,16,17,18,20,21,22,23]
    #for i in range(len(indexes)):
    #    initial[indexes[i]] = 0.05
    initial[15] = 0.25
    initial[22] = 0.25
    initial[23] = 0.25
    initial[29] = 0.25

    # generate trajectories
    value = S.value_iteration(world.p_transition, reward, discount)
    policy = S.stochastic_policy_from_value(world, value, w=weighting)
    policy_exec = T.stochastic_policy_adapter(policy)
    tjs = list(T.generate_trajectories(n_trajectories, world, policy_exec, initial, terminal))

    return tjs


def maxent_causal(world, terminal, trajectories, discount=0.7):
    """
    Maximum Causal Entropy Inverse Reinforcement Learning
    """
    # set up features: we use one feature vector per state
    features = W.state_features(world)

    # choose our parameter initialization strategy:
    #   initialize parameters with constant
    init = O.Constant(1.0)

    # choose our optimization strategy:
    #   we select exponentiated gradient descent with linear learning-rate decay
    optim = O.ExpSga(lr=O.linear_decay(lr0=0.2))

    # actually do some inverse reinforcement learning
    reward = M.irl_causal(world.p_transition, features, terminal, trajectories, optim, init, discount)

    return reward

def helper2(wrap):
    ave1, ave2, ave3, goal_frequencies, path_length_per_goal = calc(wrap[0], wrap[1], wrap[2], wrap[3], wrap[4], wrap[5], wrap[6], wrap[7])
    return (ave1, ave2, ave3, goal_frequencies, path_length_per_goal)

def alt_main2():
    reward_exp1 = [(10, 0.1), (26, 0.8), (38, 0.1)]
    reward_exp2 = [(10, 0.8), (26, 0.1), (38, 0.1)]
    reward_exp3 = [(10, 0.1), (26, 0.1), (38, 0.8)]
    terminal_exp = [10, 26, 38]

    nit = 10
    rans = [[0, 0], [10, 10], [20, 20], [30, 30], [40, 40], [50, 50], [0, 20], [0, 40], [0, 60], [0, 80], [0, 100]]
    #rans = [[0,0], [10, 10], [20, 20], [30, 30], [40, 40], [50, 50], [60, 60], [70, 70], [80, 80], [90, 90], [100, 100]]
    dtwres1 = [0] * len(rans)
    dtwres2 = [0] * len(rans)
    dtwres3 = [0] * len(rans)
    minres1 = [999999] * len(rans)
    minres2 = [999999] * len(rans)
    minres3 = [999999] * len(rans)
    maxres1 = [-999999] * len(rans)
    maxres2 = [-999999] * len(rans)
    maxres3 = [-999999] * len(rans)
    goalfreqs = [dict() for i in range(len(rans))]
    path_lengths_per_goal = [dict() for i in range(len(rans))]
    for j in range(nit):
        print("Starting iteration", j)
        world, reward, terminal = setup_mdp(reward_exp1, terminal_exp)
        world2, reward2, terminal2 = setup_mdp(reward_exp2, terminal_exp)
        world3, reward3, terminal3 = setup_mdp(reward_exp3, terminal_exp)
        trajectories1 = generate_trajectories(world, reward, terminal, 200)
        trajectories2 = generate_trajectories(world2, reward2, terminal2, 200)
        trajectories3 = generate_trajectories(world3, reward3, terminal3, 200)
        argies = []
        for i in rans:
            argies.append((100, i[0], i[1], trajectories1, trajectories2, trajectories3, world, terminal))
        p = Pool(10)
        result = p.map(helper2, argies)
        print("The result of iteration", j, "is", result)
        for i in range(len(result)):
            dtwres1[i] += result[i][0]/nit
            if result[i][0] < minres1[i]:
                minres1[i] = result[i][0]
            if result[i][0] > maxres1[i]:
                maxres1[i] = result[i][0]

            dtwres2[i] += result[i][1]/nit
            if result[i][1] < minres2[i]:
                minres2[i] = result[i][1]
            if result[i][1] > maxres2[i]:
                maxres2[i] = result[i][1]

            dtwres3[i] += result[i][2] / nit
            if result[i][2] < minres3[i]:
                minres3[i] = result[i][2]
            if result[i][2] > maxres3[i]:
                maxres3[i] = result[i][2]

            newgoalfreqs = result[i][3]
            newpathlengths = result[i][4]
            for key in newgoalfreqs:
                if key in goalfreqs[i]:
                    goalfreqs[i][key] = goalfreqs[i][key] + newgoalfreqs[key]
                    path_lengths_per_goal[i][key] = path_lengths_per_goal[i][key] + newpathlengths[key]/nit
                else:
                    goalfreqs[i][key] = newgoalfreqs[key]
                    path_lengths_per_goal[i][key] = newpathlengths[key]/nit


    print("Final result compared to expert 1:", dtwres1)
    print("Final result compared to expert 2:", dtwres2)
    print("Final result compared to expert 3:", dtwres3)
    print("---")
    print("Final minres1:", minres1)
    print("Final minres2:", minres2)
    print("Final minres3:", minres3)
    print("Final maxres1:", maxres1)
    print("Final maxres2:", maxres2)
    print("Final maxres3:", maxres3)
    for i in range(len(rans)):
        print("with", rans[i][0], "trajectories of expert 2 and", rans[i][1], "trajectories of expert 3, goalfrequencies are:", goalfreqs[i], "and path lengths are", path_lengths_per_goal[i])

    with open('allresult3.txt', 'w') as f:
        print(rans, file=f)
        print(dtwres1, file=f)
        print(dtwres2, file=f)
        print(dtwres3, file=f)
        print(minres1, file=f)
        print(minres2, file=f)
        print(minres3, file=f)
        print(maxres1, file=f)
        print(maxres2, file=f)
        print(maxres3, file=f)
        print(goalfreqs, file=f)
        print(path_lengths_per_goal, file=f)


def calc(trajs1, trajs2, trajs3, trajectories1, trajectories2, trajectories3, world, terminal_exp):
    #mix trajectories from expert 1 & 2
    trajectories = trajectories1[:trajs1].copy()
    trajectories += trajectories2[:trajs2].copy()
    trajectories += trajectories3[:trajs3].copy()
    random.shuffle(trajectories)
    #print("amount of trajectories is ", len(trajectories))

    # get retrieved reward by running causal max entropy on the mixed trajectories
    reward_maxcausal = maxent_causal(world, terminal_exp, trajectories)

    # generate trajectories on the reward recovered by IRL
    world_rec, reward_rec, terminal_rec = setup_mdp([],terminal_exp)
    trajectories_comb = generate_trajectories(world_rec, reward_maxcausal, terminal_rec, 200)

    rltrajs1 = convert_traj(trajectories1)
    rltrajs2 = convert_traj(trajectories2)
    rltrajs3 = convert_traj(trajectories3)
    irltrajs = convert_traj(trajectories_comb)
    #print(irltrajs)

    # calculate average dynamic time warping distance of IRL compared to both experts
    totalavg1 = calculate_avg_dtw(irltrajs, rltrajs1)
    totalavg2 = calculate_avg_dtw(irltrajs, rltrajs2)
    totalavg3 = calculate_avg_dtw(irltrajs, rltrajs3)
    goal_frequencies, path_length_per_goal = calculate_goal_frequencies(irltrajs)

    # plt.draw()
    # plt.show()

    return totalavg1, totalavg2, totalavg3, goal_frequencies, path_length_per_goal


# calculate the average path length per goal
def calculate_path_length_per_goal(arr):
    freqs = {}
    for i in arr:
        # recover index from coordinates
        idx = i[len(i) - 1][0] * 7 + i[len(i) - 1][1]
        if idx in freqs:
            freqs[idx] = freqs[idx] + len(i)
        else:
            freqs[idx] = len(i)


# calculate frequency of visiting each goal
# also calculates average path length per goal
def calculate_goal_frequencies(arr):
    freqs = {}
    path_lengths = {}
    for i in arr:
        #recover index from coordinates
        idx = i[len(i)-1][0] * 7 + i[len(i)-1][1]
        if idx in freqs:
            freqs[idx] = freqs[idx] + 1
            path_lengths[idx] = path_lengths[idx] + len(i)
        else:
            freqs[idx] = 1
            path_lengths[idx] = len(i)

    new_path_lengths = {}
    for key in freqs:
        new_path_lengths[key] = path_lengths[key] / freqs[key]
    return freqs, new_path_lengths


# calculates average dynamic time warping distance between 2 arrays between the elements with equal starting positions
def calculate_avg_dtw(arr1, arr2):

    sum = 0
    count = 0
    for i in arr1:
        for j in arr2:
            if i[0] == j[0]:
                distance, path = fastdtw(i, j, dist=euclidean)
                sum += distance
                count += 1
    return sum / count


#converts a trajectory that contains indices into an array of coordinates
def convert_traj(traj):
    newtraj = []
    for i in traj:
        newtraj.append(convertarr(i._t))
    return newtraj


# converts an array of 1d indices into 2d coordinates on a 7x7 grid
def convertarr(arr):
    if len(arr) == 0:
        return arr
    newarr = []
    last = None
    for i in arr:
        oldi = i[0]
        newi1 = (oldi - (oldi % 7)) / 7
        newi2 = oldi - (7 * newi1)
        newi = [newi1, newi2]
        newarr.append(newi)
        last = i[2]
    last1 = (last - (last % 7)) / 7
    last2 = last - (7 * last1)
    newarr.append([last1, last2])
    return newarr


if __name__ == '__main__':
    alt_main2()
