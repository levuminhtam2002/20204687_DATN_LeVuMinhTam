import numpy as np
import math
import copy
import matplotlib.pyplot as plt
from UAV_env import UAVEnv
chromo = 50
nmbpopu = 64
populationLimit = 128
mutation_rate = 0.3
mutation_rate_1 = 0.3
mutation_rate_2 = 0.3
generations = 400
e = 1e-3
bestState = None
bestReward = []


class Indv(object):
    def __init__(self):
        self.genes = []
        self._ep_reward = 0
        self._ep_reward_list = []
    
    def _add_state(self):
        env = UAVEnv()
        for i in range(chromo):
            action = np.random.uniform(-1, 1, size=(env.M,))
            env.act = (action + 1) / 2
            _env = copy.deepcopy(env)
            s_, r, is_terminal, step_redo, offloading_ratio_change, reset_dist = env.step(action)
            _env.sum_task_size = env.sum_task_size
            _env.e_battery_uav = env.e_battery_uav
            self._ep_reward += r
            self._ep_reward_list = np.append(self._ep_reward_list, r)
            self.genes = np.append(self.genes, _env)
            if not hasattr(_env, 'steps'):
                _env.steps = []
            _env.steps.append({
                'uav_pos': _env.loc_uav,
                'action': env.act,
                'reward': r,
                'next_state': s_,
                'ue_pos': _env.loc_ue_list.copy()  # Store UE positions
            })
            if is_terminal or step_redo:
                break

def print_best_gene(bestState):
    print("Best gene:")
    for i, gene in enumerate(bestState.genes):
        ue_id = int(gene.M * gene.act[0])
        offloading_ratio = gene.act[3]
        print(f"Step {i}:")
        print(f"  UAV Position: {gene.loc_uav}")
        print(f"  Task Size: {gene.sum_task_size}")
        print(f"  Battery: {gene.e_battery_uav}")
        print(f"  UE Locations: {gene.loc_ue_list}")
        print(f"  UE Task Sizes: {gene.task_list}")
        print(f"  UE Block Flags: {gene.block_flag_list}")
        print(f"  Actions: {gene.act}")
        print(f"  Offloading Ratio: {offloading_ratio}")
        print(f"  UE Served: {ue_id}")

def GA():
    global bestReward
    population = []
    for i in range(nmbpopu):
        idv = Indv()
        idv._add_state()
        population = np.append(population, idv)
    
    for i in range(generations):
        while len(population) < populationLimit:
            i1 = np.random.randint(len(population))
            i2 = (i1 + 1 + np.random.randint(len(population) - 1)) % len(population)
            parent1 = population[i1]
            parent2 = population[i2]
            if np.random.rand() < 0.5:
                child1, child2 = crossOver(parent1, parent2)
                if np.random.rand() < mutation_rate:
                    mutation(child1)
                    mutation(child2)
                if np.random.rand() < mutation_rate_1:
                    mutation1(child1)
                    mutation1(child2)
                if np.random.rand() < mutation_rate_2:
                    mutation2(child1)
                    mutation2(child2)
                population = np.append(population, child1)
                population = np.append(population, child2)
        
        sorted_indices = np.argsort([-idv._ep_reward for idv in population])
        population = population[sorted_indices]

        population = population[:64]
        bestState = population[0]
        bestReward = np.append(bestReward, bestState._ep_reward)
        print('Generation:', i, ' Reward: %7.2f' % bestState._ep_reward, 'mutation_rate: %.3f' % mutation_rate)
        # print_best_gene(bestState)

def crossOver(a, b):
    x = len(a.genes) if len(a.genes) < len(b.genes) else len(b.genes)
    i1 = np.random.randint(x)
    i2 = (i1 + 1 + np.random.randint(x - 1)) % x
    n1 = copy.deepcopy(a)
    n2 = copy.deepcopy(b)
    if i1 < i2:
        i = i1
        j = i2
    else:
        i = i2
        j = i1
    while i != j:
        n1.genes[i] = b.genes[i]
        n2.genes[i] = a.genes[i]
        i = i + 1
    update(n1)
    update(n2)
    return n1, n2

def update(self):
    env_ = UAVEnv()
    self._ep_reward = 0
    self._ep_reward_list = []
    for i in range(len(self.genes) - 1):
        env_ = copy.deepcopy(self.genes[i])
        if i == 0:
            env_.e_battery_uav = 50000
            env_.sum_task_size = 100 * 1048576
        else:
            env_.e_battery_uav = self.genes[i-1].e_battery_uav
            env_.sum_task_size = self.genes[i-1].sum_task_size

        dx_uav = env_.flight_speed * env_.t_fly * env_.act[2] * np.cos(env_.act[1] * 2 * np.pi)
        dy_uav = env_.flight_speed * env_.t_fly * env_.act[2] * np.sin(env_.act[1] * 2 * np.pi)
        loc_uav_after_fly_x = env_.loc_uav[0] + dx_uav
        loc_uav_after_fly_y = env_.loc_uav[1] + dy_uav

        dis_fly = np.sqrt(dx_uav * dx_uav + dy_uav * dy_uav)
        dis_fly = dis_fly + e
        e_fly = (dis_fly / env_.t_fly) ** 2 * env_.m_uav * env_.t_fly * 0.5

        if env_.act[0] == 1:
            ue_id = env_.M - 1
        else:
            ue_id = int(env_.M * env_.act[0])
        offloading_ratio = env_.act[3]
        task_size = env_.task_list[ue_id]
        block_flag = env_.block_flag_list[ue_id]
        t_server = offloading_ratio * task_size / (env_.f_uav / env_.s)
        e_server = env_.r * env_.f_uav ** 3 * t_server

        if env_.sum_task_size == 0:
            is_terminal = True
            reward = 0
        elif env_.sum_task_size - task_size < 0:
            self.task_list = np.ones(env_.M) * env_.sum_task_size
            reward = 0
        elif loc_uav_after_fly_x < 0 or loc_uav_after_fly_x > 100 or loc_uav_after_fly_y < 0 or loc_uav_after_fly_y > 100:
            delay = env_.com_delay(env_.loc_ue_list[ue_id], env_.loc_uav, offloading_ratio, task_size, block_flag)
            reward = -delay
            env_.e_battery_uav = env_.e_battery_uav - e_server
            env_.sum_task_size -= env_.task_list[ue_id]
        elif env_.e_battery_uav < e_fly or env_.e_battery_uav - e_fly < e_server:
            delay = env_.com_delay(env_.loc_ue_list[ue_id], np.array([loc_uav_after_fly_x, loc_uav_after_fly_y]), 0, task_size, block_flag)
            reward = -delay
            env_.sum_task_size -= env_.task_list[ue_id]
        else:
            delay = env_.com_delay(env_.loc_ue_list[ue_id], np.array([loc_uav_after_fly_x, loc_uav_after_fly_y]), offloading_ratio, task_size, block_flag)
            reward = -delay
            env_.e_battery_uav = env_.e_battery_uav - e_fly - e_server
            env_.sum_task_size -= env_.task_list[ue_id]
        
        env_.loc_uav = np.clip([loc_uav_after_fly_x, loc_uav_after_fly_y], 0, 100)  # Ensure UAV stays within 100x100
        self._ep_reward = self._ep_reward + reward
        self._ep_reward_list = np.append(self._ep_reward_list, reward)
        self.genes[i] = copy.deepcopy(env_)

        if not hasattr(env_, 'steps'):
            env_.steps = []
        env_.steps.append({
            'uav_pos': env_.loc_uav,
            'action': env_.act,
            'reward': reward,
            'next_state': [loc_uav_after_fly_x, loc_uav_after_fly_y]
        })
    reward = 0
    self._ep_reward = self._ep_reward + reward
    self._ep_reward_list = np.append(self._ep_reward_list, reward)
    self.genes[i] = copy.deepcopy(env_)


def mutation(self):
    i1 = np.random.randint(len(self.genes))
    i2 = (i1 + 1 + np.random.randint(len(self.genes) - 1)) % (len(self.genes))
    while True:
        env_ = self.genes[i1]
        self.genes[i1] = self.genes[i2]
        self.genes[i2] = env_
        i1 = (i1 + 1) % len(self.genes)
        i2 = (i2 - 1) % len(self.genes)
        if i1 == i2 or abs(i1 - i2) == 1:
            break
    update(self)

def mutation1(self):
    x = np.random.randint(len(self.genes), size=10)
    for i in range(len(x)):
        for j in range(self.genes[x[i]].M):
            tmp = np.random.rand(2)
            theta_ue = tmp[0] * np.pi * 2
            dis_ue = tmp[1] * self.genes[0].delta_t * self.genes[0].v_ue
            self.genes[x[i]].loc_ue_list[j][0] = self.genes[x[i]].loc_ue_list[j][0] + math.cos(theta_ue) * dis_ue
            self.genes[x[i]].loc_ue_list[j][1] = self.genes[x[i]].loc_ue_list[j][1] + math.sin(theta_ue) * dis_ue
            self.genes[x[i]].loc_ue_list[j] = np.clip(self.genes[x[i]].loc_ue_list[j], 0, 100)
        self.genes[x[i]].task_list = np.random.randint(2621440, 3145729, self.genes[0].M)
        self.genes[x[i]].block_flag_list = np.random.randint(0, 2, self.genes[0].M)
    update(self)

def mutation2(self):
    x = np.random.randint(len(self.genes), size=10)
    for i in range(len(x)):
        tmp = np.random.rand(2)
        theta_uav = tmp[0] * np.pi * 2
        dis_uav = tmp[1] * self.genes[0].flight_speed * self.genes[0].t_fly
        self.genes[x[i]].loc_uav[0] = dis_uav * math.cos(theta_uav) + self.genes[x[i]].loc_uav[0]
        self.genes[x[i]].loc_uav[1] = dis_uav * math.sin(theta_uav) + self.genes[x[i]].loc_uav[1]
        self.genes[x[i]].loc_uav = np.clip(self.genes[x[i]].loc_uav, 0, 100)
    update(self)

GA()
plt.plot(bestReward)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show() 
