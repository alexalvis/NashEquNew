import numpy as np
import pickle
import pandas
import nashpy as nash
from itertools import product
import math
import multiprocessing as mp
import time
import warnings
from copy import deepcopy as dcp
warnings.filterwarnings("ignore")

from EnvPara import EnvPara
class TwoPlayerGridGame():
    def __init__(self):
        self.Height = EnvPara.height
        self.Width = EnvPara.width
        self.A = EnvPara.A
        self.distthre = EnvPara.distthre
        self.S = self.getS()
        self.catch = self.getCatch()
        self.R = self.getR()
        self.contrRand = EnvPara.contr_rand
        self.adRand = EnvPara.ad_rand
        self.goalReachV = EnvPara.goalReachV
        self.catchV = EnvPara.catchV
        self.goal = EnvPara.goal   ##this should be a list, right now this is just single goal, maybe extend to several goals later but not now.     2019/10/11
        self.O = EnvPara.Obs
        self.P = self.getP()
        self.V = self.init_V()
        self.V_ = {}
        filename = "finalP5.pkl"
        picklefile = open(filename, "wb")
        pickle.dump(self.P, picklefile)
        picklefile.close()

    def getS(self):
        inner = []
        for p1, q1, p2, q2 in product(range(self.Height), range(self.Width), repeat = 2):
            inner.append(((p1, q1), (p2, q2)))
        return inner

    def getCatch(self):
        catch = []
        for s in self.S:
            if (abs(s[0][0] - s[1][0]) + abs(s[0][1] - s[1][1])) <= self.distthre:
                catch.append(s)
        print(len(catch))
        return catch
    def getR(self):
        R = []
        for p1, q1 in product(range(self.Height), range(self.Width)):
            R.append((p1, q1))
        return R

    def init_V(self):
        V = {}
        for s in self.S:
            if s in self.catch:
                V[s] = self.catchV
            elif s[0] in self.goal:
                V[s] = self.goalReachV
            else:
                V[s] = 0
        return V

    def trans_P(self, state, id, goal):    ##Here goal should be a list
        P = {}
        st_ct = state[0]
        st_ad = state[1]
        if id == 1:      ##The situation of control agent
            if st_ct not in goal and st_ct not in self.O and state not in self.catch:
                rand = self.contrRand
                explore = []
                for action in self.A:
                    temp = tuple(np.array(st_ct) + np.array(self.A[action]))
                    explore.append(temp)
                for action in self.A:
                    P[action] = {}
                    P[action][st_ct] = rand
                    temp_st = tuple(np.array(st_ct) + np.array(self.A[action]))
                    if temp_st in self.R and temp_st not in self.O:       ##Here obstacle is None, so it does not matter, if there are obstacles, we have to talk about what will be the situation
                        P[action][temp_st] = 1 - rand * 4
                        for st_ in explore:
                            if st_ != temp_st:
                                if st_ in self.R:
                                    P[action][st_] = rand
                                else:
                                    P[action][st_ct] += rand
                    else:
                        P[action][st_ct] = 1 - rand * 3
                        for st_ in explore:
                            if st_ != temp_st:
                                if st_ in self.R:
                                    P[action][st_] = rand
                                else:
                                    P[action][st_ct] += rand
            else:
                for action in self.A:
                    P[action] = {}
                    P[action][st_ct] = 1.0
        else:    ##The situation of advanced agent
            if state not in self.catch and st_ct not in self.O and st_ct not in goal:
                rand = self.adRand
                explore = []
                for action in self.A:
                    temp = tuple(np.array(st_ad) + np.array(self.A[action]))
                    explore.append(temp)
                for action in self.A:
                    P[action] = {}
                    P[action][st_ad] = rand
                    temp_st = tuple(np.array(st_ad) + np.array(self.A[action]))
                    if temp_st in self.R:
                        P[action][temp_st] = 1 - rand * 4
                        for st_ in explore:
                            if st_ != temp_st:
                                if st_ in self.R:
                                    P[action][st_] = rand
                                else:
                                    P[action][st_ad] += rand
                    else:
                        P[action][st_ad] = 1 - rand * 3
                        for st_ in explore:
                            if st_ != temp_st:
                                if st_ in self.R:
                                    P[action][st_] = rand
                                else:
                                    P[action][st_ad] += rand
            else:
                for action in self.A:
                    P[action] = {}
                    P[action][st_ad] = 1.0
        return P

    def getP(self):
        P = {}
        Pro_ct = {}
        Pro_ad = {}
        for s in self.S:
            P[s] = {}
            Pro_ct[s] = self.trans_P(s, 1, self.goal)
            Pro_ad[s] = self.trans_P(s, 2, self.goal)
            for a1, a2 in product(self.A, self.A):
                P[s][(a1, a2)] = {}
                for st_ct_ in Pro_ct[s][a1].keys():
                    for st_ad_ in Pro_ad[s][a2].keys():
                        if Pro_ct[s][a1][st_ct_] > 0 and Pro_ad[s][a2][st_ad_] > 0:
                            P[s][(a1,a2)][(st_ct_, st_ad_)] = Pro_ct[s][a1][st_ct_] * Pro_ad[s][a2][st_ad_]
        return P

    def createGameMatrix(self, state):
        actDict = {0 : "N", 1 : "S", 2 : "W", 3 : "E"}
        M = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                M[i][j] = self.getReward(state, actDict[i], actDict[j])
        return M

    def getReward(self, state, a1, a2):
        reward = 0
        for st_ in self.P[state][(a1, a2)].keys():
            reward += self.V[st_] * self.P[state][(a1, a2)][st_]
        return reward

    def dict2vec(self, V):
        VVec = []
        for s in self.S:
            VVec.append(V[s])
        return np.array(VVec)

    def vec2dict(self, V):    ##transfer value to self.V_
        for i in range(len(self.S)):
            self.V_[self.S[i]] = V[i]

    def checkConverge(self):
        VVec = self.dict2vec(self.V)
        VVec_ = self.dict2vec(self.V_)
        if (abs(VVec - VVec_) > 0.001).any():
            return False
        return True

def NashEqu(output, Game, state):
    M = Game.createGameMatrix(state)
    rps = nash.Game(M)
    eqs = rps.support_enumeration()
    for eq in eqs:
        policy_ct = eq[0]
        policy_ad = eq[1]
        break
    try:
        reward = rps[policy_ct, policy_ad]
        reward_ct = reward[0]
        reward_ad = reward[1]
    except UnboundLocalError:  ##Here, we can not use support_enumeration, try vertex enumeration
        rps = nash.Game(M)
        eqs = rps.vertex_enumeration()
        for eq in eqs:
            policy_ct = eq[0]
            policy_ad = eq[1]
            break
        try:
            reward = rps[policy_ct, policy_ad]
            reward_ct = reward[0]
            reward_ad = reward[1]
        except UnboundLocalError:
            print ("WDNMD")
            print("M is:", M)
    output.put(reward_ct)

def valueIter(Game):
    i = 1
    print (i, "th iteration")
    value_new = parallelComputeReward(Game)
    Game.vec2dict(value_new)
    while (not Game.checkConverge()):
        i += 1
        print(i, "th iteration")
        Game.V = dcp(Game.V_)
        value_new = parallelComputeReward(Game)
        Game.vec2dict(value_new)
    filename = "finalReward5.pkl"
    picklefile = open(filename, "wb")
    pickle.dump(Game.V, picklefile)
    picklefile.close()


def parallelComputeReward(Game):
    parallelprocess = 10
    iterationnum = len(Game.S)
    periods = math.ceil(iterationnum/parallelprocess)
    result = []
    for i in range(periods):
        output = mp.Queue()
        process = []
        for j in range(parallelprocess):
            index = j + i * parallelprocess
            if index >= len(Game.S):
                break
            process.append(mp.Process(target = NashEqu, args = (output, Game, Game.S[index])))
        for p in process:
            p.start()
        for p in process:
            p.join()

        result.extend([output.get() for p in process])
        time.sleep(1)
    return result

if __name__ == '__main__':
    Game = TwoPlayerGridGame()
    valueIter(Game)
    # filename = "finalReward.pkl"
    # with open(filename, "rb") as f:
    #     reward = pickle.load(f)
    # Game.V = reward
    # state = ((0, 0),(1, 1))
    # M = Game.createGameMatrix(state)
    # print(M)
    # rps = nash.Game(M)
    # eqs = rps.support_enumeration()
    # for eq in eqs:
    #     policy_ct = eq[0]
    #     policy_ad = eq[1]
    #     break
    # print(policy_ct)
    # print(policy_ad)


