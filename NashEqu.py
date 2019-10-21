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
import time
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
        # filename = "reward/4thReward.pkl"
        # with open(filename, "rb") as f:
        #     self.V = pickle.load(f)
        self.V = self.init_V()
        self.V_ = {}
        filename = "finalP11.pkl"
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
        # print(len(catch))
        return catch
    def getR(self):
        R = []
        for p1, q1 in product(range(self.Height), range(self.Width)):
            R.append((p1, q1))
        return R

    def init_V(self):
        V = {}
        filename = "Set95.pkl"
        with open(filename, "rb") as f:
            amsurewin = pickle.load(f)
        for s in self.S:
            if s in self.catch:
                V[s] = self.catchV
            elif s in amsurewin:
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
        # print("V_ is: ", self.V_[(2, 1), (0, 2)])

    def checkConverge(self):
        VVec = self.dict2vec(self.V)
        VVec_ = self.dict2vec(self.V_)
        if (abs(VVec - VVec_) > 0.001).any():
            return False
        return True

def NashEqu(output, Game, state, j):
    M = Game.createGameMatrix(state)
    flag , V = saddlePoint(M)
    if flag:
        output.put((j, V))
    else:
        # if np.linalg.det(M) != 0:
        #     iden = np.array([1, 1, 1, 1])
        #     V = 1 / (iden.dot(np.linalg.inv(M)).dot(iden.T))
        #     output.put((j, V))
        #     print("Not singular Matrix, use Game theory method, V is:", V)
        #     if V>100 or V<-100:
        #         print("M is:", M)
        #         print("wdnmd")
        # else:
        rps = nash.Game(M)
        # print("state is: ", state, "M is: ", M)
        eqs = rps.support_enumeration()
        flag_su = 0
        for eq in eqs:
            # policy_ct = np.round(eq[0], 6)
            # policy_ad = np.round(eq[1], 6)
            policy_ct = eq[0]
            policy_ad = eq[1]
            reward = rps[policy_ct, policy_ad]
            reward_ct = reward[0]
            if math.isnan(reward_ct) == False and math.isinf(reward_ct) == False and abs(reward_ct) <=100 and sum(abs(policy_ct))<1.1 and sum(abs(policy_ad))<1.1:
                flag_su = 1
                break

        if flag_su == 1:
            # print("Use support_enumeration, V is:", reward_ct)
            output.put((j, reward_ct))
        else:
        # except UnboundLocalError:  ##Here, we can not use support_enumeration, try vertex enumeration
            rps = nash.Game(M)
            eqs = rps.vertex_enumeration()
            flag_ver = 0
            for eq in eqs:
                # policy_ct = np.round(eq[0], 6)
                # policy_ad = np.round(eq[1], 6)
                policy_ct = eq[0]
                policy_ad = eq[1]
                reward = rps[policy_ct, policy_ad]
                reward_ct = reward[0]
                if math.isnan(reward_ct) == False and math.isinf(reward_ct) == False and abs(reward_ct) <=100 and sum(abs(policy_ct))<1.1 and sum(abs(policy_ad))<1.1:
                    flag_ver = 1
                    break
            if flag_ver == 1:
                # print("Use vertex_enumeration, V is:", reward_ct)
                output.put((j, reward_ct))
            else:
                print ("WDNMD")
                print("M is:", M)
                print("state is:", state)
    # print("state finish:", state)
         ##j here is used to keep multiprocessing in order

def saddlePoint(M):
    flag = False
    V = 0
    for i in range(len(M)):
        for j in range(len(M[0])):
            if M[i][j] == np.min(M[i,:]) and M[i][j] == np.max(M[:,j]):
                flag = True
                V = M[i][j]
                break
    return flag, V

"""
def NashEqu(output, Game, state, j):
    M = Game.createGameMatrix(state)
    flag , V = saddlePoint(M)
    if flag:
        output.put((j, V))
    else:
        print("Not saddle point, state is:", state)
        print("M is:", M)
        iden = np.array([1, 1, 1, 1])
        try:
            V = 1 / (iden.dot(np.linalg.pinv(M)).dot(iden.T))
            q = V * np.linalg.pinv(M).dot(iden.T)
            q_ = np.linalg.pinv(M).dot(iden.T) * V
            print("V is:", V)
            print("q is:", q)
            print("q_ is:", q_)
            output.put((j, V))
        except np.linalg.LinAlgError:
            print(M)
            print(state)
            input("111")
"""

def valueIter(Game, threadnumber):
    index = 1
    print (index, "th iteration")
    value_new = parallelComputeReward(Game, threadnumber)
    Game.vec2dict(value_new)
    filename_temp = "reward/" + str(index) + "thReward.pkl"
    picklefile = open(filename_temp, "wb")
    pickle.dump(Game.V, picklefile)
    picklefile.close()
    while (not Game.checkConverge()):
        index += 1
        print(index, "th iteration")
        Game.V = dcp(Game.V_)
        filename_temp = "reward/" + str(index) + "thReward.pkl"
        picklefile = open(filename_temp, "wb")
        pickle.dump(Game.V, picklefile)
        picklefile.close()
        value_new = parallelComputeReward(Game, threadnumber)
        Game.vec2dict(value_new)
    filename = "finalReward11.pkl"
    picklefile = open(filename, "wb")
    pickle.dump(Game.V, picklefile)
    picklefile.close()


def parallelComputeReward(Game, threadnumber):
    parallelprocess = threadnumber
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
            process.append(mp.Process(target = NashEqu, args = (output, Game, Game.S[index], j)))
        for p in process:
            p.start()
        for p in process:
            p.join()
        temp_res = [None] * len(process)
        for p in process:
            res = output.get()
            temp_res[res[0]] = res[1]
        result.extend(temp_res)
        # time.sleep(1)
    # print("In multiprocessing is: ", result[65])
    return result

if __name__ == '__main__':
    threadnumber = 20
    localtime = time.asctime(time.localtime(time.time()))
    print("Start time is:", localtime)
    Game = TwoPlayerGridGame()
    valueIter(Game, threadnumber)
    # M =[[0, 0,  0, 0],[100, 0,  100,  100],[100, 0, 100, 100],[0,0, 0, 0]]
    # M = Game.createGameMatrix(((0,7), (1,6)))
    # print (M)
    # rps = nash.Game(M)
    # eqs = rps.support_enumeration()
    # for eq in eqs:
    #     print("111")
    #     policy_ct = eq[0]
    #     policy_ad = eq[1]
    #     print(type(policy_ct))
    #     print(policy_ad)
    #     reward = rps[policy_ct, policy_ad]
    #     print(reward)
    # localtime = time.asctime(time.localtime(time.time()))
    print("End time is:", localtime)
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


