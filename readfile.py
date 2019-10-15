import pickle
filename = "finalReward.pkl"
with open(filename, "rb") as f:
    reward = pickle.load(f)

filename = "finalP.pkl"
with open(filename, "rb") as f:
    P = pickle.load(f)
print(len(reward))
for s in reward.keys():
    print (s, " reward is: ", reward[s])
print (P[((0, 0),(1, 1))])