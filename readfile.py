import pickle
filename = "finalReward.pkl"
with open(filename, "rb") as f:
    reward = pickle.load(f)

for s in reward.keys():
    print (s, " reward is: ", reward[s])