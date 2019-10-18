import pickle
filename = "finalReward3.pkl"
with open(filename, "rb") as f:
    reward = pickle.load(f)

filename = "finalP3.pkl"
with open(filename, "rb") as f:
    P = pickle.load(f)
# print(len(reward))
for key, value in reward.items():
    print (key, ":    ", value)

# for s in reward.keys():
#     print (s, " reward is: ", reward[s])
print (P[((2, 1),(0, 2))])