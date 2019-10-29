from matplotlib import pyplot as plt
import numpy
import pickle
def makeLiest(s):
    res = []
    for i in range(706):
        filename = str(i) + "thReward.pkl"
        with open(filename, "rb") as f:
            reward = pickle.load(f)
        value = reward[s]
        res.append(value)
    return res

def draw(res):
    fig, ax1 = plt.subplots()
    plt.plot(range(len(res)), res, label = 'value', linewidth = 2.0, linestyle = '--')
    plt.show()

if __name__ == "__main__":
    state = ((0, 0),(1, 1))
    res = makelist(state)
    draw(res)