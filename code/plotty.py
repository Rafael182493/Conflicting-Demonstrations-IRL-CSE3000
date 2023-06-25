import matplotlib
import matplotlib.pyplot as plt

with open("allresult.txt", "r") as f:
    lines = f.readlines()
    evalled = []
    for line in lines:
        a = eval(line)
        evalled.append(eval(line))
    x = evalled[0]
    arr1 = evalled[1][:5]
    arr2 = evalled[2]
    mins1 = evalled[3]
    mins2 = evalled[4]
    maxs1 = evalled[5]
    maxs2 = evalled[6]
    goal_frequencies = evalled[7]
    path_lengths_per_goal = evalled[8]

ax = plt.figure(num='Distance from IRL to original trajectories')
plt.plot(x, arr1)
plt.plot(x, arr2)
plt.fill_between(x, mins1, maxs1, color=matplotlib.colors.to_rgba('blue', 0.075))
plt.fill_between(x, mins2, maxs2, color=matplotlib.colors.to_rgba('red', 0.075))

plt.xlabel("Amount of trajectories from expert 2")
plt.ylabel("Dynamic Time Warping Distance")
plt.title('Dynamic time warping distance of IRL with conflicting demonstrations')

print(goal_frequencies)

arr1 = []
arr2 = []
for i in goal_frequencies:
    total = i[12]+i[36]
    arr1.append(i[12]/total)
    arr2.append(i[36]/total)

#use for rewards on 12, 24 & 40
#for i in goal_frequencies:
#    total = i[12]+i[40]+i[24]
#    arr1.append((i[12] + i[40])/total)
#    arr2.append(i[24]/total)

ax = plt.figure(num='Goal Frequencies')
plt.plot(x, arr1)
plt.plot(x, arr2)
#plt.fill_between(x, mins1, maxs1, color=matplotlib.colors.to_rgba('blue', 0.075))
#plt.fill_between(x, mins2, maxs2, color=matplotlib.colors.to_rgba('red', 0.075))

plt.xlabel("Amount of trajectories from expert 2")
plt.ylabel("Frequency of visiting goal")
plt.title('Goal Frequencies')

plt.show()







