import matplotlib
import matplotlib.pyplot as plt

with open("allresult3.txt", "r") as f:
    lines = f.readlines()
    evalled = []
    for line in lines:
        a = eval(line)
        evalled.append(eval(line))
    x = evalled[0]
    arr1 = evalled[1]
    arr2 = evalled[2]
    arr3 = evalled[3]
    mins1 = evalled[4]
    mins2 = evalled[5]
    mins3 = evalled[6]
    maxs1 = evalled[7]
    maxs2 = evalled[8]
    maxs3 = evalled[9]
    goal_frequencies = evalled[10]
    path_lengths_per_goal = evalled[11]

arr1new = arr1[:6]
mins1new = mins1[:6]
maxs1new = maxs1[:6]
arr2new = [arr1[0]] + arr1[6:]
mins2new = [mins1[0]] + mins1[6:]
maxs2new = [maxs1[0]] + maxs1[6:]

for i in range(len(x)):
    x[i] = x[i][0]+x[i][1]

xx = x[:6]

ax = plt.figure(num='Distance from IRL to original trajectories, 3 experts')
plt.plot(xx, arr1new)
plt.plot(xx, arr2new)
plt.fill_between(xx, mins1new, maxs1new, color=matplotlib.colors.to_rgba('blue', 0.075))
plt.fill_between(xx, mins2new, maxs2new, color=matplotlib.colors.to_rgba('red', 0.075))

plt.xlabel("Amount of trajectories from expert 2 and 3")
plt.ylabel("Dynamic Time Warping Distance")
plt.title('Distance from IRL to original trajectories, 3 experts')

print(goal_frequencies)

filtered_goal_frequencies = []
for i in goal_frequencies:
    total = i[10]+i[26]+i[38]
    filtered_goal_frequencies.append(i[26]/total)

arr1 = filtered_goal_frequencies[:6]
arr2 = [filtered_goal_frequencies[0]] + filtered_goal_frequencies[6:]

ax = plt.figure(num='Goal Frequencies, 3 experts')
plt.plot(xx, arr1)
plt.plot(xx, arr2)
#plt.fill_between(x, mins1, maxs1, color=matplotlib.colors.to_rgba('blue', 0.075))
#plt.fill_between(x, mins2, maxs2, color=matplotlib.colors.to_rgba('red', 0.075))

plt.xlabel("Combined amount of trajectories from expert 2 and 3")
plt.ylabel("Frequency of visiting goal")
plt.title('Goal Frequencies, 3 experts')

plt.show()