# Libraries
import numpy as np
import matplotlib.pyplot as plt

# Make a random dataset:
times = [15.8, 17.1, 22.1,
         18.0, 20.3, 28.6,
         18.7, 22.0, 32.8,
         20.8, 23.1, 37.0]
bars = ('L2K8', 'L2K16', 'L2K32',
        'L3K8', 'L3K16', 'L3K32',
        'L4K8', 'L4K16', 'L4K32',
        'L5K8', 'L5K16', 'L5K32')

bar_colors = ['#7FB17F', '#7FB17F', '#7FB17F',
              '#8080BD', '#8080BD', '#8080BD',
              '#FF7F7F', '#FF7F7F', '#FF7F7F',
              '#7FDEFF', '#7FDEFF', '#7FDEFF']

class_1 = [0.5206, 0.9137, 0.5018,
           0.9069, 0.9400, 0.9157,
           0.9682, 0.9495, 0.9565,
           0.9633, 0.9496, 0.9694]
class_2 = [0.9573, 0.9532, 0.9482,
           0.9625, 0.9611, 0.9588,
           0.9587, 0.9678, 0.9657,
           0.9636, 0.9707, 0.9684]
class_3 = [0.7402, 0.7669, 0.9436,
           0.8043, 0.9085, 0.9396,
           0.9428, 0.9774, 0.9676,
           0.9651, 0.9800, 0.9820]
class_4 = [0.0762, 0.0795, 0.0688,
           0.7820, 0.8853, 0.8193,
           0.9787, 0.9778, 0.9787,
           0.9850, 0.9699, 0.9809]
class_5 = [0.0941, 0.8625, 0.0968,
           0.7925, 0.8904, 0.9565,
           0.8978, 0.9709, 0.9826,
           0.9381, 0.9790, 0.9878]
class_6 = [0.9227, 0.9787, 0.9802,
           0.9043, 0.9862, 0.9909,
           0.9763, 0.9937, 0.9945,
           0.9949, 0.9949, 0.9950]
average = np.mean([class_1, class_2, class_3, class_4, class_5, class_6], axis=0).tolist()

y_pos = np.arange(len(bars))

fig, ax1 = plt.subplots()
# Create bars
barplot = ax1.bar(y_pos, times, color=bar_colors, alpha=0.7)




# Create names on the x-axis
plt.title('Average configuration processing time for 1 image', fontsize=18)
plt.xlabel('CNN configuration', fontsize=14)
ax1.set_ylabel('Time, [ms]', fontsize=13)
ax1.set_xlabel('CNN configuration', fontsize=13)
plt.xticks(y_pos, bars, rotation='vertical', fontsize=10)
plt.grid(color='r', linestyle='-', alpha=0.1)

cm = plt.get_cmap('gist_rainbow')

colors = []
for i in range(5):
    colors.append(cm(i / 5.0))
colors.append((253.0 / 255.0, 1.0 / 255.0, 255.0 / 255.0, 255.0 / 255.0))

ax2 = ax1.twinx()
ax2.set_ylim(0.45, 1.001)
line = ax2.plot(y_pos, class_1, label='Class 1', alpha=0.4)
line[0].set_color(colors[0])
line[0].set_linestyle('dashed')
line = ax2.plot(y_pos, class_2, label='Class 2', alpha=0.4)
line[0].set_color(colors[1])
line[0].set_linestyle('dashed')
line = ax2.plot(y_pos, class_3, label='Class 3', alpha=0.4)
line[0].set_color(colors[2])
line[0].set_linestyle('dashed')
line = ax2.plot(y_pos, class_4, label='Class 4', alpha=0.4)
line[0].set_color(colors[3])
line[0].set_linestyle('dashed')
line = ax2.plot(y_pos, class_5, label='Class 5', alpha=0.4)
line[0].set_color(colors[4])
line[0].set_linestyle('dashed')
line = ax2.plot(y_pos, class_6, label='Class 6', alpha=0.4)
line[0].set_color(colors[5])
line[0].set_linestyle('dashed')
line = ax2.plot(y_pos, average, label='Average')
line[0].set_color('#FFA500')
line[0].set_linewidth(5)
#line[0].set_linestyle('dashed')

ax2.legend(bbox_to_anchor=(1.37, 1.00))

ax2.set_ylabel('Dice score', fontsize=14)

def autolabel(rects):
    for idx, rect in enumerate(barplot):
        height = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width() / 2., 2,
                 times[idx],
                 ha='center', va='bottom', rotation=90, fontsize=12)

autolabel(barplot)

#plt.tight_layout()
plt.savefig('dagm.png', bbox_inches='tight', dpi=400)
plt.close(fig)

