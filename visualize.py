from collections import Counter
import matplotlib.pyplot as plt
from utils import get_data

def bar_chart(dataset):
    labels = []
    for i in dataset:
        labels.extend(i[1])

    count_label = Counter(labels)
    count_label = dict(count_label.most_common())
    tags = list(count_label.keys())[1:]  # Except tag O
    values = list(count_label.values())[1:]

    # Figure Size
    fig, ax = plt.subplots(figsize=(16, 9))

    # Horizontal Bar Plot
    ax.barh(tags, values)

    # Remove axes splines
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)

    # Remove x, y Ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    # Add padding between axes and labels
    ax.xaxis.set_tick_params(pad=5)
    ax.yaxis.set_tick_params(pad=10)

    # Add x, y gridlines
    ax.grid(visible=True, color='grey',
            linestyle='-.', linewidth=0.5,
            alpha=0.2)

    # Show top values
    ax.invert_yaxis()

    # Add annotation to bars
    for i in ax.patches:
        plt.text(i.get_width() + 0.2, i.get_y() + 0.5,
                 str(round((i.get_width()), 2)),
                 fontsize=10, fontweight='bold',
                 color='grey')

    # Add Plot Title
    ax.set_title('Number of instances in each tag',
                 loc='left', )

    plt.show()


def pos_neg_bar(dataset):
    num_pos = 0
    for ex in dataset:
        tags = ex[1]
        # print(tags)
        for tag in tags:
            if tag != 'O':
                num_pos += 1
                break

    return num_pos, len(dataset) - num_pos

# print(pos_neg_bar(examples))







