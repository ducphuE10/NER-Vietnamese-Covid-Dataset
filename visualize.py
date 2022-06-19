from dataset import Dataset
from collections import Counter
import matplotlib.pyplot as plt

corpus = Dataset(train_path='./PhoNER_COVID19/data/word/train_word.conll',
                 val_path='./PhoNER_COVID19/data/word/dev_word.conll',
                 test_path='./PhoNER_COVID19/data/word/test_word.conll',
                 batch_size=64)

label = []
for i in range(len(corpus.train_dataset)):
    label.extend(corpus.train_dataset[i].tag)

count_label = Counter(label)

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


# Show Plot
plt.show()