data = """
Epoch: 000 | Mean training loss:  0.1098
Epoch: 001 | Mean training loss:  0.0665
Epoch: 002 | Mean training loss:  0.0591
Epoch: 003 | Mean training loss:  0.0496
Epoch: 004 | Mean training loss:  0.0513
Epoch: 005 | Mean training loss:  0.0421
Epoch: 006 | Mean training loss:  0.0467
Epoch: 007 | Mean training loss:  0.0371
Epoch: 008 | Mean training loss:  0.0400
Epoch: 009 | Mean training loss:  0.0368
Epoch: 010 | Mean training loss:  0.0341
Epoch: 011 | Mean training loss:  0.0372
Epoch: 012 | Mean training loss:  0.0309
Epoch: 013 | Mean training loss:  0.0336
Epoch: 014 | Mean training loss:  0.0345
Epoch: 015 | Mean training loss:  0.0301
Epoch: 016 | Mean training loss:  0.0351
Epoch: 017 | Mean training loss:  0.0285
Epoch: 018 | Mean training loss:  0.0326
Epoch: 019 | Mean training loss:  0.0319
"""

losses = []
for line in data.split("\n"):
    if len(line) > 1:
        losses.append(float(line.split(" ")[-1]))

import matplotlib.pyplot as plt


print(losses)

fig, ax = plt.subplots(1, 1, figsize=(8,4))

ax.plot([*range(1,len(losses)+1)] ,losses)
ax.set_title(f"Training losses")
ax.set_xlabel("epochs")
ax.set_ylabel("loss")
ax.set_xticks([*range(1,len(losses)+1)])
fig.savefig("figs/training_losses.png")
plt.show()
