import matplotlib.pyplot as plt
import numpy as np

# fig, ax = plt.subplots(figsize=(3, 3))

# t = np.arange(0.0, 5.0, 0.01)
# s = np.cos(2*np.pi*t)
# line, = ax.plot(t, s, lw=2)

# # ax.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
# #             arrowprops=dict(facecolor='black', shrink=0.25))

# ax.annotate('local max', xy=(2, 1), xycoords='data',
#             xytext=(0.1, .90), textcoords='axes fraction',
#             va='top', ha='left',
#             arrowprops=dict(facecolor='black', shrink=0.05))

# ax.set_ylim(-2, 2)
# plt.show()

# import matplotlib.patches as mpatches
# fig, ax = plt.subplots(figsize=(3, 3))
# arr = mpatches.FancyArrowPatch((1.25, 1.5), (1.75, 1.5),
#                                arrowstyle='->,head_width=.15', mutation_scale=20)
# ax.add_patch(arr)
# ax.annotate("label", (.5, .5), xycoords=arr, ha='center', va='bottom')
# ax.set(xlim=(1, 2), ylim=(1, 2))


# fig = plt.figure()
# ax = fig.add_subplot(projection='polar')
# r = np.arange(0, 1, 0.001)
# theta = 2 * 2*np.pi * r
# line, = ax.plot(theta, r, color='#ee8d18', lw=3)

# ind = 800
# thisr, thistheta = r[ind], theta[ind]
# ax.plot([thistheta], [thisr], 'o')
# ax.annotate('a polar annotation',
#             xy=(thistheta, thisr),  # theta, radius
#             xytext=(0.05, 0.05),    # fraction, fraction
#             textcoords='figure fraction',
#             arrowprops=dict(facecolor='black', shrink=0.05),
#             horizontalalignment='left',
#             verticalalignment='bottom')


# fig, ax = plt.subplots(figsize=(3, 3))
# x = [1, 3, 5, 7, 9]
# y = [2, 4, 6, 8, 10]
# annotations = ["A", "B", "C", "D", "E"]
# ax.scatter(x, y, s=20)

# for xi, yi, text in zip(x, y, annotations):
#     ax.annotate(text,
#                 xy=(xi, yi), xycoords='data',
#                 xytext=(1.5, 1.5), textcoords='offset points')
    



# fig, ax = plt.subplots(figsize=(5, 5))
# t = ax.text(0.5, 0.5, "Direction",
#             ha="center", va="center", rotation=45, size=15,
#             bbox=dict(boxstyle="round,pad=0.3,rounding_size=(8, 5)",
#                       fc="lightblue", ec="steelblue", lw=2))


# from matplotlib.path import Path


# def custom_box_style(x0, y0, width, height, mutation_size):
#     """
#     Given the location and size of the box, return the path of the box around it.

#     Rotation is automatically taken care of.

#     Parameters
#     ----------
#     x0, y0, width, height : float
#        Box location and size.
#     mutation_size : float
#         Mutation reference scale, typically the text font size.
#     """
#     # padding
#     mypad = 0.3
#     pad = mutation_size * mypad
#     # width and height with padding added.
#     width = width + 2 * pad
#     height = height + 2 * pad
#     # boundary of the padded box
#     x0, y0 = x0 - pad, y0 - pad
#     x1, y1 = x0 + width, y0 + height
#     # return the new path
#     return Path([(x0, y0), (x1, y0), (x1, y1), (x0, y1),
#                  (x0-pad, (y0+y1)/2), (x0, y0), (x0, y0)],
#                 closed=True)

# fig, ax = plt.subplots(figsize=(3, 3))
# ax.text(0.5, 0.5, "Test", size=30, va="center", ha="center", rotation=30,
#         bbox=dict(boxstyle=custom_box_style, alpha=0.2))


# fig, ax = plt.subplots(figsize=(3, 3))
# ax.annotate("",
#             xy=(0.2, 0.2), xycoords='data',
#             xytext=(0.8, 0.8), textcoords='data',
#             arrowprops=dict(arrowstyle="<|-|>", connectionstyle="arc3"))


# fig, ax = plt.subplots(figsize=(3, 3))

# ax.annotate("Test",
#             xy=(0.2, 0.2), xycoords='data',
#             xytext=(0.8, 0.8), textcoords='data',
#             size=10, va="center", ha="center",
#             arrowprops=dict(arrowstyle="simple",
#                             connectionstyle="arc3,rad=-0.2"))


# fig, ax = plt.subplots(figsize=(3, 3))

# ann = ax.annotate("Test",
#                   xy=(0.2, 0.2), xycoords='data',
#                   xytext=(0.8, 0.8), textcoords='data',
#                   size=20, va="center", ha="center",
#                   bbox=dict(boxstyle="round4", fc="w"),
#                   arrowprops=dict(arrowstyle="-|>",
#                                   connectionstyle="arc3,rad=-0.2",
#                                   fc="w"))



# fig, ax = plt.subplots(figsize=(3, 3))

# ann = ax.annotate("Test",
#                   xy=(0.2, 0.2), xycoords='data',
#                   xytext=(0.8, 0.8), textcoords='data',
#                   size=20, va="center", ha="center",
#                   bbox=dict(boxstyle="round4", fc="w"),
#                   arrowprops=dict(arrowstyle="-|>",
#                                   connectionstyle="arc3,rad=0.2",
#                                   relpos=(0., 0.),
#                                   fc="w"))

# ann = ax.annotate("Test",
#                   xy=(0.2, 0.2), xycoords='data',
#                   xytext=(0.8, 0.8), textcoords='data',
#                   size=20, va="center", ha="center",
#                   bbox=dict(boxstyle="round4", fc="w"),
#                   arrowprops=dict(arrowstyle="-|>",
#                                   connectionstyle="arc3,rad=-0.2",
#                                   relpos=(1., 0.),
#                                   fc="w"))



# from matplotlib.offsetbox import AnchoredText

# fig, ax = plt.subplots(figsize=(3, 3))
# at = AnchoredText("Figure 1a",
#                   prop=dict(size=15), frameon=True, loc='upper left')
# at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
# ax.add_artist(at)


# from matplotlib.patches import Circle
# from mpl_toolkits.axes_grid1.anchored_artists import AnchoredDrawingArea

# fig, ax = plt.subplots(figsize=(3, 3))
# ada = AnchoredDrawingArea(40, 20, 0, 0,
#                           loc='upper right', pad=0., frameon=False)
# p1 = Circle((10, 10), 10)
# ada.drawing_area.add_artist(p1)
# p2 = Circle((30, 10), 5, fc="r")
# ada.drawing_area.add_artist(p2)
# ax.add_artist(ada)


from matplotlib.patches import Ellipse
# from mpl_toolkits.axes_grid1.anchored_artists import AnchoredAuxTransformBox

# fig, ax = plt.subplots(figsize=(3, 3))
# box = AnchoredAuxTransformBox(ax.transData, loc='upper left')
# el = Ellipse((0, 0), width=0.1, height=0.4, angle=30)  # in data coordinates!
# box.drawing_area.add_artist(el)
# ax.add_artist(box)


# from matplotlib.offsetbox import (AnchoredOffsetbox, DrawingArea, HPacker,
#                                   TextArea)

# fig, ax = plt.subplots(figsize=(3, 3))

# box1 = TextArea(" Test: ", textprops=dict(color="k"))
# box2 = DrawingArea(60, 20, 0, 0)

# el1 = Ellipse((10, 10), width=16, height=5, angle=30, fc="r")
# el2 = Ellipse((30, 10), width=16, height=5, angle=170, fc="g")
# el3 = Ellipse((50, 10), width=16, height=5, angle=230, fc="b")
# box2.add_artist(el1)
# box2.add_artist(el2)
# box2.add_artist(el3)

# box = HPacker(children=[box1, box2],
#               align="center",
#               pad=0, sep=5)

# anchored_box = AnchoredOffsetbox(loc='lower left',
#                                  child=box, pad=0.,
#                                  frameon=True,
#                                  bbox_to_anchor=(0., 1.02),
#                                  bbox_transform=ax.transAxes,
#                                  borderpad=0.,)

# ax.add_artist(anchored_box)
# fig.subplots_adjust(top=0.8)



# x = np.linspace(-1, 1)

# fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))
# ax1.plot(x, -x**3)
# ax2.plot(x, -3*x**2)
# ax2.annotate("",
#              xy=(0, 0), xycoords=ax1.transData,
#              xytext=(0, 0), textcoords=ax2.transData,
#              arrowprops=dict(arrowstyle="<->")) 


# fig, ax = plt.subplots(figsize=(3, 3))

# t1 = ax.text(0.05, .05, "Text 1", va='bottom', ha='left')
# t2 = ax.text(0.90, .90, "Text 2", ha='right')
# t3 = ax.annotate("Anchored to 1 & 2", xy=(0, 0), xycoords=(t1, t2),
#                  va='bottom', color='tab:orange',)

# plt.show()


import pandas as pd

tips = pd.read_csv("examples/tips.csv")
# print(tips.head())

party_counts = pd.crosstab(tips["day"], tips["size"])
party_counts = party_counts.reindex(index=["Thur", "Fri", "Sat", "Sun"])
party_counts = party_counts.loc[:, 2:5]

# print(party_counts)

party_pcts = party_counts.div(party_counts.sum(axis="columns"), axis="index")

# print(party_counts.sum(axis="columns"))
# print(party_pcts)
plt.style.use('grayscale')

# party_pcts.plot.bar(stacked=True)

import seaborn as sns
tips["tip_pct"] = tips["tip"] / (tips["total_bill"] - tips["tip"])
# print(tips.head())
# sns.barplot(x="tip_pct", y="day", data=tips, orient="h")

# sns.barplot(x="tip_pct", y="day", hue="time", data=tips, orient="h")

# tips["tip_pct"].plot.hist(bins=50)
# tips["tip_pct"].plot.density()



# comp1 = np.random.standard_normal(200)
# comp2 = 10 + 2 * np.random.standard_normal(200)
# values = pd.Series(np.concatenate([comp1, comp2]))
# sns.histplot(values, bins=100, color="black")

 
macro = pd.read_csv("examples/macrodata.csv")
data = macro[["cpi", "m1", "tbilrate", "unemp"]]
# print(data.tail())

trans_data = np.log(data).diff().dropna()
# print(trans_data.tail())


# ax = sns.regplot(x="m1", y="unemp", data=trans_data)
# ax.set_title("Changes in log(m1) versus log(unemp)")

# sns.pairplot(trans_data, diag_kind="kde", plot_kws={"alpha": 0.2})

# sns.catplot(x="day", y="tip_pct", hue="time", col="smoker",
# kind="bar", data=tips[tips.tip_pct < 1])


# sns.catplot(x="day", y="tip_pct", row="time",
# col="smoker",
# kind="bar", data=tips[tips.tip_pct < 1])


sns.catplot(x="tip_pct", y="day", kind="box",
data=tips[tips.tip_pct < 0.5])
plt.show()