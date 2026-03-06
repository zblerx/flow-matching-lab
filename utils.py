import matplotlib.pyplot as plt

def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02}:{m:02}:{s:02}"

def live_plot(data_dict, figsize=(7,5), title=''):
    if not hasattr(live_plot, "fig"):
        live_plot.fig, live_plot.ax = plt.subplots(figsize=figsize)
        live_plot.lines = {}

        for label in data_dict:
            line, = live_plot.ax.plot([], [], label=label)
            live_plot.lines[label] = line

        live_plot.ax.legend(loc="upper right")
        live_plot.ax.grid(True)
        live_plot.ax.set_xlabel("epoch")

    for label, data in data_dict.items():
        live_plot.lines[label].set_data(range(len(data)), data)

    live_plot.ax.relim()
    live_plot.ax.autoscale_view()
    live_plot.ax.set_title(title)

    plt.pause(.1)


