import csv
import logging
import numpy as np
from tabulate import tabulate
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from matplotlib.patches import Wedge
import matplotlib.animation as animation

# Anomaly detection parameters
WINDOW_SIZE = 60
THRESHOLD_ZSCORE = 2
THRESHOLD_SMA = 18
K_NEIGHBOURS = 14
RADIUS_R = 7

# Anomaly simulation parameters and statistics
ANOMALY_CHANCE = 0.02
MAX_PAST_DATA = 500

anomaly_stats = {"actual": 0, "zscore": {}, "sma": {}, "storm": {}}
for algorithm in ["zscore", "sma", "storm"]:
    anomaly_stats[algorithm]["true_positive"] = 0
    anomaly_stats[algorithm]["true_negative"] = 0
    anomaly_stats[algorithm]["false_positive"] = 0
    anomaly_stats[algorithm]["false_negative"] = 0

# Logging configuration
logging.basicConfig(
    filename="anomalies.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    filemode="w",
)

with open("anomalies.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Time", "Value", "Score", "Algorithm"])


class AnomalyDetector:
    """Detects anomalies in a data stream using different algorithms.

    The class supports three algorithms: Z-score, Simple Moving Average (SMA), and STORM.
    """

    def __init__(self, window_size=50, threshold=3, k=3, R=5, algorithm=None):
        """Initializes the anomaly detector.

        :param window_size: Number of recent data points to consider.
        :param threshold: Threshold for detecting anomalies (used by both algorithms).
        :param k: Number of nearest neighbors to consider (used by the STORM algorithm).
        :param R: Radius for the STORM algorithm.
        :param algorithm: The algorithm to use for anomaly detection.
        """
        self.window_size = window_size
        self.threshold = threshold
        self.data_window = deque(maxlen=window_size)
        self.sma_window = deque(maxlen=window_size)
        self.k = k
        self.R = R
        self.algorithm = algorithm

    def detect(self, value):
        """Detects if the incoming value is an anomaly based on the specified algorithm.

        :param value: The latest data point from the stream.
        :return: Tuple (is_anomaly: bool, score: float)
        """
        if self.algorithm:
            algorithm = self.algorithm
        else:
            algorithm = "storm"
        match algorithm:
            case "zscore":
                return self._zscore_detection(value)
            case "storm":
                return self._storm_detection(value)
            case "sma":
                return self._sma_detection(value)
            case _:
                raise ValueError(f"Unknown algorithm: {algorithm}")

    def _zscore_detection(self, value):
        """Z-score-based anomaly detection.

        :param value: The latest data point from the stream.
        :return: Tuple (is_anomaly: bool, z_score: float)
        """
        self.data_window.append(value)

        # Not enough data to compute reliable statistics
        if len(self.data_window) < self.window_size:
            return False, 0

        mean = np.mean(self.data_window)
        std = np.std(self.data_window)

        # Handle zero division error
        if std == 0:
            return False, 0

        z_score = (value - mean) / std
        is_anomaly = abs(z_score) > self.threshold
        return is_anomaly, z_score

    def _sma_detection(self, value):
        """Simple Moving Average (SMA)-based anomaly detection.

        :param value: The latest data point from the stream.
        :return: Tuple (is_anomaly: bool, deviation: float)
        """
        self.sma_window.append(value)

        # wait until we have enough data points
        if len(self.sma_window) < self.window_size:
            return False, 0

        sma = np.mean(self.sma_window)
        deviation = abs(value - sma)

        is_anomaly = deviation > self.threshold
        return is_anomaly, deviation

    def _storm_detection(self, value):
        """STORM-based distance outlier detection.

        :param value: The latest data point from the stream.
        :return: Tuple (is_anomaly: bool, distance to k-nearest neighbors)
        """
        self.data_window.append(value)

        # wait until the simulation is stable
        if len(self.data_window) < self.window_size:
            return False, 0

        distances = [abs(value - point) for point in self.data_window]
        distances.sort()

        neighbors_within_R = sum(1 for d in distances if d <= self.R)

        is_anomaly = neighbors_within_R < self.k
        return is_anomaly, neighbors_within_R


def log_anomaly(time, value, score, algorithm):
    """Logs an anomaly event using the logging module and writes it to a CSV file.

    :param time: The timestamp of the detected anomaly.
    :param value: The value of the detected anomaly.
    :param score: The score used by the algorithm.
    :param algorithm: The name of the algorithm that detected the anomaly.
    """
    logging.info(
        f"Anomaly detected by {algorithm} at time {time} with value {value} and score {score}"
    )

    with open("anomalies.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([time, value, score, algorithm])


def data_stream():
    """Simulates a continuous data stream with seasonal patterns, noise, and occasional anomalies."""
    t = 0
    while True:
        seasonal = 10 * np.sin(0.1 * t)
        noise = np.random.normal(0, 1)
        anomaly = 0
        if np.random.rand() < ANOMALY_CHANCE:
            anomaly = np.random.normal(15, 5)
        value = seasonal + noise + anomaly
        yield value, anomaly != 0
        t += 1


def display_stats(stats):
    """Displays the confusion matrix for each algorithm using the tabulate library.

    :param stats: The dictionary containing the confusion matrix statistics for each algorithm.
    """
    headers = [
        "Algorithm",
        "True Positive",
        "True Negative",
        "False Positive",
        "False Negative",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
    ]
    table = []

    for algorithm in ["zscore", "sma", "storm"]:
        true_positive, true_negative, false_positive, false_negative = (
            stats[algorithm]["true_positive"],
            stats[algorithm]["true_negative"],
            stats[algorithm]["false_positive"],
            stats[algorithm]["false_negative"],
        )
        accuracy = (
            (true_positive + true_negative)
            / (true_positive + true_negative + false_positive + false_negative)
            if (true_positive + true_negative + false_positive + false_negative) > 0
            else 0
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if (true_positive + false_positive) > 0
            else 0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if (true_positive + false_negative) > 0
            else 0
        )
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        row = [
            algorithm.capitalize(),
            true_positive,
            true_negative,
            false_positive,
            false_negative,
            accuracy,
            precision,
            recall,
            f1_score,
        ]
        table.append(row)

    print(tabulate(table, headers=headers, tablefmt="grid"))

    print(f"\nTotal actual anomalies in the stream: {stats['actual']}")


def draw_pie_marker(ax, x, y, algorithms, size=50):
    """Draws a pie-shaped marker representing overlapping anomalies at the same point.

    :param ax: The axes on which to draw the pie marker.
    :param x: The x-coordinate of the anomaly point.
    :param y: The y-coordinate of the anomaly point.
    :param algorithms: A list of algorithms that detected the anomaly (e.g., ['zscore', 'sma']).
    :param size: The size of the pie marker.
    """
    colors = {
        "actual": "green",
        "zscore": "orange",
        "sma": "lightblue",
        "storm": "magenta",
    }

    # Divide the full circle (360 degrees) into equal slices based on the number of algorithms
    num_algorithms = len(algorithms)
    angle_step = 360 / num_algorithms
    start_angle = 0

    # Draw each slice as a Wedge (pie slice)
    for algorithm in algorithms:
        wedge = Wedge(
            center=(x, y),
            r=size / 15,
            theta1=start_angle,
            theta2=start_angle + angle_step,
            color=colors[algorithm],
            lw=0,
        )
        ax.add_patch(wedge)
        start_angle += angle_step


def add_custom_legend(ax):
    """Adds a custom legend to the ax showing the meaning of pie marker colors."""
    legend_elements = [
        Patch(facecolor="green", edgecolor="black", label="Actual Anomaly"),
        Patch(facecolor="orange", edgecolor="black", label="Z-Score Anomaly"),
        Patch(facecolor="lightblue", edgecolor="black", label="SMA Anomaly"),
        Patch(facecolor="magenta", edgecolor="black", label="STORM Anomaly"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", title="Anomaly Detection")


def animate(
    i,
    axes: list[Axes],
    detectors,
    data_gen,
    xdata,
    ydata,
    zscore_data,
    sma_data,
    storm_data,
    lines,
    scatters,
):
    """Updates the plot with the latest data point and anomaly detection results.

    :param i: The current frame number.
    :param axes: The list of axes objects for the subplots.
    :param detectors: The dictionary of anomaly detectors.
    :param data_gen: The generator for the data stream.
    :param xdata: The x-coordinates of the data points.
    :param ydata: The y-coordinates of the data points.
    :param zscore_data: The z-score values for each data point.
    :param sma_data: The SMA deviation values for each data point.
    :param storm_data: The number of k-nearest neighbors for each data point.
    :param lines: The list of line objects for the plots.
    :param scatters: The list of scatter objects for the anomaly markers.
    :return: The updated line and scatter objects.
    """
    value, actual_anomaly = next(data_gen)
    xdata.append(i)
    ydata.append(value)

    is_anomaly_zscore, zscore = detectors["zscore"].detect(value)
    is_anomaly_sma, sma = detectors["sma"].detect(value)
    is_anomaly_storm, storm = detectors["storm"].detect(value)

    zscore_data.append(zscore)
    sma_data.append(sma)
    storm_data.append(storm)

    for line, score_data in zip(lines, [ydata, zscore_data, sma_data, storm_data]):
        line.set_data(xdata, score_data)

    for scatter, is_anomaly, score in zip(
        scatters,
        [is_anomaly_zscore, is_anomaly_sma, is_anomaly_storm],
        [zscore, sma, storm],
    ):
        if is_anomaly:
            scatter.set_offsets(
                np.ma.append(scatter.get_offsets(), [[i, score]], axis=0)
            )

    # Handle the pie marker creation for overlapping anomalies
    triggered_algorithms = []
    if is_anomaly_zscore:
        triggered_algorithms.append("zscore")
    if is_anomaly_sma:
        triggered_algorithms.append("sma")
    if is_anomaly_storm:
        triggered_algorithms.append("storm")
    if actual_anomaly:
        triggered_algorithms.append("actual")

    if triggered_algorithms:
        draw_pie_marker(axes[0], i, value, triggered_algorithms, size=50)

    if actual_anomaly:
        anomaly_stats["actual"] += 1
        log_anomaly(i, value, 0, "actual")

    for is_anomaly, score, algorithm in zip(
        [is_anomaly_zscore, is_anomaly_sma, is_anomaly_storm],
        [zscore, sma, storm],
        ["zscore", "sma", "storm"],
    ):
        if is_anomaly:
            log_anomaly(i, value, score, algorithm)
        anomaly_stats[algorithm]["true_positive"] += (
            actual_anomaly == True and is_anomaly == True
        )
        anomaly_stats[algorithm]["true_negative"] += (
            actual_anomaly == False and is_anomaly == False
        )
        anomaly_stats[algorithm]["false_positive"] += (
            actual_anomaly == False and is_anomaly == True
        )
        anomaly_stats[algorithm]["false_negative"] += (
            actual_anomaly == True and is_anomaly == False
        )

    # Rescale axes
    for ax, score_data in zip(axes, [ydata, zscore_data, sma_data, storm_data]):
        ax.relim()
        ax.autoscale_view()
        ax.set_xlim(max(-5, i - MAX_PAST_DATA), i + 5)

    return lines + scatters


if __name__ == "__main__":
    detectors = {
        "zscore": AnomalyDetector(
            window_size=WINDOW_SIZE, threshold=THRESHOLD_ZSCORE, algorithm="zscore"
        ),
        "sma": AnomalyDetector(
            window_size=WINDOW_SIZE, threshold=THRESHOLD_SMA, algorithm="sma"
        ),
        "storm": AnomalyDetector(
            window_size=WINDOW_SIZE, k=K_NEIGHBOURS, R=RADIUS_R, algorithm="storm"
        ),
    }

    data_gen = data_stream()
    xdata, ydata = [], []
    zscore_data, sma_data, storm_data = [], [], []

    fig, axes = plt.subplots(4, 1, figsize=(15, 12), num="Anomaly Detection Simulation")
    ax_data, ax_zscore, ax_sma, ax_storm = axes

    # Real-Time Data Plot
    ax_data.set_title("Real-Time Data Stream")
    ax_data.set_xlabel("Time")
    ax_data.set_ylabel("Value")
    (line_data,) = ax_data.plot([], [], lw=1, color="black")
    add_custom_legend(ax_data)

    # Z-score plot
    ax_zscore.set_title("Z-Score-Based Anomaly Detection")
    ax_zscore.set_xlabel("Time")
    ax_zscore.set_ylabel("Z-Score")
    (line_zscore,) = ax_zscore.plot([], [], lw=1, color="orange")
    scatter_zscore = ax_zscore.scatter([], [], color="red", s=50, label="Anomaly")
    ax_zscore.axhline(y=THRESHOLD_ZSCORE, color="green", linestyle="--")
    ax_zscore.axhline(y=-THRESHOLD_ZSCORE, color="green", linestyle="--")
    ax_zscore.legend(loc="upper left")

    # SMA plot
    ax_sma.set_title("SMA-Based Anomaly Detection")
    ax_sma.set_xlabel("Time")
    ax_sma.set_ylabel("SMA Deviation")
    (line_sma,) = ax_sma.plot([], [], lw=1, color="lightblue")
    scatter_sma = ax_sma.scatter([], [], color="red", s=50, label="Anomaly")
    ax_sma.axhline(y=THRESHOLD_SMA, color="green", linestyle="--")
    ax_sma.legend(loc="upper left")

    # STORM plot
    ax_storm.set_title("STORM-Based Anomaly Detection")
    ax_storm.set_xlabel("Time")
    ax_storm.set_ylabel("k-Nearest Neighbors")
    (line_storm,) = ax_storm.plot([], [], lw=1, color="magenta")
    scatter_storm = ax_storm.scatter([], [], color="red", s=50, label="Anomaly")
    ax_storm.axhline(y=K_NEIGHBOURS, color="green", linestyle="--")
    ax_storm.legend(loc="upper left")

    lines = [line_data, line_zscore, line_sma, line_storm]
    scatters = [scatter_zscore, scatter_sma, scatter_storm]

    ani = animation.FuncAnimation(
        fig,
        animate,
        fargs=(
            list(axes),
            detectors,
            data_gen,
            xdata,
            ydata,
            zscore_data,
            sma_data,
            storm_data,
            lines,
            scatters,
        ),
        interval=50,
        cache_frame_data=False,
    )

    plt.tight_layout()

    try:
        plt.show()
    except KeyboardInterrupt:
        print("Animation stopped by the user.")
    except Exception as e:
        print(f"An error occurred during animation: {e}")
    finally:
        print("Anomaly statistics:")
        display_stats(anomaly_stats.copy())
