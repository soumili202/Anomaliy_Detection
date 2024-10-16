## Running the Script

### Prerequisites
Ensure you have Python 3.1x installed along with the following required libraries:
```
numpy
matplotlib
tabulate
```
You can install them using:
```bash
pip install -r requirements.txt
```

### Running the Simulation
1. Download or clone the script.
2. Open a terminal in the project directory.
3. Run the Python script:
   ```bash
   python main.py
   ```
4. The simulation will start and display four real-time plots showing the incoming data stream and the anomalies detected by each algorithm.

### Outputs

- **Real-time plots**: Shows the real-time data stream along with detected anomalies using Z-score, SMA, and STORM.
- **Console output**: Displays the confusion matrix and performance metrics for each algorithm.
- **Log and CSV files**: Detected anomalies are logged in `anomalies.log` and written to `anomalies.csv` for later analysis.

### Explanation of Algorithms
1. **Z-Score**: Measures the number of standard deviations a point deviates from the mean. A point is flagged as an anomaly if its Z-score exceeds a specified threshold.
2. **Simple Moving Average (SMA)**: Flags points that deviate significantly from the moving average of a recent window of values.
3. **STORM**: A distance-based anomaly detection method that identifies outliers by calculating the number of neighboring points within a specified radius.
