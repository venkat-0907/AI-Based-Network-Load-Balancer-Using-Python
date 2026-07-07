# AI-Based Network Load Balancer Using Python

Major Project — Python, Scikit-learn

This is a load balancer that uses machine learning to decide which server should get the next request, instead of just going round robin like most basic load balancers do.

- Developed an intelligent load balancer to distribute network traffic across multiple servers.
- Applied machine learning techniques to optimize traffic routing and resource allocation.
- Monitored server performance metrics such as latency and bandwidth for efficient load distribution.
- Visualized network traffic and system performance using Python tools.

I did this because in most networking courses/projects, load balancing algorithms are just fixed rules (round robin, least connections, etc.) and they don't actually look at whether a server is healthy before sending it more work. So I wanted to try replacing that rule with something that learns from server behavior instead.

## What it actually does

There are two models working together here.

The first one looks at a server's current stats (CPU, memory, active connections, latency, bandwidth, error rate) and tags it as Low, Medium or High load. The second one takes those same stats and predicts the actual response time in milliseconds if that server got the next request right now.

The load balancer uses both of these together — it skips servers tagged High load, and among the rest, picks whichever one the model thinks will respond fastest. Every time a new request comes in, all the servers get re-scored, so if one starts struggling mid-way, the next requests just naturally stop going to it.

## Dataset

I looked for a real public dataset of server telemetry to train this on, but most of the ones out there (Google's cluster traces, Alibaba's, etc.) are huge multi-gigabyte files meant for cluster scheduling research, not really usable for a project like this. So I wrote a script that generates realistic server telemetry data instead — it has day/night traffic patterns, weekday vs weekend differences, random traffic spikes, and response times that increase sharply once a server's CPU gets close to maxing out, which is closer to how real servers behave than a straight line would be.

The generated dataset has 6,720 rows across 5 servers, sampled every 15 minutes over two simulated weeks. It's in data/network_load_dataset.csv, and the script that makes it is data/generate_dataset.py. If you want to swap in real server logs later, you just need columns for cpu usage, memory usage, active connections, requests per second, latency, disk io, bandwidth, error rate, and the two targets (response time and load status).

## Models used

The classifier is a RandomForestClassifier and the regressor is a GradientBoostingRegressor, both from scikit-learn. Trained on an 80/20 split. Once trained they get saved as .joblib files in the models folder so you don't need to retrain every time.

Classifier came out at 96.8% accuracy, with precision/recall/F1 all around 96-97% weighted, and ROC-AUC of about 0.998. The regressor got an MAE of 3.47ms, RMSE of 4.65ms, and R² of 0.994. That R² is honestly higher than I expected — probably because the response time formula in my simulated data is fairly smooth, so on messier real-world data I'd expect it to be somewhat lower.

To actually check if this was worth doing, I wrote a separate simulation that runs 500 requests through the same 5 servers twice, once with plain round robin and once with the AI routing, and compares them. Round robin averaged 156ms per request, the AI-based routing averaged 143ms — about an 8% improvement, and it was also a lot more consistent (less variance), mainly because it stopped sending traffic to the two weaker servers once they started struggling, while round robin just kept hitting them equally no matter what.

## Files in here

data/generate_dataset.py builds the training dataset.
src/train_model.py trains both models and saves the graphs, metrics and the models themselves.
src/predict.py is the script you actually use to get a routing decision for a set of servers.
src/load_balancer_simulator.py runs the round robin vs AI comparison.
src/make_screenshots.py just turns the console output into screenshot images for this README.

outputs/graphs has all the charts, outputs/screenshots has the terminal screenshots, outputs/logs has the raw text/csv logs from every run.

## Running it

Clone it, then:

```
pip install -r requirements.txt
python3 data/generate_dataset.py
python3 src/train_model.py
```

Then to get a routing decision for a batch of servers:

```
python3 src/predict.py --batch data/sample_live_servers.csv
```

or check one server manually:

```
python3 src/predict.py --single --cpu 91 --mem 85 --conn 650 --rps 350 --latency 48 --disk 20 --bandwidth 190 --error 3.2
```

and to run the round robin vs AI comparison:

```
python3 src/load_balancer_simulator.py
```

## Graphs

confusion matrix
![](outputs/graphs/confusion_matrix.png)

ROC curves
![](outputs/graphs/roc_curves.png)

feature importance for the classifier
![](outputs/graphs/feature_importance_classifier.png)

feature importance for the regressor
![](outputs/graphs/feature_importance_regressor.png)

predicted vs actual response time
![](outputs/graphs/regression_predicted_vs_actual.png)

residuals for the regressor
![](outputs/graphs/regression_residuals.png)

load class distribution in the dataset
![](outputs/graphs/class_distribution.png)

feature correlation heatmap
![](outputs/graphs/correlation_heatmap.png)

AI routing vs round robin response time comparison
![](outputs/graphs/routing_comparison.png)

how requests got split across servers under each strategy
![](outputs/graphs/requests_per_server.png)

## Screenshots

training run
![](outputs/screenshots/01_training_console.png)

prediction run
![](outputs/screenshots/02_prediction_console.png)

simulation run
![](outputs/screenshots/03_simulator_console.png)

## Tech used

Python, scikit-learn, pandas, numpy, matplotlib, seaborn, joblib.

## What I'd still like to add

Right now this is all simulated — it'd be a lot more interesting to actually wire it up to real servers using something like FastAPI or NGINX with Lua scripts instead of just running the simulation. I'd also like to try a time-series model like an LSTM so it predicts load a few minutes ahead instead of just reacting to current stats, and eventually test it on real production data if I can get access to some.

## License

MIT, do whatever you want with it.
