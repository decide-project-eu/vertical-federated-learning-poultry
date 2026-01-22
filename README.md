# vertical-federated-learning-poultry
Proof of concept of vertical federated learning using Dutch poultry database with antibiotics data.

The approach is adapted from this tutorial, using the Message API: https://github.com/adap/flower/tree/main/examples/vertical-fl

In this setup, two clients are simulated. They have data on the same broiler flocks, but each client holds different information (columns). A third party, the server, holds the labels, which in this case is whether or not flocks were treated with antibiotics after week 1. The server and client 2 could be the same data owner, holding information on antibiotic use, but in the current setup they are simulated as two separate data owners.


![https://github.com/decide-project-eu/vertical-federated-learning-poultry/_static/setup.png](https://github.com/decide-project-eu/vertical-federated-learning-poultry/blob/main/_static/setup.png)


For the vertical federated analysis to work, each data owner needs to have information on identical flocks, ordered in the same way. In the current setup, it is assumed that each data owner has a flock ID to achieve this. Data is also cleaned and filtered (free range and organic farms are removed) before clients are simulated.


## Set up the project

This project runs on python 3.11. Using higher versions of Python causes dependency issues with Ray when running the project with the simulation engine. A virtual environment is recommended.

The project should have the following structure:
```
vertical-fl
├── vertical_fl
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your model, training and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
├── data/train.csv
└── README.md
```
**Note: the data train.csv is not shared.**

Install the dependencies defined in pyproject.toml as well as the mlxexample package.
```
pip install -e .
```

## Run with simulation engine
In the terminal, make sure that you are in the vertical-fl-own folder. Start the simulation engine with:

```
flwr run .
```
The output is a summary, including prediction accuracy, sensitivity and specificity.


