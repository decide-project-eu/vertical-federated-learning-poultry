# vertical-federated-learning-poultry
Proof of concept of vertical federated learning using Dutch poultry database with antibiotics data.

The approach is adapted from this tutorial: https://github.com/adap/flower/tree/main/examples/vertical-fl

## Set up the project

This project runs on python 3.11. Using higher versions of Python causes dependency issues with Ray when running the project with the simulation engine. A virtual environment is recommended.

The project should have the following structure:
```
vertical-fl-own
├── vertical_fl_own
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
The output is a summary, including prediction accuracy of each round.



## Run with deployment
To be able to run a project with deployment, the following should be in the pyproject.toml file:
```
[tool.flwr.federations.local-deployment]
address = "127.0.0.1:9093"
insecure = true
```

Open a terminal and create a superlink:
```
flower-superlink --insecure
```
Then, in a new terminal, create a supernode:
```
flower-supernode --insecure --superlink 127.0.0.1:9092 --clientappio-api-address 127.0.0.1:9094 --node-config "partition-id=0 num-partitions=3"
```
Create as many supernodes as the number of clients you have. This example project is based on 3 supernodes. Make sure you change the API address for each supernode (9094, 9095, 9096).
Then, in a new terminal, run the project:
```
flwr run vertical-fl-own local-deployment --stream
```



