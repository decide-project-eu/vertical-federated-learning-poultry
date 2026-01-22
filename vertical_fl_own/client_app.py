from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
#
from vertical_fl_own.task import load_data   # If ClientModel is in Task, then also import here

# This is the model that the client runs. Input = features of the data that one client has.
# output = embeddings (I set it to 2)
class ClientModel(nn.Module):
    def __init__(self, input_size):
        super(ClientModel, self).__init__()   # was: super().__init__()
        self.fc = nn.Linear(input_size, 2)  # it was self.fc = nn.Linear(input_size, 4)

    def forward(self, x):
        return self.fc(x)


class FlowerClient(NumPyClient):
    def __init__(self, v_split_id, data, lr):
        self.v_split_id = v_split_id
        # scaler, but I already scaled the data in task!
        # self.data = torch.tensor(StandardScaler().fit_transform(data)).float()
        self.data = torch.tensor(data).float()
        self.model = ClientModel(input_size=self.data.shape[1])
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)  # NOT CHANGED

    #this is not used in VFL:
    def get_parameters(self, config):
        pass

    def fit(self, parameters, config):
        embedding = self.model(self.data)
        return [embedding.detach().numpy()], 1, {} 

    # flower is made for horizontal FL and then you would do evaluation here. But for VFL we 
    # use it for back-propagation
    def evaluate(self, parameters, config):
        self.model.zero_grad()
        embedding = self.model(self.data)
        embedding.backward(torch.from_numpy(parameters[int(self.v_split_id)]))
        self.optimizer.step()
        return 0.0, 1, {}


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    partition, v_split_id = load_data(partition_id, num_partitions=num_partitions)
    lr = context.run_config["learning-rate"]
    return FlowerClient(v_split_id, partition, lr).to_client()


app = ClientApp(
    client_fn=client_fn,
)
