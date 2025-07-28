import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays


class ServerModel(nn.Module):
    # this is the model that the server side is running, so the input is the embeddings from the clients
    def __init__(self) -> None:  # was: def __init__(self, input_size):
        super(ServerModel, self).__init__()
        self.fc = nn.Linear(6, 1)  # was: (input_size, 1). If I do that then use self.model = ServerModel(12) or correct number in class Strategy
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        return self.sigmoid(x)


class Strategy(fl.server.strategy.FedAvg):
    def __init__(self, labels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = ServerModel()  # was ServerModel(12) but mine is 6?
            # but in Youtube version this is empty
        # convert PyTorch tensors to Numpy Array:
        self.initial_parameters = ndarrays_to_parameters(
            [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        )
        #self.optimizer = torch.optim.LBFGS(self.model.parameters(), lr=1) # was: optim.SGD(self.model.parameters(), lr=0.01)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()
        self.label = torch.tensor(labels).float().unsqueeze(1)

    def aggregate_fit(self, rnd, results, failures):
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        embedding_results = [
            torch.from_numpy(parameters_to_ndarrays(fit_res.parameters)[0])
            for _, fit_res in results
        ]
       
        embeddings_aggregated = torch.cat(embedding_results, dim=1)
        embedding_server = embeddings_aggregated.detach().requires_grad_()

        # OWN ADDITION (BECAUSE I USE A DIFFERENT OPTIMISER)
        # If using SGD then outcomment this block and use block between # ---
        # def closure():
        #     self.optimizer.zero_grad()
        #     output = self.model(embedding_server)
        #     loss = self.criterion(output, self.label)
        #     loss.backward()
        #     return loss
        #     # Perform optimizer step with the closure
        # self.optimizer.step(closure)
    
        # ---
	    # embeddings_aggregated is a tensor that holds the aggregated parameters.
	    # .detach() ensures that this tensor is not part of the computation graph (which is important for optimization).
	    #.requires_grad_() ensures that this tensor will track gradients during backpropagation (as we need to compute the gradient to update the model during training).
        output = self.model(embedding_server)
        loss = self.criterion(output, self.label)
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()
        # ---

        # gradients to send back to each client
        grads = embedding_server.grad.split([2, 2, 2], dim=1)
        #number of features per client: 3, 6, 7 (?)
        # # DEBUGGING
        # for idx, grad in enumerate(grads):
        #     print(f"Gradient shape for client {idx}: {grad.shape}")  
        np_grads = [grad.numpy() for grad in grads]
        parameters_aggregated = ndarrays_to_parameters(np_grads)

        # evaluation
        with torch.no_grad():
            correct = 0
            output = self.model(embedding_server)
            predicted = (output > 0.5).float()

            correct += (predicted == self.label).sum().item()

            accuracy = correct / len(self.label) * 100

        metrics_aggregated = {"accuracy": accuracy}

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        rnd,
        results,
        failures,
    ):
        return None, {}
        # this is because flower is made with horizontal FL in mind. We do evaluation on server side
        # instead of client side. So client evaluation sends back None.
