import io

import torch
from flwr.app import Array, ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.clientapp import ClientApp

from vertical_fl_own.task import ClientModel, load_data

# Flower ClientApp
app = ClientApp()

# Load data once, then cache
def get_cached_data(context, partition_id, feature_dims):

    if "data_full" not in context.state:
        data = load_data(partition_id, feature_dims)
        context.state["data_full"] = ArrayRecord({"data": Array(data.numpy())})

    return torch.from_numpy(context.state["data_full"]["data"].numpy()).float()


@app.query("generate_embeddings")
def get_emb(msg: Message, context: Context):
    """Generate embeddings."""

    # Read from config
    partition_id = context.node_config["partition-id"]
    feature_splits: str = context.run_config["feature-splits"]
    feature_dims = [int(dim) for dim in feature_splits.split(",")]
    out_feature_dim_clientapp: int = context.run_config["out-feature-dim-clientapp"] #the dimension of embedding vector after forward pass

    # Get cached full data, then split in train-val with indices from server
    data_full = get_cached_data(context, partition_id, feature_dims)
    indices = torch.tensor(msg.content["config"]["indices"])
    data = data_full[indices]
    
    model = ClientModel(
        input_size = data.shape[1],
        out_feat_dim = out_feature_dim_clientapp
    )

    # Load model from state if available
    if model_record := context.state.get("model", None):
        model.load_state_dict(model_record.to_torch_state_dict())

    # Do forward pass
    embedding = model(data)

    # Construct and return reply Message
    model_record = ArrayRecord({
        "embedding": Array(embedding.detach().numpy()),
        "ids": Array(indices.numpy())
    })

    content = RecordDict({
        "arrays": model_record,
        "config": ConfigRecord({"pos": partition_id}),
    })

    return Message(content=content, reply_to=msg)


@app.train("apply_gradients")
def apply_grad(msg: Message, context: Context):
    """Apply gradients to local model."""

    # Read from config
    lr = context.run_config["learning-rate"]
    partition_id = context.node_config["partition-id"]
    out_feature_dim_clientapp: int = context.run_config["out-feature-dim-clientapp"]
    feature_splits: str = context.run_config["feature-splits"]
    feature_dims = [int(dim) for dim in feature_splits.split(",")]
    
    # Get cached full data, then split in train-val with indices from server
    data_full = get_cached_data(context, partition_id, feature_dims)
    indices = torch.tensor(msg.content["config"]["indices"])
    data = data_full[indices]
    
    model = ClientModel(
        input_size = data.shape[1],
        out_feat_dim = out_feature_dim_clientapp
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer.zero_grad()                                       

    # Load model from state if available
    if model_record := context.state.get("model", None):
        model.load_state_dict(model_record.to_torch_state_dict())

    # Load optimizer state from state if available
    if optimizer_state_record := context.state.get("optimizer_state", None):
        buffer = io.BytesIO(optimizer_state_record["serialized"])
        optimizer.load_state_dict(torch.load(buffer))

    # Do forward pass
    embedding = model(data)

    # Get gradients from message and apply them
    embedding.backward(
        torch.from_numpy(msg.content["gradients"]["local-gradients"].numpy())
    )

    optimizer.step()

    # Save updated model in state for next round
    context.state["model"] = ArrayRecord(model.state_dict())
    # (Optional) Save the optimizer state. Not all optimizers have state, but Adam does.
    torch.save(optimizer.state_dict(), buffer := io.BytesIO())
    context.state["optimizer_state"] = ConfigRecord({"serialized": buffer.getvalue()})

    # Construct and return reply Message
    return Message(content=RecordDict(), reply_to=msg)
