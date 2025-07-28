from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from vertical_fl_own.strategy import Strategy
from vertical_fl_own.task import process_dataset


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour."""

    # Get dataset
    processed_df, _ = process_dataset()

    # Define the strategy
    strategy = Strategy(processed_df["AntibioticsAfterWeek1"].values)

    # Construct ServerConfig
    num_rounds = context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Start Flower server
app = ServerApp(server_fn=server_fn)
