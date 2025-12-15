import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))


import torch
from dig.xgraph.method import SubgraphX

from src.gnn_training.training import GNN_Classifier
from torch.utils.data import Dataset
from torch_geometric.data import Data

class MarginalSubgraphDataset(Dataset):
    def __init__(self, data, exclude_mask, include_mask, subgraph_build_func):
        self.num_nodes = data.num_nodes
        self.X = data.x
        self.edge_index = data.edge_index
        self.device = self.X.device

        self.label = data.y
        self.exclude_mask = torch.tensor(exclude_mask).type(torch.float32).to(self.device)
        self.include_mask = torch.tensor(include_mask).type(torch.float32).to(self.device)
        self.subgraph_build_func = subgraph_build_func

    def __len__(self):
        return self.exclude_mask.shape[0]

    def __getitem__(self, idx):
        exclude_graph_X, exclude_graph_edge_index = self.subgraph_build_func(self.X, self.edge_index, self.exclude_mask[idx])
        include_graph_X, include_graph_edge_index = self.subgraph_build_func(self.X, self.edge_index, self.include_mask[idx])
        exclude_data = Data(x=exclude_graph_X, edge_index=exclude_graph_edge_index)
        include_data = Data(x=include_graph_X, edge_index=include_graph_edge_index)
        return exclude_data, include_data

# Monkey-patch DIG's class
import dig.xgraph.method.shapley
dig.xgraph.method.shapley.MarginalSubgraphDataset = MarginalSubgraphDataset

# ---- Minimal test: adjust these to your real data/model ----
# Dummy graph data (replace with a real Data object from your dataset)
x = torch.randn(10, 768)  # 10 nodes, 768 features
y = torch.tensor([1])     # single label
edge_index = torch.randint(0, 10, (2, 20))  # 20 edges

data = Data(x=x, edge_index=edge_index, y=y)

# Dummy model (replace with your trained model and correct dimensions)
model = GNN_Classifier(
    input_dim=768,
    hidden_dim=64,
    output_dim=2,
    num_layers=2,
    dropout=0.1,
    module='GCNConv',
    layer_norm=False,
    residual=False,
    pooling='max'
)

# Put model in eval mode and on CPU for test
model.eval()
device = torch.device('cpu')
model = model.to(device)

data = data.to(device)

class UniversalDataModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        # If called with a single argument that is a Data object
        if len(args) == 1 and hasattr(args[0], 'x') and hasattr(args[0], 'edge_index'):
            data = args[0]
            return self.model(
                x=data.x,
                edge_index=data.edge_index,
                batch=getattr(data, 'batch', None)
            )
        # If called with x, edge_index, (batch)
        elif len(args) >= 2:
            # Pass through to model, assuming same signature
            return self.model(*args, **kwargs)
        # If called with only keyword arguments
        elif 'data' in kwargs:
            data = kwargs['data']
            return self.model(
                x=data.x,
                edge_index=data.edge_index,
                batch=getattr(data, 'batch', None)
            )
        else:
            raise TypeError("Unsupported input signature for model wrapper.")
        
wrapped_model = UniversalDataModelWrapper(model)



# ---- SubgraphX explainer ----
explainer = SubgraphX(
    model=wrapped_model,
    num_classes=2,
    device=device,
    num_hops=1,
    rollout=10,
    min_atoms=1,
    c_puct=1,
    expand_atoms=1,
    local_radius=1,
    sample_num=1,
    save_dir=None
)

# Run explanation (adjust arguments as needed for your DIG version)
explanation, related_pred = explainer.explain(
    x=data.x,
    edge_index=data.edge_index,
    label=int(data.y.item()),
    max_nodes=5
)
print('Explanation:', explanation)
print('Related Prediction:', related_pred)
# ---- End of minimal test ----
