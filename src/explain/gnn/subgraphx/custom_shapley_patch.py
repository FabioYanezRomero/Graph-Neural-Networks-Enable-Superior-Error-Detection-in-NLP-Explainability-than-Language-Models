import torch
from torch_geometric.data import Dataset

class CustomMarginalSubgraphDataset(Dataset):
    def __init__(self, data, exclude_mask, include_mask, subgraph_build_func):
        super().__init__()
        self.num_nodes = data.num_nodes
        self.X = data.x
        self.edge_index = data.edge_index
        self.device = self.X.device
        self.label = data.y
        self.exclude_mask = torch.tensor(exclude_mask).type(torch.float32).to(self.device)
        self.include_mask = torch.tensor(include_mask).type(torch.float32).to(self.device)
        self.subgraph_build_func = subgraph_build_func
        self.length = self.exclude_mask.shape[0]

    def __len__(self):
        return self.length

    def len(self):
        return self.__len__()

    def __getitem__(self, idx):
        return self.get(idx)

    def get(self, idx):
        # Return exclude_data and include_data for the idx-th mask
        exclude_mask = self.exclude_mask[idx]
        include_mask = self.include_mask[idx]
        exclude_data = self.subgraph_build_func(self.X, self.edge_index, exclude_mask)
        include_data = self.subgraph_build_func(self.X, self.edge_index, include_mask)
        return exclude_data, include_data

def value_func_wrapper(model):
    """
    Returns a value function that unpacks a Data object and calls the model with the correct arguments.
    Use this with DIG explainers and marginal_contribution.
    """
    def value_func(data):
        # Handles both single Data and batched Data
        if hasattr(data, 'batch'):
            return model(data.x, data.edge_index, data.batch)
        else:
            return model(data.x, data.edge_index)
    return value_func

def patch_marginal_contribution():
    """
    Monkey-patch dig.xgraph.method.shapley.marginal_contribution to use CustomMarginalSubgraphDataset.
    Call this at the start of your pipeline.
    """
    import dig.xgraph.method.shapley as shapley_mod
    def marginal_contribution(data, exclude_mask, include_mask, model, subgraph_build_func):
        value_func = value_func_wrapper(model)
        dataset = CustomMarginalSubgraphDataset(data, exclude_mask, include_mask, subgraph_build_func)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
        marginal_contribution_list = []
        for exclude_data, include_data in dataloader:
            exclude_values = value_func(exclude_data)
            include_values = value_func(include_data)
            margin_values = include_values - exclude_values
            marginal_contribution_list.append(margin_values)
        marginal_contributions = torch.cat(marginal_contribution_list, dim=0)
        return marginal_contributions
    shapley_mod.marginal_contribution = marginal_contribution
