import networkx as nx
from src.graph_builders.constituency import ConstituencyTreeGenerator

# Test sentence
sentence = "Safety Net (Forbes.com) Forbes.com - After earning a PH.D. in Sociology, Danny Bazil Riley started to work as the general manager at a commercial real estate firm at an annual base salary of #36;70,000. Soon after, a financial planner stopped by his desk to drop off brochures about insurance benefits available through his employer. But, at 32, 'buying insurance was the furthest thing from my mind,' says Riley."

generator = ConstituencyTreeGenerator(device="cpu")
graphs = generator.get_graph([sentence])
G = graphs[0]

# Print nodes sorted by id
nodes_sorted = sorted(G.nodes(data=True), key=lambda x: x[1]['id'])
for key, data in nodes_sorted:
    print(f"Node key: {key}, id: {data['id']}, label: {data['label']}")

# Check ids are strictly sequential
ids = [data['id'] for _, data in nodes_sorted]
assert ids == list(range(len(ids))), f"IDs are not strictly sequential: {ids}"
print("IDs are strictly sequential and contiguous.")
