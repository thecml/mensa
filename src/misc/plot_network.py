import matplotlib.pyplot as plt
import networkx as nx

def plot_model_architecture():
    G = nx.DiGraph()
    
    # Define the nodes
    input_node = 'Input Layer\n(nf)'
    shared_layers = 'Shared Layers\n(Linear, BN, Dropout, etc)'
    residual = 'Residual Connection'
    concat = 'Concatenation\n(Input + Residual)'
    
    # Event specific layers
    n_events = 3
    event_nodes = {}
    intermediate_nodes = {}
    output_nodes = {}
    
    for event in range(1, n_events + 1):
        event_nodes[event] = f'Event {event} Branch'
        intermediate_nodes[event] = [f'Event {event} Intermediate Layer {i+1}' for i in range(2)]
        output_nodes[event] = f'Event {event} Output Layer\n(k1, k2)'
    
    # Add nodes to the graph
    G.add_node(input_node)
    G.add_node(shared_layers)
    G.add_node(residual)
    G.add_node(concat)
    
    for event in range(1, n_events + 1):
        G.add_node(event_nodes[event])
        for node in intermediate_nodes[event]:
            G.add_node(node)
        G.add_node(output_nodes[event])
    
    # Define edges
    G.add_edge(input_node, shared_layers)
    G.add_edge(shared_layers, residual)
    G.add_edge(residual, concat)
    
    for event in range(1, n_events + 1):
        G.add_edge(concat, intermediate_nodes[event][0])
        for i in range(len(intermediate_nodes[event]) - 1):
            G.add_edge(intermediate_nodes[event][i], intermediate_nodes[event][i+1])
        G.add_edge(intermediate_nodes[event][-1], output_nodes[event])
    
    # Define node positions
    pos = {}
    pos[input_node] = (0, 3)
    pos[shared_layers] = (0, 2)
    pos[residual] = (0, 1)
    pos[concat] = (0, 0)
    
    for event in range(1, n_events + 1):
        pos[event_nodes[event]] = (event * 2, -1)
        for i, node in enumerate(intermediate_nodes[event]):
            pos[node] = (event * 2, -2 - i)
        pos[output_nodes[event]] = (event * 2, -4)
    
    # Draw the graph
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_size=5000, node_color='lightblue', font_size=10, font_weight='bold', edge_color='gray')
    plt.title('Net2Improved Architecture')
    plt.show()

plot_model_architecture()