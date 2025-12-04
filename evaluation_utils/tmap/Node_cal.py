import numpy as np

def analyze_node_distribution(types, s, t):

    ones_nodes = np.where(types == 1)[0]


    zero_nodes = np.where(types == 0)[0]


    neighbors = {i: set() for i in range(len(types))}
    

    for start, end in zip(s, t):
        neighbors[start].add(end)
        neighbors[end].add(start)
    

    count_ones = len(ones_nodes)


    count_one_connected_to_one_zero = 0


    count_one_connected_to_only_ones = 0

    for node in ones_nodes:

        node_neighbors = neighbors[node]
        
 
        zero_neighbors = len([n for n in node_neighbors if types[n] == 0])
        
        if zero_neighbors == 0:
            count_one_connected_to_only_ones += 1
        elif zero_neighbors == 1:
            count_one_connected_to_one_zero += 1

    return count_ones, count_one_connected_to_one_zero, count_one_connected_to_only_ones

# example
types = np.array([0,0,1,1])
s = np.array([0,1,0,2])
t = np.array([1,2,2,3])


count_ones, count_one_connected_to_one_zero, count_one_connected_to_only_ones = analyze_node_distribution(types, s, t)


print(f"Node total: {count_ones}")
print(f"Node A: {count_ones-count_one_connected_to_only_ones}")
print(f"Node B: {count_one_connected_to_only_ones}")
