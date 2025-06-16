import json
import os
import random
import pickle
import networkx as nx

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit, QuantumRegister, transpile, execute
import qiskit
from qiskit.converters import circuit_to_dag
from qiskit.visualization import dag_drawer
from qiskit.quantum_info import Operator, Pauli, SparsePauliOp
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import (
    NoiseModel, ReadoutError, pauli_error, depolarizing_error
)
from qiskit import IBMQ, Aer
from qiskit.providers.fake_provider import (
    FakeWashingtonV2, FakeCairoV2, FakeHanoi, FakeHanoiV2
)
from qiskit_aer.primitives import Estimator


from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Pauli
from qiskit.result import marginal_counts

import torch
from torch_geometric.data import InMemoryDataset, Data
import torch, math
from torch.autograd.variable import Variable
import torch.nn as nn


import numpy as np
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import RemoveBarriers

from qiskit.providers.aer.noise import NoiseModel, ReadoutError, pauli_error, depolarizing_error, errors

def add_depolar_error(allowed_gates, single_qubit_gates, p_depolarize_single=8e-4, p_depolarize_double=0.004, random_round=0, noise_model=None,
                      qubit_num=6):
    if noise_model is None:
        noise_model = NoiseModel()
    assert random_round <= len(allowed_gates), 'random round should be smaller than the number of allowed gates'
    random_e_single= random.uniform(0, p_depolarize_single)
    random_e_double= random.uniform(0, p_depolarize_double)
    if qubit_num > 1:
        gate_name = random.sample(allowed_gates, random_round)
    else:
        print('we are operating on a single qubit circuit')
        gate_name = random.sample(single_qubit_gates, random_round)
    for gate_i in gate_name:
        if gate_i in single_qubit_gates:
            depolarizing_error_1q = errors.depolarizing_error(random_e_single, 1)
            noise_model.add_all_qubit_quantum_error(depolarizing_error_1q, gate_i)
        else: 
            assert qubit_num > 1, 'qubit num should be larger than 1'
            depolarizing_error_2q = errors.depolarizing_error(random_e_double, 2)
            noise_model.add_quantum_error(depolarizing_error_2q, gate_i, [0, 1])
    return noise_model

def add_pauli_error(single_qubit_gates, error_rate=8e-4, random_round=0, noise_model=None):
    assert noise_model is not None, f'pauli noise should be added to incoherent noise'

    assert random_round <= len(single_qubit_gates), 'random round should be smaller than the number of allowed gates'
    random_error_rate = random.uniform(0, error_rate)
    random_gate_list = random.sample(single_qubit_gates, random_round)
    for gate_i in random_gate_list:
        pauli_error_1q = errors.pauli_error([('X', random_error_rate), ('I', 1 - random_error_rate)])
        noise_model.add_all_qubit_quantum_error(pauli_error_1q, gate_i)
    return noise_model

def add_readout_error(readout, NUM_QUBITS, noise_model=None):
    assert noise_model is not None, f'readout noise should be added to incoherent noise'

    assert readout != [0, 0]
    readout_error = ReadoutError([[1-readout[0], readout[0]], [readout[1], 1-readout[1]]])
    for qubit in range(NUM_QUBITS):
        noise_model.add_readout_error(readout_error, [qubit])

    return noise_model

def expectation_value_fast(state_counts):
    total_counts = sum(state_counts.values())
    expectation = 0
    for state, count in state_counts.items():
        state = state.split(' ')[-1]
        state_int = int(state, 2)
        num_ones = bin(state_int).count('1')
        outcome = (-1) ** num_ones
        probability = count / total_counts
        expectation += outcome * probability
    return expectation

def normalize_dict(x_dict):
    total_error_counts = sum(x_dict.values())
    return {state: count / total_error_counts for state, count in x_dict.items()}




def get_gate_matrix(qc, allowed_gates=['cx', 'rx', 'ry', 'rz', 'cz', 'x', 'y', 'z', 'h', 'swap', 'rzz']):
    """
    Generate a gate matrix for the given quantum circuit, including layer information for each gate.
    Gate types and qubit indices are encoded, with an additional column for gate parameters (theta).
    """
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import RemoveBarriers
    pass_manager = PassManager(RemoveBarriers())
    qc_no_barrier = pass_manager.run(qc)
    dag = circuit_to_dag(qc_no_barrier)
    gate_data = qc_no_barrier.data
    
    h = len(gate_data) 
    m = len(allowed_gates)
    
    gate_matrix = np.zeros((h + qc.num_qubits * 2 + 1, m + qc.num_qubits + 2 + 1)) 

    op_offset = {gate: i + 1 for i, gate in enumerate(allowed_gates)}

    layers = dag.layers()
    
    layer_idx = 1
    gate_idx = qc.num_qubits
    pe_input = np.zeros((h + qc.num_qubits * 2 + 1, 3))

    for i in range(qc.num_qubits):
        pe_input[i] = [i+1, 0, layer_idx]

    layer_idx += 1
    for layer in layers:
        for node in layer['graph'].topological_op_nodes():
            if node.name in allowed_gates:
                g_name = node.name
                if qc.num_qubits == 50:
                    qubits = [q._index for q in node.qargs]
                else:
                    qubits = [q.index for q in node.qargs]
                param = node.op.params[0] if node.op.params else 0

                for qubit in qubits:
                    qubit_idx = qubit + m + 2
                    gate_matrix[gate_idx, qubit_idx] = 1
                gate_matrix[gate_idx, op_offset[g_name]] = 1
                gate_matrix[gate_idx, -1] = param / (2 * np.pi)

                if len(qubits) == 1:
                    pe_input[gate_idx] = [qubits[0]+1, 0, layer_idx]
                else:
                    assert len(qubits) == 2
                    pe_input[gate_idx] = [qubits[0]+1, qubits[1]+1, layer_idx]
                    
                gate_idx += 1
            else:
                assert False, f'Allowed gates does not include all the gate types of the circ, {node.name}.'
        layer_idx += 1
    
    print('current layer', layer_idx)
    for i in range(qc.num_qubits):
        gate_matrix[i, 0] = 1
        gate_matrix[h + qc.num_qubits + i, m + 1] = 1
        
        gate_matrix[i, i + m + 2] = 1
        gate_matrix[h + qc.num_qubits + i, i + m + 2] = 1

        pe_input[gate_idx] = [i+1, 0, layer_idx] 
        gate_idx += 1
    assert gate_idx == gate_matrix.shape[0] - 1, f'not all cols in gate mat has been assigned values'

    for i in range(gate_matrix.shape[1]):
        gate_matrix[gate_idx, i] = random.random()
    pe_input[gate_idx] = [0, 0, 0]
    
    return gate_matrix, pe_input

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:int(pe[:,1::2].shape[1])])
        self.register_buffer('pe', pe)
        
    def get_from_pe(self, idx_tensor):
        idx_tensor = idx_tensor.reshape(-1)
        
        return Variable(self.pe[idx_tensor], requires_grad=False)
        
def get_idx_from_coordinates(pe_input, num_bits, max_depth):
    return pe_input[:,0] * (num_bits + 1) * max_depth + pe_input[:, 1] * max_depth + pe_input[:, 2]



def circuit_to_graph(circuit):
    G = nx.DiGraph()

    for i, qubit in enumerate(circuit.qubits):
        start_node = f"q[{i}]_start"
        G.add_node(start_node, color='green', label=f"q[{i}]")
    
    qubit_last_node = {f"q[{i}]": f"q[{i}]_start" for i in range(len(circuit.qubits))}

    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import RemoveBarriers
    pass_manager = PassManager(RemoveBarriers())
    qc_no_barrier = pass_manager.run(circuit)
    dag = circuit_to_dag(qc_no_barrier)
    
    gate_data = qc_no_barrier.data
    layers = dag.layers()
    idx = qc_no_barrier.num_qubits
    for layer in layers:
        for gate in layer['graph'].topological_op_nodes():    
            gate_name = gate.name
            if circuit.num_qubits == 50:
                qargs = [f"q[{q._index}]" for q in gate.qargs]
            else:
                qargs = [f"q[{q.index}]" for q in gate.qargs]
            gate_node = f"{gate_name}_{idx}"
            G.add_node(gate_node, color='blue', label=gate_name)

            for q in qargs:
                if not G.has_edge(qubit_last_node[q], gate_node):
                    G.add_edge(qubit_last_node[q], gate_node, label=q)
                qubit_last_node[q] = gate_node
            idx += 1

    for i, qubit in enumerate(circuit.qubits):
        end_node = f"q[{i}]_end"
        G.add_node(end_node, color='red', label=f"q[{i}]")
        last_node = qubit_last_node[f"q[{i}]"]
        G.add_edge(last_node, end_node, label=f"q[{i}]")

    new_node = 'special_node'
    G.add_node(new_node, color='yellow')

    for node in G.nodes:
        if node != new_node:
            G.add_edge(node, new_node)

    return G



def get_adjacency_matrix(G):
    adj_matrix = nx.adjacency_matrix(G).todense()
    return np.array(adj_matrix)


def compute_pauli_x_ev_prob(qc, num_qubits, noise_model, shots=1000):
    """Compute the expectation value of Pauli-X on each qubit in a circuit.
    
    Args:
        num_qubits (int): Number of qubits in the circuit.
        qc (QuantumCircuit): The quantum circuit to run.
        shots (int): Number of shots to run the circuit.
        
    Returns:
        ev_vector (list): A list of expectation values for X on each qubit.
    """
    ev_vector = []
    backend = Aer.get_backend('qasm_simulator', noise_model=noise_model)

    for qubit in range(num_qubits):
        qc_copy = qc.copy()
        
        qc_copy.h(qubit)
        
        qc_copy.measure_all()

        counts = execute(qc_copy, backend, shots=shots, noise_model=noise_model).result().get_counts()
        marginalized_counts = marginal_counts(counts, indices=[qubit])

        P_plus = marginalized_counts.get('0', 0) / shots
        P_minus = marginalized_counts.get('1', 0) / shots

        ev_x = P_plus - P_minus
        ev_vector.append(ev_x)
    
    return ev_vector

def get_edge_feat_evs(exp_dir, cir_i, qc, n_qubits, noise_model, args):
    if os.path.exists(f'{exp_dir}/evs_feat/xi_ev/data_{cir_i}.pkl'):
        print(f'xi for circ {cir_i} existed, skip!' )
        x_ev_mat = pickle.load(open(f'{exp_dir}/evs_feat/xi_ev/data_{cir_i}.pkl', 'rb'))
    else:
        print('generating xi')
        x_ev_mat = compute_pauli_x_ev_prob(qc, n_qubits, noise_model=noise_model, shots=args.shots)
        os.makedirs(f'{exp_dir}/evs_feat/xi_ev', exist_ok=True)
        pickle.dump(x_ev_mat, open(f'{exp_dir}/evs_feat/xi_ev/data_{cir_i}.pkl', 'wb'))

    if os.path.exists(f'{exp_dir}/evs_feat/yi_ev/data_{cir_i}.pkl'):
        print(f'yi for circ {cir_i} existed, skip!' )
        y_ev_mat = pickle.load(open(f'{exp_dir}/evs_feat/yi_ev/data_{cir_i}.pkl', 'rb'))
    else:
        print('generating yi')
        y_ev_mat = compute_pauli_y_ev_prob(qc, n_qubits, noise_model=noise_model, shots=args.shots)
        os.makedirs(f'{exp_dir}/evs_feat/yi_ev', exist_ok=True)
        pickle.dump(y_ev_mat, open(f'{exp_dir}/evs_feat/yi_ev/data_{cir_i}.pkl', 'wb'))

    if os.path.exists(f'{exp_dir}/evs_feat/zi_ev/data_{cir_i}.pkl'):
        print(f'zi for circ {cir_i} existed, skip!' )
        z_ev_mat = pickle.load(open(f'{exp_dir}/evs_feat/zi_ev/data_{cir_i}.pkl', 'rb'))
    else:
        print('generating zi')
        z_ev_mat = compute_pauli_z_ev_prob(qc, n_qubits, noise_model=noise_model, shots=args.shots)
        os.makedirs(f'{exp_dir}/evs_feat/zi_ev', exist_ok=True)
        pickle.dump(z_ev_mat, open(f'{exp_dir}/evs_feat/zi_ev/data_{cir_i}.pkl', 'wb'))

    return x_ev_mat, y_ev_mat, z_ev_mat


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    


def ideal_executor(circuit, error_saving_path, circuit_i):
    cir_cp = circuit.copy()
    cir_cp.save_state()
    job = qiskit.execute(
        experiments=cir_cp,
        backend=AerSimulator(method='statevector'),
    )
    density_mat = job.result().get_statevector()
    ZZ = Operator(Pauli('Z'*(circuit.num_qubits)))
    ideal_ev = density_mat.expectation_value(ZZ).real
    os.makedirs(f'{error_saving_path}/no_error_state_vector', exist_ok=True)
    pickle.dump(ideal_ev, open(f'{error_saving_path}/no_error_state_vector/data_{circuit_i}.pkl', 'wb'))
    print(f'circuit {circuit_i}, ideal ev --> {ideal_ev}')

    return density_mat.expectation_value(ZZ), density_mat.probabilities_dict()

def noisy_executor(noise_model, qc, shots, error_saving_path, circuit_i):
    qc_to_measured = qc.copy()
    qc_to_measured.measure_all()
    os.makedirs(f'{error_saving_path}/error_state_vector', exist_ok=True)
    os.makedirs(f'{error_saving_path}/error_counts', exist_ok=True)
    
    simulator = Aer.get_backend('qasm_simulator', noise_model=noise_model)
    counts_noise = execute(qc_to_measured, simulator, noise_model=noise_model, shots=shots).result().get_counts()
    noisy_value = expectation_value_fast(counts_noise)
    print(f'circuit {circuit_i}, noisy ev -->{noisy_value}' )
    pickle.dump(noisy_value, open(f'{error_saving_path}/error_state_vector/data_{circuit_i}.pkl', 'wb'))



def ideal_noisy_ev_gen(qc, circ_idx, noise_model, shots, exp_dir):
    if os.path.exists(f'{exp_dir}/error_state_vector/data_{circ_idx}.pkl'):
        print(f'Noisy and ideal ev for circ {circ_idx} existed, skip!')
        return None
        
    ideal_executor(qc, error_saving_path=exp_dir, circuit_i=circ_idx)
    noisy_executor(noise_model=noise_model, qc=qc, shots=shots, error_saving_path=exp_dir, circuit_i=circ_idx)



def compute_pauli_y_ev_prob(circuit, num_qubits, noise_model, shots=1000):
    """Compute the expectation values for Pauli-Y applied to each qubit.
    
    Args:
        circuit (QuantumCircuit): The quantum circuit to run.
        num_qubits (int): The number of qubits in the circuit.
        shots (int): The number of shots for the measurement (default: 8000).
        
    Returns:
        ev_vector (list): A list of expectation values for Y applied to each qubit.
    """
    ev_vector = []
    backend = Aer.get_backend('qasm_simulator', noise_model=noise_model)

    for qubit in range(num_qubits): 
        qc_copy = circuit.copy()

        qc_copy.sdg(qubit)
        qc_copy.h(qubit)

        qc_copy.measure_all()

        counts = execute(qc_copy, backend, shots=shots, noise_model=noise_model).result().get_counts()

        marginalized_counts = marginal_counts(counts, indices=[qubit])

        P_0 = marginalized_counts.get('0', 0) / shots
        P_1 = marginalized_counts.get('1', 0) / shots

        ev_y = P_0 - P_1
        ev_vector.append(ev_y)

    return ev_vector

def compute_pauli_z_ev_prob(circuit, num_qubits, noise_model, shots=1000):
    """Compute the expectation values for Pauli-Z applied to each qubit.
    
    Args:
        circuit (QuantumCircuit): The quantum circuit to run.
        num_qubits (int): The number of qubits in the circuit.
        shots (int): The number of shots for the measurement (default: 8000).
        
    Returns:
        ev_vector (list): A list of expectation values for Z applied to each qubit.
    """
    ev_vector = []
    backend = Aer.get_backend('qasm_simulator', noise_model=noise_model)
    qc = circuit.copy()
    qc.measure_all()

    counts = execute(qc, backend, shots=shots, noise_model=noise_model).result().get_counts()
    for qubit in range(num_qubits):
        marginalized_counts = marginal_counts(counts, indices=[qubit])
        P_0 = marginalized_counts.get('0', 0) / shots
        P_1 = marginalized_counts.get('1', 0) / shots

        ev_z = P_0 - P_1
        ev_vector.append(ev_z)

    return ev_vector



def floyd_warshall(adj_matrix, base_dir, cir_i):

    if os.path.exists(f'{base_dir}/floyd_warshall/data_{cir_i}.pkl'):
        print('loading shortest path for circ', cir_i)
        dist = pickle.load(open(f'{base_dir}/floyd_warshall/data_{cir_i}.pkl', 'rb'))
        return dist

    print('no shortest path available for circ', cir_i, 'generating')

    n = len(adj_matrix)
    
    dist = np.copy(adj_matrix)
    
    dist[dist == 0] = 510
    np.fill_diagonal(dist, 0)
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    
    os.makedirs(f'{base_dir}/floyd_warshall', exist_ok=True)
    pickle.dump(dist, open(f'{base_dir}/floyd_warshall/data_{cir_i}.pkl', 'wb'))
    return dist

def get_qubits_operated_on(node, node_feature_matrix, gate_feature_offset, num_qubits):
    return [q for q in range(num_qubits) if node_feature_matrix[node, gate_feature_offset + q] == 1]

def get_edge_feature_dict(adj_matrix, node_feature_matrix, qc, allowed_gates, x_ev_matrix, y_ev_matrix, z_ev_matrix):
    """
    Construct a dictionary where keys are edges (node1, node2) and values are the edge features.

    Args:
        adj_matrix (np.ndarray): The adjacency matrix of the graph.
        node_feature_matrix (np.ndarray): Node feature matrix containing qubit information.
        qc (QuantumCircuit): The quantum circuit.
        allowed_gates (list): List of allowed gates.

    Returns:
        dict: A dictionary with keys as (node1, node2) pairs and values as [feature1, feature2, feature3].
    """
    edge_feature_dict = {}
    num_nodes = adj_matrix.shape[0]
    num_qubits = qc.num_qubits
    gate_feature_offset = len(allowed_gates) + 2

    for node1 in range(num_nodes):
        for node2 in range(num_nodes):
            if adj_matrix[node1][node2] == 1:  # Edge exists
                edge_features = [0] * 6  # Initialize with 6 zero features

                assert node1 != num_nodes - 1, 'no edge comes from the special node'
                if node2 == num_nodes - 1:
                    edge_feature_dict[(node1, node2)] = edge_features
                    continue
                
                qubits_node1 = get_qubits_operated_on(node1, node_feature_matrix, gate_feature_offset, num_qubits)
                qubits_node2 = get_qubits_operated_on(node2, node_feature_matrix, gate_feature_offset, num_qubits)

                assert len(qubits_node1) in [1, 2], f'Invalid node feature at node1: {qubits_node1}'
                assert len(qubits_node2) in [1, 2], f'Invalid node feature at node2: {qubits_node2}'

                intersection = set(qubits_node1) & set(qubits_node2)
                assert len(intersection) in [1, 2], f'Invalid node feature at intersection: {qubits_node1}, {qubits_node2}'
                for i, ii in enumerate(intersection):
                    edge_features[i*3:i*3+3] = [x_ev_matrix[ii], y_ev_matrix[ii], z_ev_matrix[ii]]
                edge_feature_dict[(node1, node2)] = edge_features

    assert len(edge_feature_dict) == np.sum(adj_matrix)
    
    return edge_feature_dict

def expectation_value_fast(state_counts):
    total_counts = sum(state_counts.values())
    expectation = 0
    for state, count in state_counts.items():
        state = state.split(' ')[-1]
        state_int = int(state, 2)
        num_ones = bin(state_int).count('1')
        outcome = (-1) ** num_ones
        probability = count / total_counts
        expectation += outcome * probability

    return expectation