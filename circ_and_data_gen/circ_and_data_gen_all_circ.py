import os
import sys
import math
import time
import random
import pickle
import json
import warnings
import numpy as np
from collections import Counter, OrderedDict
from itertools import combinations, chain
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import networkx as nx
import torch
from json import JSONEncoder

import qiskit
from qiskit import Aer, QuantumCircuit, QuantumRegister, ClassicalRegister, execute, transpile, IBMQ
from qiskit.visualization import circuit_drawer, dag_drawer
from qiskit.circuit import Parameter
from qiskit.circuit.random import random_circuit
from qiskit.converters import circuit_to_dag
from qiskit.providers.aer.noise import NoiseModel, ReadoutError, pauli_error, depolarizing_error, errors
from qiskit.providers.aer.noise import coherent_unitary_error
import qiskit.quantum_info as qi
from qiskit.providers.fake_provider import FakeWashingtonV2, FakeCairoV2, FakeLima, FakeHanoi, FakeHanoiV2
from qiskit.quantum_info import Operator, Pauli, SparsePauliOp, Statevector, state_fidelity
from qiskit.providers.aer import AerSimulator
from qiskit_aer.primitives import Estimator

import circ_parser
from data_gen_func import *

# Remove deprecated warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def data_generator(base_dir, exp_dir, args, allowed_gates):

    os.makedirs(exp_dir, exist_ok=True)
    print('generating final dataset to:', exp_dir)

    total_circ_num = args.total_circ_num
    n_qubits = args.qubit_num
    max_gate_length = json.load(open(f'{exp_dir}/config.json', 'r'))['max_gate_length']
    all_cir_dict = pickle.load(open(f'{base_dir}/all_circ_dict.pkl', 'rb'))
    

    print("Simulation Parameters:")
    print("----------------------")
    print(f"Total circuits: {total_circ_num}")
    print(f"Number of qubits: {n_qubits}")
    print(f"Max gate length: {max_gate_length}")
    print(f"Noise type is: {args.noise_type}")

    noise_model = pickle.load(open(f'{exp_dir}/noise_model.pkl', 'rb'))


    for cir_i in range(total_circ_num):

        if os.path.exists(f'{exp_dir}/dataset/data_{cir_i}.pkl'):
            print(f'data {cir_i} existed, skip!')
            continue
        qc = all_cir_dict[cir_i]['circ']
        if args.circ_type == 'trotter':
            tro_step =  all_cir_dict[cir_i]['size']
            print('step size for current circ is,', tro_step)
        else:
            tro_step = qc.size()

        cirq_graph = circuit_to_graph(qc)
        adj_mat = get_adjacency_matrix(cirq_graph)
        node_feat_mat, pe_input = get_gate_matrix(qc, allowed_gates=allowed_gates)

        qc_appended = qc.copy()
        # get edge feature
        x_ev_mat, y_ev_mat, z_ev_mat = get_edge_feat_evs(exp_dir, cir_i, qc_appended, args.qubit_num, noise_model, args)
        edge_feat_dict = get_edge_feature_dict(adj_mat, node_feat_mat, qc, allowed_gates, x_ev_mat, y_ev_mat, z_ev_mat)

        # get node index
        node_idx = [i for i in range(adj_mat.shape[0])]

        # get shortest path between each node pairs
        shortest_path_mat = floyd_warshall(adj_mat, base_dir, cir_i)


        # get ideal ev & noisy ev
        exact_value = pickle.load(open(f'{exp_dir}/no_error_state_vector/data_{cir_i}.pkl', 'rb'))
        noisy_value = pickle.load(open(f'{exp_dir}/error_state_vector/data_{cir_i}.pkl', 'rb'))

        # get positional encoding for each node
        max_depth = max_gate_length*2
        # print(n_qubits+1, max_depth, 'sdaks')
        max_len = max_depth * (n_qubits + 1) * (n_qubits + 1)
        # print(pe_input.shape)
        idx_tensor = get_idx_from_coordinates(pe_input, n_qubits, max_depth)
        pe_module = PositionalEncoding(node_feat_mat.shape[1], max_len=max_len)
        pe_feat_mat = pe_module.get_from_pe(idx_tensor)

        # checkers
        assert pe_feat_mat.shape == node_feat_mat.shape, f'{node_feat_mat.shape, pe_feat_mat.shape}'
        assert adj_mat.shape[0] == node_feat_mat.shape[0]

        # data structure
        save_data = {'x': node_feat_mat, 'adj': adj_mat, 'y': exact_value, 'noise_y': noisy_value, "x_ev_matrix": x_ev_mat, 'y_ev_matrix':y_ev_mat, 'z_ev_matrix': z_ev_mat,
                                 'edge_feature': edge_feat_dict, 'node_idx': node_idx, 'pe_feat': pe_feat_mat.tolist(), 'shortest_path': shortest_path_mat,
                                 'circ_idx': cir_i, 'trotter_step': tro_step}
        os.makedirs(f'{exp_dir}/dataset', exist_ok=True)
        pickle.dump(save_data, open(f'{exp_dir}/dataset/data_{cir_i}.pkl', 'wb'))

def config_gen(args, exp_dir, all_circ_dict):
    if os.path.exists(f'{exp_dir}/config.json'):
        print('Config file existed!')
        return None
    if args.circ_type == 'random':
        max_gate_length = args.max_gate_length # * 2
    else:
        max_gate_length = 0
        for i in range(args.total_circ_num):
            if max_gate_length < all_circ_dict[i]['circ'].size():
                max_gate_length = all_circ_dict[i]['circ'].size()
    config = {
        'max_gate_length': max_gate_length,
        'shots': args.shots,
        'depolar_round': args.depolar_round,
        'pauli_er': args.pauli_er,
        'depolar_er_1q': args.dep_1q,
        'depolar_er_2q': args.dep_2q,
        'cohe_theta': args.cohe_theta,
        'readout_er': args.readout_er
    }
    json.dump(config, open(f'{exp_dir}/config.json', 'w'), indent=4)

from qiskit.quantum_info import Pauli, Operator

def trotter_step(qc, qubits, J, h, delta_t):
    # Z_i Z_j interaction terms (Ising interactions)
    for i in range(len(qubits) - 1):
        qc.cx(qubits[i], qubits[i + 1])
        qc.rz(-2 * J * delta_t, qubits[i + 1])
        qc.cx(qubits[i], qubits[i + 1])
    
    # X_i transverse field terms
    for i in qubits:
        qc.rx(-2 * h * delta_t, i)

def build_ising_trotter_circuit(num_qubits, total_circ_nums, J_bound, h_bound, t_bound, N_list=range(1, 20)):
    all_circ_dict = {}
    circ_idx = 0
    while circ_idx < total_circ_nums:
    # Initialize quantum circuit
        J = random.uniform(J_bound[0], J_bound[1])
        h = random.uniform(h_bound[0], h_bound[1])
        t = random.uniform(t_bound[0], t_bound[1])
        assert total_circ_nums % len(N_list) == 0, f'the total circ num is not divisble for the N list, {total_circ_nums} and {len(N_list)}'
        for N in N_list:
            qc = QuantumCircuit(num_qubits)
            # Time step for each Trotter step
            delta_t = t / N
            # Add Trotter steps
            for _ in range(N):
                trotter_step(qc, range(num_qubits), J, h, delta_t)
            print(f'circ: {circ_idx}, J, h, t, {J} {h} {t}, N_step: {N}')
            all_circ_dict[circ_idx] = {'circ': qc, 'circ_type': 'trotter', 'size': N, 'para': {'J': J, 'h': h, 't': t}}
            circ_idx += 1
    
    return all_circ_dict

def random_circuit_gen(max_num_qubits, max_gate_length, allowed_gates, total_circ_num, qubit_type='max'):
    all_circ_dict = {}
    for i in range(total_circ_num):
        if qubit_type == 'max':
            num_qubits = random.choice(range(2, max_num_qubits+1))
        else:
            assert qubit_type == 'all'
            num_qubits = max_num_qubits
        gate_length = random.choice(range(10, max_gate_length))
        qc = QuantumCircuit(max_num_qubits)
        for _ in range(gate_length):
            gate = random.choice(allowed_gates)
            qubit_indices = random.sample(range(num_qubits), 2)
            angle = random.uniform(0, 2 * math.pi)  # For rotation gates
            if gate == 'x':
                qc.x(qubit_indices[0])
            elif gate == 'y':
                qc.y(qubit_indices[0])
            elif gate == 'z':
                qc.z(qubit_indices[0])
            elif gate == 'h':
                qc.h(qubit_indices[0])
            elif gate == 'rx':
                qc.rx(angle, qubit_indices[0])
            elif gate == 'ry':
                qc.ry(angle, qubit_indices[0])
            elif gate == 'rz':
                qc.rz(angle, qubit_indices[0])
            elif gate == 'rzz':
                qc.rzz(angle, qubit_indices[0], qubit_indices[1])
            elif gate == 'cx':
                qc.cx(qubit_indices[0], qubit_indices[1])
            elif gate == 'cz':
                qc.cz(qubit_indices[0], qubit_indices[1])
            elif gate == 'swap':
                qc.swap(qubit_indices[0], qubit_indices[1])
            elif gate == 'id':
                qc.id(qubit_indices[0])
            elif gate == 'sx':
                qc.sx(qubit_indices[0])
            else:
                assert False, f'{gate} gate has not been implemented.'
        print(f'circ {i}, qc.size: {qc.size()}, qubit_actual: {num_qubits}, qubit_qc: {qc.num_qubits}')
        all_circ_dict[i] = {'circ': qc, 'circ_type': 'random', 'size': qc.size(), 'para': None, 'actual_qubit_num': num_qubits}
        assert qc.size() == gate_length, 'wrong gate length generation'
    return all_circ_dict




def amplify_noise_backend_via_noise_scale(backend, qubit_num, noise_scale=1):
    for i in range(qubit_num):
        backend.target['measure'][(i,)].error = backend.target['measure'][(i,)].error * noise_scale
        backend.target['id'][(i,)].error = backend.target['id'][(i,)].error * noise_scale
        backend.target['sx'][(i,)].error = backend.target['sx'][(i,)].error * noise_scale

    return backend

def circ_generator(base_dir, exp_dir, args, allowed_gates, double_q_gates, single_q_gates, noise_scale=1):
    total_circ_num = args.total_circ_num
    qubit_num = args.qubit_num
    max_gate_length = args.max_gate_length
    max_trotter_steps = args.max_trotter_steps
    qubit_type = args.qubit_type
    circ_type = args.circ_type
    noise_type = args.noise_type
    shots = args.shots
    depolar_round = args.depolar_round

    pauli_er = args.pauli_er
    depolar_er_1q = args.dep_1q
    depolar_er_2q = args.dep_2q
    cohe_theta = args.cohe_theta
    readout_er = args.readout_er
    exp_type = args.exp_type

    print("Simulation Parameters:")
    print("----------------------")
    print(f"Total circuits: {total_circ_num}")
    print(f"Number of qubits: {qubit_num}")
    print(f"Max gate length: {max_gate_length}")
    print(f"Max Trotter steps: {max_trotter_steps}")
    print(f"Qubit type: {qubit_type}")
    print(f"Circuit type: {circ_type}")
    print(f"Noise type: {noise_type}")
    print(f"Exp type: {exp_type}")
    print(f"Shots: {shots}")
    print(f"Rounds for adding depolar: {depolar_round}")
    print("\nNoise Error Rates:")
    print("------------------")
    print(f"Pauli error rate: {pauli_er}")
    print(f"1-qubit depolarizing error rate: {depolar_er_1q}")
    print(f"2-qubit depolarizing error rate: {depolar_er_2q}")
    print(f"Coherent error angle (radians): {cohe_theta}")
    print(f"Readout error rates: {readout_er}")



    if os.path.exists(f'{base_dir}/all_circ_dict.pkl'):
        all_circ_dict = pickle.load(open(f'{base_dir}/all_circ_dict.pkl', 'rb'))
        print(f'{len(all_circ_dict.keys())} circ dict existed, load')
    else:
        print('generating new circs to file:', base_dir)
        if circ_type == 'random':
            print(f'{qubit_num} bit random circs')
            all_circ_dict = random_circuit_gen(qubit_num, max_gate_length, allowed_gates, double_q_gates, total_circ_num, qubit_type=qubit_type)
        elif circ_type == 'trotter':
            J_bound = [0.1, 5]
            h_bound = [0.1, 5]
            t_bound = [1, 10]
            all_circ_dict = build_ising_trotter_circuit(qubit_num, total_circ_num, J_bound, h_bound, t_bound, N_list=range(1, max_trotter_steps))
        else:
            raise ValueError("No available circ generator")
        pickle.dump(all_circ_dict, open(f'{base_dir}/all_circ_dict.pkl', 'wb'))

    if os.path.exists(f'{exp_dir}/noise_model.pkl'):
        print('noise model existed, load')
        noise_model = pickle.load(open(f'{exp_dir}/noise_model.pkl', 'rb'))
    else:
        print("No noise model available, generating")
        # incoherent
        assert noise_type == 'incoherent' or 'all_noise' or 'provider', 'invalid noise type input'
        if noise_type == 'provider':
            print('Customized noise model')
            noise_model = add_depolar_error(allowed_gates, single_q_gates, depolar_er_1q, depolar_er_2q,
                                            random_round=depolar_round, qubit_num=qubit_num)
            noise_model = add_pauli_error(single_q_gates, pauli_er, random_round=len(single_q_gates), noise_model=noise_model)
            if noise_type == 'all_noise':
                noise_model = add_readout_error(readout_er, qubit_num, noise_model=noise_model)
        else:
            print('using fake provider')
            print('current noise scale, ', noise_scale)
            noisy_backend = FakeHanoiV2()
            noisy_backend = amplify_noise_backend_via_noise_scale(noisy_backend, qubit_num, noise_scale=noise_scale)
            noise_model = NoiseModel.from_backend(noisy_backend)

        pickle.dump(noise_model, open(f'{exp_dir}/noise_model.pkl', 'wb'))
    return all_circ_dict, noise_model



if __name__ == "__main__":
    allowed_gates = ['x', 'y', 'z', 'h', 'rx', 'ry', 'rz', 'cx', 'cz', 'swap', 'rzz']
    double_q_gates = ['cx', 'cz', 'swap', 'rzz']
    single_q_gates = [x for x in allowed_gates if x not in double_q_gates]

    args = circ_parser.parse_args()

    base_dir = f'/tmp/{args.exp_type}/{args.qubit_type}_{args.qubit_num}q/{args.circ_type}'
    exp_dir = f'{base_dir}/{args.noise_type}'
    os.makedirs(exp_dir, exist_ok=True)

    
    if args.exp_type != 'amplify':
        all_circ_dict, noise_model= circ_generator(base_dir, exp_dir, args, allowed_gates, double_q_gates, single_q_gates)
        
        config_gen(args, exp_dir, all_circ_dict)
        for i in range(args.total_circ_num):
            ideal_noisy_ev_gen(all_circ_dict[i]['circ'], i, noise_model, args.shots, exp_dir)
            
        data_generator(base_dir, exp_dir, args, allowed_gates)
    else:
        print(f'handling special setting: {args.exp_type}')
        for noise_scale in range(1, 4):
            exp_dir = f'{base_dir}/{args.noise_type}/noise_scale_{noise_scale}'
            os.makedirs(exp_dir, exist_ok=True)
            all_circ_dict, noise_model= circ_generator(base_dir, exp_dir, args, allowed_gates, double_q_gates, single_q_gates, noise_scale=noise_scale)
            for i in range(args.total_circ_num):
                ideal_noisy_ev_gen(all_circ_dict[i]['circ'], i, noise_model, args.shots, exp_dir)
            config_gen(args, exp_dir, all_circ_dict)
            data_generator(base_dir, exp_dir, args, allowed_gates)
        
            
        