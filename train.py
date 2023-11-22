import argparse
import os
import sys
import time
import dgl
import numpy as np
import torch
import torch.nn.functional as F
from model import HTCNS
from utils.data import load_data
from utils.pytorchtools import EarlyStopping
from tqdm import tqdm

def calculate_score(adj_list, node_types, node_sequence):
    num_unique_types = len(set([node_types[node] for node in node_sequence]))
    counts = np.bincount([node_types[node] for node in node_sequence])
    probabilities = counts / np.sum(counts)
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))  
    last_node = node_sequence[-1]
    degree_inverse_log = -np.log2(adj_list.getrow(last_node).sum() + 1) 
    score = num_unique_types * degree_inverse_log * entropy
    return score

def sea_score(adj_list, node_types, current_node, target_node, depth, current_sequence, max_score, best_node, visited):
    if current_node == target_node:
        score = calculate_score(adj_list, node_types, current_sequence)
        if score > max_score[0]:
            max_score[0] = score
            best_node[0] = current_sequence[-1]
        return

    if depth == 0:
        return

    neighbors = adj_list.getrow(current_node).indices

    for neighbor in neighbors:
        if neighbor not in visited:
            current_sequence.append(neighbor)
            visited.add(neighbor)
            sea_score(adj_list, node_types, neighbor, target_node, depth - 1, current_sequence, max_score, best_node, visited)
            current_sequence.pop()
            visited.remove(neighbor)

def sample_sequence_with_score(adj_list, node_types, max_sequence_length, nodetmp_num):
    N = adj_list.shape[0]
    length = max_sequence_length
    sequence_matrix = np.zeros((nodetmp_num, length), dtype=int)

    for n in tqdm(range(nodetmp_num)):
        node_sequence = [n]
        visited = {n}

        neighbors = adj_list.getrow(n).indices
        for neighbor in neighbors:
            if len(node_sequence) < length:
                node_sequence.append(neighbor)
                visited.add(neighbor)

        while len(node_sequence) < length:
            max_score = [float('-inf')]
            best_node = [-1]

            for node in node_sequence:
                sea_score(adj_list, node_types, node, n, 2, [node], max_score, best_node, visited)

            if best_node[0] == -1:
                break

            node_sequence.append(best_node[0])
            visited.add(best_node[0])

        sequence_matrix[n] = node_sequence

    return sequence_matrix

sys.path.append('utils/')

def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)

def run_HTCNS(args):

    if not os.path.exists('checkpoint/'):
        os.makedirs('checkpoint/')

    feats_type = args.feats_type
    features_list, adjM, labels, train_val_test_idx, dl = load_data(args.dataset)
    device = torch.device('cuda:' + str(args.device)
                          if torch.cuda.is_available() else 'cpu')
    features_list = [mat2tensor(features).to(device)
                     for features in features_list]
    node_cnt = [features.shape[0] for features in features_list]
    sum_node = 0
    for x in node_cnt:
        sum_node += x
    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1 or feats_type == 5:
        save = 0 if feats_type == 1 else 2
        in_dims = []
        for i in range(0, len(features_list)):
            if i == save:
                in_dims.append(features_list[i].shape[1])
            else:
                in_dims.append(10)
                features_list[i] = torch.zeros(
                    (features_list[i].shape[0], 10)).to(device)
    elif feats_type == 2 or feats_type == 4:
        save = feats_type - 2
        in_dims = [features.shape[0] for features in features_list]
        for i in range(0, len(features_list)):
            if i == save:
                in_dims[i] = features_list[i].shape[1]
                continue
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(
                indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(
                indices, values, torch.Size([dim, dim])).to(device)

    labels = torch.LongTensor(labels).to(device)
    train_idx = train_val_test_idx['train_idx']
    train_idx = np.sort(train_idx)
    val_idx = train_val_test_idx['val_idx']
    val_idx = np.sort(val_idx)
    test_idx = train_val_test_idx['test_idx']
    test_idx = np.sort(test_idx)

    g = dgl.DGLGraph(adjM+(adjM.T))
    g = dgl.remove_self_loop(g)

    node_type = [i for i, z in zip(range(len(node_cnt)), node_cnt) for x in range(z)]
    node_seq = sample_sequence_with_score(adjM, node_type, args.len_seq, features_list[0].shape[0])

    g = g.to(device)
    train_seq = node_seq[train_idx]
    val_seq = node_seq[val_idx]
    test_seq = node_seq[test_idx]

    micro_f1 = torch.zeros(args.repeat)
    macro_f1 = torch.zeros(args.repeat)

    num_classes = dl.labels_train['num_classes']
    type_emb = torch.eye(len(node_cnt)).to(device)
    node_type = torch.tensor(node_type).to(device)

    for i in range(args.repeat):
        
        net = HTCNS(g, num_classes, in_dims, args.hidden_dim, args.num_layers, args.num_gnns, args.num_heads, args.dropout,
                    temper=args.temperature, num_type=len(node_cnt), beta = args.beta)

        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

        net.train()
        early_stopping = EarlyStopping(patience=args.patience, verbose=True, save_path='checkpoint/HTCNS_{}_{}_{}.pt'.format(args.dataset, args.num_layers, args.device))
        for epoch in range(args.epoch):
            t_start = time.time()
            net.train()
            logits = net(features_list, train_seq, type_emb, node_type, args.l2norm)
            logp = F.log_softmax(logits, 1)
            train_loss = F.nll_loss(logp, labels[train_idx])
            optimizer.zero_grad() 
            train_loss.backward()
            optimizer.step()
            t_end = time.time()
            print('Epoch {:05d} | Train_Loss: {:.4f} | Time: {:.4f}'.format(
                epoch, train_loss.item(), t_end-t_start))
            t_start = time.time()
            net.eval()
            with torch.no_grad():
                logits = net(features_list, val_seq, type_emb, node_type, args.l2norm)
                logp = F.log_softmax(logits, 1)
                val_loss = F.nll_loss(logp, labels[val_idx])
                pred = logits.cpu().numpy().argmax(axis=1)
                onehot = np.eye(num_classes, dtype=np.int32)
                pred = onehot[pred]
                print(dl.evaluate_valid(pred, dl.labels_train['data'][val_idx]))
    
            scheduler.step(val_loss)
            t_end = time.time()
            print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
                epoch, val_loss.item(), t_end - t_start))
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print('Early stopping!')
                break
        net.load_state_dict(torch.load(
            'checkpoint/HTCNS_{}_{}_{}.pt'.format(args.dataset, args.num_layers, args.device)))
        net.eval()
        with torch.no_grad():
            logits = net(features_list, test_seq, type_emb, node_type, args.l2norm)
            test_logits = logits
            if args.mode == 1:
                pred = test_logits.cpu().numpy().argmax(axis=1)
                dl.gen_file_for_evaluate(test_idx=test_idx, label=pred, file_name=f"{args.dataset}_{i+1}.txt")
            else:
                pred = test_logits.cpu().numpy().argmax(axis=1)
                onehot = np.eye(num_classes, dtype=np.int32)
                pred = onehot[pred]
                result = dl.evaluate_valid(pred, dl.labels_test['data'][test_idx])
                print(result)
                micro_f1[i] = result['micro-f1']
                macro_f1[i] = result['macro-f1']
    print('Micro-f1:' , micro_f1)
    print('Macro-f1:' , macro_f1)
    print('Micro-f1: %.4f, std: %.4f' % (micro_f1.mean().item(), micro_f1.std().item()))
    print('Macro-f1: %.4f, std: %.4f' % (macro_f1.mean().item(), macro_f1.std().item()))

 
if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='HTCNS')
    ap.add_argument('--feats-type', type=int, default=3)
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--hidden-dim', type=int, default=256)
    ap.add_argument('--dataset', type=str, default = 'AMiner')
    ap.add_argument('--num-heads', type=int, default=2)
    ap.add_argument('--epoch', type=int, default=1000)
    ap.add_argument('--patience', type=int, default=50)
    ap.add_argument('--repeat', type=int, default=5)
    ap.add_argument('--num-layers', type=int, default=2)
    ap.add_argument('--num-gnns', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=2023)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--weight-decay', type=float, default=0)
    ap.add_argument('--len-seq', type=int, default=5)
    ap.add_argument('--l2norm', type=bool, default=True)
    ap.add_argument('--mode', type=int, default=0)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--beta', type=float, default=1.0)
    args = ap.parse_args()
    run_HTCNS(args)

