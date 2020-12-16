import os
from collections import defaultdict, Counter
import numpy as np
from sortedcontainers import SortedList
import pickle

TRAIN_SIZE = 854
VALID_SIZE = 284
TEST_SIZE = 284
# types table
# 0 -
# 1 apache2
# 2 back
# 3 dict
# 4 guest
# 5 httptunnel-e
# 6 ignore
# 7 ipsweep
# 8 mailbomb
# 9 mscan
# 10 neptune
# 11 nmap
# 12 pod
# 13 portsweep
# 14 processtable
# 15 rootkit
# 16 saint
# 17 satan
# 18 smurf
# 19 smurfttl
# 20 snmpgetattack
# 21 snmpguess
# 22 teardrop
# 23 warez
# 24 warezclient
attack_types = ['-', 'apache2', 'back', 'dict', 'guest', 'httptunnel-e', 'ignore', 'ipsweep', 'mailbomb', 'mscan',
                'neptune', 'nmap', 'pod', 'portsweep', 'processtable', 'rootkit', 'saint', 'satan', 'smurf', 'smurfttl',
                'snmpgetattack', 'snmpguess', 'teardrop', 'warez', 'warezclient']

# type_table_sdp = defaultdict(set)
# type_table_sd = defaultdict(set)
# type_table_dp = defaultdict(set)
# type_table_sp = defaultdict(set)
# type_table_s = defaultdict(set)
# type_table_d = defaultdict(set)
# type_table_p = defaultdict(set)
# s_table_dp = defaultdict(set)
# sdpc_set = set()
# time_list_sdp = defaultdict(SortedList)
# tsdpc_list = SortedList()
#
# for i in range(TRAIN_SIZE):
#     with open(os.path.join('train', 'train_%03d.txt' % i)) as f:
#         data = f.readlines()
#     for line in data:
#         if line[-1] == '\n':
#             line = line[:-1]
#         info_list = line.split('\t')
#         s = int(info_list[0])
#         d = int(info_list[1])
#         p = int(info_list[2])
#         t = int(info_list[3])
#         c = attack_types.index(info_list[4])
#         type_table_sdp[(s, d, p)].add(c)
#         type_table_sd[(s, d)].add(c)
#         type_table_dp[(d, p)].add(c)
#         type_table_sp[(s, p)].add(c)
#         type_table_s[s].add(c)
#         type_table_d[d].add(c)
#         type_table_p[p].add(c)
#         s_table_dp[(d, p)].add(s)
#         sdpc_set.add((s, d, p, c))
#         time_list_sdp[(s, d, p)].add(t)
#         tsdpc_list.add((t, s, d, p, c))
#
# type_list_sd = defaultdict(list)
# type_list_sp = defaultdict(list)
# type_list_dp = defaultdict(list)
# type_list_s = defaultdict(list)
# type_list_d = defaultdict(list)
# type_list_p = defaultdict(list)
#
# for s, d, p, c in sdpc_set:
#     type_list_sd[(s, d)].append(c)
#     type_list_sp[(s, p)].append(c)
#     type_list_dp[(d, p)].append(c)
#     type_list_s[s].append(c)
#     type_list_d[d].append(c)
#     type_list_p[p].append(c)
#
# with open('train.pkl', 'wb') as f:
#     pickle.dump([type_table_sdp,
#                  type_table_sd,
#                  type_table_sp,
#                  type_table_dp,
#                  type_table_s,
#                  type_table_d,
#                  type_table_p,
#                  s_table_dp,
#                  sdpc_set,
#                  time_list_sdp,
#                  tsdpc_list,
#                  type_list_sd,
#                  type_list_sp,
#                  type_list_dp,
#                  type_list_s,
#                  type_list_d,
#                  type_list_p], f)
with open('train.pkl', 'rb') as f:
    (type_table_sdp,
     type_table_sd,
     type_table_sp,
     type_table_dp,
     type_table_s,
     type_table_d,
     type_table_p,
     s_table_dp,
     sdpc_set,
     time_list_sdp,
     tsdpc_list,
     type_list_sd,
     type_list_sp,
     type_list_dp,
     type_list_s,
     type_list_d,
     type_list_p) = pickle.load(f)

# train_sdp_total = set()
# train_sd_total = set()
# train_dp_total = set()
# train_sp_total = set()
# train_s_total = set()
# train_d_total = set()
# train_p_total = set()
# for i in range(TRAIN_SIZE):
#     with open(os.path.join('train', 'train_%03d.txt' % i)) as f:
#         data = f.readlines()
#     for line in data:
#         if line[-1] == '\n':
#             line = line[:-1]
#         info_list = line.split('\t')
#         s = int(info_list[0])
#         d = int(info_list[1])
#         p = int(info_list[2])
#         # t = int(info_list[3])
#         train_sdp_total.add((s, d, p))
#         train_sd_total.add((s, d))
#         train_dp_total.add((d, p))
#         train_sp_total.add((s, p))
#         train_s_total.add(s)
#         train_d_total.add(d)
#         train_p_total.add(p)
#
# with open('train_1.pkl', 'wb') as f:
#     pickle.dump([train_sdp_total,
#                  train_sd_total,
#                  train_dp_total,
#                  train_sp_total,
#                  train_s_total,
#                  train_d_total,
#                  train_p_total], f)
with open('train_1.pkl', 'rb') as f:
    (train_sdp_total,
     train_sd_total,
     train_dp_total,
     train_sp_total,
     train_s_total,
     train_d_total,
     train_p_total) = pickle.load(f)

# Counter(len(v) for v in type_table_sdp.values())
# Counter({1: 308996, 2: 77})
# Counter(len(v) for v in type_table_sd.values())
# Counter({1: 54422, 2: 448, 3: 56, 4: 16, 6: 2})
# Counter(len(v) for v in type_table_dp.values())
# Counter({1: 247259, 2: 4587, 3: 23, 4: 1})
# Counter(len(v) for v in type_table_sp.values())
# Counter({1: 185023, 2: 3971})
# Counter(len(v) for v in type_table_s.values())
# Counter({1: 9198, 2: 214, 4: 21, 3: 20, 5: 11, 6: 8, 7: 5, 9: 1})
# Counter(len(v) for v in type_table_d.values())
# Counter({1: 20982, 2: 285, 3: 33, 4: 32, 5: 9, 7: 3, 6: 2, 13: 1, 18: 1})
# Counter(len(v) for v in type_table_p.values())
# Counter({1: 16530, 2: 6386, 3: 1257, 4: 378, 5: 105, 6: 18, 7: 1})

# valid_sdp_total = set()
# valid_sd_total = set()
# valid_dp_total = set()
# valid_sp_total = set()
# valid_s_total = set()
# valid_d_total = set()
# valid_p_total = set()
# valid_time_list_sdp = defaultdict(SortedList)
#
# for i in range(VALID_SIZE):
#     with open(os.path.join('valid_query', 'valid_query_%03d.txt' % i)) as f:
#         query_data = f.readlines()
#     for line in query_data:
#         if line[-1] == '\n':
#             line = line[:-1]
#         info_list = line.split('\t')
#         s = int(info_list[0])
#         d = int(info_list[1])
#         p = int(info_list[2])
#         t = int(info_list[3])
#
#         valid_sdp_total.add((s, d, p))
#         valid_sd_total.add((s, d))
#         valid_dp_total.add((d, p))
#         valid_sp_total.add((s, p))
#         valid_s_total.add(s)
#         valid_d_total.add(d)
#         valid_p_total.add(p)
#         valid_time_list_sdp[(s, d, p)].add(t)
#
# with open('valid.pkl', 'wb') as f:
#     pickle.dump([valid_sdp_total,
#                  valid_sd_total,
#                  valid_dp_total,
#                  valid_sp_total,
#                  valid_s_total,
#                  valid_d_total,
#                  valid_p_total,
#                  valid_time_list_sdp], f)

# test_sdp_total = set()
# test_sd_total = set()
# test_dp_total = set()
# test_sp_total = set()
# test_s_total = set()
# test_d_total = set()
# test_p_total = set()
# test_time_list_sdp = defaultdict(SortedList)
#
# for i in range(TEST_SIZE):
#     with open(os.path.join('test_query', 'test_query_%03d.txt' % i)) as f:
#         query_data = f.readlines()
#     for line in query_data:
#         if line[-1] == '\n':
#             line = line[:-1]
#         info_list = line.split('\t')
#         s = int(info_list[0])
#         d = int(info_list[1])
#         p = int(info_list[2])
#         t = int(info_list[3])
#
#         test_sdp_total.add((s, d, p))
#         test_sd_total.add((s, d))
#         test_dp_total.add((d, p))
#         test_sp_total.add((s, p))
#         test_s_total.add(s)
#         test_d_total.add(d)
#         test_p_total.add(p)
#         test_time_list_sdp[(s, d, p)].add(t)
#
# with open('test.pkl', 'wb') as f:
#     pickle.dump([test_sdp_total,
#                  test_sd_total,
#                  test_dp_total,
#                  test_sp_total,
#                  test_s_total,
#                  test_d_total,
#                  test_p_total,
#                  test_time_list_sdp], f)

with open('valid.pkl', 'rb') as f:
    (valid_sdp_total,
     valid_sd_total,
     valid_dp_total,
     valid_sp_total,
     valid_s_total,
     valid_d_total,
     valid_p_total,
     valid_time_list_sdp) = pickle.load(f)

with open('test.pkl', 'rb') as f:
    (test_sdp_total,
     test_sd_total,
     test_dp_total,
     test_sp_total,
     test_s_total,
     test_d_total,
     test_p_total,
     test_time_list_sdp) = pickle.load(f)

valid_sd_GNN = np.load('valid.npy', allow_pickle=True).item()
test_sd_GNN = np.load('test.npy', allow_pickle=True).item()

special_types = {7, 16, 18, 19}
vague_types = {7, 17, 18, 19, 23, 24}


def unique_type_s(source):
    return len(type_table_s[source]) == 1


def unique_type_d(dest):
    return len(type_table_d[dest]) == 1 and type_table_d[dest].isdisjoint({18, 19, 23, 24})


def unique_type_p(port):
    return len(type_table_p[port]) == 1 and type_table_p[port].isdisjoint({18, 19, 23, 24})


def possible_for_s(connection_type, source):
    # in valid set, only s == 5954 such that type_table_s[s] is disjoint with some corresponding possible answers
    return (connection_type in type_table_s[source]) or (source == 5954 and connection_type in {0, 3, 7})


def unique_type_dp(source, dest, port):
    return len(type_table_dp[(dest, port)]) == 1 and possible_for_s(min(type_table_dp[(dest, port)]), source)


def unique_type_sp(source, port):
    return len(type_table_sp[(source, port)]) == 1


def unique_type_sd(source, dest):
    return len(type_table_sd[(source, dest)]) == 1 and min(type_table_sd[(source, dest)]) not in vague_types


sd_23_24 = [sd for sd in train_sd_total if type_table_sd[sd] == {23, 24}]
time_type_23_24 = defaultdict(SortedList)
for t, s, d, p, c in tsdpc_list:
    if (s, d) in sd_23_24:
        time_type_23_24[(s, d)].add((t, p, c))
for s, d, p in valid_sdp_total:
    if (s, d) in sd_23_24 and len(valid_time_list_sdp[(s, d, p)]) <= 4:
        for t in valid_time_list_sdp[(s, d, p)]:
            time_type_23_24[(s, d)].add((t, p, 0))
for s, d, p in test_sdp_total:
    if (s, d) in sd_23_24:
        # c = min(possible_answers[(s,d,p)].intersection({23, 24}))
        for t in test_time_list_sdp[(s, d, p)]:
            time_type_23_24[(s, d)].add((t, p, -1))
# for sd in sd_23_24:
#     print(sd)
#     for t, p, c in time_type_23_24[sd]:
#         print(t, p, c)
#     print('---------------------------------------------------')


prediction_valid = dict.fromkeys(valid_sdp_total)
for sdp in valid_sdp_total:
    s, d, p = sdp
    sd = (s, d)
    dp = (d, p)
    sp = (s, p)
    if s not in train_s_total:  # s, never seen
        prediction_valid[sdp] = {0}
    elif sdp in train_sdp_total:  # sdp, seen before
        prediction_valid[sdp] = type_table_sdp[sdp].copy() if len(type_table_sdp[sdp]) == 1 else {0}
    elif unique_type_s(s):  # s, unique type
        prediction_valid[sdp] = type_table_s[s].copy()
    elif type_table_s[s] == {18, 19}:
        prediction_valid[sdp] = {19} if 19 in type_table_dp[dp] else {18}
    elif unique_type_d(d):  # d, unique type
        prediction_valid[sdp] = type_table_d[d].copy()
    elif type_table_d[d] == {7, 16}:
        prediction_valid[sdp] = {0} if 16 in type_table_s[s] else {7}
    elif unique_type_p(p):  # p, unique type
        prediction_valid[sdp] = type_table_p[p].copy()
    elif 17 in type_table_sd[sd] or (17 in type_table_s and valid_sd_GNN[sd][17] >= 0.1):
        timestamps = valid_time_list_sdp[sdp]
        connection_times = len(timestamps)
        span = timestamps[-1] - timestamps[0]
        prediction_valid[sdp] = {17} if connection_times > 1000 and span <= 0.1 * connection_times else {0}
    elif unique_type_dp(s, d, p):  # dp, unique type
        prediction_valid[sdp] = type_table_dp[dp].copy()
    elif unique_type_sp(s, p):  # sp, unique type
        prediction_valid[sdp] = type_table_sp[sp].copy()
    # elif type_table_sd[sd] in ({23}, {24}):  # 23, 24
    #     timestamps = valid_time_list_sdp[sdp]
    #     connection_times = len(timestamps)
    #     span = timestamps[-1] - timestamps[0]
    #     if connection_times == 1 or connection_times >= 5:
    #         prediction_valid[sdp] = {0}
    #     # elif connection_times == 2 and span == 0:
    #     #     prediction_valid[sdp] = {23}
    #     else:
    #         prediction_valid[sdp] = {24} if span >= 3 else {23}
    # elif type_table_sd[sd] == {23, 24}:
    #     timestamps = valid_time_list_sdp[sdp]
    #     connection_times = len(timestamps)
    #     span = timestamps[-1] - timestamps[0]
    #     if connection_times >= 5:
    #         prediction_valid[sdp] = {0}
    #     elif span >= 1:
    #         prediction_valid[sdp] = {24}
    #     elif connection_times == 1:
    #         t = timestamps[0]
    #         count_c = len(time_type_23_24[sd])
    #         index_c = time_type_23_24[sd].index((t, p, 0))
    #         if index_c < count_c - connection_times and \
    #                 time_type_23_24[sd][index_c - 1][-1] == time_type_23_24[sd][index_c + connection_times][-1] == 24:
    #             prediction_valid[sdp] = {24}
    #         else:
    #             prediction_valid[sdp] = {0}
    #     else:
    #         prediction_valid[sdp] = {23}
    elif type_table_sd[sd] in ({23}, {24}, {23, 24}):  # 23, 24
        timestamps = valid_time_list_sdp[sdp]
        connection_times = len(timestamps)
        span = timestamps[-1] - timestamps[0]
        if connection_times == 1 or connection_times >= 5:
            prediction_valid[sdp] = {0}
        else:
            prediction_valid[sdp] = {24} if span >= 1 else {23}
    # elif type_table_sd[sd] in ({0, 23}, {0, 24}, {0, 23, 24}, {0, 13, 23, 24}, {0, 6, 7, 16, 23, 24},
    #                            {0, 6, 7, 16, 23, 24}, {24, 11, 23}, {}):
    # elif not type_table_sd[sd].isdisjoint({23, 24}):
    #     timestamps = valid_time_list_sdp[sdp]
    #     connection_times = len(timestamps)
    #     span = timestamps[-1] - timestamps[0]
    #     prediction_valid[sdp] = {24} if (connection_times, span) in \
    #                                     [(4, 16), (4, 17), (3, 16), (3, 12), (3, 13)] else {0}
    elif unique_type_sd(s, d):  # sd, unique type
        prediction_valid[sdp] = type_table_sd[sd].copy()
    elif type_table_sd[sd] == {12, 13}:
        timestamps = valid_time_list_sdp[sdp]
        connection_times = len(timestamps)
        span = timestamps[-1] - timestamps[0]
        if (2 <= connection_times <= 3 and 35 <= span <= 50) or \
                (27 <= connection_times <= 30 and 56 <= span >= 59) or \
                (connection_times == 2 and span == 20):
            prediction_valid[sdp] = {13}
        elif connection_times >= 10 and span <= 1:
            prediction_valid[sdp] = {12}
        else:
            prediction_valid[sdp] = {0}
    elif 13 in type_table_sd[sd]:
        timestamps = valid_time_list_sdp[sdp]
        connection_times = len(timestamps)
        span = timestamps[-1] - timestamps[0]
        prediction_valid[sdp] = {13} if (2 <= connection_times <= 3 and 35 <= span <= 50) or \
                                        (27 <= connection_times <= 30 and 56 <= span >= 59) else {0}
    elif type_table_sd[sd] == {3, 18}:
        prediction_valid[sdp] = {0} if 18 in type_table_dp[dp] else {3}
    elif 3 in type_table_sd[sd] or (3 in type_table_s and valid_sd_GNN[sd][3] >= 0.1):
        timestamps = valid_time_list_sdp[sdp]
        connection_times = len(timestamps)
        span = timestamps[-1] - timestamps[0]
        prediction_valid[sdp] = {3} if 51 <= span <= 59 and 11 <= connection_times <= 14 else {0}
    elif 12 in type_table_sd[sd]:
        timestamps = valid_time_list_sdp[sdp]
        connection_times = len(timestamps)
        span = timestamps[-1] - timestamps[0]
        prediction_valid[sdp] = {12} if span <= 1 and connection_times >= 10 else {0}
    elif type_table_sd[sd] == {20, 21}:
        timestamps = valid_time_list_sdp[sdp]
        connection_times = len(timestamps)
        span = timestamps[-1] - timestamps[0]
        if connection_times == 1:
            prediction_valid[sdp] = {0}
        elif connection_times <= 13 and span >= 20:
            prediction_valid[sdp] = {20}
        elif connection_times >= 37 and 49 <= span <= 59:
            prediction_valid[sdp] = {21}
        else:
            prediction_valid[sdp] = {0}
    elif 6 in type_table_sd[sd]:
        timestamps = valid_time_list_sdp[sdp]
        connection_times = len(timestamps)
        span = timestamps[-1] - timestamps[0]
        prediction_valid[sdp] = {6} if connection_times == 2 and 38 <= span <= 42 else {0}
    elif 16 in type_table_sd[sd]:
        timestamps = valid_time_list_sdp[sdp]
        connection_times = len(timestamps)
        span = timestamps[-1] - timestamps[0]
        if connection_times == 2 and span == 10:
            prediction_valid[sdp] = {16}
        elif connection_times == 3 and \
                (timestamps[1] - timestamps[0], timestamps[2] - timestamps[1]) in ((10, 10), (9, 10), (10, 9)):
            prediction_valid[sdp] = {16}
        else:
            prediction_valid[sdp] = {0}
    elif 1 in type_table_sd[sd]:  # 1
        timestamps = valid_time_list_sdp[sdp]
        connection_times = len(timestamps)
        span = timestamps[-1] - timestamps[0]
        prediction_valid[sdp] = {1} if connection_times >= 100 and span <= connection_times / 5 else {0}
    elif 2 in type_table_s[s]:  # 2
        timestamps = valid_time_list_sdp[sdp]
        connection_times = len(timestamps)
        span = timestamps[-1] - timestamps[0]
        prediction_valid[sdp] = {2} if 60 <= connection_times <= 100 and 58 <= span <= 59 else {0}
    # elif 4 in type_table_sd[sd]:  # 4
    #     prediction_valid[sdp] = {0}
    elif 5 in type_table_sd[sd]:  # 5
        timestamps = valid_time_list_sdp[sdp]
        connection_times = len(timestamps)
        span = timestamps[-1] - timestamps[0]
        prediction_valid[sdp] = {5} if connection_times == 2 and span == 30 else {0}
    elif 8 in type_table_sd[sd]:  # 8
        timestamps = valid_time_list_sdp[sdp]
        connection_times = len(timestamps)
        span = timestamps[-1] - timestamps[0]
        prediction_valid[sdp] = {8} if 47 <= connection_times <= 49 and 57 <= connection_times <= 59 else {0}
    elif 14 in type_table_sd[sd]:  # 14
        timestamps = valid_time_list_sdp[sdp]
        connection_times = len(timestamps)
        span = timestamps[-1] - timestamps[0]
        prediction_valid[sdp] = {14} if connection_times == 15 and 56 <= connection_times <= 57 else {0}
    # elif 15 in type_table_sd[sd]:  # 15
    #     prediction_valid[sdp] = {0}
    else:
        prediction_valid[sdp] = {0}

A_type_count = B_type_count = score_sum_A = score_sum_B = 0
# valid_res = defaultdict(set)
for i in range(VALID_SIZE):
    with open(os.path.join('valid_query', 'valid_query_%03d.txt' % i)) as f:
        query_data = f.readlines()
    with open(os.path.join('valid_answer', 'valid_answer_%03d.txt' % i)) as f:
        answer_data = f.readlines()
    if not answer_data:
        ground_truth = set()
    else:
        ground_truth = set([attack_types.index(t) for t in answer_data[0].split('\t')])
    pred = set()
    for line in query_data:
        if line[-1] == '\n':
            line = line[:-1]
        info_list = line.split('\t')
        s = int(info_list[0])
        d = int(info_list[1])
        p = int(info_list[2])
        # t = int(info_list[3])
        pred.update(prediction_valid[(s, d, p)])
        # valid_res[(s, d, p)].update(ground_truth)
    pred.discard(0)
    if 24 in pred:
        pred.discard(23)
    if not answer_data:
        ground_truth = set()
        B_type_count += 1
        prec = 1 / (1 + len(pred))
        recall = 1
        score_sum_B += 2 * prec * recall / (prec + recall)
    else:
        ground_truth = set([attack_types.index(t) for t in answer_data[0].split('\t')])
        A_type_count += 1
        inter = pred.intersection(ground_truth)
        prec = (1 + len(inter)) / (1 + len(pred))
        recall = (1 + len(inter)) / (1 + len(ground_truth))
        score_sum_A += 2 * prec * recall / (prec + recall)
    print('@ valid #{}\tGT: {}\tpred: {}\tredundant: {}\tmissing: {}'
          .format(i, ground_truth, pred, pred - ground_truth, ground_truth - pred))
final_score = (score_sum_A / A_type_count + score_sum_B / B_type_count) / 2
print('final score:', final_score)

prediction_test = dict.fromkeys(test_sdp_total)
for sdp in test_sdp_total:
    s, d, p = sdp
    sd = (s, d)
    dp = (d, p)
    sp = (s, p)
    if s not in train_s_total:  # s, never seen
        prediction_test[sdp] = {0}
    elif sdp in train_sdp_total:  # sdp, seen before
        prediction_test[sdp] = type_table_sdp[sdp].copy() if len(type_table_sdp[sdp]) == 1 else {0}
    elif unique_type_s(s):  # s, unique type
        prediction_test[sdp] = type_table_s[s].copy()
    elif type_table_s[s] == {18, 19}:
        prediction_test[sdp] = {19} if 19 in type_table_dp[dp] else {18}
    elif unique_type_d(d):  # d, unique type
        prediction_test[sdp] = type_table_d[d].copy()
    elif type_table_d[d] == {7, 16}:
        prediction_test[sdp] = {0} if 16 in type_table_s[s] else {7}
    elif unique_type_p(p):  # p, unique type
        prediction_test[sdp] = type_table_p[p].copy()
    elif 17 in type_table_sd[sd] or (17 in type_table_s and test_sd_GNN[sd][17] >= 0.1):
        timestamps = test_time_list_sdp[sdp]
        connection_times = len(timestamps)
        span = timestamps[-1] - timestamps[0]
        prediction_test[sdp] = {17} if connection_times > 1000 and span <= 0.1 * connection_times else {0}
    elif unique_type_dp(s, d, p):  # dp, unique type
        prediction_test[sdp] = type_table_dp[dp].copy()
    elif unique_type_sp(s, p):  # sp, unique type
        prediction_test[sdp] = type_table_sp[sp].copy()
    elif type_table_sd[sd] in ({23}, {24}, {23, 24}):  # 23, 24
        timestamps = test_time_list_sdp[sdp]
        connection_times = len(timestamps)
        span = timestamps[-1] - timestamps[0]
        if connection_times == 1 or connection_times >= 5:
            prediction_test[sdp] = {0}
        else:
            prediction_test[sdp] = {24} if span >= 1 else {23}
    elif unique_type_sd(s, d):  # sd, unique type
        prediction_test[sdp] = type_table_sd[sd].copy()
    elif type_table_sd[sd] == {12, 13}:
        timestamps = test_time_list_sdp[sdp]
        connection_times = len(timestamps)
        span = timestamps[-1] - timestamps[0]
        if (2 <= connection_times <= 3 and 35 <= span <= 50) or \
                (27 <= connection_times <= 30 and 56 <= span >= 59) or \
                (connection_times == 2 and span == 20):
            prediction_test[sdp] = {13}
        elif connection_times >= 10 and span <= 1:
            prediction_test[sdp] = {12}
        else:
            prediction_test[sdp] = {0}
    elif 13 in type_table_sd[sd]:
        timestamps = test_time_list_sdp[sdp]
        connection_times = len(timestamps)
        span = timestamps[-1] - timestamps[0]
        prediction_test[sdp] = {13} if (2 <= connection_times <= 3 and 35 <= span <= 50) or \
                                       (27 <= connection_times <= 30 and 56 <= span >= 59) else {0}
    elif type_table_sd[sd] == {3, 18}:
        prediction_test[sdp] = {0} if 18 in type_table_dp[dp] else {3}
    elif 3 in type_table_sd[sd] or (3 in type_table_s and test_sd_GNN[sd][3] >= 0.1):
        timestamps = test_time_list_sdp[sdp]
        connection_times = len(timestamps)
        span = timestamps[-1] - timestamps[0]
        prediction_test[sdp] = {3} if 51 <= span <= 59 and 11 <= connection_times <= 14 else {0}
    elif 12 in type_table_sd[sd]:
        timestamps = test_time_list_sdp[sdp]
        connection_times = len(timestamps)
        span = timestamps[-1] - timestamps[0]
        prediction_test[sdp] = {12} if span <= 1 and connection_times >= 10 else {0}
    elif type_table_sd[sd] == {20, 21}:
        timestamps = test_time_list_sdp[sdp]
        connection_times = len(timestamps)
        span = timestamps[-1] - timestamps[0]
        if connection_times == 1:
            prediction_test[sdp] = {0}
        elif connection_times <= 13 and span >= 20:
            prediction_test[sdp] = {20}
        elif connection_times >= 37 and 49 <= span <= 59:
            prediction_test[sdp] = {21}
        else:
            prediction_test[sdp] = {0}
    elif 6 in type_table_sd[sd]:
        timestamps = test_time_list_sdp[sdp]
        connection_times = len(timestamps)
        span = timestamps[-1] - timestamps[0]
        prediction_test[sdp] = {6} if connection_times == 2 and 38 <= span <= 42 else {0}
    elif 16 in type_table_sd[sd]:
        timestamps = test_time_list_sdp[sdp]
        connection_times = len(timestamps)
        span = timestamps[-1] - timestamps[0]
        if connection_times == 2 and span == 10:
            prediction_test[sdp] = {16}
        elif connection_times == 3 and \
                (timestamps[1] - timestamps[0], timestamps[2] - timestamps[1]) in ((10, 10), (9, 10), (10, 9)):
            prediction_test[sdp] = {16}
        else:
            prediction_test[sdp] = {0}
    elif 1 in type_table_sd[sd]:  # 1
        timestamps = test_time_list_sdp[sdp]
        connection_times = len(timestamps)
        span = timestamps[-1] - timestamps[0]
        prediction_test[sdp] = {1} if connection_times >= 100 and span <= connection_times / 5 else {0}
    elif 2 in type_table_s[s]:  # 2
        timestamps = test_time_list_sdp[sdp]
        connection_times = len(timestamps)
        span = timestamps[-1] - timestamps[0]
        prediction_test[sdp] = {2} if 60 <= connection_times <= 100 and 58 <= span <= 59 else {0}
    # elif 4 in type_table_sd[sd]:  # 4
    #     prediction_test[sdp] = {0}
    elif 5 in type_table_sd[sd]:  # 5
        timestamps = test_time_list_sdp[sdp]
        connection_times = len(timestamps)
        span = timestamps[-1] - timestamps[0]
        prediction_test[sdp] = {5} if connection_times == 2 and span == 30 else {0}
    elif 8 in type_table_sd[sd]:  # 8
        timestamps = test_time_list_sdp[sdp]
        connection_times = len(timestamps)
        span = timestamps[-1] - timestamps[0]
        prediction_test[sdp] = {8} if 47 <= connection_times <= 49 and 57 <= connection_times <= 59 else {0}
    elif 14 in type_table_sd[sd]:  # 14
        timestamps = test_time_list_sdp[sdp]
        connection_times = len(timestamps)
        span = timestamps[-1] - timestamps[0]
        prediction_test[sdp] = {14} if connection_times == 15 and 56 <= connection_times <= 57 else {0}
    # elif 15 in type_table_sd[sd]:  # 15
    #     prediction_test[sdp] = {0}
    else:
        prediction_test[sdp] = {0}

os.makedirs('test_answer', exist_ok=True)
for i in range(TEST_SIZE):
    with open(os.path.join('test_query', 'test_query_%03d.txt' % i)) as f:
        query_data = f.readlines()
    pred = set()
    for line in query_data:
        if line[-1] == '\n':
            line = line[:-1]
        info_list = line.split('\t')
        s = int(info_list[0])
        d = int(info_list[1])
        p = int(info_list[2])
        pred.update(prediction_test[(s, d, p)])
    pred.discard(0)
    if 24 in pred:
        pred.discard(23)
    print('@ test #{}\tpred: {}'.format(i, pred))
    with open(os.path.join('test_answer', 'test_answer_%03d.txt' % i), 'w+') as f:
        pred = list(pred)
        for j in range(len(pred)):
            f.write(attack_types[pred[j]])
            if j != len(pred) - 1:
                f.write('\t')
