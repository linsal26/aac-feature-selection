# This code is developed by Md Abdus Salam (Lincoln) (mdsalam@uta.edu/lincoln.cse.uta@gmail.com). Use of any of the materials in this codebase
# in any program/paper without prior permission from the authoer is prohibited. You can change  parameters and run the program to compare various methods
# Date : September 14, 2021

import feature_importance

import math

import pandas as pd
import numpy as np

import time
from datetime import datetime
import os
from functools import reduce

# Legends
# aac - Attribute average conflict based method
# g3 - G3-error based method
# urs - Uniform random sampling based method
# Swope - Adaptive random sampling top-$k$ feature selection as proposed in https://dl.acm.org/doi/abs/10.1145/3448016.3457255?casa_token=WD7S49Hs3i4AAAAA:0W6GIP0LCY7NrJmYfbyk1T5IWcfJVV6bzfBBsMIctIVkAw6Wst5H2kUpma0CTjReCG04ex5tBYKOdQ
# mi - Exact MI calculation based metho
measures_to_compute = ['aac',
                       'g3', 'urs', 'swope', 'mi']


file_name_column = list()
top_k_column = list()


# dictionary of ndcg columns for different measures
# each row corresponds to each file
ndcg_score_column = dict()


# dictionary of precision columns for different measures
# each row corresponds to each file
precision_score_column = dict()


# dictionary of average precision columns for different measures
# each row corresponds to each file
ap_score_column = dict()


# dictionary of mi_sum columns for different measures
# each row corresponds to each file
mi_sum_column = dict()


# attr_sorting_time_column = list()
# avg_time_column = list()


# dictionary of runtime columns for different measures
# each row corresponds to each file
run_time_column = dict()

data_directory = 'data_files/'

output_folder = 'output_csv_files/'

sorted_features = dict()
run_time = dict()

ndcg_score_stats = dict()
precision_at_k_stats = dict()
ap_stats = dict()
mi_sum = dict()
attr_mi_map = dict()


def get_date_time_for_file_prefix():
    now = datetime.now()
    return now.strftime("%m_%d_%Y_%H_%M")


def reset_global_csv_lists():
    global file_name_column, top_k_column

    del file_name_column[:]
    del top_k_column[:]


def calculate_precision(gt, test):
    gt_set = set(gt)
    test_set = set(test)

    common = gt_set.intersection(test_set)

    return len(common) / len(test_set)


def calculate_avg_precision_top_k(fo, gt, test):
    n_k = 0
    prec_k = 0

    gt_set = set(gt)
    test_set = set(test)
    common = gt_set.intersection(test_set)

    for k in range(1, len(test) + 1):
        top_k_list = test[:k]

        for index, item in enumerate(top_k_list):
            if k != index + 1:
                continue
            if item in common:
                # fo.write("item {} found, index:{}".format(item, index + 1))

                current_gt = gt[:k]
                current_test = test[:k]

                # fo.write("current gt: {}".format(current_gt))
                # fo.write("current test: {}".format(current_test))

                n_k += 1
                current_prec = calculate_precision(current_gt, current_test)
                prec_k += current_prec

                # current_gt_set = set(current_gt)
                # current_test_set = set(current_test)
                # current_common = current_gt_set.intersection(current_test_set)

                # n_k += 1
                # prec_k += len(current_common) / len(current_test_set)

                # fo.write("n_k : {}".format(n_k))
                # fo.write("prec_k: {}".format(prec_k))

            # else:
                # fo.write("item {} was not found, index:{}".format(
                #     item, index + 1))
                # fo.write("\n#################\n")

    if n_k == 0:
        avg_prec = 0
    else:
        avg_prec = prec_k / n_k

    # fo.write("Avg precision : {}".format(avg_prec))

    return avg_prec


def calculate_avg_precision(fo, file, K):
    global ap_stats

    ap_stats[file] = dict()

    num_attrs = len(sorted_features[file]['mi'])

    k = 0

    if num_attrs > K:
        k = K
    else:
        k = num_attrs

    ground_truth = sorted_features[file]['mi'][:k]

    for method in measures_to_compute:
        if method != 'mi':
            ap_stats[file][method] = calculate_avg_precision_top_k(
                fo, ground_truth, sorted_features[file][method][:k])

    return


def write_to_csv_avg_precision():
    for file in ap_stats.keys():
        file_name_column.append(file)

        for method in measures_to_compute:
            if method != 'mi':
                if method not in ap_score_column:
                    ap_score_column[method] = list()

                ap_score_column[method].append(ap_stats[file][method])

    dict_for_df = dict()
    dict_for_df['file_name'] = file_name_column

    for method in measures_to_compute:
        if method != 'mi':
            dict_for_df[method] = ap_score_column[method]

    df = pd.DataFrame(dict_for_df)

    prefix = get_date_time_for_file_prefix()
    file_name = output_folder + prefix + \
        "_feature_selection_quality_avg_precision" + ".csv"
    df.to_csv(file_name, index=False)


def calculate_precision_at_k(fo, file, K):
    global precision_at_k_stats

    if file not in precision_at_k_stats.keys():
        precision_at_k_stats[file] = dict()

    num_attrs = len(sorted_features[file]['mi'])
    if num_attrs > K:
        num_attrs = K

    for k in range(1, num_attrs + 1):
        ground_truth = sorted_features[file]['mi'][:k]

        precision_at_k_stats[file][k] = dict()
        for method in measures_to_compute:
            if method != 'mi':
                current_item = sorted_features[file][method][:k]
                precision_at_k_stats[file][k][method] = calculate_precision(
                    ground_truth, current_item)

    return


def write_to_csv_precision_at_k():
    for file in precision_at_k_stats.keys():
        for k in precision_at_k_stats[file].keys():
            file_name_column.append(file)
            top_k_column.append(k)

            for method in measures_to_compute:
                if method != 'mi':
                    if method not in precision_score_column:
                        precision_score_column[method] = list()
                    precision_score_column[method].append(
                        precision_at_k_stats[file][k][method])

    dict_for_df = dict()
    dict_for_df['file_name'] = file_name_column
    dict_for_df['top_k'] = top_k_column

    for method in measures_to_compute:
        if method != 'mi':
            dict_for_df[method] = precision_score_column[method]

    df = pd.DataFrame(dict_for_df)

    prefix = get_date_time_for_file_prefix()

    file_name = output_folder + prefix + \
        "_feature_selection_quality_precision" + ".csv"
    df.to_csv(file_name, index=False)


def find_sum_mi(fo, file, method, k):
    total = 0
    for item in sorted_features[file][method][:k]:
        total += attr_mi_map[item]
    return total


def calculate_mi_sum(fo, file, K):
    mi_sum[file] = dict()

    num_attrs = len(sorted_features[file]['mi'])
    if num_attrs > K:
        num_attrs = K
    # item_score_map = dict()
    item_score_list_mi_desc = list()

    for k in range(1, num_attrs + 1):

        mi_sum[file][k] = dict()

        for method in measures_to_compute:
            mi_sum[file][k][method] = find_sum_mi(fo, file, method, k)


def write_to_csv_mi_sum():
    for file in mi_sum.keys():
        for k in mi_sum[file].keys():
            file_name_column.append(file)
            top_k_column.append(k)

            for method in measures_to_compute:
                if method not in mi_sum_column:
                    mi_sum_column[method] = list()

                mi_sum_column[method].append(mi_sum[file][k][method])

    dict_for_df = dict()
    dict_for_df['file_name'] = file_name_column
    dict_for_df['top_k'] = top_k_column

    for method in measures_to_compute:
        dict_for_df[method] = mi_sum_column[method]

    df = pd.DataFrame(dict_for_df)

    # prefix = str(datetime.date(datetime.now()))

    prefix = get_date_time_for_file_prefix()

    file_name = output_folder + prefix + \
        "_feature_selection_quality_mi_sum" + ".csv"
    df.to_csv(file_name, index=False)


def calculate_dcg(test, k):
    total_score = 0
    for i in range(1, k + 1):
        rel = test[i - 1]
        discounted_val = math.log(i + 1, 2)
        score = rel / discounted_val
        total_score += score
    return total_score


def ndcg_score(gt, test, k):
    test_dcg = calculate_dcg(test, k)
    gt_dcg = calculate_dcg(gt, k)
    return test_dcg / gt_dcg


def calculate_ndcg_score(fo, file, K):
    global attr_mi_map

    ndcg_score_stats[file] = dict()

    num_attrs = len(sorted_features[file]['mi'])
    if num_attrs > K:
        num_attrs = K
    # item_score_map = dict()
    item_score_list_mi_desc = list()

    for index, item in enumerate(sorted_features[file]['mi']):
        # print(index, item)
        item_score_list_mi_desc.append(attr_mi_map[item])
        # item_score_map[item] = attr_mi_map[item]

    # gt_rank = list(item_score_map.values())

    gt_rank = item_score_list_mi_desc

    for k in range(1, num_attrs + 1):
        ground_truth = gt_rank[:k]

        ndcg_score_stats[file][k] = dict()

        rank_list = {}

        for method in measures_to_compute:
            if method != 'mi':
                if method not in rank_list:
                    rank_list[method] = list()

                for item in sorted_features[file][method][:k]:
                    rank_list[method].append(attr_mi_map[item])

                ndcg_score_stats[file][k][method] = ndcg_score(
                    ground_truth, rank_list[method], k)


def write_to_csv_ndcg():
    for file in ndcg_score_stats.keys():
        for k in ndcg_score_stats[file].keys():
            file_name_column.append(file)
            top_k_column.append(k)

            for method in measures_to_compute:
                if method != 'mi':
                    if method not in ndcg_score_column:
                        ndcg_score_column[method] = []

                    ndcg_score_column[method].append(
                        ndcg_score_stats[file][k][method])

    dict_for_df = dict()
    dict_for_df['file_name'] = file_name_column
    dict_for_df['top_k'] = top_k_column

    for method in measures_to_compute:
        if method != 'mi':
            dict_for_df[method] = ndcg_score_column[method]

    df = pd.DataFrame(dict_for_df)

    prefix = get_date_time_for_file_prefix()

    # prefix = str(datetime.date(datetime.now()))
    file_name = output_folder + prefix + \
        "_feature_selection_quality_ndcg_score" + ".csv"
    df.to_csv(file_name, index=False)


def write_to_file(fo, file_prefix, method):
    global sorted_features, run_time

    fo.write("\n.........................................")
    fo.write("\nFeatures Sorted by {}:".format(method))
    fo.write("\n..........................................")
    for attr in sorted_features[file_prefix][method]:
        fo.write("\n" + str(attr))
    fo.write("\n")

    fo.write("\nSorted Featureset creation time (sec) using {} : {}".format(
        method, str(run_time[file_prefix][method])))

    fo.write("\n\n")


def calculate_runtime_mi(fo, file_prefix, df, X, Z, k, mi_calculation_using_joint_entropy, cache, iterations=1):
    global sorted_features, run_time
    global attr_mi_map

    # mi_original indicates whether MI(X,Z), or MI(Z,X) will be calculated
    # if true, MI(X,Z) is calculated --> will take more time for feature selection for MI as H(X) need to be calculated for every attr X and used in eqn MI(X,Z) = H(X) - H(X|Z)
    # if false, MI(Z,X) is calculated --> will take less time for feature selection for MI as H(Z) is calculated once and used for MI calculation for every attr X, MI(Z,X) = H(Z) - H(Z|X)
    mi_original = False

    # if true, sklearn's mi library function will be used
    mi_sklearn = False

    # should be false when computing mi for the whole dataset
    use_sample = False

    # np.random.seed(1234)

    run_time_iteration = list()

    for i in range(iterations):

        # using MI
        start_time = time.perf_counter()

        # sort_feature_by_mi_desc will return a tuple
        # 1st element is sorted attributes by descending MI
        # 2nd element is dictionary of attribute,MI pair
        mi_stats = feature_importance.sort_feature_by_mi_desc(
            fo, use_sample,  df, X, Z, k, mi_original, mi_calculation_using_joint_entropy, mi_sklearn, cache)

        # fo.write("iteration {}\n".format(i))
        # fo.write("mi_stats:\n")
        # for item in mi_stats:
        #     fo.write(str(item))
        #     fo.write("\n")
        # print("\n **** \n")

        end_time = time.perf_counter()

        sorted_features[file_prefix]['mi'] = mi_stats[0]
        attr_mi_map = mi_stats[1]

        # sorted_feature_current_data['mi'] = feature_importance.sort_feature_by_mi_desc(
        #     df, X, Z_mi, k)

        run_time_iteration.append(end_time - start_time)

    # print("\n .. for MI , run_time_iteration list ...\n")
    # print(run_time_iteration)
    # print("\n")

    avg_runtime_mi = reduce(
        lambda a, b: a + b, run_time_iteration) / len(run_time_iteration)
    run_time[file_prefix]['mi'] = avg_runtime_mi


def cacluate_runtime_mi_sampled(fo, file_prefix, sample_size, df, X, Z, k, mi_calculation_using_joint_entropy, cache, iterations=1):
    global sorted_features, run_time
    global attr_mi_map

    # mi_original indicates whether MI(X,Z), or MI(Z,X) will be calculated
    # if true, MI(X,Z) is calculated --> will take more time for feature selection for MI as H(X) need to be calculated for every attr X and used in eqn MI(X,Z) = H(X) - H(X|Z)
    # if false, MI(Z,X) is calculated --> will take less time for feature selection for MI as H(Z) is calculated once and used for MI calculation for every attr X, MI(Z,X) = H(Z) - H(Z|X)
    mi_original = False

    # indicates whether MI will be calculated on sample/full dataset
    use_sample = True

    # np.random.seed(1234)

    df_sampled = df.sample(n=sample_size)

    # if true, sklearn's mi library function will be used; this is slower than our implemetnation of MI
    mi_sklearn = False

    run_time_iteration = list()

    for i in range(iterations):

        start_time = time.perf_counter()

        # df_sampled = df.sample(n=sample_size)

        # urs_stats = feature_importance.sort_feature_by_mi_desc(
        #     fo, use_sample,  df_sampled, X, Z, k, mi_original, mi_calculation_using_joint_entropy, mi_sklearn, cache)

        sorted_features[file_prefix]['urs'] = feature_importance.sort_feature_by_mi_desc(
            fo, use_sample,  df_sampled, X, Z, k, mi_original, mi_calculation_using_joint_entropy, mi_sklearn, cache)

        # sorted_features[file_prefix]['urs'] = feature_importance.sort_feature_by_urs(
        #     fo,  df_sampled, X, Z, k, mi_original, mi_calculation_using_joint_entropy, cache)

        end_time = time.perf_counter()

        # sorted_features[file_prefix]['urs'] = urs_stats[0]

        run_time_iteration.append(end_time - start_time)

    avg_runtime_urs = reduce(
        lambda a, b: a + b, run_time_iteration) / len(run_time_iteration)

    run_time[file_prefix]['urs'] = avg_runtime_urs


def cacluate_runtime_swope(fo, file_prefix, df, X, Z, k, p_f, epsilon, cache, iterations=1):
    global sorted_features, run_time
    global attr_mi_map

    # np.random.seed(1234)

    df_shuffled = df.sample(frac=1)

    run_time_iteration = list()

    for i in range(iterations):

        # using ac
        start_time = time.perf_counter()

        sorted_features[file_prefix]['swope'] = feature_importance.sort_feature_by_swope(
            fo,  df_shuffled, X, Z, k, p_f, epsilon, cache)
        end_time = time.perf_counter()

        run_time_iteration.append(end_time - start_time)

    avg_runtime_swope = reduce(
        lambda a, b: a + b, run_time_iteration) / len(run_time_iteration)

    run_time[file_prefix]['swope'] = avg_runtime_swope


def calculate_runtime_g3(fo, file_prefix, df, X, Z, k, cache, iterations=1):
    global sorted_features, run_time
    global attr_mi_map

    # np.random.seed(1234)

    # if sampling_for_conflict:
    #     sample_size = round(len(df.index) * pct_sample)
    #     df = df.sample(n=sample_size)

    run_time_iteration = list()

    for i in range(iterations):

        # using ac
        start_time = time.perf_counter()
        sorted_features[file_prefix]['g3'] = feature_importance.sort_feature_by_g3(
            fo,  df, X, Z, k, cache)
        end_time = time.perf_counter()

        run_time_iteration.append(end_time - start_time)

    avg_runtime_g3 = reduce(
        lambda a, b: a + b, run_time_iteration) / len(run_time_iteration)

    run_time[file_prefix]['g3'] = avg_runtime_g3


def calculate_runtime_aac(fo, file_prefix, df, X, Z, k, cache, iterations=1):

    global sorted_features, run_time
    global attr_mi_map

    # np.random.seed(1234)

    # if sampling_for_conflict:
    #     sample_size = round(len(df.index) * pct_sample)
    #     df = df.sample(n=sample_size)

    run_time_iteration = list()

    for i in range(iterations):

        # using aac
        start_time = time.perf_counter()
        sorted_features[file_prefix]['aac'] = feature_importance.sort_feature_by_aac(
            fo,  df, X, Z, k, cache)
        end_time = time.perf_counter()
        run_time_iteration.append(end_time - start_time)

    avg_runtime_aac = reduce(
        lambda a, b: a + b, run_time_iteration) / len(run_time_iteration)

    run_time[file_prefix]['aac'] = avg_runtime_aac


def calculate_runtime(fo, k, iterations, file_prefix, df, dependent_variable_name, p_f, epsilon, sampling_for_conflict, pct_sample, mi_calculation_using_joint_entropy, cache):
    global sorted_features, run_time, measures_to_compute
    global attr_mi_map

    df_original = df

    sample_size = round(len(df.index) * pct_sample)

    if sampling_for_conflict:
        df = df.sample(n=sample_size)

    sorted_feature_current_data = dict()

    run_time_current_data = dict()

    Z_mi = [dependent_variable_name]
    Z = dependent_variable_name
    X = list(df.columns)
    X.remove(Z_mi[0])
    df["probability"] = 1.0 / len(df.index)

    if file_prefix not in sorted_features.keys():
        sorted_features[file_prefix] = dict()

    if file_prefix not in run_time.keys():
        run_time[file_prefix] = dict()

    if 'urs' in measures_to_compute:
        cacluate_runtime_mi_sampled(
            fo, file_prefix, sample_size, df, X, Z, k, mi_calculation_using_joint_entropy, cache, iterations)

    if 'g3' in measures_to_compute:
        calculate_runtime_g3(
            fo, file_prefix, df, X, Z, k, cache, iterations)

    if 'aac' in measures_to_compute:
        calculate_runtime_aac(
            fo, file_prefix, df, X, Z, k, cache, iterations)

    if 'swope' in measures_to_compute:
        cacluate_runtime_swope(
            fo, file_prefix, df_original, X, Z_mi, k, p_f, epsilon, cache, iterations)

    calculate_runtime_mi(fo, file_prefix, df, X, Z,
                         k, mi_calculation_using_joint_entropy, cache, iterations)

    fo.write("\n#####################{}########################".format(file_prefix))

    for method in measures_to_compute:
        write_to_file(fo, file_prefix, method)

    # file_name_column.append(file_name)
    # iteration_column.append(i)
    # attr_sorting_time_column.append(run_time)

def write_to_csv_run_time(k):

    # print('printing run_time')
    # print(run_time)
    for file in run_time.keys():
        file_name_column.append(file)
        top_k_column.append(k)

        for method in measures_to_compute:
            if method not in run_time_column:
                run_time_column[method] = []

            run_time_column[method].append(run_time[file][method])

    dict_for_df = dict()
    dict_for_df['file_name'] = file_name_column
    dict_for_df['top_k'] = top_k_column

    for method in measures_to_compute:
        dict_for_df[method] = run_time_column[method]

    df = pd.DataFrame(dict_for_df)

    # prefix = str(datetime.date(datetime.now()))

    prefix = get_date_time_for_file_prefix()

    file_name = output_folder + prefix + \
        "_feature_selection_run_time.csv"
    df.to_csv(file_name, index=False)


def main():

    global output_folder

    data_files = os.listdir(data_directory)

    K = 10

    # this is used for Swope
    epsilon = 0.5

    # if True, pct_sample of original data will be used to calculate Ac/Aac/MI
    # if False, full dataset will be used to calculate Ac/Aac/MI
    sampling_for_conflict = False

    # percentage of records used for sampling
    pct_sample = 0.65

    # indicates whether one-shot uniform random sampling (Urs) is used a measurement
    using_sampling = True

    # if True, Original MI is calculated using joint entropy formula
    # MI(a,b) = H(a) + H(b) - H(a,b)
    # else, Original MI is calculated using MI formula
    # MI(a,b) = sum_(a)sum_(b)p(a,b)log_2(p(a,b)/p(a) * p(b))
    mi_calculation_using_joint_entropy = True

    # how many iterations for running each method; an average runtime is calculated based on this
    runtime_iterations = 3

    # contains the text file listing the top-k attributes for different methods along with score
    resultFolder = 'text_output'

    # contains csv files genreated for the quality and speedup measurements
    output_folder += 'sampling_frac_point_' + \
        str(pct_sample).rsplit('.')[1] + '/'

    dir_exists = os.path.isdir(output_folder)
    if not dir_exists:
        print("No, {} does not exist".format(output_folder))
        print("trying to create the folder ..")
        os.mkdir(output_folder)

    if using_sampling:
        output_folder += 'using_sampling/'
        dir_exists = os.path.isdir(output_folder)
        if not dir_exists:
            print("No, {} does not exist".format(output_folder))
            print("trying to create the folder ..")
            os.mkdir(output_folder)

    dependent_variable_name = "Outcome"

    for file in data_files:

        # if file not in ['madelon.csv']:
        #     continue

        file_name = data_directory + file

        file_name_parts = file.rsplit('.', 1)
        file_prefix = file_name_parts[0]

        print("Processing ... : " + file_prefix)

        df = pd.read_csv(file_name)

        print('file name : {}, shape : {}'.format(file_name, df.shape))

        num_attrs = df.shape[1] - 1
        # run_time = 0

        if 'probability' in df.columns:
            df.drop('probability', axis=1, inplace=True)

        date_time_prefix = get_date_time_for_file_prefix()

        fo = open(resultFolder + "/" + date_time_prefix + "_" + str(file_prefix) + "_top_" +
                  str(K) + ".txt", "w")

        fo.write("\n")

        # should be set as True for saving  and resuing H(Z) in exact MI calculation
        cache = True

        # used for Swope
        p_f = 1.0 / len(df.index)

        calculate_runtime(fo, num_attrs, runtime_iterations, file_prefix, df,
                          dependent_variable_name, p_f, epsilon, sampling_for_conflict, pct_sample, mi_calculation_using_joint_entropy, cache)

        calculate_ndcg_score(fo, file_prefix, K)

        calculate_precision_at_k(fo, file_prefix, K)

        calculate_avg_precision(fo, file_prefix, K)

        calculate_mi_sum(fo, file_prefix, K)

    reset_global_csv_lists()

    write_to_csv_run_time(K)

    reset_global_csv_lists()

    write_to_csv_ndcg()

    reset_global_csv_lists()

    write_to_csv_precision_at_k()

    reset_global_csv_lists()

    write_to_csv_avg_precision()

    reset_global_csv_lists()

    write_to_csv_mi_sum()


if __name__ == "__main__":
    main()
