# GiG
import math
from math import log, ceil, sqrt
import numpy as np


# MI(X,Z) = MI(Z,X) ; instead of calculating MI(X,Z) we are going to calculate MI(Z,X)
# as MI(Z,X) = H(Z) - H(Z|X), and MI(Z,Y) = H(Z) - H(Z|Y), so we need to calculate
# H(Z) once and store it in the cache; entropy_cache dictionary will serve this purpose
# If we calculate MI(X,Z) = H(X) - H(X|Z), then entropy of each attribute will be
# cached, but that will be of no use as we cannot reuse those
entropy_cache = {}


def compute_entropy(fo, df, variables, cache=False):
    global entropy_cache
    df_id = id(df)
    if df_id not in entropy_cache:
        entropy_cache[df_id] = {}

    cache_of_df = entropy_cache[df_id]
    variables_as_key = tuple(sorted(variables))
    # cache hit
    if variables_as_key in cache_of_df:
        return cache_of_df[variables_as_key]

    val_map = {}
    if len(variables) < 2:
        variable_to_consider = variables[0]
        for val in df[variable_to_consider]:
            if val not in val_map:
                val_map[val] = 1
            else:
                val_map[val] += 1
        # print(val_map)
    else:

        # no_of_vars = len(variables)
        # print(no_of_vars)
        # print(range(no_of_vars))

        for item in df[variables].itertuples(index=False):
            vals = ''
            vals += str(item[0]) + str(item[1])
            if vals not in val_map:
                val_map[vals] = 1
            else:
                val_map[vals] += 1
        # print(val_map)

    N = len(df)

    entropy = 0.0
    for key in val_map:
        p = val_map[key] / N
        entropy += (-1) * p * log(p, 2)

    entropy = abs(entropy)

    # print("Entropy of {} is: {}".format(variables,entropy))

    return entropy


def compute_conditional_entropy(fo, left_variables, right_variables, mi_original, cache=False):

    conditional_entropy = 0

    all_vars_val_map = {}
    right_var_val_map = {}

    all_vars = left_variables + right_variables
    # print(all_vars)

    for item in df[all_vars].iterrows():
        # get right variable value
        # print(item[1][right[0]])

        cooccurred_values = str(
            item[1][left_variables[0]]) + str(item[1][left_variables[0]])
        right_var_val = str(item[1][left_variables[0]])

#         for i in range(no_of_vars):
#             cooccurred_values += str(item[1][all_vars[i]])
#             val_list.append(item[1][all_vars[i]])
#             print(item[1][all_vars[i]])
#         print("********")

#         val = str(val_list)
#         print(val)

        # print(cooccurred_values)

        # print("_________")
        if cooccurred_values not in all_vars_val_map:
            all_vars_val_map[cooccurred_values] = 1
        else:
            all_vars_val_map[cooccurred_values] += 1

        if right_var_val not in right_var_val_map:
            right_var_val_map[right_var_val] = 1
        else:
            right_var_val_map[right_var_val] += 1

    # print(all_vars_val_map)
    # print(right_var_val_map)

    N = len(df)

    for all_var_val in all_vars_val_map:

        right_var_val = all_var_val[1]

        p_all_var_val = all_vars_val_map[all_var_val] / N

        p_right_var_val = right_var_val_map[right_var_val] / N

        if p_right_var_val > 0 and p_all_var_val > 0:
            conditional_entropy += (-1) * p_all_var_val * \
                math.log((p_all_var_val / p_right_var_val), 2)
            # print('conditional_entropy: {}'.format(conditional_entropy))

    return conditional_entropy


def compute_mutual_information(fo, df, left_variables, right_variables, mi_original, mi_calculation_using_joint_entropy, cache=False):

    if mi_calculation_using_joint_entropy:
        entropy_a = compute_entropy(fo, df, left_variables, cache)
        entropy_b = compute_entropy(fo, df, right_variables, cache)
        a_b = left_variables + right_variables
        entropy_a_b = compute_entropy(fo, df, a_b, cache)

        # print("ent a: {}".format(entropy_a))
        # print("ent b: {}".format(entropy_b))
        # print("ent a_b: {}".format(entropy_a_b))

        mi = entropy_a + entropy_b - entropy_a_b

        # print("mi = {}".format(mi))

    else:
        mi = compute_entropy(fo, df, left_variables, cache) - \
            compute_conditional_entropy(
            fo, df, left_variables, right_variables, mi_original, cache)

    return mi


def compute_entropy_lower_bound(fo, df, variables, M, N, lambda_, p,  cache=False):
    entropy = compute_entropy(fo, df, variables, cache)
    # print("entropy is : {}".format(entropy))
    # print("lambda_ is : {}".format(lambda_))
    entropy_lb = entropy - lambda_
    return entropy_lb


def compute_entropy_upper_bound(fo, df, variables, M, N, lambda_, b_attr, p, cache=False):
    entropy = compute_entropy(fo, df, variables, cache)
    entropy_ub = entropy + lambda_ + b_attr
    return entropy_ub


def compute_lambda(M, N, beta_, p):
    a = M * (N - M) * log(2 / p)
    b = 2 * (N - (1 / 2)) * (1 - (1 / (2 * max(M, N - M))))
    lambda_ = beta_ * (sqrt(a / b))
    return lambda_


def compute_b_attr(df, variables, M, N):

    u_alpha = len(df[variables].unique())
    # print("u_alpha .. : {}".format(u_alpha))
    return log(1 + ((u_alpha - 1) * (N - M) / M * (N - 1)), 2)


def compute_b_attr_Z_pair(df, attr, Z, M, N):
    all_vars = [attr] + [Z]

    u_alpha_Z_pair = len(df[all_vars].drop_duplicates())
    # print("u_alpha_Z_pair .. : {}".format(u_alpha_Z_pair))
    return log(1 + ((u_alpha_Z_pair - 1) * (N - M) / M * (N - 1)), 2)


def compute_ac(df,  attr, target, cache=False):

    # N = len(df)
    variables = [attr, target]
    val_map = dict()
    for row in df[variables].itertuples(index=False):
        if row[0] not in val_map:
            val_map[row[0]] = dict()
            val_map[row[0]][row[1]] = 1
#             val_map[item[attr]]['conf'] = 0
        else:
            if row[1] not in val_map[row[0]]:
                val_map[row[0]][row[1]] = 1
#                 val_map[item[attr]]['conf'] = 0

            else:
                val_map[row[0]][row[1]] += 1

    # print(val_map)

    ac = 0
    for val in val_map:
        values_to_compare = val_map[val].values()
        if len(values_to_compare) > 1:
            ac += min(values_to_compare)

    return ac


def compute_aac(df,  attr, target, cache=False):

    N = len(df)
    variables = [attr, target]
    val_map = dict()
    for row in df[variables].itertuples(index=False):
        # item = row[1]
        if row[0] not in val_map:
            val_map[row[0]] = dict()
            val_map[row[0]][row[1]] = 1
#             val_map[item[attr]]['conf'] = 0
        else:
            if row[1] not in val_map[row[0]]:
                val_map[row[0]][row[1]] = 1
#                 val_map[item[attr]]['conf'] = 0

            else:
                val_map[row[0]][row[1]] += 1

    # print(val_map)

    aac = 0
    for val in val_map:
        values_to_compare = val_map[val].values()
        if len(values_to_compare) > 1:
            aac += min(values_to_compare) * \
                (sum(values_to_compare) / N)

    return aac
