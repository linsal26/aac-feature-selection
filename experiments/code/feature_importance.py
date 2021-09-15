import mi_strategy_utils

from math import log, ceil
from sklearn.feature_selection import mutual_info_classif

import heapq


def get_largest_no_of_distinct_vals(df, X):

    # u_max will contain maximum no of distinct values among attributes in X
    u_max = 0

    for attr in X:
        # get the unique values of current attribute
        u_attr = df[attr].unique()

        # compare the no. of unique values of current attribute with u_max
        # if it is greater than u_max, then update u_max with this value
        if len(u_attr) > u_max:
            u_max = len(u_attr)

    return u_max

def sort_feature_by_mi_desc(fo, use_sample,  df, X, Z, k, mi_original, mi_calculation_using_joint_entropy, mi_sklearn, cache):

    attribute_mi_dict = {}

    if use_sample:
        attribute_urs_dict = {}
        for attr in X:
            feature = [attr]
            if mi_original:
                attribute_urs_dict[attr] = mi_strategy_utils.compute_mutual_information(
                    fo,  df, feature, [Z], mi_original, mi_calculation_using_joint_entropy, cache)
            else:
                attribute_urs_dict[attr] = mi_strategy_utils.compute_mutual_information(
                    fo,  df, [Z], feature, mi_original, mi_calculation_using_joint_entropy, cache)

        fo.write("\n.........................\n")
        fo.write("Urs MI:")
        fo.write(str(attribute_urs_dict))
        fo.write("\n.........................\n")

        return heapq.nlargest(k, attribute_urs_dict.keys(), key=lambda attr: (attribute_urs_dict[attr]))

    else:

        if mi_sklearn:

            X_arr = df[X].to_numpy()
            y_arr = df[Z].to_numpy()

            # discrete_features = [*range(len(X))]
            mi_sklearn_list = mutual_info_classif(
                X_arr, y_arr, discrete_features=[*range(len(X))])

            for index, attr in enumerate(X):
                # print(index, attr)
                # print('score of {}:{}'.format(attr, mi_sklearn_list[index]))
                attribute_mi_dict[attr] = mi_sklearn_list[index]

            # fo.write("\n.. mi_sklearn on original data...\n:")
            # for attr in attribute_mi_dict:
            #     fo.write("{},{}\n".format(attr, str(attribute_mi_dict[attr])))

        else:
            for attr in X:
                feature = [attr]
                if mi_original:
                    attribute_mi_dict[attr] = mi_strategy_utils.compute_mutual_information(
                        fo,  df, feature, [Z], mi_original, mi_calculation_using_joint_entropy, cache)
                else:
                    attribute_mi_dict[attr] = mi_strategy_utils.compute_mutual_information(
                        fo,  df, [Z], feature, mi_original, mi_calculation_using_joint_entropy, cache)

            # attribute_mi_dict = {k: v for k, v in sorted(
            #     attribute_mi_dict.items(), key=lambda x: (x[1]), reverse=True)}

        fo.write("\n.........................\n")
        fo.write("Original MI:")
        fo.write("\n No of records considered: {}\n".format(len(df)))
        fo.write(str(attribute_mi_dict))
        fo.write("\n.........................\n")

        return (heapq.nlargest(k, attribute_mi_dict.keys(), key=lambda attr: (attribute_mi_dict[attr])), attribute_mi_dict)


def sort_feature_by_urs(fo,  df, X, Z, k, mi_original, mi_calculation_using_joint_entropy, cache):
    attribute_urs_dict = {}

    for attr in X:
        feature = [attr]
        if mi_original:
            attribute_urs_dict[attr] = mi_strategy_utils.compute_mutual_information(
                fo,  df, feature, [Z], mi_original, mi_calculation_using_joint_entropy, cache)
        else:
            attribute_urs_dict[attr] = mi_strategy_utils.compute_mutual_information(
                fo,  df, [Z], feature, mi_original, mi_calculation_using_joint_entropy, cache)

    fo.write("\n.........................\n")
    fo.write("Urs MI:")
    fo.write(str(attribute_urs_dict))
    fo.write("\n.........................\n")

    return (heapq.nlargest(k, attribute_urs_dict.keys(), key=lambda attr: (attribute_urs_dict[attr])), attribute_urs_dict)


def sort_feature_by_swope(fo,  df, X, Z, k, p_f, epsilon, cache):

    # fo.write("\n######## swope calculation ###########\n")
    # print("p_f: {}, epsilon:{}".format(p_f, epsilon))
    # print("\n")
    # print("X: {}".format(X))
    # print("Z: {}".format(Z))

    H_lb = {}
    H_ub = {}
    MI_lb = {}
    MI_ub = {}

    b_prime = {}

    target_name = Z[0]

    # df_shuffled = df.sample(frac=1)

    N = len(df.index)

    # h is the no of attributes / independent variables
    h = len(X)

    # u_max is the max support size among all attributes in X, which is
    # the largest no. of distinct values of attribute in X
    u_max = get_largest_no_of_distinct_vals(df, X)

    # print("u_max: {}".format(u_max))

    M0 = log(h * log(N) / p_f) * pow(log(N), 2) / pow(log(u_max, 2), 2)

    # print("Calculated M0: {}".format(M0))

    M = ceil(M0)

    i_max = ceil(log((N / M0), 2)) + 1

    # print("i_max: {}".format(i_max))

    p_f_prime = p_f / (3 * i_max * (h - 1))

    # print("p_f_prime: {}".format(p_f_prime))

    # R will be the final list of top-k attributes to return
    R = []

    count = 0
    while M <= N:
        count += 1
        # print("M = {}".format(M))

        # fo.write("\n iteration {}".format(count))
        # fo.write("\n-----------------------------------\n")
        # fo.write("\n M = {} \n".format(M))
        # fo.write("\n Number of attrs to process: {} \n".format(len(X)))

        # get the first M records from df_shuffled, here df_shuffled is passed as df from calling module
        df_sampled = df.head(M)

        # fo.write("\n Number of records to process: {} \n".format(len(df_sampled)))

        beta_ = log(M / (M - 1), 2) + log((M - 1), 2) / M

        # calculate H_Z_lower, H_Z_upper, Z_b, and lambda_ by lemma 3 with p equal to p_f_prime
        p = p_f_prime

        lambda_ = mi_strategy_utils.compute_lambda(M, N, beta_, p)

        b_Z = mi_strategy_utils.compute_b_attr(
            df_sampled, target_name, M, N)  # this is b_alpha_t

        H_Z_lower = mi_strategy_utils.compute_entropy_lower_bound(
            fo,  df_sampled, Z, M, N, lambda_, p, cache)
        H_Z_upper = mi_strategy_utils.compute_entropy_upper_bound(
            fo,  df_sampled, Z, M, N, lambda_, b_Z, p, cache)

        # print("H_Z_lower: {}, H_Z_upper: {}, lambda_ : {}, b_Z:{}".format(
        #     H_Z_lower, H_Z_upper, lambda_, b_Z))

        H_lb[Z[0]] = H_Z_lower
        H_ub[Z[0]] = H_Z_upper

        for attr in X:
            b_attr = mi_strategy_utils.compute_b_attr(
                df_sampled, attr, M, N)  # this is b_alpha
            H_attr_lower = mi_strategy_utils.compute_entropy_lower_bound(
                fo,  df_sampled, [attr], M, N, lambda_, p, cache)
            H_attr_upper = mi_strategy_utils.compute_entropy_upper_bound(
                fo,  df_sampled, [attr], M, N, lambda_, b_attr, p, cache)

            H_lb[attr] = H_attr_lower
            H_ub[attr] = H_attr_upper

            # this is b(aplha_t, alpha) ; alpha = attr, alpha_t = taret/Z
            b_attr_Z_pair = mi_strategy_utils.compute_b_attr_Z_pair(
                df_sampled, attr, target_name, M, N)

            attr_Z_pair = [attr] + Z

            H_attr_Z_lower = mi_strategy_utils.compute_entropy_lower_bound(
                fo,  df_sampled, attr_Z_pair, M, N, lambda_, p, cache)
            H_attr_Z_upper = mi_strategy_utils.compute_entropy_upper_bound(
                fo,  df_sampled, [attr], M, N, lambda_, b_attr_Z_pair, p, cache)

            # print("for attr : {}, H_attr_lower: {}, H_attr_upper: {}, b_attr_Z_pair:{}".format(
            #     attr, H_attr_lower, H_attr_upper, b_attr_Z_pair))
            # print("... H_attr_Z_lower : {}, H_attr_Z_upper:{}".format(
            #     H_attr_Z_lower, H_attr_Z_upper))

            # print("\n")

            mi_attr_Z_lower = H_Z_lower + H_attr_lower - H_attr_Z_upper
            mi_attr_Z_upper = H_Z_upper + H_attr_upper - H_attr_Z_lower

            MI_lb[attr] = mi_attr_Z_lower
            MI_ub[attr] = mi_attr_Z_upper

            # fo.write("\nfor {} - MI_ub: {}, MI_lb: {}\n".format(attr,
            #                                                     MI_ub[attr], MI_lb[attr]))

            b_prime_attr = b_Z + b_attr + b_attr_Z_pair

            b_prime[attr] = b_prime_attr

            # h_lb[attr] = H_attr_lower
            # h_ub[attr] = H_attr_upper

        # print("h_lb: ")
        # print(H_lb)

        # print("h_ub: ")
        # print(H_ub)

        # print("MI_lb:")
        # print(MI_lb)

        # print("MI_ub:")
        # print(MI_ub)

        # print("b_prime:")
        # print(b_prime)

        # R is the list of top-k attributes from X according to MI_ub
        R = heapq.nlargest(k, MI_ub.keys(), key=lambda attr: (MI_ub[attr]))

        # print("R : ")
        # print(R)
        # print("last element of R: {} ".format(R[-1]))

        # get the k-th largest MI_ub
        MI_ub_k = MI_ub[R[-1]]

        b_prime_max = 0
        # get b_prime_max for aplha \in R
        for attr in R:

            # print("attr, b_prime_attr: {},{}".format(attr, b_prime[attr]))

            if b_prime[attr] > b_prime_max:
                b_prime_max = b_prime[attr]

        # print("b_prime_max = {}".format(b_prime_max))

        # print("\n\n ... while loop ran {} times".format(count))

        if (MI_ub_k - 6 * lambda_ - b_prime_max) / MI_ub_k >= (1 - epsilon):
            # print('reached first condition')
            # fo.write("\n ..  reached first condition .. \n")
            # fo.write("While loop ran {} times\n".format(count))
            # for attr in R:
            #     fo.write("\nfor {}- MI_ub: {}, MI_lb: {}\n".format(
            #         attr, MI_ub[attr], MI_lb[attr]))
            return R
        elif M < N:
            M = min(N, 2 * M)
            # print('doubling M .. ')
            # print("M = {}".format(M))
            # fo.write("\n doubling M ..")
            # fo.write(" M = {}\n".format(M))
        else:
            # print('reached else .. breaking..')
            break

        # print(".. printing MI_ub..")
        # for item in MI_ub:
        #     print(item, MI_ub[item])

        # P is the list of top-k attributes from X according to MI_lb
        P = heapq.nlargest(k, MI_lb.keys(), key=lambda attr: (MI_lb[attr]))

        # get the k-th largest lower bound for MI of attributes
        MI_lb_k = MI_lb[P[-1]]

        for attr in X:
            if MI_ub[attr] < MI_lb_k:
                X.remove(attr)

    # fo.write("While loop ran {} times".format(count))

    # fo.write("\n######## end of calculation###########\n")

    return R


def sort_feature_by_g3(fo,  df, X, Z, k, cache):
    ac_dict = {}

    for attr in X:
        ac_dict[attr] = mi_strategy_utils.compute_g3(
            df,  attr, Z, cache)
    fo.write("\n.........................\n")
    fo.write("G3 Error:")
    fo.write("\n No of records considered: {}\n".format(len(df)))
    fo.write(str(ac_dict))
    fo.write("\n.........................\n")

    return heapq.nsmallest(k, ac_dict.keys(), key=lambda attr: (ac_dict[attr]))


def sort_feature_by_aac(fo,  df, X, Z, k, cache):
    aac_dict = {}

    for attr in X:
        aac_dict[attr] = mi_strategy_utils.compute_aac(
            df,  attr, Z, cache)

    fo.write("\n.........................\n")
    fo.write("Expected Error:")
    fo.write("\n No of records considered: {}\n".format(len(df)))
    fo.write(str(aac_dict))
    fo.write("\n.........................\n")

    return heapq.nsmallest(k, aac_dict.keys(), key=lambda attr: (aac_dict[attr]))
