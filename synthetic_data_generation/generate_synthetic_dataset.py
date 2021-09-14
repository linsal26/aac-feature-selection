# This code is developed by Md Abdus Salam (Lincoln) (mdsalam@uta.edu/lincoln.cse.uta@gmail.com). Use of any of the materials in this codebase
# in any program/paper without prior permission from the authoer is prohibited.
# Date : September 14, 2021

import numpy as np
import pandas as pd
import os


def generate_target_variable(N, p_0):
    z_0 = int(N * p_0) * [0]
    # print(len(z_0))
    z_1 = int(N - len(z_0)) * [1]

    # print(z_0, len(z_0))
    # print(z_1, len(z_1))

    z = z_0 + z_1

    return z


def generate_attribute(N, p):
    attr_val = []
    for i in range(0, len(p)):
        # print(i, p[i])
        if i == len(p) - 1:
            attr_val += int(N - len(attr_val)) * [i]
        else:
            attr_val += int(N * p[i]) * [i]

    return attr_val

def main():

    # no_of_records = [1000000, 1500000, 2000000, 2500000, 3000000]
    no_of_records = [1000000]

    # no_of_attrs = [50, 60, 70, 80, 90, 100]
    no_of_attrs = [50]

    # p_0_list = [.1, .2, .3, .4, .5, .6, .7, .8, .9]

    # p_0_list = [.1, .2, .3, .4, .5, .6, .7]

    # this is the probability of z=0
    p_0_list = [.000005]

    # this is the difference of  probability of 0-values between consecutive attributes
    # i.e. p(x2=0) - p(x1=0)
    probability_difference = .000001

    # this is the value of p(x0=0) i.e., probablity of 0 value for the first attribute
    starting_probabaility = .999999

    prob_diff_suffix = str(probability_difference).rsplit('.')[-1]

    output_csv_folder = 'binary_attributes_prob_diff_' + prob_diff_suffix + '/'

    if not os.path.isdir(output_csv_folder):
        print("No, the folder {} does not exist".format(output_csv_folder))
        print("\n.. creating the folder ..\n")
        os.mkdir(output_csv_folder)

    for N in no_of_records:

        for p_0 in p_0_list:

            for M in no_of_attrs:

                p0_suffix = str(p_0).rsplit('.')[-1]

                attr_dict = {}

                # target variable Z is named as 'Outcome' in the data file
                attr_dict['Outcome'] = generate_target_variable(N, p_0)

                # print("z = {}".format(z))

                start_p = starting_probabaility

                list_of_p = []

                for i in range(M):
                    p = start_p - (i * probability_difference)

                    distribution = [p, 1 - p]

                    list_of_p.append(distribution)

                # print(list_of_p)
                # for index, item in enumerate(list_of_p):
                #     print("{},{}".format(item[0], item[1]))
                # print("\n***********************\n")

                # exit()

                for i in range(0, len(list_of_p)):
                    var_name = 'x' + str(i)
                    attr_dict[var_name] = generate_attribute(N, list_of_p[i])

                # for k in attr_dict:
                    # print("printing .. {}".format(k))
                    # print(attr_dict[k])
                    # print(len(attr_dict[k]))
                    # print("\n....\n")

                # print(attr_dict)

                # x1 = generate_attribute(N, p)

                # print("x1")
                # print(x1)

                df = pd.DataFrame(attr_dict)

                # print(df)

                output_file_name = output_csv_folder + 'N_' + str(N) + '_M_' + \
                    str(len(list_of_p)) + '_starting_prob_' + str(starting_probabaility) + '_prob_diff_' + str(probability_difference) + '_z0_point_' + p0_suffix + \
                    '_binary_controlled_data_with_shuffle.csv'

                df_shuffled = df.sample(frac=1)

                df_shuffled.to_csv(output_file_name, index=False)

                # print(df['x0'].value_counts())

                # df.to_csv(output_file_name, index=False)

    return


if __name__ == "__main__":
    main()
