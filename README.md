# aac-feature-selection
Experimental comparison of the new measure Attribute average Conflict with exact MI and other relevant methods


# Requirements for running the code
- Python 3.6.9  (https://www.python.org/downloads/release/python-369/)
- Following packages need to be installed
    - numpy (https://numpy.org/install/)
    - pandas (https://pandas.pydata.org/docs/getting_started/index.html)
    - scikit-learn (https://scikit-learn.org/stable/install.html)

1. For generating synthetic data, run the code 'synthetic_data_generation/generate_synthetic_dataset.py'. Changing different parameters inside the code
   will produce dataset with different distribution. Please see the comment in the code for parameter setting.

2. Once a datafile is generated, copy the file to the location - 'experiments/code/data_files/', then running the file 'experiments/code/evaluation.py'
   will produce experimental result in csv format. 
 
3. Make sure 'output_csv_files' and 'text_output' directories are present in the same location of evaluation.py file.

4. The output csv files will be located inside 'experiments/code/output_csv_files/'. Text file with the generated features along with scores
   using various methods will be generated inside 'experiments/code/text_output/' directory.

5. The program expects target variable (Z) name as 'Outcome'. Please rename the class/target column in the data file as 'Outcome' if any other name is present.

6. The program expects discrete attribute and target variable. You need to discretize any continuous valued-attribute using suitable method before using in this program.
   Put the datafile inside 'experiments/data_files/' directory before running evaluation.py

7. Real world datasets stated in the paper are provided under '/experiments/data/'  directory. Please unzip any zipped file and place under 'experiments/code/data_files/' directory in order to run the experiment using the file.


