import numpy as np

def log_data_quantities(y):
    number_of_zeros = np.sum(y == 0) 
    number_of_ones = np.sum(y == 1)
    number_of_twos = np.sum(y == 2)

    print(f"Number of zeros is {number_of_zeros}, Number of ones is {number_of_ones}, Number of twos is {number_of_twos}")