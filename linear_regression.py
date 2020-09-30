#Michael Rao
#1001558150

import os
import sys
import numpy 

#Load file 
def load_file(file_name):
    dataset = list()
    i = 0
    with open(file_name, 'r') as file:
        lines = file.readlines()
        for line in lines:
            i = i + 1
            dataset.append(line.split())
    return dataset, i

#Convert string to float
def create_floats(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Calculate activation function matrix
def act_funct(dataset, degree):
    act_matr = list()

    for x in range(len(dataset)):
        act_vect = list()
        for y in range(degree + 1):
            act_vect.append(dataset[x][0]**(y))
        act_matr.append(act_vect)
    return(act_matr)

def linear_regression():
    if(len(sys.argv) < 5):
        print("Insufficient command line args")
        exit()

    training_file = sys.argv[1]
    test_file = sys.argv[2]
    degree = int(sys.argv[3])
    lambda_in = int(sys.argv[4])

    training_set, train_size = load_file(training_file)
    test_set, test_size = load_file(test_file)
    test_label = list()
    t = list()

    for i in range(len(training_set[0])):
        create_floats(training_set, i)

    for i in range(len(test_set[0])):
        create_floats(test_set, i)

    for i in range(len(test_set)):
        test_label.append(test_set[i][-1])
        del(test_set[i][-1])

    for i in range(len(training_set)):
        t.append(training_set[i][-1])
    
    test_set = numpy.array(act_funct(test_set,degree))
    t = numpy.array(t)
    act_matrix = numpy.array(act_funct(training_set, degree))

    # Identity matrix
    Id_mat = numpy.identity(act_matrix.shape[1])
    
    #phi transpose times phi
    phiT_x_phi = numpy.matmul(numpy.transpose(act_matrix),act_matrix)

    #lambda times ident matrix
    lambda_ID = lambda_in * Id_mat

    #phi transpose times t
    phiT_t = numpy.matmul(numpy.transpose(act_matrix),t)
    w = numpy.matmul(numpy.linalg.pinv(numpy.add(lambda_ID,phiT_x_phi)),phiT_t)

    for i in range(len(w)):
        print("w{0:d}={1:.4f}".format(i,w[i]))

    for i in range(len(test_set)):
        output = numpy.matmul(test_set[i], w)
        print("ID={0:5d}, output={1:14.4f}, target value =  {2:14.4f}, squared error = {3:14.4f}".format(i+1, output,test_label[i], (test_label[i] - output)**2))
   
if __name__ == '__main__':
    linear_regression()