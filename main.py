import json
import numpy as np


def main():
    print("Input the dataset you want to use:\n")
    print("Ex: CS170_Small_DataSet_85")


    "On the _th level of the search tree"
    "--Considering adding the xxx feature"
    "On level x I added feature x to current set"



    return

if __name__ == "__main__":
    main()

size = 0

def accuracy(data, current_set, feature_to_add):
    # accuracy = rand
    return


# // printing
    "On the _th level of the search tree"
    "--Considering adding the xxx feature"
    "On level x I added feature x to current set"

# search algorithm
def feature_search_demo(data):
    curr_set_of_features = [] #initialize an empty set

    for i in size(data, 1) - 1:
        print("On the ", num2str(i), "th level of the search tree")
        feature_to_add_at_this_level = []
        best_accuracy_so_far = 0

        for k in size(data, 1)-1:
            if isempty(intersect(curr_set_of_features_k))

            print("Considering adding the", num2str(k) ,"feature")
            accuracy = leave_one_out_cross_validation(data, curr_set_of_features, k+1)

            if accuracy > best_accuracy_so_far:
                best_accuracy_so_far = accuracy
                feature_to_add_at_this_level = k;

        curr_set_of_features(i) = feature_to_add_at_this_level
        print("On level", num2str(i), "I added feature ", num2str(feature_to_add_at_this_level), " to current set")

# forward search
#backwards search is same, just change the inequalities

def num2str(index):
    return

#accuracy function
# def leave_one_out_cross_validation(features):
#     #echo the numbers it returned to the screen
#     return 


def int2str(i):
    return

#accuracy 
# echo numbers it returned to the screen
def leave_one_out_cross_validation(data, current_set, feature_to_add):
    # data = load('filename)
    # data = []
    # with open(filename, "r") as f:
    number_correctly_classified = 0

    for i in range(data.shape[0]):
        object_to_classify = data[i, 1:]; # to end
        label_object_to_classify = data(i,1)

        nearest_neighbor_distance = float('inf')
        nearest_neighbor_location = float('inf')
        # print("Looping over i, at the ", int2str(i), " location")
        # print("The ", int2str(i), "th object is in class ", num2str(label_object_to_classify))
        for k in range(data.shape[0]):
            print(["Ask if", int2str(i), "is nearest neighbor with ", int2str(k)])
            if k != i: # to prevent comparing to self
                
                distance = np.sqrt(np.sum((object_to_classify - data[k, 1:]) ** 2))

                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = data(nearest_neighbor_location, 1)
        # print(["Object ", num2str(i), "is class ", num2str(label_object_to_classify)])
        # print("Its nearest_neighbor is ", num2str(nearest_neighbor_location), " which is in class ", num2str(nearest_neighbor_label))        
        if label_object_to_classify == nearest_neighbor_label:
            number_correctly_classified += 1
    accuracy = number_correctly_classified / data.shape[0]


#early abandoning for omtimization, use global variable inside the fns


def label_object_to_classify():
    return