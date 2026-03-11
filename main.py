import json
import numpy as np
import time

size = 0

# // printing
    # "On the _th level of the search tree"
    # "--Considering adding the xxx feature"
    # "On level x I added feature x to current set"

global_accuracy = 0; 
# forward algorithm
def forward_search(data):
    curr_set_of_features = [] #initialize an empty set
    best_features = []
    best_accuracy = 0

    for i in range(data.shape[1] - 1):
        print(f"On the {i+1}th level of the search tree")
        feature_to_add_at_this_level = []
        best_accuracy_so_far = 0

        for k in range(1, data.shape[1]):
            # if is empty
            if k not in curr_set_of_features:
                # print(f"Considering adding the {k} feature")
                accuracy = leave_one_out_cross_validation(data, curr_set_of_features, k)
                print(f"Using feature {set(curr_set_of_features + [k])}, the accuracy is {accuracy*100:.1f}%")
                if accuracy > best_accuracy_so_far:
                    best_accuracy_so_far = accuracy
                    feature_to_add_at_this_level = k

        if feature_to_add_at_this_level is not None:
            curr_set_of_features.append(feature_to_add_at_this_level) 
            print(f"Feature set {set(curr_set_of_features)} was best, accuracy is {best_accuracy_so_far*100:.1f}%")

            if best_accuracy_so_far > best_accuracy:
                best_accuracy = best_accuracy_so_far
                best_features = list(curr_set_of_features)
        else:
            print("(Warning: Accuracy has decreased! Continuing search in case of local maxima)")

        # curr_set_of_features[i] = feature_to_add_at_this_level
        # print("On level", num2str(i), "I added feature ", num2str(feature_to_add_at_this_level), " to current set")
    print(f"\nFinished search!\n The best feature subset is {set(best_features)}, which has an accuracy of {best_accuracy*100:.1f}%")


# forward search
# backwards search is same, just change the inequalities
# starts full, then remove 
def backward_elimination(data):
    curr_set_of_features = list(range(1, data.shape[1]))
    best_features = list(curr_set_of_features)
    best_accuracy = 0

    for i in range(data.shape[1] - 1):
        print(f"On the {i+1}th level of the search tree")
        feature_to_remove = None
        best_accuracy_so_far = 0

        for k in curr_set_of_features:
            # if is empty
            temp_set = [f for f in curr_set_of_features if f != k]
            if not temp_set:
                continue

            accuracy = leave_one_out_cross_validation(data, temp_set[:-1], temp_set[-1])
            # print(f"Removing feature {k}, remaining {set(temp_set)}, accuracy is {accuracy*100:.1f}%")
            print(f"Using feature {set(curr_set_of_features + [k])}, the accuracy is {accuracy*100:.1f}%")

            if accuracy > best_accuracy_so_far:
                best_accuracy_so_far = accuracy
                feature_to_remove = k

        if feature_to_remove is not None:
            curr_set_of_features.remove(feature_to_remove) 
            print(f"On level {i}, feature '{feature_to_remove}' was removed in current set.")
            print(f"Feature set {set(curr_set_of_features)} was best, accuracy is {best_accuracy_so_far*100:.1f}%")

            if best_accuracy_so_far > best_accuracy:
                best_accuracy = best_accuracy_so_far
                best_features = list(curr_set_of_features)
        else:
            print("(Warning: Accuracy has decreased! Continuing search in case of local maxima)")

    print(f"\nFinished search!\n The best feature subset is {set(best_features)}, which has an accuracy of {best_accuracy*100:.1f}%")
    return

def num2str(i):
    # print("filler for num2str")
    return str(i)

def int2str(i):
    # print("filler for int2str")
    return str(i)

#accuracy 
# echo numbers it returned to the screen
def leave_one_out_cross_validation(data, current_set, feature_to_add):
    number_correctly_classified = 0

    for i in range(data.shape[0]):

        features = current_set + [feature_to_add]
        object_to_classify = data[i, features]
        label_object_to_classify = data[i, 0]

        nearest_neighbor_distance = float('inf')
        nearest_neighbor_location = float('inf')
        # print("Looping over i, at the ", int2str(i), " location")
        # print("The ", int2str(i), "th object is in class ", num2str(label_object_to_classify))
        for k in range(data.shape[0]):
            # print("Ask if", int2str(i), "is nearest neighbor with ", int2str(k))
            if k != i: # to prevent comparing to self
                
                distance = np.sqrt(np.sum((object_to_classify - data[k, features]) ** 2))

                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = data[nearest_neighbor_location, 0]
        
        # print(["Object ", num2str(i), "is class ", num2str(label_object_to_classify)])
        # print("Its nearest_neighbor is ", num2str(nearest_neighbor_location), " which is in class ", num2str(nearest_neighbor_label))        
        
        if label_object_to_classify == nearest_neighbor_label:
            number_correctly_classified += 1
    accuracy = number_correctly_classified / data.shape[0]
    return accuracy


#early abandoning for omtimization, use global variable inside the fns

def label_object_to_classify():
    return

# main :)
def main():
    data = []
    print("Welcome to the Feature Selection Algorithm")
    print("Input name of the dataset you want to use:")
    print("Ex: CS170_Small_DataSet__85.txt \n")
    #CS170_Small_DataSet__85.txt
    #CS170_Large_DataSet__98.txt
    #SanityCheck_DataSet__1.txt
    #SanityCheckDataSet__2.txt

    filename = input().strip()
    
    if(not filename):
        print("FILE NOT FOUND")
        return       

    data = np.loadtxt(filename)
    n_features = data.shape[1]-1
    n_instances = data.shape[0]

    print("Input the number of the algorithm you want to run")
    print("1) Forward Selection")
    print("2) Backward Selection \n")
    
    option = input().strip()


    print(f"This dataset has {n_features} features (not including the class attribute), with {n_instances}.\n")

    features = list(range(1, n_features+1))
    accuracy = leave_one_out_cross_validation(data, features[:-1], features[-1])
    print(f"Running nearest neighbor with all {n_features} features, using leaving-one-out evaluation, I get an accuracy of {accuracy*100:.1f}% ")
    print("Beginning search...\n")

    if(option == "1"):
        print("Forward Selection Algorithm\n")
        start_time = time.time()
        forward_search(data)
        end_time = time.time()

        print(f"Total time: ~ {end_time - start_time:.2f} seconds")

    elif(option == "2"):
        print("Backward Selection Algorithm\n")
        start_time = time.time()
        backward_elimination(data)
        end_time = time.time()

        print(f"Total time: ~ {end_time - start_time:.2f} seconds")

    else:
        print("Invalid input. Only enter 1 or 2\n")
        return


    # print(f"Running nearest neighbor with all {n_features} features, using {evaluation} evaluation, I get an accuracy of {accuracy}% ")

    # "On the _th level of the search tree"
    # "--Considering adding the xxx feature"
    # "On level x I added feature x to current set"

    # return

if __name__ == "__main__":
    main()