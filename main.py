import numpy as np
import time

# default rate for stats
def default_rate(data):
    labels = data[:, 0] 
    unique, counts = np.unique(labels, return_counts=True)  # count each class
    most_common_count = np.max(counts)
    return most_common_count / len(labels)

# forward algorithm, start with no features
# gradually adds features
def forward_search(data):
    curr_set_of_features = [] #initialize an empty set
    best_features = []
    best_accuracy = 0

    # runs once per level in search
    for i in range(data.shape[1] - 1):
        print(f"On the {i+1}th level of the search tree")
        feature_to_add_at_this_level = []
        best_accuracy_so_far = 0

        # adding features not already in set and evaluates the accuracy
        for k in range(1, data.shape[1]):
            # if is empty
            if k not in curr_set_of_features:
                # calculate accuracy
                accuracy = leave_one_out_cross_validation(data, curr_set_of_features, k)
                print(f"Using feature {set(curr_set_of_features + [k])}, the accuracy is {accuracy*100:.1f}%")
                if accuracy > best_accuracy_so_far:
                    best_accuracy_so_far = accuracy
                    feature_to_add_at_this_level = k

        # add more features if not empty
        if feature_to_add_at_this_level is not None:
            curr_set_of_features.append(feature_to_add_at_this_level) 
            print(f"Feature set {set(curr_set_of_features)} was best, accuracy is {best_accuracy_so_far*100:.1f}%")

            # if the best accuracy so far > the overall best accuracy, set as new overall best
            if best_accuracy_so_far > best_accuracy:
                best_accuracy = best_accuracy_so_far
                best_features = list(curr_set_of_features)
        else:
            print("(Warning: Accuracy has decreased! Continuing search in case of local maxima)")

    print(f"\nFinished search!\n The best feature subset is {set(best_features)}, which has an accuracy of {best_accuracy*100:.1f}%")

# forward search
# backwards search is same, just change the inequalities
# starts full, then gradually removes features
def backward_elimination(data):
    curr_set_of_features = list(range(1, data.shape[1]))
    best_features = list(curr_set_of_features)
    best_accuracy = 0

    # same as forward but reversed, one level per removal
    for i in range(data.shape[1] - 1):
        print(f"On the {i+1}th level of the search tree")
        feature_to_remove = None
        best_accuracy_so_far = 0

        # tries to remove features gradually and evaluates accuracy 
        for k in curr_set_of_features:
            # if is empty
            temp_set = [f for f in curr_set_of_features if f != k]
            if not temp_set:
                continue

            # calculate accuracy
            accuracy = leave_one_out_cross_validation(data, temp_set[:-1], temp_set[-1])
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

# can be omitted, converts to strings
def num2str(i):
    return str(i)
def int2str(i):
    return str(i)

# accuracy 
# echo numbers it returned to the screen
def leave_one_out_cross_validation(data, current_set, feature_to_add):
    number_correctly_classified = 0

    # iterates over every data point
    for i in range(data.shape[0]):
        features = current_set + [feature_to_add]
        object_to_classify = data[i, features]
        label_object_to_classify = data[i, 0]

        nearest_neighbor_distance = float('inf')
        nearest_neighbor_location = float('inf')
        # compare set with every other feature point excluding self
        for k in range(data.shape[0]):
            if k != i: # to prevent comparing to self
                distance = np.sqrt(np.sum((object_to_classify - data[k, features]) ** 2))
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = data[nearest_neighbor_location, 0]
                
        if label_object_to_classify == nearest_neighbor_label:
            number_correctly_classified += 1
    accuracy = number_correctly_classified / data.shape[0]
    return accuracy

# main :)
def main():
    data = []
    print("Welcome to the Feature Selection Algorithm")
    print("Input name of the dataset you want to use:")
    print("Ex: CS170_Small_DataSet__85.txt \n")

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

    # print default rate 
    rate = default_rate(data)
    print(f"Default rate: {rate*100:.1f}%\n")

    print(f"This dataset has {n_features} features (not including the class attribute), with {n_instances}.\n")

    # test features
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
    
if __name__ == "__main__":
    main()