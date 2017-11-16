import numpy as np
from collections import Counter, defaultdict


def occurrences(list1):
    no_of_examples = len(list1)
    prob = dict(Counter(list1))
    for key in prob.keys():
        prob[key] = prob[key] / float(no_of_examples)
    return prob


def naive_bayes(training, outcome, new_sample):
    classes = np.unique(outcome)
    rows, cols = np.shape(training)
    likelihoods = {}
    for cls in classes:
        likelihoods[cls] = defaultdict(list)

    class_probabilities = occurrences(outcome)

    for cls in classes:
        row_indices = np.where(outcome == cls)[0]
        subset = training[row_indices, :]
        r, c = np.shape(subset)
        for j in range(0, c):
            likelihoods[cls][j] += list(subset[:, j])

    for cls in classes:
        for j in range(0, cols):
            likelihoods[cls][j] = occurrences(likelihoods[cls][j])

    results = {}
    for cls in classes:
        class_probability = class_probabilities[cls]
        for i in range(0, len(new_sample)):
            relative_values = likelihoods[cls][i]
            if new_sample[i] in relative_values.keys():
                class_probability *= relative_values[new_sample[i]]
            else:
                class_probability *= 0
            results[cls] = class_probability
    print("Probability of FLU ==> ",results)
    total=0
    for i in results:
        total+=pow(results[i],2)
    total=pow(total,0.5)
    for i in results:
        results[i]=results[i]/total
    print("Normalized Probability of FLU ==> ",results)
    max=0
    ans=0
    for i in results:
        if(results[i]>max):
            ans=i
            max=results[i]
        #print(max,ans)
    if ans == 0:
        print("Happy Mood")
    else:
        print("Sad Mood")

if __name__ == "__main__":
    training = np.asarray(((1, 0, 1, 1,1), (1, 1, 0, 0,0), (1, 0, 2, 1,1), (0, 1, 1, 1,0), (0, 0, 0, 0,0), (0, 1, 2, 1,1),
                           (0, 1, 2, 0,0), (1, 1, 1, 1,1)));
    outcome = np.asarray((0, 1, 1, 1, 0, 1, 0, 1))
    new_sample = np.asarray((1, 0, 2, 0,1))
    print("\tFight TV-watching	Homework	No-Games	New Skill Learning	Sad")
    print("\tYes   No            Strong        No           Yes            Yes/NO")
    print("\tIf Yes then Sad Mood If No then Happy Mood\n\n")
    naive_bayes(training, outcome, new_sample)