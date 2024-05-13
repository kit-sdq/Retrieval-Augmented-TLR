import csv


def calculate_f1(results: list[tuple[str, str]], ground_truth_path: str, print_pairs: bool = True, reverse: bool = False) -> (float, float, float):
    with open(ground_truth_path, 'r') as file:
        reader = csv.reader(file)
        ground_truth = [tuple(row) for row in reader]

    if reverse:
        ground_truth = [(row[1], row[0]) for row in ground_truth]

    print("--------------------------------")
    print(len(ground_truth))
    print(len(results))

    true_pos = [prediction for prediction in results if prediction in ground_truth]
    false_pos = [prediction for prediction in results if prediction not in ground_truth]
    false_neg = [truth for truth in ground_truth if truth not in results]

    if print_pairs:
        print("--------------------------------")
        print("TRUE POSITIVES")
        for pair in true_pos:
            print(pair[0] + " : " + pair[1])

        print("--------------------------------")
        print("FALSE POSITIVES")
        for pair in false_pos:
            print(pair[0] + " : " + pair[1])

        print("--------------------------------")
        print("FALSE NEGATIVES")
        for pair in false_neg:
            print(pair[0] + " : " + pair[1])


    tp = len(true_pos)
    fp = len(results) - tp
    fn = len(ground_truth) - tp

    # Calculate precision
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    # Calculate recall
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    # Calculate F1 score
    f1 = (2 * (precision * recall)) / (precision + recall) if (precision + recall) > 0 else 0

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    return (precision, recall, f1)
