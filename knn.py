import csv
import time
import random


def read_dataset(file_path):
    dataset = []
    with open(file_path) as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            if row:
                datapoint = [float(value) for value in row[:4]]
                datapoint.append(row[4])  # Adicionando a classe
                dataset.append(datapoint)
    return dataset


def euclidean_distance(subj1, subj2):
    distance = sum((a - b) ** 2 for a, b in zip(subj1, subj2[:-1]))
    return distance ** 0.5


def predict_class(point, data_train, k):
    distances_and_classes = [(euclidean_distance(
        point, train_point), train_point[-1]) for train_point in data_train]
    distances_and_classes.sort(key=lambda x: x[0])
    nearest_classes = [cls for _, cls in distances_and_classes[:k]]

    class_counts = {}
    for cls in nearest_classes:
        class_counts[cls] = class_counts.get(cls, 0) + 1

    return max(class_counts, key=class_counts.get)


def calculate_accuracy(data_test, data_train, k):
    correct_predictions = 0
    total_predictions = len(data_test)

    for test_point in data_test:
        predicted_class = predict_class(test_point, data_train, k)
        if predicted_class == test_point[-1]:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions * 100
    return accuracy


def main():
    start_time = time.time()
    file_path = "./iris/iris.data"
    dataset = read_dataset(file_path)
    random.shuffle(dataset)

    test_percentage = 0.3
    test_size = int(len(dataset) * test_percentage)
    data_test = dataset[:test_size]
    data_train = dataset[test_size:]

    k_values = [1, 3, 5, 7, 9, 11, 13, 15]
    max_accuracy = 0
    best_k = 1
    print("\nDataset lido")
    print("-------------------------------------------------------")
    print("Algoritmo do KNN para classificação com a IrisDatabase")
    print("-------------------------------------------------------")

    for k in k_values:
        accuracy = calculate_accuracy(data_test, data_train, k)
        print(f"k = {k} | Acertos: {accuracy:.2f}%")
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_k = k

    print(f"\nMelhor valor de K: {best_k} \nPrecisão de: {max_accuracy:.2f}%")
    end_time = time.time()
    print(f"\nTempo de execução: {end_time - start_time:.3f} segundos")


if __name__ == "__main__":
    main()
