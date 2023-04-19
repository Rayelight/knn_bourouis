import csv
import math

try:
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
except:
    pass

def extract_class(rows):
    classes = []
    rows_n = []
    for row in rows:
        rows_n.append(row[0:-1])
        classes.append(int(row[-1]))
    return rows_n, classes


def read_csv(filename):
    file = open(filename)
    csvreader = csv.reader(file, delimiter=';')

    rows = []
    for row in csvreader:
        row = list(map(float, row))
        rows.append(row)

    file.close()

    return rows


def euclidian_dist(p1, p2):
    dist = 0
    for i in range(len(p1)):
        dist += (p1[i] - p2[i]) ** 2
    return math.sqrt(dist)


def knn(k, datas, classes, points):
    classification = []

    for point in points:
        dists = []
        for data in datas:
            dists.append(euclidian_dist(point, data))

        nearest = []
        for _ in range(k):
            # formule pour trouver l'index le plus petit
            ind = min(zip(dists, range(len(dists))))[1]
            # eliminer la distance la plus petite pour qu'elle ne soit pas prise en compte dans la recherche du prochains point le plus proche
            dists[ind] = max(dists)
            # recuperer la classe dont l index correspond a la plus petite distance
            nearest.append(classes[ind])

        # ajoute dans le tableau des classes a predir l'element le plus present dans le tableau nearest
        classification.append(max(set(nearest), key=nearest.count))

    return classification


# variations du parametres k du knn pour tester le plus intéressant
def knn_comparaison(data):
    data, classes = extract_class(data)
    x_train, x_test, y_train, y_test = train_test_split(data, classes, train_size=int(len(classes) * 0.9),
                                                        random_state=0)

    knns = []
    scores = []
    for k in range(8):
        predicted = knn(k + 1, x_train, y_train, x_test)
        score = sum([x == y for x, y in zip(predicted, y_test)]) / len(predicted)

        scores.append(score)
        knns.append(knn)

        print("k: ", k + 1, " score: ", score)

    return knns, scores


# utilisé a titre de comparaison pour verifier que l'algorithme a bien fonctionné
def sklearn_knn_predict(k, data, points):
    data, classes = extract_class(data)

    # instanciation et définition du k
    knn = KNeighborsClassifier(n_neighbors=k)
    # training
    knn.fit(data, classes)

    return knn.predict(points)


def write_csv(filename, data):
    with open(filename, 'w', newline="") as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow([str(row)])


def main():
    data = read_csv("data/data.txt")
    knn_comparaison(data)
    datas, classes = extract_class(data)

    finalTest = read_csv("data/finalTest.txt")

    inducted_points = knn(3, datas, classes, finalTest)
    try:
        inducted_sk = sklearn_knn_predict(3, data, finalTest)
    except:
        inducted_sk = inducted_points
    finally:
        print(all(inducted_points == inducted_sk))

    write_csv("Bourouis_Bonnett_TDK.txt", inducted_points)


if __name__ == "__main__":
    main()
