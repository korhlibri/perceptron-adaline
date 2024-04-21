import random
import matplotlib.pyplot as plt

# populate example database with random data for case 1
class case1_data_generation:
    filename = ""

    def __init__(self, filename):
        self.filename = filename

    def populate(self, rows):
        row = [0]*5

        i = 0
        while i < rows:

            j = 0
            while j < len(row)-1:
                row[j] = random.choice([0, 1])
                j += 1
            
            if (row[0] and row[1]) or (row[2] and not row[3]):
                row[-1] = 1
            else:
                row[-1] = 0
            
            with open(self.filename, "a") as f:
                f.write(",".join(map(str, row)) + "\n")
            
            i += 1

# create logistic map and the x values for case 2
class case2_data_generation:
    filename = ""

    def __init__(self, filename):
        self.filename = filename
    
    def logistic_formula(self, x):
        return 4.0 * x * (1 - x)

    def logistic_map(self):
        x0 = random.random()
        x = x0
        i = 0
        while i < 20:
            x = self.logistic_formula(x)
            i += 1
        i = 0
        with open("logistic_map.csv", "a") as f:
            while i < 1103:
                x = self.logistic_formula(x)
                f.write(str(x) + "\n")
                i += 1

    def logistic_values(self):
        with open("logistic_map.csv", "r") as f:
            with open(self.filename, "a") as v:
                logistics = f.readlines()
                i = 0
                while i + 3 < len(logistics):
                    v.write("{},{},{},{}\n".format(logistics[i][:-1], logistics[i+1][:-1], logistics[i+2][:-1], logistics[i+3][:-1]))
                    i += 1

class Perceptron:
    weight = []
    rate = 0
    epochs = 0

    error = []

    def __init__(self, rate, epochs):
        self.rate = rate
        self.epochs = epochs

    def predict(self, row):
        activation = self.weight[0]

        i = 0
        while i < len(row)-1:
            activation += self.weight[i+1] * row[i]
            i += 1

        return 1 if activation >= 0.0 else 0

    def fit(self, dataset):
        self.error = []
        self.weight = [0.0 for i in range(len(dataset[0]))]

        for epoch in range(self.epochs):
            error_sum = 0

            for row in dataset:
                prediction = self.predict(row)
                error = row[-1] - prediction
                error_sum += error**2
                self.weight[0] += self.rate * error

                i = 0
                while i < len(row)-1:
                    self.weight[i + 1] += self.rate * error * row[i]
                    i += 1
                
            self.error.append(error_sum)
            
        self.plot_error()

    def plot_error(self):
        plt.plot(self.error)
        plt.grid()
        plt.xlabel("Epochs")
        plt.ylabel("Error Amount")
        plt.show()
                

class Adaline:
    weight = []
    rate = 0
    epochs = 0

    error = []

    def __init__(self, rate, epochs):
        self.rate = rate
        self.epochs = epochs
    
    def activate(self, row):
        activation = self.weight[0]

        i = 0
        while i < len(row)-1:
            activation += self.weight[i+1] * row[i]
            i += 1
        
        return activation

    def predict(self, row):
        return 1 if self.activate(row) >= 0.5 else 0

    def fit(self, dataset):
        self.error = []
        self.weight = [0.0 for i in range(len(dataset[0]))]

        for epoch in range(self.epochs):
            error_sum = 0

            for row in dataset:
                prediction = self.activate(row)
                error = row[-1] - prediction
                error_sum += error**2
                self.weight[0] += self.rate * error

                i = 0
                while i < len(row)-1:
                    self.weight[i + 1] += self.rate * error * row[i]
                    i += 1
                
            self.error.append(error_sum)
            
        self.plot_error()

    def plot_error(self):
        plt.plot(self.error)
        plt.grid()
        plt.xlabel("Epochs")
        plt.ylabel("Error Amount")
        plt.show()

class case1_prediction:
    file = ""

    def __init__(self, file):
        self.file = file

    def predict(self, model):
        with open(self.file, "r") as f:
            rows = f.readlines()

            true_positive = 0
            false_positive = 0
            true_negative = 0
            false_negative = 0

            for row in rows:
                prepared_row = [int(x) for x in row[:-1].split(",")]
                predicted = model.predict(prepared_row)
                # print("{}. Predicted: {}. Expected: {}. Error: {}".format(prepared_row, predicted, prepared_row[-1], predicted != prepared_row[-1]))
                if predicted == prepared_row[-1]:
                    if predicted == 1:
                        true_positive += 1
                    else:
                        true_negative += 1
                else:
                    if predicted == 1:
                        false_positive += 1
                    else:
                        false_negative += 1
            precision = (true_positive / (true_positive + false_positive))*100
            recall = (true_positive / (true_positive + false_negative))*100
            f_measure = ((2 * true_positive) / (2 * true_positive + false_positive + false_negative))*100
            accuracy = ((true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative))*100
            print("\nPrecision: {:.2f}%, Recall: {:.2f}%, F measure: {:.2f}%".format(precision, recall, f_measure))
            print("Mean accuracy rating: {:.2f}%".format(accuracy))

class case2_prediction:
    file = ""
    y = []
    predicted_y = []
    error = []

    def __init__(self, file):
        self.file = file

    def predict(self, model):
        self.y = []
        self.predicted_y = []
        self.error = []

        with open(self.file, "r") as f:
            for row in f.readlines()[1000:]:
                prepared_row = [float(x) for x in row.split(",")]
                prepared_row = [prepared_row[0], prepared_row[1], prepared_row[2], prepared_row[0]**2, prepared_row[1]**2, prepared_row[2]**2, prepared_row[3]]
                self.y.append(prepared_row[-1])
                self.predicted_y.append(model.activate(prepared_row))
                self.error.append((self.y[-1] - self.predicted_y[-1])**2)
        
        self.plot_prediction()
    
    def plot_prediction(self):
        plt.plot(self.y, self.y, label="Expected")
        plt.plot(self.predicted_y, self.y, label="Predicted")
        plt.grid()
        plt.legend(loc="best")
        plt.show()

dataset = [[0,0,0,0,0],
[0,0,0,1,0],
[0,0,1,0,1],
[0,0,1,1,0],
[0,1,0,0,0],
[0,1,0,1,0],
[0,1,1,0,1],
[0,1,1,1,0],
[1,0,0,0,0],
[1,0,0,1,0],
[1,0,1,0,1],
[1,0,1,1,0],
[1,1,0,0,1],
[1,1,0,1,1],
[1,1,1,0,1],
[1,1,1,1,1]]

filename = "chaos_values.csv"

def case1_perceptron():
    filename = "trainingdata.csv"
    perceptron = Perceptron(0.01, 100)
    perceptron.fit(dataset)

    test = case1_prediction(filename)
    test.predict(perceptron)

def case1_adaline():
    filename = "trainingdata.csv"
    adaline = Adaline(0.01, 100)
    adaline.fit(dataset)

    test = case1_prediction(filename)
    test.predict(adaline)

def case2_adaline():
    filename = "chaos_values.csv"
    adaline = Adaline(0.001, 1000)

    prepared_dataset = []

    with open(filename, "r") as f:
        prepared_dataset = [list(map(float, x[:-1].split(","))) for x in f.readlines()[:999]]
    
    prepared_dataset_redundancy = []
    for row in prepared_dataset:
        row_redundancy = [row[0], row[1], row[2], row[0]**2, row[1]**2, row[2]**2, row[3],]
        prepared_dataset_redundancy.append(row_redundancy)

    adaline.fit(prepared_dataset_redundancy)

    test = case2_prediction(filename)
    test.predict(adaline)

# case1_perceptron()
# case1_adaline()
case2_adaline()