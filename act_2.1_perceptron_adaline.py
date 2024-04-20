import random
import matplotlib.pyplot as plt

# populate example database with random data
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
            while i < 1000:
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
                
            print("Epoch {}: Error Sum = {}".format(epoch+1, error_sum))

class Adaline:
    weight = []
    rate = 0
    epochs = 0

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
                
            print("Epoch {}: Error Sum = {}".format(epoch+1, error_sum))

class TestDataPrediction:
    file = ""

    def __init__(self, file):
        self.file = file

    def predict(self, model):
        with open(self.file, "r") as f:
            rows = f.readlines()

            accurate_total = 0
            inaccurate_total = 0

            k = 3
            curk = 0
            row = 0

            while curk < k:
                accurate = 0
                inaccurate = 0

                while row < (len(rows)/3)*(curk+1):
                    prepared_row = [int(x) for x in rows[row][:-1].split(",")]
                    predicted = model.predict(prepared_row)
                    # print("{}. Predicted: {}. Expected: {}. Error: {}".format(prepared_row, predicted, prepared_row[-1], predicted != prepared_row[-1]))
                    if predicted == prepared_row[-1]:
                        accurate += 1
                    else:
                        inaccurate += 1
                    row += 1
                print("Accuracy rating with test data k={}: {:.2f}%".format(curk+1, accurate/(accurate+inaccurate)*100))
                accurate_total += accurate
                inaccurate_total += inaccurate
                curk += 1
            print("\nMean accuracy rating: {:.2f}%".format(accurate_total/(accurate_total+inaccurate_total)*100))


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

# CASE 1 PREPARATION
# prepare_test_data = case1_data_generation(filename)
# prepare_test_data.populate(1000)

# CASE 1 FITTING
# perceptron = Perceptron(0.01, 100)
# perceptron.fit(dataset)

# print()

# adaline = Adaline(0.1, 100)
# adaline.fit(dataset)

# CASE 1 TESTING
# test = TestDataPrediction(filename)
# print("\nPerceptron")
# test.predict(perceptron)
# print("\nAdaline")
# test.predict(adaline)

# CASE 2 PREPARATION
# logic_map = case2_data_generation("chaos_values.csv")
# logic_map.logistic_map()
# logic_map.logistic_values()

# CASE 2 FITTING
prepared_dataset = []
with open(filename, "r") as f:
    prepared_dataset = [list(map(float, x[:-1].split(","))) for x in f.readlines()]

adaline = Adaline(0.001, 500)
adaline.fit(prepared_dataset)

y = []
predicted_y = []

for row in prepared_dataset:
    y.append(row[-1])
    predicted_y.append(adaline.activate(row))

plt.plot(y, y)
plt.plot(predicted_y, y)
plt.show()