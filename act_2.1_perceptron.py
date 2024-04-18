from random import choice

# populate example database with random data
class Formula:
    filename = ""

    def __init__(self, filename):
        self.filename = filename

    def populate(self, rows):
        row = [0]*5

        i = 0
        while i < rows:

            j = 0
            while j < len(row)-1:
                row[j] = choice([0, 1])
                j += 1
            
            if (row[0] and row[1]) or (row[2] and not row[3]):
                row[-1] = 1
            else:
                row[-1] = 0
            
            with open(self.filename, "a") as f:
                f.write(",".join(map(str, row)) + "\n")
            
            i += 1

# stocastic gradient descent
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
                error_sum += error
                self.weight[0] += self.rate * error

                i = 0
                while i < len(row)-1:
                    self.weight[i + 1] += self.rate * error * row[i]
                    i += 1
                
            print("Epoch {}: Error Sum = {}".format(epoch+1, error_sum))

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

filename = "testdata.csv"

# prepare_test_data = Formula(filename)
# prepare_test_data.populate(1000)

perceptron = Perceptron(0.01, 20)
perceptron.fit(dataset)
# print()
# print(perceptron.weight)
# print()

with open("testdata.csv", "r") as f:
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
            predicted = perceptron.predict(prepared_row)
            # print("{}. Predicted: {}. Expected: {}. Error: {}".format(prepared_row, predicted, prepared_row[-1], predicted != prepared_row[-1]))
            if predicted == prepared_row[-1]:
                accurate += 1
            else:
                inaccurate += 1
            row += 1
        print("Perceptron accuracy rating with test data k={}: {:.2f}%".format(curk+1, accurate/(accurate+inaccurate)*100))
        accurate_total += accurate
        inaccurate_total += inaccurate
        curk += 1
    print("\nMedian perceptron accuracy rating: {:.2f}%".format(accurate_total/(accurate_total+inaccurate_total)*100))