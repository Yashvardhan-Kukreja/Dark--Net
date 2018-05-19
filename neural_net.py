import numpy as np

def sigmoid(x):
    return (1/(1 + np.exp(-x)))

# Training data containing input rgb values of dark colors. A 3x20 matrix
dark_cols = [[139, 137, 137],
             [205, 192, 176],
             [139, 69, 19],
             [49, 79, 79],
             [105, 105, 105],
             [25, 25, 112],
             [72, 61, 139],
             [0, 0, 205],
             [0, 100, 0],
             [85, 107, 47],
             [34, 139, 34],
             [218, 165, 32],
             [184, 134, 11],
             [178, 34, 34],
             [176, 48, 96],
             [165, 42, 42],
             [148, 0, 211],
             [160, 32, 240],
             [153, 50, 204],
             [70, 130, 180]]

# Training data containing input rgb values of light colors. A 3x20
light_cols = [[250, 235, 215],
              [253, 245, 230],
              [255, 250, 205],
              [255, 228, 225],
              [135, 206, 250],
              [175, 238, 238],
              [127, 255, 212],
              [152, 251, 152],
              [250, 250, 0],
              [255, 215, 0],
              [255, 192, 203],
              [255, 105, 180],
              [218, 112, 214],
              [255, 69, 0],
              [255, 14, 0],
              [238, 130, 238],
              [238, 232, 170],
              [240, 230, 140],
              [245, 222, 179],
              [244, 164, 96]]

# Training data containing output values. 0 for dark, 1 for light. A 1x40 matrix
output = [[0],
          [0],
          [0],
          [0],
          [0],
          [0],
          [0],
          [0],
          [0],
          [0],
          [0],
          [0],
          [0],
          [0],
          [0],
          [0],
          [0],
          [0],
          [0],
          [0],
          [1],
          [1],
          [1],
          [1],
          [1],
          [1],
          [1],
          [1],
          [1],
          [1],
          [1],
          [1],
          [1],
          [1],
          [1],
          [1],
          [1],
          [1],
          [1],
          [1]]

# Input training data
X = np.array(dark_cols+light_cols)

# Output training data
Y = np.array(output)

# Defining the learning rate
lr = 0.1

# Initialising random weights for the neural net
weights0 = 2*np.random.random((3, 40)) - 1
weights1 = 2*np.random.random((40, 1)) - 1


for i in range(10000):

    l0 = X   ## Input layer
    l1 = sigmoid(np.dot(l0, weights0))  ## First hidden layer
    l2 = sigmoid(np.dot(l1, weights1))  ## Second hidden layer


    ''' Applying backpropagation :'( '''

    # Finding the error and delta for second hidden layer
    l2_err = Y - l2
    l2_delta = l2_err * (l2*(1-l2))

    # Finding the error and delta value for first hidden layer
    l1_err = l2_delta.dot(weights1.T)
    l1_delta = l1_err * (l1*(1-l1))

    # Using the respective delta values to modify the weights
    weights0 += l0.T.dot(l1_delta)*lr
    weights1 += l1.T.dot(l2_delta)*lr

    # Printing mean error at each 100th iteration
    if i % 1000 == 0:
        print("Error : " + str(np.mean(np.abs(l2_err))))

# Printing the final accuracy of the neural net after training it
print("\nAccuracy of the neural net: " + str(1 - (np.mean(np.abs(l2_err)))))

# Taking user input
print("\nEnter some rgb values of a random color to know whether it's a light or dark color\n")
r_val = int(input("Enter the R value: "))
g_val = int(input("Enter the G value: "))
b_val = int(input("Enter the B value: "))

color = [r_val, g_val, b_val]

# Predicting the output
l1 = sigmoid(np.dot(color, weights0))
l2 = sigmoid(np.dot(l1, weights1))

# Printing the predicted output
if (l2[0] < 0.5):
    print("\nThe color is: Dark")
else:
    print("\nThe color is: Light")