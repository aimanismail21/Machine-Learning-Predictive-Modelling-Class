import numpy as np
matrix = [[0.52106591, -0.37741762],
          [-0.26934744, -0.92329566],
          [0.5804131, -0.02449161],
          [0.56485654, -0.06694199]]

flower = [-1.1430e+00, -1.3198e-01, -1.3402e+00, -1.3154e+00]
a = np.array(matrix)
b = np.array(flower)

print(b.dot(a))

# Exercise 12
flower_2 = [-1.3853e+00,  3.2841e-01, -1.3971e+00, -1.3154e+00]
c = np.array(flower_2)
print(c.dot(a))

x_result = flower[0] * matrix[0][0] + flower[1] * matrix[1][0] + flower[2] *\
           matrix[2][0] + flower[3] * matrix[3][0]
y_result = flower[0] * matrix[0][1] + flower[1] * matrix[1][1] + flower[2] * \
           matrix[2][1] + flower[3] \
           * matrix[3][1]

print(f"{x_result} and {y_result}")
