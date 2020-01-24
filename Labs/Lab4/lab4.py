import matplotlib.pyplot as plt

# Plot scatter of actual values.
x = [0.19, 0.28, 0.35, 0.37, 0.4, 0.18]
y = [0.13, 0.12, 0.35, 0.3, 0.37, 0.1]
plt.scatter(x, y, color='green', label='Sample Data')

# Plot prediction line.
x_init = [0.19, 0.28, 0.35, 0.37, 0.4, 0.18]
y_predicted = [0.136751684,
               0.221175409,
               0.286838306,
               0.305599134,
               0.333740375,
               0.12737127
               ]
plt.plot(x_init, y_predicted, color='blue', label='y=0.938041*x - 0..04148')

# Show average
x3 = [0, 0.50]
y3 = [0.195714286, 0.195714286]
plt.plot(x3, y3, '--', color='Black', label='Average Y Level')

# Add a legend, axis labels, and title.
plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title('Growth of Y over X')

plt.show()
