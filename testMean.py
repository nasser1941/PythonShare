from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np

variable = norm.rvs(size=1000)

# Fit a normal distribution to the data:
mu, std = norm.fit(variable)

# Plot the histogram.
plt.hist(variable, bins=25, normed=True, color='g')

# Plot the PDF.
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
print x
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
plt.title(title)
plt.show()
