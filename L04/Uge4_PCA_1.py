# pylint: disable=no-member
#%% Imports and data preparation

import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import matplotlib
import matplotlib.pyplot as plt

# Load digit data and split into training and test data
digits = load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#%% Plot single digit
def plot_digit(digit, cmpsize=64, title=None):
    s = int(np.sqrt(cmpsize))
    image = digit.reshape(s, s)
    plt.figure(figsize=(2,2))
    plt.title(title)
    plt.imshow(image, cmap = matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()

digit_index = 123
plot_digit(X[digit_index])
print(y[digit_index])

#%% Create PCAs with different number of components
def create_pca(data, cmps):
    pca = PCA(n_components=cmps)
    pca.fit(data)
    return pca

dimensions_sq = [n**2 for n in range(1,7)] # 4, 9, 16 ...
dimensions_2n = [2**n for n in range(1,7)] # 2, 4, 8 ...
dimensions = sorted(set(dimensions_sq + dimensions_2n))

pca = {n: create_pca(X_train, n) for n in dimensions}

X_reduced = {n: pca[n].fit_transform(X_train) for n in dimensions}

print(f"Created {len(dimensions)} pca's of training data with dimensions {dimensions}")

#%% Plot the standard deviation and explained variance pr dimension
def plot_pca_dim_variance(pca):
    varians = pca.explained_variance_
    s = np.sqrt(varians)

    plt.figure(figsize=(8,4))
    plt.suptitle(f"Components: {pca.n_components}", x=0.5, y=1.05)
    
    plt.subplot(1, 2, 1, adjustable='box')
    plt.plot(s)
    plt.title('Standard afvigelser (spredning) \nlangs de forskellige dimensioner')
    plt.xlabel('Dimension nummer')
    
    plt.subplot(1, 2, 2, adjustable='box')
    plt.plot(np.cumsum(pca.explained_variance_)/np.sum(pca.explained_variance_))
    plt.title('Explained variance - procentdel')
    plt.xlabel('Dimension nummer')
    
for n in [32, 25, 16, 8, 4]:
    plot_pca_dim_variance(pca[n])

#%% Split data into which digit it "should" represent
def mkdict1(x_reduced): # Method 1
    xdict={i:[] for i in range(10)}
    for i, x in enumerate(x_reduced):
        xdict[y_train[i]].append(x)
    return {i:np.array(xdict[i]) for i in range(10)}

def mkdict2(x_reduced): # Method 2
    return {
        digit: np.array([x for xi, x in enumerate(x_reduced) if y_train[xi] == digit]) 
        for digit in range(10)
    }

#%% Time the two different mothods of creating dict
from timeit import timeit
r1 = timeit(lambda: mkdict1(X_reduced[2]), number=100)
r2 = timeit(lambda: mkdict2(X_reduced[2]), number=100)
print(f"mkdict1 = {r1}")
print(f"mkdict2 = {r2}")
print("the fastest is {}".format("mkdict1" if r1 < r2 else "mkdict2"))

#%% Plot the two-component representation of digits and color by 'actual' digit
npdict = mkdict1(X_reduced[2])
def plot_two_component_digits(digits):
    for dnum in digits:
        plt.scatter(npdict[dnum][:,0], npdict[dnum][:,1], s=2, label=str(dnum))
    # plt.axis('equal')
    plt.legend()
    plt.show()

plot_two_component_digits([1, 2, 3])
plot_two_component_digits([4, 5, 6])
plot_two_component_digits([7, 8, 9])

#%% plot every pairwise digit combination
pairperms = [(a, b) for a in range(10) for b in range(a+1, 10)] 

for pair in pairperms:
    plot_two_component_digits(pair)
#%% 
# plot 3-pair combinations with (probably) high/good seperation
pl ={
    0:[], 1:[],
    2:[6],
    3:[4,6],
    4:[5,7,8,9],
    5:[6],
    6:[7,8],
    7:[], 8:[], 9:[]
}

for n in range(10):
    for m in pl[n]:
        plot_two_component_digits([0, n, m])
#%% Recreate / recover original
X_recovered = {n: pca[n].inverse_transform(X_reduced[n]) for n in dimensions}

#%% Plot 16-component PCA representation of digits as grid
indexes = [1,2,3,41,42,43]

for i in indexes:
    plot_digit(X_train[i], title=y_train[i])
    # print(y_train[i])
    for d in dimensions_sq:
        plot_digit(X_reduced[d][i], d, title=y_train[i])

#%%
digit_index = 5

plot_digit(X[digit_index])
print(y[digit_index])
plot_digit(X_recovered[16][digit_index])
