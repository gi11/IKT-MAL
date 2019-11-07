

### Importing the data

The data is loaded from the csv file into a pandas dataframe
```python
data_csv_path = "path/to/SpotifyFeatures.csv"
spotifyDBData = pd.read_csv(data_csv_path, sep=',', header=0)
```

### Distribution of popularity

```python
# Lets look at popularity - this would be expected to resemble a normal distribution
trackPopularity = spotifyDBData['popularity']

# Find the statistical properties of the popularity
mu = np.mean(trackPopularity)
sigma = np.std(trackPopularity)
sigma2 = np.var(trackPopularity)
median = np.median(trackPopularity)

# Create an object to plot into
fig, ax = plt.subplots(1, 1, figsize=[10,6])

# Now plot the popularity data to get an idea of it's distribution
ax.hist(trackPopularity,bins=100, density=True, label='Track Popularity') # Normalises the histogram
plt.xlabel('Popularity rating')
plt.ylabel('Normalised Counts')
ax.axvline(mu, color='b', label = "Mean")
ax.axvline(median, color='r', label="Median")

# Try and fit a gausian distribution
xarr = np.linspace(np.max(trackPopularity), np.min(trackPopularity), 500)
ax.plot(xarr, norm.pdf(xarr, mu, sigma), label='True gausian PDF')
ax.legend()
```

### Genre Pre-processing



### Popularity distribution by Genre

```python
Genres = spotifyDBData['genre'] # Series containing genres of each track
UniqueGenres = Genres.unique()  # Contains the name of each genre included in DB

print(UniqueGenres) # Verify 

cols = 4   # How many subplots pr row
width = 15 # Width of figure
prop = 1/3 # Subplot proportions, width/height ratio of subfigures

rows = int(len(UniqueGenres)/cols)+1
height = (rows/cols)*width*prop

fig, ax = plt.subplots(rows, cols, figsize=(width,height))
plt.subplots_adjust(wspace=0.2, hspace=1)
for index, genre in enumerate(UniqueGenres):
    row, col = int(index/cols), index % cols
    genre_tracks = spotifyDBData.loc[spotifyDBData['genre'] == genre]
    popularity = genre_tracks['popularity']
    title = genre + ", N = " + str(len(popularity))
    ax[row,col].hist(popularity, bins=40, density=True, label='Track Popularity')
    ax[row,col].set_title(title)
```

### Audio Features by genre

### Correlation between audio-features


