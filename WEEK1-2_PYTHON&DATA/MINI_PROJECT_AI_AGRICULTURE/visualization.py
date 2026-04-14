import matplotlib.pyplot as plt# Matplotlib is used for data visualization

def histogram_ph(df):# Create a histogram to visualize the distribution of pH values
    plt.figure()
    plt.hist(df["ph"], bins=20)#bins allow to group reading data
    plt.title("Histogram ph")
    plt.xlabel("PH values")
    plt.ylabel("Y values")
    plt.show()

def scatter_ph_humidity(df):# Create a histogram to visualize the distribution of pH values
    plt.figure()
    plt.scatter(df["ph"], df["humidite"], alpha=0.5)#alpha is used for transparancy like this you can see density zones
    plt.title("Scatter ph and Humidity")
    plt.xlabel("PH values")
    plt.ylabel("Humidity values")
    plt.show()

def lineplot_ph(df):#we create a line plot graph to visualize ph evolution
    plt.figure()
    plt.plot(df["ph"])
    plt.title("PH evolution")
    plt.xlabel("Samples")
    plt.ylabel("PH values")
    plt.grid()
    plt.show()
