import matplotlib.pyplot as plt

def histogram_ph(df):
    plt.figure()
    plt.hist(df["ph"], bins=20)#bins allow to group reading data
    plt.title("Histogram ph")
    plt.xlabel("PH values")
    plt.ylabel("Y values")
    plt.show()

def scatter_ph_humidity(df):
    plt.figure()
    plt.scatter(df["ph"], df["humidite"], alpha=0.5)#alpha is used for transparancy like this you can see density zones
    plt.title("Scatter ph and Humidity")
    plt.xlabel("PH values")
    plt.ylabel("Humidity values")
    plt.show()

def lineplot_ph(df):
    plt.figure()
    plt.boxplot(df["ph"])
    plt.title("Line Plot ph")
    plt.xlabel("PH values")
    plt.ylabel("Y values")
    plt.show()
