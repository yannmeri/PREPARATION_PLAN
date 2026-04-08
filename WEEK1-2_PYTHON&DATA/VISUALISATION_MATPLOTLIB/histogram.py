import matplotlib.pyplot as plt

def create_histogram(ph):#This function creates a histogram of pH values

    plt.hist(ph)#Create the histogram using pH values
    plt.title("Data Distribution")#Add a title to the graph
    plt.xlabel("pH values")#Label the X-axis
    plt.ylabel("Frequency")#Label the Y-axis
    plt.show()#Display the graph


if __name__ == "__main__":

    ph_values = [5.5, 6.2, 6.8, 7.5, 6.0, 6.3, 5.8]
    create_histogram(ph_values)

    #Most soils are neutral, with a few acidic values and one basic value.