import matplotlib.pyplot as plt
from read_csv_file import read_csv_file
from histogram import create_histogram
from scatter_plot import relation_ph_humidity


def line_plot_ph(ph):#This function plots pH values to observe their variation
    plt.figure()#Create a new figure (a new graph window)
    plt.plot(ph)#Draw a line graph using pH values
    plt.title("pH Values Distribution")#Add a title to the graph
    plt.xlabel("Samples")#Label the X-axis (index of samples)
    plt.ylabel("pH values")#Label the Y-axis
    plt.grid()#Add a grid for better readability
    plt.show()#Display the graph


if __name__ == "__main__":
    df = read_csv_file()

    plt.figure()#Create a new figure for the histogram
    create_histogram(df["ph"])

    plt.figure()#Create a new figure for the scatter plot
    relation_ph_humidity(df["ph"], df["humidite"])

    line_plot_ph(df["ph"])