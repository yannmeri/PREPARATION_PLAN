import matplotlib.pyplot as plt#Matplotlib is used to visualize data with graphs

def create_graph(x, y):

    plt.plot(x, y)# Draw a line graph using x and y values
    plt.title("First Graph")#Add a title to the graph
    plt.xlabel("X values")#Label the X-axis
    plt.ylabel("Y values")#Label the Y-axis
    plt.grid()#Add a grid to make the graph easier to read
    plt.show()#Display the graph


if __name__ == "__main__":

    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    create_graph(x, y)