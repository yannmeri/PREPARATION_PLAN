import matplotlib.pyplot as plt

def relation_ph_humidity(ph, humidity):#This function shows the relationship between pH and humidity

    plt.title("Relation between pH and humidity")#Add a title to the graph
    plt.xlabel("pH values")#Label the X-axis (pH)
    plt.ylabel("Humidity values")#Label the Y-axis (humidity)
    plt.scatter(ph, humidity)#Create a scatter plot to show the relationship between pH and humidity
    plt.show()#Display the graph


if __name__ == "__main__":

    ph = [5.5, 6.2, 6.8, 7.5, 6.0]
    humidity = [15, 22, 30, 10, 25]
    relation_ph_humidity(ph, humidity)

    #OBSERVATION:
    #Soils with a pH between 6 and 7 tend to have higher humidity.
    #However, when the pH is above 7, humidity tends to decrease.