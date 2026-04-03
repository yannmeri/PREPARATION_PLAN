from read_csv_file import read_csv_file #I import the read_csv_file function from the read_csv_file file.

class Sol: #it is a class for detecting the type of the soil and propose cultures
    def __init__(self, type_soil, ph, humidity): #it is the initial function to create object later
        self.type = type_soil
        self.ph = ph
        self.humidity = humidity

    def classifier_ph(self): #here, we classify soil basing on their ph
        if (self.ph < 6):
            return ("acide")
        elif(self.ph > 7):
            return ("basique")
        else:
            return ("neutre")

    def est_fertile(self): #we check if a soil is fertile or not basing on its ph and its humidity
        return (6 <= self.ph <= 7 and self.humidity >= 20)

    def state_soil(self): #we give the state of the soil: fertile or infertile
        if (self.est_fertile()):
            return ("fertile soil")
        else:
            return ("infertile soil")

    def recommander_culture(self): #we recommand basing on reality the appropriate culture for each type of soil
        cultures = {
        "clay_s" : ["rice", "cabbage", "spinash", "beans", "fruits trees (orange tree, mango tree, apple tree)"],
        "sandy_s": ["peanut", "cassava", "watermelon", "sweet potatoes", "carrot"],
        "loamy_s" : ["maize", "tomatoes", "beans", "lettuce", "rice (with irrigation)"]
        }
        if (self.type == "argileux"):
            return (cultures["clay_s"])
        elif (self.type == "sableux"):
            return (cultures["sandy_s"])
        else:
            return (cultures["loamy_s"])


def create_object(df): #here, we create object from the class Sol
    nb_t = 0
    nb_f = 0
    with open("resultats.txt", "w") as f: #open is used to open an existing file or creating one if it does not exist yet, "w" is used to write inside this file
        for row in df.itertuples(index=False): #itertuples is used to read each line of my dataframe(df) and "index=False" precise that we do not need any index here, only, the content either type_soil, ph or humidity
            type_soil = row.type
            ph = float(row.ph)
            humidite = float(row.humidite)

            type_of_culture = Sol(type_soil, ph, humidite) #the object here is type_of_culture
            if (type_of_culture.est_fertile()):
                nb_t += 1 #we are counting how many soils are fertile.
            else:
                nb_f += 1 #we are counting how many soils are infertile.
            f.write(f"Sol {type_soil} (ph {ph}, humidite {humidite}) -> {type_of_culture.classifier_ph()}, {type_of_culture.state_soil()}, culture proposed: {type_of_culture.recommander_culture()}\n") #we are writing inside the .txt file the results we obtained.

        f.write(f"\nThere are {nb_t} fertile soils and {nb_f} infertile soils.\n") #we are writing inside the .txt file the results we obtained.


if __name__=="__main__":
    df = read_csv_file()
    create_object(df)