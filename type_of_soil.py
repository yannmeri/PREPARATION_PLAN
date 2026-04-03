from read_csv_file import read_csv_file

class Sol:
    def __init__(self, type_soil, ph, humidity):
        self.type = type_soil
        self.ph = ph
        self.humidity = humidity

    def classifier_ph(self):
        if (self.ph < 6):
            return ("acide")
        elif(self.ph > 7):
            return ("basique")
        else:
            return ("neutre")

    def est_fertile(self):
        return (6 <= self.ph <= 7 and self.humidity >= 20)

    def state_soil(self):
        if (self.est_fertile()):
            return ("fertile soil")
        else:
            return ("infertile soil")

    def recommander_culture(self):
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


def create_object(df):
    nb_t = 0
    nb_f = 0
    with open("resultats.txt", "w") as f:
        for row in df.itertuples(index=False):
            #print(row)
            type_soil = row.type
            ph = float(row.ph)
            humidite = float(row.humidite)

            type_of_culture = Sol(type_soil, ph, humidite)
            if (type_of_culture.est_fertile()):
                nb_t += 1
            else:
                nb_f += 1
            f.write(f"Sol {type_soil} (ph {ph}, humidite {humidite}) -> {type_of_culture.classifier_ph()}, {type_of_culture.state_soil()}, culture proposed: {type_of_culture.recommander_culture()}\n")

        f.write(f"\nThere are {nb_t} fertile soils and {nb_f} infertile soils.\n")


if __name__=="__main__":
    df = read_csv_file()
    create_object(df)