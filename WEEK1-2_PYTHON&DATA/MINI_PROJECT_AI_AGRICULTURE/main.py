from analysis import read_csv, clean_missed_value, classify_ph, check_fertility, compute_stats, correlation_ph_humidity, recommend_crop, suggest_improvement
#I import functions inside analysis.py file

import pandas as pd
from visualization import histogram_ph, scatter_ph_humidity, lineplot_ph #I import functions inside visualization.py file

if __name__=="__main__":
    df = read_csv()
    df = clean_missed_value(df)
    df["ph_type"] = df["ph"].apply(classify_ph)
    df = check_fertility(df)

    pd.set_option('display.max_rows', None)#to print all rows of the dataframe
    print(df)

    dict_analysis = compute_stats(df)

    print(f"The mean ph is: {dict_analysis['mean ph']} .\nThe mean humidity is: {dict_analysis['mean humidity']} .\nThere are {dict_analysis['number fertile soil']} fertile soils.\nThere are {dict_analysis['number clay soil']} clay soils, {dict_analysis['number sandy soil']} sandy soils, and {dict_analysis['number loamy soil']} loamy soils.\n")

    correlation_ph_humidity(df)

    histogram_ph(df)
    scatter_ph_humidity(df)
    lineplot_ph(df)

    with open("results.txt", "w") as file:#I create and open a file.txt to write the recommended crops inside
        for row in df.itertuples(index=False):#I take each row of the dataframe and index=False to avoid to get index of each row
            cultures = recommend_crop(row.type)
            suggestions = suggest_improvement(row.ph, row.humidite)

            file.write(
                f"Soil type: {row.type} | pH: {row.ph} | Humidity: {row.humidite}\n"
                f"Recommended crops: {cultures} | {suggestions}\n\n"
            )
            