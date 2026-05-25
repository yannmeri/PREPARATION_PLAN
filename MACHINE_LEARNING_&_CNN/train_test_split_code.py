import pandas as pd#this library helps creating a datafrram

data = {
    "ph": [6.5, 7.0, 5.8, 6.2, 7.1],
    "humidite": [25, 30, 15, 20, 35],
    "yield": [90, 100, 60, 80, 110]
}

df = pd.DataFrame(data)#Creating df dataframe here

print(df)

#This is my first method to split X and y

#df_x = df[["ph", "humidite"]].copy() #I copy ph and humidite columns from the original dataframe to a new one
#df_y = df[["yield"]].copy() #I copy yield columns from the original dataframe to a new one

#This is my second method to split X and y
#Ps: the mark ':' means all.
df_x = df.iloc[:,:2]#before the comma (',') means all the lines and after the comma (',') means all columns until index 2
df_y = df.iloc[:, 2:]#before the comma (',') means all the lines and after the comma (',') means all columns from index 2

print(df_x)
print(df_y)

#I split the dataframe into two categories train and test dataframe
#Ps: the mark ':' means all.

train_dataset = df.iloc[:3,:]#before the comma (',') means all the lines until index 3 and after the comma (',') means all columns
test_dataset = df.iloc[3:,:]#before the comma (',') means all the lines from index 3 and after the comma (',') means all columns

print(train_dataset)
print(test_dataset)