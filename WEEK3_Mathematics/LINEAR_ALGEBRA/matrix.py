import numpy as np#NumPy is used for numerical computations
import pandas as pd#Pandas is used to manipulate datasets
import numpy as np#NumPy is used for numerical computations

def read_csv_file():#Load the CSV file into a DataFrame.
    df = pd.read_csv("mini_projet_sols_csv.csv")

    return (df)

def create_matrix(M):#Convert a list into a NumPy matrix
    matrix = np.array(M)
    
    return (matrix)

if __name__=="__main__":#I test my program
    
    df = read_csv_file()

    M = [
        [6.5, 25],
        [7.2, 15],
        [5.8, 30]
        ]

    matrix = create_matrix(M)

    print(f"Matrix shape: {matrix.shape}\n")#(rows, columns)
    print(f"Second row: {matrix[1,:]}\n")#matrix[line, column], ':' = select all columns for line 2 here
    print(f"Ph column: {matrix[:,0]}\n")#matrix[line, column], ':' = select all lines for column 1 here

    df_matrix = create_matrix(df[["ph", "humidite"]].values)#Convert DataFrame to matrix

    # Transpose
    df_matrix_trans = df_matrix.T #I transposed df_matrix
    print(f"Dataframe matrix:\n{df_matrix}\n")
    print(f"Dataframe matrix transpose:{df_matrix_trans}\n")

    # Matrix multiplication
    prod_matrix = np.dot(df_matrix, df_matrix_trans)#Matrix product
    print(f"Prod matrix:\n{prod_matrix}")

