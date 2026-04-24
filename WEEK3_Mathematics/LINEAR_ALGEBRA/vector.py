import numpy as np#NumPy is used for numerical computations

def create_vector(v1, v2):#Create two NumPy vectors from lists
    vec_1 = np.array(v1)
    vec_2 = np.array(v2)
    
    return (vec_1, vec_2)

def scalar_product(vec_1, vec_2):#Compute the dot product between two vectors
    return (np.dot(vec_1, vec_2))

def predict_fertility(weights, ph_soil):#Compute a score using weights and soil features (pH, humidity)
        return (np.dot(weights, ph_soil))


if __name__=="__main__":#I test my program
    v1 = [6, 25]
    v2 = [7, 20]

    vec_1, vec_2 = create_vector(v1, v2)
    add_vec = vec_1 + vec_2
    mult_vec1 = vec_1 * 2
    norme_vec1 = np.sqrt(vec_1[0]**2 + vec_1[1]**2)
    norme_vec2 = np.sqrt(vec_2[0]**2 + vec_2[1]**2)

    print(f"Vector 1 + Vector 2: {add_vec}\n")
    print(f"Vector 1 * 2: {mult_vec1}\n")
    print(f"Norme Vector 1: {norme_vec1}\n")
    print(f"Norme Vector 2: {norme_vec2}\n")

    scalar_vec = scalar_product(vec_1, vec_2)
    print(f"Scalar product: {scalar_vec}")

    #Interpretation
    print(f"It measures similarity and is used to compute predictions in models.\n")
    
    weights = [0.4, 0.6]
    ph_soil = [6.5, 25]
    score = predict_fertility(weights, ph_soil)
    print(f"Score: {score}")
    # ❗ Important correction here
    if score > 20:
        print("Soil is likely fertile.")
    else:
        print("Soil is likely infertile.")