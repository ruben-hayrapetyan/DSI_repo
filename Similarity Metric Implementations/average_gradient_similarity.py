def average_gradient(data, T, b):
    #from t = 1 to t = T-1
    #sample a mini batch B from the training set and get b gradients
    count = 0 #some placeholder
    for t in range(T):
        #do soemthing
        count += 1 #some placeholder
    return 0

def find_a(public_g, private_g):
    #optimize so that given public_g and private_g
    #a * (public_g)^2 <= (private_g)^2 <= 1/a * (public_g)^2
    return 1