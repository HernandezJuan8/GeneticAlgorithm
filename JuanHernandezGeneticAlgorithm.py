#Juan Hernandez
#Genetic Algorithm using floating point vector 
from asyncio.windows_events import NULL
from numpy.random import randint
import numpy as np
import math

#tournament Selector function which will select futtest individuals picked at random
# Tournamen ssize by default is set to 7
def tournamentSelector(pop,fitness, t = 7):
    best = randint(len(pop))
    for i in randint(0,len(pop),t-1):
        if fitness[i] > fitness[best]:
            best = i
    return pop[best]

#makes a list of num fittest individuals and allows for duplicates
def tournament(pop,fitness,num,t = 7):
    x = []
    for i in range(num):
        x.append(tournamentSelector(pop,fitness,t))
    return x

# Length of the vector
floatVectorLength = 20
#Legal Min for a number in a vector
floatMin = -5.12
#Legal Max for a number in a vector
floatMax = 5.12

#Creates a floating point vector of floatVectorLength and filled with
#uniform random numbers in the range appropriate.
def creator():
    x = []
    for i in range(floatVectorLength):
        x.append(np.random.uniform(floatMin,floatMax))
    return x

#Numbers that were just mad up, can be changed or tweaked

#Per gene probabilityu of crossover in uniform crossover
crossoverProbability = 0.1
#Per gene probability of mutation in gaussian convolution
mutationProbability = 0.1
#Per gene mutation variance in gaussian convolution
mutationVariance = 0.02

#Performs uniform crossover between 2 individuals modifying in place
def uniformCrossover(ind1, ind2):
    for i in range(floatVectorLength):
        if np.random.uniform() <= crossoverProbability:
            ind1[i],ind2[i] = ind2[i],ind1[i]
    return None

# Generates a random number under gaussian distribution with given mean
# and variance using
#Box-Muller-Marsaglia Method
def gaussianRandom(mean,variance):
    x = 0
    w = 0
    while not(0 < w and w < 1):
        x = np.random.uniform(-1.0,1.0)
        y = np.random.uniform(-1.0,1.0)
        w = math.pow(x,2) + math.pow(y,2)
    g = mean + (x * math.sqrt(variance) * math.sqrt(-2 * (math.log(w)/w)))
    return g

#Performs gaussian convolution mutation on an individual modifying in place
def gaussianConvolution(ind):
    for i in range(floatVectorLength):
        if mutationProbability >= np.random.uniform():
            n = gaussianRandom(0,mutationVariance)
            while not (floatMin <= ind[i] + n and ind[i]+n <= floatMax):
                n = gaussianRandom(0,mutationVariance)
    return None

#Copies individuals and crosses them over then mutates the children
def floatVectorModifier(ind1,ind2):
        c1 = ind1.copy()
        c2 = ind2.copy()
        uniformCrossover(c1,c2)
        gaussianConvolution(c1)
        gaussianConvolution(c2)
        x = [c1 , c2]
        return x


#Fitness evaluations Functions
#All these functions are set fopr maximize instead of minimize 
#since we are looking for higher numbers

#These Functions were takes from section 11.2.2 of Essentials of Metaheuristics
#These funcations were changed to maximize instead of the original minimize so
#Higher number is better
def sumF(ind):
    sum = 0
    for i in range(len(ind)):
        sum += ind[i]
    return sum

def stepF(ind):
    tot = 6 * len(ind)
    sum = 0
    for i in range(len(ind)):
        sum += math.floor(ind[i])
    tot += sum
    return tot

def sphereF(ind):
    sum = 0
    for i in range(len(ind)):
        sum += math.pow(ind[i],2)
    sum *= -1
    return sum

def rosenbrockF(ind):
    sum = 0
    for i in range(len(ind)-1):
        sum += math.pow(1-ind[i],2) + 100 * math.pow(ind[i+1]-math.pow(ind[i],2),2)
    sum *= -1
    return sum

def rastriginF(ind):
    tot = 10 * len(ind)
    sum =0
    for i in range(len(ind)):
        sum += math.pow(ind[i],2) - 10 * math.cos(2*math.pi*ind[i])
    tot += sum
    tot *= -1
    return tot

def schwefelF(ind):
    sum = 0
    for i in range(len(ind)):
        sum += (ind[i] * 100 * -1) *math.sin(math.sqrt(abs(ind[i]*100)))
    sum *= -1
    return sum


pop = []

#Sets up the population with stats
def setup(popSize):
    global pop
    for i in range(popSize):
        pop.append(creator())
    return None

#Prints the best individual in a generation
#with all their stats
def printer(pop,fitness,gen):
    bestInd, bestFit = NULL,NULL
    for i in range(len(pop)):
        if not bestInd:
            bestInd = pop[i]
            bestFit = fitness[i]
        if bestFit < fitness[i]:
            bestInd = pop[i]
            bestFit = fitness[i]
    print("Best Individual of Generation "+str(gen)+":")
    print("Fitness: "+str(bestFit))
    print("Individual: "+ str(bestInd))
    return None

def evolve(gen,popSize,fitFunc,ts=7):
    setup(popSize)
    global pop
    for i in range(gen):
        print("Starting Generation "+ str(i))
        fit =[]
        for j in range(popSize):
            fit.append(fitFunc(pop[j]))
        nextGen = []
        printer(pop,fit,i)
        for x in range(int(popSize/2)):
            t = tournament(pop,fit,2)
            temp = floatVectorModifier(t[0],t[1])
            
            nextGen.append(temp[0])
            nextGen.append(temp[1])
        pop = nextGen

        print("Ending Generation "+ str(i))
    return None
        
def main():
    evolve(1000,500,schwefelF,7)

if __name__ == "__main__":
    main()


