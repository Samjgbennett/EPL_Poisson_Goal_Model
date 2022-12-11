import numpy as np
from scipy.stats import poisson

# Let us assume that we have the average number of goals per game for each team, based on past data
team_A_goals_per_game = 1.5 
team_B_goals_per_game = 2.2

#define intensities (lambda & meu)

def lambd(alpha,beta,gamma):
    alpha*beta*gamma


def meu(alpha,beta):
    alpha*beta

def tau(x,y,lambd,rho,meu):
    if x==y==0:
        return 1 - lambd*meu*rho
    elif (x==0 & y==1):
        return 1+lambd*rho
    elif (x==1 & y==0):
        return 1+meu*rho
    elif (x==y==1):
        1- rho
    else:
        return 1


def theta(epsilon, t):
    return np.exp(-epsilon*t)


def Likelihood(alpha,beta,rho,gamma):
    for i in (1,11):
        return (tau()*np.exp(-lambd())*lambd()*np.exp(-meu())*meu())**theta()

# We can then calculate the probability of each team scoring a goal in the football game
# using a bivariate Poisson distribution


#pmf is probability mass function
prob_team_A_scores = poisson.pmf(1, team_A_goals_per_game) 
prob_team_B_scores = poisson.pmf(1, team_B_goals_per_game) 

# We can then use the probabilities of each team scoring a goal to predict
# the total number of goals that will occur in the football game
total_goals = 0
for i in range(1, 11):
    # Calculate the probability of i goals occurring in the game
    prob_i_goals = np.power(prob_team_A_scores, i) * np.power(prob_team_B_scores, i) 
    # Add the probability to the total number of goals
    total_goals += prob_i_goals
    


# Print the total number of goals
print("The total number of goals that will occur in the football game is:", total_goals)