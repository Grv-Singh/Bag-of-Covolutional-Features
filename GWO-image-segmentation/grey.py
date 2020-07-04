import random
import numpy as np
import math

import time
import numpy
from otsu import otsu, fast_ostu

class GWO:
    def __init__(self, image):

        self.image= image
        # self.N = N
        self.Max_iter=1000
        self.lb=-100
        self.ub=100
        self.dim=1  
        self.SearchAgents_no=5
        self.Alpha_pos=numpy.zeros(self.dim)
        self.Alpha_score=float("inf")
        
        self.Beta_pos=numpy.zeros(self.dim)
        self.Beta_score=float("inf")
        
        self.Delta_pos=numpy.zeros(self.dim)
        self.Delta_score=float("inf")
        
        #Initialize the positions of search agents
        self.Positions=numpy.random.uniform(0,1,(self.SearchAgents_no,self.dim)) *(self.ub-self.lb)+self.lb
        
        self.Convergence_curve=numpy.zeros(self.Max_iter)



    def fitness(self, wolf):
        fitness = fast_ostu(self.image, wolf)
        return fitness


    def hunt(self):
        for l in range(0,self.Max_iter):
            for i in range(0,self.SearchAgents_no):
                
                # Return back the search agents that go beyond the boundaries of the search space
                self.Positions[i,:]=numpy.clip(self.Positions[i,:], self.lb, self.ub)

                # Calculate objective function for each search agent
                fitness=self.fitness(self.Positions[i])
                
                # Update Alpha, Beta, and Delta
                if fitness<self.Alpha_score :
                    self.Alpha_score=fitness; # Update alpha
                    self.Alpha_pos=self.Positions[i,:].copy()
                
                
                if (fitness>self.Alpha_score and fitness<self.Beta_score ):
                    self.Beta_score=fitness  # Update beta
                    self.Beta_pos=self.Positions[i,:].copy()
                
                
                if (fitness>self.Alpha_score and fitness>self.Beta_score and fitness<self.Delta_score): 
                    self.Delta_score=fitness # Update delta
                    self.Delta_pos=self.Positions[i,:].copy()
                
            
            
            
            a=2-l*((2)/self.Max_iter); # a decreases linearly fron 2 to 0
            
            # Update the Position of search agents including omegas
            for i in range(0,self.SearchAgents_no):
                for j in range (0,self.dim):     
                               
                    r1=random.random() # r1 is a random number in [0,1]
                    r2=random.random() # r2 is a random number in [0,1]
                    
                    A1=2*a*r1-a; # Equation (3.3)
                    C1=2*r2; # Equation (3.4)
                    
                    D_alpha=abs(C1*self.Alpha_pos[j]-self.Positions[i,j]); # Equation (3.5)-part 1
                    X1=self.Alpha_pos[j]-A1*D_alpha; # Equation (3.6)-part 1
                               
                    r1=random.random()
                    r2=random.random()
                    
                    A2=2*a*r1-a; # Equation (3.3)
                    C2=2*r2; # Equation (3.4)
                    
                    D_beta=abs(C2*self.Beta_pos[j]-self.Positions[i,j]); # Equation (3.5)-part 2
                    X2=self.Beta_pos[j]-A2*D_beta; # Equation (3.6)-part 2       
                    
                    r1=random.random()
                    r2=random.random() 
                    
                    A3=2*a*r1-a; # Equation (3.3)
                    C3=2*r2; # Equation (3.4)
                    
                    D_delta=abs(C3*self.Delta_pos[j]-self.Positions[i,j]); # Equation (3.5)-part 3
                    X3=self.Delta_pos[j]-A3*D_delta; # Equation (3.5)-part 3             
                    
                    self.Positions[i,j]=(X1+X2+X3)/3  # Equation (3.7)
                
            
        
        
            self.Convergence_curve[l]=self.Alpha_score;


    def result_curve(self):
        return self.Convergence_curve    

