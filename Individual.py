import random

class Individual:
    #fitness
    fitness = 0

    #abc
    abc = None

    #init
    def __init__(self, genotyp=None, fitness = None, abc=None):
        self.genotyp = genotyp
        self.fitness = fitness
        self.abc = abc

    def copy(self):
        copy = Individual( [self.genotyp[0].copy(), self.genotyp[1].copy()], fitness=self.fitness , abc = self.abc )
        return copy

    def equals(self, other):
        return self.genotyp == other.genotyp

def randomGenotyp(minPitch, maxPitch, num_notes):
    pitches = random.choices(list(range(minPitch,maxPitch+1)), k=num_notes)
    durations = random.choices([1,2,4], k=num_notes) #ran.integers(1,5,num_notes).tolist()
    return [pitches,durations]
    
def uniformGenotyp(minPitch, maxPitch, num_notes):
    pitches = [0] * num_notes # only note C
    durations = [2] * num_notes # only quaters
    return [pitches,durations]
