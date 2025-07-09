import argparse

# parse args
def get_args(parser):
    parser.add_argument('--test-mode', action='store_true', default=False, help="enables test mode (performs only 5 iterations and prints debug messages)") 
    parser.add_argument('--budget', type=int, default=1000000//2*5, help="The number of individual fitness evaluations (iterations=budget//lamda)") 
    parser.add_argument('--init-length', type=int,default=1, help="Melody lengths in the initial population (default=1)")
    parser.add_argument('--init-strategy', type=str, choices={"uniform", "random"}, default="uniform", help="How to generate initial population")
    parser.add_argument('--selection', type=str, choices={"best", "tournament"}, default="tournament", help="Choose selection strategy")
    parser.add_argument('--max-measures', type=int,default=-1, help="Limit number of generated measures to this. -1 allows for endless melodies")
    parser.add_argument('--mu', type=int,default=10, help="number of individuals in a Generation")
    parser.add_argument('--rho', type=int,default=2, help="number parents from the population")
    parser.add_argument('--lamda', type=int,default=50, help="number of offsprings")
    parser.add_argument('--mutation-type', type=str, choices={"random", "local"}, default="local", help="Just random mutations or domain-aware to keep changes local")
    parser.add_argument('--recombination-type', type=str, choices={"crossover", "excerpt"}, default="excerpt", help="Use 1-point crossover or copy excerpts from A to B")
    #parser.add_argument("--mutate-rate",type=float, default=0.1, help="Die Rate an Mutationen")
    parser.add_argument("--rest-rate",type=float, default=0.0625, help="Probability to generate a rest out of a note.")
    parser.add_argument("--recombination-rate",type=float, default=0.33, help="Probability of performing recombination of two individuals.")
    parser.add_argument("--chromatic-mode", action='store_true', default=False, help = "Use all 12 notes instead of just C major (ie. white keys on piano)")
    parser.add_argument('--clamp-model', type=str, default='clamp_sander-wood/clamp-small-512', help="The CLaMP model file. Pass 'random' for random fitness.")
    parser.add_argument('--torch-device', type=str, default='cuda', help="The device to use, e.g., 'cuda:1', 'cpu', etc.")
    parser.add_argument('--prompt', type=str, default="prompts/default.txt", help="The query (text prompt) to use. May be a string or a file (.txt).")
    parser.add_argument('--working-dir', type=str, default="./results/", help="Where temporary results should go")
    parser.add_argument('--final-dir', type=str, default=None, help="Where results are finally copied to (if not provided, final results can be found in working dir).")
    parser.add_argument('--tag', type=str, default="", help="An additional tag to identify experiment runs (just for convenience).")
    parser.add_argument("--log-to-file", action='store_true', default=False, help = "Redirect stdout+stderr to log file in working directory.")
    parser.add_argument("--detailed-statistics", action='store_true', default=False, help = "Log more detailed statistics in every iteration.")
    
    return parser

#format args into variables (exits if --help or -h is passed)
args = get_args(argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        ).parse_args()

# imports
import sys
import os
import shutil
import socket

import time
from time import time as now
from unidecode import unidecode

from tqdm import tqdm
import random
from myutils import *

HOSTNAME = socket.gethostname()
def dbg_print(*args):
    pass
        
MU = args.mu
RHO = args.rho
if MU == 1:
    print("No population (mu=1), therefore recombination was disabled (rho=1).")
    RHO=1
LAMBDA = args.lamda
BUDGET = args.budget

if args.test_mode:
    print("## TEST MODE enabled ##")
    BUDGET = 5 * LAMBDA

    HOSTNAME = "testmode"
    def dbg_print(*args):
        print(*args)
iterations = BUDGET // LAMBDA

# Make paths
EXPERIMENT_TAG = args.tag +"/"+ str(time.strftime("%Y%m%d_%H%M%S")) + "_" + HOSTNAME
PTH_OUT = args.working_dir+"/"+HOSTNAME+"/"+EXPERIMENT_TAG+"/" # = 'out/'
PTH_OUT_GEN = 'genotyps/'
PTH_ABCS = 'abcs/'
PTH_LAST_GEN = 'lastGen/'
FILENAME_FITNESS = 'fitness.csv'
FILENAME_STATS = 'stats.csv'
PTH_FINAL = None if args.final_dir is None else args.final_dir+"/"+EXPERIMENT_TAG+"/"

# make necessary folders
for directory in [ PTH_OUT, PTH_OUT+PTH_ABCS, PTH_OUT+PTH_OUT_GEN,
    PTH_OUT+PTH_LAST_GEN, PTH_OUT+PTH_LAST_GEN+PTH_ABCS, PTH_OUT+PTH_LAST_GEN+PTH_OUT_GEN ]:
    os.makedirs(directory)

LOG_FILENAME = "log.txt"
LOG_TO_FILE_MODE = args.log_to_file
if LOG_TO_FILE_MODE:
    print("stdout+stderr redirected to:", PTH_OUT+"/"+LOG_FILENAME)
    stdout_redirect_file = open(PTH_OUT+"/"+LOG_FILENAME, 'wt')
    sys.stdout = stdout_redirect_file
    sys.stderr = stdout_redirect_file
LOG_DETAILED_STATS = args.detailed_statistics

print(f"Iterations: {iterations}")    

CLAMP_MODEL = args.clamp_model
PROMPT = args.prompt

# load query
if PROMPT.endswith(".txt"):
    with open(PROMPT, 'r', encoding='utf-8') as f:
        PROMPT = f.read()
PROMPT = unidecode(PROMPT)

INIT_LENGTH = args.init_length
INIT_STRATEGY = args.init_strategy
MAX_MEASURES = args.max_measures

#MUTATE_RATE = args.mutate_rate
REST_RATE = args.rest_rate
RECOMBINATION_RATE = args.recombination_rate
RECOMBINATION_TYPE = args.recombination_type
MUTATION_TYPE = args.mutation_type
SELECTION = args.selection

TORCH_DEVICE = args.torch_device

print("Output of this run is stored at:", PTH_OUT)
if not PTH_FINAL is None:
    os.makedirs(PTH_FINAL)
    print("Results will finally be copied to:", PTH_FINAL)

# do heavy imports later, so that --help is fast
from Individual import *
from Encoding import *

#creating clamp oject
if CLAMP_MODEL == "random":
    clamp = None
    print("## RANDOM FITNESS MODE enabled ##")
else:
    print("Loading pytorch and clamp..")
    from clamp_semantic import CLaMP_Semantic
    clamp = CLaMP_Semantic(clamp_model_name = CLAMP_MODEL, torch_device=TORCH_DEVICE, query=PROMPT)

#ran = np.random.default_rng()
encoding = Encoding_2_Oct_Dia
CHORMATIC_MODE = args.chromatic_mode
if CHORMATIC_MODE:
    encoding = Encoding_2_Oct_Chroma_Sharps
print("Note encoding:", encoding)

abcgen = ABC_Generator(encoding)
interval = (-1,len(encoding)-2)

if REST_RATE== -1:
    REST_RATE =  1.0/len(encoding)

def paramToStr():
    r = "random"
    out = "iterations = "+str(iterations)+"\n"
    out += "BUDGET = "+str(BUDGET)+"\n"
    out += "MU = "+str(MU)+"\n"
    out += "RHO = "+str(RHO)+"\n"
    out += "LAMBDA = "+str(LAMBDA)+"\n"
    out += "\n"
    out += "REST_RATE = "+str(REST_RATE)+"\n"
    out += "RECOMBINATION_RATE = "+str(RECOMBINATION_RATE)+"\n"
    out += "RECOMBINATION_TYPE = "+str(RECOMBINATION_TYPE)+"\n"
    out += "MUTATION_TYPE = "+str(MUTATION_TYPE)+"\n"
    out += "SELECTION = "+str(SELECTION)+"\n"
    out += "\n"
    #out += "VOICE_COUNT = "+str(VOICE_COUNT)+"\n"
    out += "INIT_LENGTH = "+str(INIT_LENGTH)+"\n"
    out += "INIT_STRATEGY = "+str(INIT_STRATEGY)+"\n"
    out += "MAX_MEASURES = "+str(MAX_MEASURES)+"\n"
    kod = ""
    if encoding == Encoding_2_Oct_Chroma_Sharps :
        kod = "Encoding_2_Oct_Chroma_Sharps"
    elif encoding == Encoding_2_Oct_Chroma_Flats:
        kod = "Encoding_2_Oct_Chroma_Flats"
    else:
        kod = "Encoding_2_Oct_Dia"
    out+= "Encoding: "+kod+"\n"
    out += "\n"
    out += "CLAMP_MODEL = "+str(CLAMP_MODEL)+"\n"
    out += "PROMPT = "+str(PROMPT)+"\n"
    out += "\n"
    out += "HOSTNAME = "+str(socket.gethostname())+"\n"
    out += "EXPERIMENT_TAG = "+str(EXPERIMENT_TAG)+"\n"
    out += "\n"
    out += "argv = "+str(sys.argv)+"\n"
    return out

def outputIndivudual(individual, gen, stayCounter, prePath="", best_fitness_so_far=None):
    
    path = PTH_OUT+prePath+FILENAME_FITNESS
    addToCSV(individual,stayCounter, gen, path)
    
    if best_fitness_so_far is None or individual.fitness > best_fitness_so_far:
        if individual.abc is None:
            individual.abc = abcgen.encode_voice(individual.genotyp)
        
        path = str(gen)+".abc"
        writeStringToFile(PTH_OUT+prePath+PTH_ABCS+path,individual.abc)
        
        path = PTH_OUT+ prePath + PTH_OUT_GEN + str(gen)
        saveGenotyp(individual.genotyp,path)
    
def evaluate_clamp_fast(offsprings, gen_number):
    '''
    Evaluate the fitness of a population (parallelized on gpu, no file writing)
    '''
    abc_strings = []
    for i in range(len(offsprings)):
        # generate file in inference folder
        offsprings[i].abc = abcgen.encode_voice(offsprings[i].genotyp)
        abc_strings.append(offsprings[i].abc)
        
    sims = clamp.zero_shot_fast(abc_strings)
    for i in range(len(offsprings)):
        offsprings[i].fitness = sims[i]
   
# EA functions
def random_fitness(offsprings, gen_number):
    '''
    Evaluate the fitness as random value
    '''
    for i in range(len(offsprings)):
        offsprings[i].fitness = random.random()+gen_number
        
def evaluate_fitness(offsprings, gen_number):
    if clamp is None:
        random_fitness(offsprings, gen_number)
    else:
        evaluate_clamp_fast(offsprings,gen_number)

def recombinate(parents):
    start_time = now()
    if len(parents) == 1:
        child = Individual(parents[0].copy())
    else:
        if RECOMBINATION_TYPE == "excerpt":
            child = ExcerptRecombination(parents[0],parents[1])
        elif RECOMBINATION_TYPE == "crossover":
            child = OnePointRecombination(parents[0],parents[1])
        else:
            assert False, "Recombination type unknown:"+RECOMBINATION_TYPE

    dbg_print("recombination took:",now()-start_time,"s")
    return child

def OnePointRecombination(parent1, parent2):
    o = parent1.copy()
    point = random.randint(1, len(o.genotyp[0])-1)
    o.genotyp[0][point:] = parent2.genotyp[0][point:]
    o.genotyp[1][point:] = parent2.genotyp[1][point:]
    return o

def TwoPointRecombination(parent1, parent2):
    o = parent1.copy()
    lengths1 = parent1.genotyp[1]
    lengths2 = parent2.genotyp[1]
    point1 = random.randint(1, len(lengths1))
    point2 = random.randint(0, len(lengths2)-1)
    
    # allow slicing only on quarter beats
    while sum(lengths1[:point1])%2 != 0:
        dbg_print(f"Changing point1 (point1={point1}, lengths1={lengths1})")
        point1 = random.randint(1, len(lengths1))
    while sum(lengths2[:point2])%2 != 0:
        dbg_print(f"Changing point2 (point2={point2}, lengths2={lengths2})")
        point2 = random.randint(0, len(lengths2)-1)
    
    o.genotyp[0] = parent1.genotyp[0][:point1] + parent2.genotyp[0][point2:]
    o.genotyp[1] = parent1.genotyp[1][:point1] + parent2.genotyp[1][point2:]
    return o
    
def ExcerptRecombination(parent1, parent2):
    o = parent1.copy()
    pitches1 = parent1.genotyp[0]
    pitches2 = parent2.genotyp[0]
    lengths1 = parent1.genotyp[1]
    lengths2 = parent2.genotyp[1]

    if len(lengths1) == 1 and lengths1[0] == 1:
        lengths1[0] = 2
    if len(lengths2) == 1 and lengths2[0] == 1:
        lengths2[0] = 2

    dbg_print(f"Genotypes before recombination: {parent1.genotyp}, {parent2.genotyp}")

    excerpt_length = random.randint(2,6)
    point2a = random.randint(0, max(0,len(lengths2)-excerpt_length))
    point2b = point2a + excerpt_length
    point1a = random.randint(0, max(1,len(lengths1)-(excerpt_length)))
    point1b = point1a+excerpt_length

    safety_count = 0
    while sum(lengths1[:point1a]) % 2 != 0 or sum(lengths2[point2a:point2b]) % 2 != 0 or sum(lengths2[:point2a]) % 2 != 0 or sum(lengths1[point1b:]) % 2 != 0:
        excerpt_length = random.randint(2,6)
        point2a = random.randint(0, max(0,len(lengths2)-excerpt_length))
        point2b = point2a + excerpt_length
        point1a = random.randint(0, max(1,len(lengths1)-(excerpt_length)))
        point1b = point1a+excerpt_length
        
        safety_count += 1
        if safety_count > 10000:
            print("WARNING: Something went wrong during execution of ExcerptRekombination(). Could be a BUG! Using fail safe now by just returning copy of parent1.")
            return o

    pitches_new = pitches1[:point1a] + pitches2[point2a:point2b] + pitches1[point1b:]
    lengths_new = lengths1[:point1a] + lengths2[point2a:point2b] + lengths1[point1b:]
    
    difference = sum(lengths_new) - sum(parent1.genotyp[1])
    dbg_print("inserting excerpt made difference of:", difference)

    o.genotyp[0] = pitches_new
    o.genotyp[1] = lengths_new

    dbg_print("Following excerpt was inserted:", lengths1[:point1a], lengths2[point2a:point2b], lengths1[point1b:])
    dbg_print(f"Gentype after recombination: {o.genotyp}")
    assert difference % 2 == 0, f"Difference is not even: difference={difference}"

    return o

def binomial(n):
    p = 1 / n
    k = 0  # success count
    for _ in range(n):
        if random.random() < p:
            k += 1
    return k

def mutate(offspring): 
    dbg_print(f"Genotyp before mutation: {offspring.genotyp}")
    
    start_time = now()
    if MUTATION_TYPE == "local":
        length = len(offspring.genotyp[0])*2
        for i in range(max(1,binomial(length))):
            if random.random() < 0.5:
                dbg_print(f"start mutate_pitches ({i})")
                offspring = mutate_pitches(offspring)
            else:
                dbg_print(f"start mutate_lengths ({i})")
                offspring = mutate_lengths(offspring)
    elif MUTATION_TYPE == "random":
        length = len(offspring.genotyp[0])*2
        for i in range(max(1,binomial(length))):
            offspring = mutate_random(offspring)
    else:
        assert False, "Mutation type unknown:"+MUTATION_TYPE
    
    dbg_print("mutatation took:",now()-start_time,"s")
    dbg_print(f"Genotyp after mutation: {offspring.genotyp}")
    return offspring

def mutate_random(offspring):
    point = random.randint(0,len(offspring.genotyp[0])-1)
    
    if random.random() < 0.5:
        # random mutate pitches
        if random.random() < REST_RATE:
            offspring.genotyp[0][point] = -1
        else:
            offspring.genotyp[0][point] = random.randint(0, interval[1])
    else:
        #random mutate lengths
        offspring.genotyp[1][point] = random.choice([1,2,4])

    return offspring

def mutate_pitches(offspring):
    pitches = offspring.genotyp[0]
    lengths = offspring.genotyp[1]
    #for i in range(len(höhen)):
    #    if random.random() < MUTATE_RATE:
    
    # choose mutation index
    i = random.randint(0,len(pitches)-1)

    if pitches[i] == -1:
        # repeat a note, do not choose randomly anymore
        #pitches[i] = ran.integers(0,interval[1],endpoint=True)
        dbg_print("Rest becomes note")
        pitches[i] = int(random.choice([pitches[i-1], pitches[(i+1) % len(pitches)]]))
    elif len(pitches)>1 and i>=1 and random.random() < REST_RATE and pitches[i]>-1:
        dbg_print("Note becomes Rest (but never first note)")
        pitches[i] = -1
    else:
        dbg_print("Note pitch gets mutated")
        p = pitches[i]
        while p == pitches[i] or pitches[i]>interval[1] or pitches[i]<0 :
            pitches[i] = int(random.gauss(mu=pitches[i],sigma=2))

    offspring.genotyp[0] = pitches
    offspring.genotyp[1] = lengths
    
    return offspring
    
def mutate_lengths(offspring):
    pitches = offspring.genotyp[0]
    lengths = offspring.genotyp[1]
    
    sum_before = sum(lengths)
    
    # choose mutation index
    i = random.randint(0,len(lengths)-1)
    
    ### Version without dotted quarter notes
    # case: half note
    if lengths[i] == 4:
        # Half to quarter
        lengths = lengths[:i] + [2,2] + lengths[i+1:]
        pitches = pitches[:i] + pitches[i:i+1]*2 + pitches[i+1:]
    # case: quarter note
    elif lengths[i] == 2:
        if random.random() < 0.5:
            # quarter to two eigths
            dbg_print("quarter to two eigths")
            lengths = lengths[:i] + [1,1] + lengths[i+1:]
            pitches = pitches[:i] + pitches[i:i+1]*2 + pitches[i+1:]
        else:
            # two quarter to one half note
            dbg_print("two quarter to one half note")
            if i+1 < len(lengths) and lengths[i+1] == 2 and sum(lengths[:i])%4==0:
                lengths = lengths[:i] + [4] + lengths[i+2:]
                pitches = pitches[:i] + random.choice([pitches[i:i+1], pitches[i+1:i+2]]) + pitches[i+2:]
            elif i-1 >= 0 and lengths[i-1] == 2 and sum(lengths[:i])%4==2:
                lengths = lengths[:i-1] + [4] + lengths[i+1:]
                pitches = pitches[:i-1] + random.choice([pitches[i-1:i], pitches[i:i+1]]) + pitches[i+1:]
            else:
                dbg_print("This should not have happened.. mutating again.")
                return mutate_lengths(offspring)
    # case: two eighth notes
    elif lengths[i] == 1:
        # two eighth notes to one quarter note
        if i+1 < len(lengths) and lengths[i+1] == 1 and sum(lengths[:i])%2==0:
            lengths = lengths[:i] + [2] + lengths[i+2:]
            pitches = pitches[:i] + random.choice([pitches[i:i+1], pitches[i+1:i+2]]) + pitches[i+2:]
        elif i-1 >= 0 and lengths[i-1] == 1 and sum(lengths[:i])%2==1:
            lengths = lengths[:i-1] + [2] + lengths[i+1:]
            pitches = pitches[:i-1] + random.choice([pitches[i-1:i], pitches[i:i+1]]) + pitches[i+1:]
        else:
            dbg_print("This should not have happened.. mutating again.")
            return mutate_lengths(offspring)
    else:
        dbg_print(f"Note length {lengths[i]} unknown.")
    
    offspring.genotyp[0] = pitches
    offspring.genotyp[1] = lengths
    
    # assure the mutation affects the melody only locally
    assert sum(lengths) == sum_before
    assert len(pitches) == len(lengths)
    
    return offspring

def roulette_selection(population):
    maximum = sum(individual.fitness for individual in population)
    pick = random.uniform(0,maximum-1)
    current = 0
    for individual in population:
        current += individual.fitness
        if current > pick:
            return individual
        
def marriage_select(pop):
    parents = []
    for i in range(RHO):
        parent = roulette_selection(pop)
        while parent in parents:
            parent = roulette_selection(pop)
        parents.append(parent)
    return parents
    
def best_select(population, mu=MU):
    
    next_pop = remove_doubles(population)
    next_pop = sorted(next_pop, key=lambda ind: ind.fitness, reverse=True)[:mu]
    if len(next_pop) < mu:
        dbg_print(f"WARNING: next_pop:{len(next_pop)} < mu:{mu}")
    return next_pop
    
def tournament_select(population, mu=MU, k=2):
    next_pop = []
    t_count = 0
    while len(next_pop) < mu:
        tournament = random.sample(population, k=k)
        # select best
        winner = max(tournament, key=lambda ind: ind.fitness)
        next_pop.append(winner)
        
        next_pop = remove_doubles(next_pop)
        t_count += 1
    dbg_print(f"{t_count} tournaments needed to generate {mu} offsprings (tournament size={k}).")
    return next_pop

def remove_doubles(population):
    pop_set = []  
    seen = set()
    not_seen_count = 0
    for ind in population:
        ind_tuple = (tuple(ind.genotyp[0]), tuple(ind.genotyp[1]))
        if ind_tuple not in seen:
            pop_set.append(ind)
            seen.add(ind_tuple)
        else:
            not_seen_count += 1
    dbg_print(f"{not_seen_count} individuals were doubles.")
    return pop_set

def copyPop(population):
    copy = list()
    for i in population:
        copy.append(i)
    return copy


def log_detailed_stats(population, n_gen=0, first=False):

    if not LOG_DETAILED_STATS:
        return

    # Calculate worst, best and mean of the fitnesses
    worst = min(population, key=lambda ind: ind.fitness).fitness
    best = max(population, key=lambda ind: ind.fitness).fitness
    mean = sum(ind.fitness for ind in population) / len(population)
    
    # Write stats to file
    with open(PTH_OUT+FILENAME_STATS, mode='a', newline='') as file:
        writer = csv.writer(file)
        if first:
            writer.writerow(['n_gen', 'min', 'max', 'mean'])
        writer.writerow([n_gen, worst, best, mean])

def trim_list_by_sum(lst, max_sum):
    while sum(lst[:-1]) >= max_sum:
        dbg_print("trimming..")
        lst.pop()
    return lst


# main programm
if __name__ == "__main__": 
    '''
    Main programm
    '''

    writeStringToFile(PTH_OUT+"parameter.txt", paramToStr())
    writeStringToFile(PTH_OUT+"prompt.txt", str(PROMPT))

    #erstelle erste Generation
    population = list()
    for i in range(0,MU):
        if INIT_STRATEGY == "uniform":
            g = uniformGenotyp(interval[0],interval[1], INIT_LENGTH)
        elif INIT_STRATEGY == "random":
            g = randomGenotyp(interval[0],interval[1], INIT_LENGTH)
        population.append(Individual(g))
    
    evaluate_fitness(population,0)
    Best = max(population, key=lambda ind: ind.fitness).copy()
    best_fitness = Best.fitness
    stayCounter = 0
    outputIndivudual(Best,stayCounter,0)

    log_detailed_stats(population,n_gen=0,first=True)

    print("Starting Evolution..")
    for i in tqdm(range(1,iterations+1), miniters=max(1,iterations//100)):#, disable=LOG_TO_FILE_MODE): 
        offsprings = []
        #generate Offsprings
        for j in range(LAMBDA):
            #parents = marriage_select(population,RHO)
            parents = random.choices(population,k=RHO)

            if RHO > 1 and random.random() < RECOMBINATION_RATE:
                offspring = recombinate(parents)
            else:
                offspring = parents[0].copy()
            
            offspring = mutate(offspring)
    
            if MAX_MEASURES > 0:
                offspring.genotyp[1] = trim_list_by_sum(offspring.genotyp[1], max_sum=MAX_MEASURES*8)
                offspring.genotyp[0] = offspring.genotyp[0][:len(offspring.genotyp[1])]
            
            offsprings.append(offspring)
        
        evaluate_fitness(offsprings, i)
        
        if SELECTION == "best":
            population = best_select(population + offsprings)
        elif SELECTION == "tournament":
            population = tournament_select(population + offsprings)
        else:
            print("Selection strategy not implemented:", SELECTION)
            sys.exit(1)
        
        pBest = max(population, key=lambda ind: ind.fitness).copy()
        if not pBest.equals(Best):
            Best = pBest
            if not clamp is None:
                outputIndivudual(Best,i,stayCounter,best_fitness_so_far=best_fitness)
            if Best.fitness > best_fitness:
                best_fitness = Best.fitness
            stayCounter = 0
        else:
            stayCounter += 1

        log_detailed_stats(population, n_gen=i)

    #outputIndivudual(Best, iterations , stayCounter)

    bests = sorted(population, key=lambda ind: ind.fitness,reverse=True)
    #print(bests)
    for i in range(len(bests)):
        outputIndivudual(bests[i], i , 0, prePath=PTH_LAST_GEN )
    
    # convert all generated abc files of last generation to pdfs and midis
    try:
        convert_bash_script = "bash abctopdf/abc2midi_and_ps.sh"
        abc_path = PTH_OUT+PTH_LAST_GEN+PTH_ABCS
        os.system(convert_bash_script+" "+abc_path+"*.abc"+" >> "+abc_path+"/"+LOG_FILENAME+" 2>&1")
    except:
        print("Error when calling converter script:", convert_bash_script)

    # store results in a final destination
    if not PTH_FINAL is None:
        print(f"copying results from '{PTH_OUT}' to '{PTH_FINAL}'")
        shutil.copytree(PTH_OUT, PTH_FINAL, dirs_exist_ok=True)
        
    print("Program exits without errors.")
    if LOG_TO_FILE_MODE:
        stdout_redirect_file.close()
        
