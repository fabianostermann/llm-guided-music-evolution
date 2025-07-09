import os, sys, argparse
import json

from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statistics import fmean as mean

# get args
def get_args(parser):
    parser.add_argument("paths", nargs='+', default=[])
    parser.add_argument("--legend", nargs='+', default=[])
    parser.add_argument('--output', '-o', type=str, default=None, help="output figure to file instead of showing")
    parser.add_argument('--force', '-f', action='store_true', default=False, help="force overwriting of existing files")
    parser.add_argument('--mean-only', action='store_true', default=False, help="do not plot all single runs, just the mean of them")
    parser.add_argument('--description', type=str, nargs='+', default=None, help="Additional text that will be shown below the plot (each string in a new line)")
    parser.add_argument('--property', '-p', type=str, choices={"pitches", "durations", "lengths", "intervals", "fitness"}, default="fitness", help="the statistical property to plot")
    return parser

def load_genotyp_from_file(file):
    with open(file) as f:
        content = f.read()
        genotyp = json.loads(content)
    return genotyp

def load_genotyp_stats(paths, property, iterations, BUDGET, LAMBDA):
    stats = []
    for path in paths.split("+"):
        for folder in os.listdir(path):
            genotyp_folder = path+"/"+folder+"/genotyps/"
            if os.path.exists(genotyp_folder):
                df = pd.DataFrame(columns=[0,1])
                for i in range(iterations):
                    next_file = genotyp_folder+f"/{i}.json"
                    if os.path.exists(next_file):
                        genotyp = load_genotyp_from_file(next_file)
                        df.loc[len(df)] = [i, make_stat(genotyp, property=property)]
            
            # repeat last value
            last_row_value = df.iloc[-1, 1]
            df.loc[len(df)] = [iterations, last_row_value]

            df[0] = (df[0] * LAMBDA).astype(int)
            stats.append(df)

    return stats

def make_stat(genotyp, property):
    pitches = genotyp[0]
    durations = genotyp[1]

    pitches_ignore_rests = [p for p in pitches if p >= 0]

    if property == "pitches":
        if len(pitches_ignore_rests) >= 1:
            return mean(pitches_ignore_rests)
        else:
            return -1
    elif property == "durations":
        return mean(durations)
    elif property == "lengths":
        return sum(durations)
    elif property == "intervals":
        if len(pitches_ignore_rests) >= 2:
            intervals = [abs(pitches_ignore_rests[i] - pitches_ignore_rests[i+1]) for i in range(len(pitches_ignore_rests) - 1)]
            return mean(intervals)
        else:
            return -1
    else:
        assert False, f"Property {property} unknown."

def load_fitness_file(file):
    return pd.read_csv(file, header=None)

def read_parameters(path):
    for folder in os.listdir(path):
        with open(path+"/"+folder+'/parameter.txt', 'r') as file:
            params = file.read()
            iterations = int(params.split('iterations = ')[1].splitlines()[0])
            LAMBDA = int(params.split('LAMBDA = ')[1].splitlines()[0])
            
            BUDGET = None
            try:
                BUDGET = int(params.split('BUDGET = ')[1].splitlines()[0])
            except:
                pass
            assert BUDGET is None or BUDGET == iterations * LAMBDA, "Parameters are not consistent"
        return { "iterations": iterations, "BUDGET": BUDGET, "LAMBDA": LAMBDA }

def load_fitness(paths, iterations, BUDGET, LAMBDA):
    fitness = []
    params_loaded = False
    
    for path in paths.split("+"):
        for folder in os.listdir(path):
            fitness_file = path+"/"+folder+"/fitness.csv"
            df = load_fitness_file(fitness_file)

            #print(df)
            #print(f"it:{iterations}, l:{LAMBDA}")

            # repeat last value
            last_row_value = df.iloc[-1, 1]
            df.loc[len(df)] = [iterations, last_row_value, None]

            #print(df)
            
            df[0] = (df[0] * LAMBDA).astype(int)
            fitness.append(df)

        #print(df)
    return fitness

# Funktion zur Interpolation der y-Werte
def interpolate_steps(x, y, x_common):
    y_common = []
    for xc in x_common:
        # Finde den letzten Wert in x, der kleiner oder gleich xc ist
        idx = np.searchsorted(x, xc, side='right') - 1
        y_common.append(y[idx])
    return y_common

def make_plot(fitness_dict, legend, property="fitness", description=None, mean_only=False):
    fig, ax = plt.subplots()

    print("Plotting..")
    for path, fitnesses in tqdm(zip(legend, fitness_dict.values())):

        X, Y = [], []
        for f in fitnesses:
            X.append(f[0].values)
            Y.append(f[1].values)

        x_common = sorted(set().union(*X))
        all_y_common = []
        curr_color = None
        for i in range(len(X)):
            x = X[i]
            y = Y[i]
            y_common = interpolate_steps(x, y, x_common)
            all_y_common.append(y_common)
            #if curr_color is None:
            if not mean_only:
              baseline = ax.step(x_common, y_common, where='post', label=None, color=curr_color, alpha=0.1)[0]
              curr_color = baseline.get_color()

        # Berechne den Mittelwert, das Minimum und das Maximum für jedes x_common
        mean_y = np.mean(all_y_common, axis=0)
        
        min_y = np.min(all_y_common, axis=0)
        mask = all_y_common < mean_y
        filtered_values = np.where(mask, all_y_common, np.nan)
        min_y_average = np.nanmean(filtered_values, axis=0)
        
        max_y = np.max(all_y_common, axis=0)
        mask = all_y_common > mean_y
        filtered_values = np.where(mask, all_y_common, np.nan)
        max_y_average = np.nanmean(filtered_values, axis=0)
        
        # Mittelwertslinie und range plotten
        baseline = ax.step(x_common, mean_y, where='post', label=path, linewidth=2, color=curr_color)[0]#, alpha=1)
        curr_color = baseline.get_color()
        ax.fill_between(x_common, min_y_average, max_y_average, step='post', label=None, color=curr_color, alpha=0.1)

    plt.ylabel(property)
    plt.xlabel("budget")
    plt.legend()

    if property == "pitches":
        ax.set_yticks(list(range(0,15)))
        ax.yaxis.set_major_formatter(lambda x, pos: "CDEFGAB"[x%7] + '\''*(x//7+1))
    elif property == "durations":
        ax.set_yticks([1,2,4])
        ax.set_yticklabels(["Eighth note", "Quarter note", "Half note"])
    elif property == "lengths":
        #max_bars = 15
        #plt.ylim(bottom=0, top=8*max_bars)
        #ax.set_yticks(list(range(0,8*max_bars,8)))
        ax.yaxis.set_major_formatter(lambda x, pos: "Bar " + str(int(x//8+1)))

    if not description is None and not description == "":
        ax.text(0.05,-0.05, description, size=10, ha="left", va="top",
         transform=ax.transAxes, wrap=True, linespacing=1.66)

    # prepare output
    fig.tight_layout()

    if args.output is None:
        plt.show()
    else:
        if os.path.exists(args.output):
            if not args.force:
                print("File exists:", args.output)
                answer = input('Force overwrite? (Y/n) ')
                if answer.lower() != "y" and answer.lower() != "yes":
                    print("Abort.")
                    sys.exit(0)
        plt.savefig(args.output)#, bbox_inches='tight')
        print("Figure written to file:", args.output)


if __name__ == "__main__": 

    if len(sys.argv) <= 1:
        print("Please provide 1 or more paths to experiment data folders.\nFor usage info use --help.")
        sys.exit(1)

    # process arguments
    args = get_args(argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)).parse_args()
    print("Considering data from following paths:")
    print(' '.join(args.paths))

    for paths in args.paths:
        for path in paths.split("+"):
            assert os.path.exists(path), f"Path does not exist: {path}"

    params = read_parameters(path)

    print("Process data..")
    stats = {}
    for path in tqdm(args.paths):
        if args.property == "fitness":
            stats[path] = load_fitness(path, **params)
        else:
            stats[path] = load_genotyp_stats(path, property=args.property, **params)

    description = ""
    if not args.description is None:
        description += "\n" + '\n'.join(args.description)

    make_plot(stats, args.legend if len(args.legend)>0 else stats.keys(), property=args.property, description=description, mean_only=args.mean_only)
                
