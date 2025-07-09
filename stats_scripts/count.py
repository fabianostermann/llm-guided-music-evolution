import os, sys, argparse
import json, yaml

from statistics import fmean as mean
import matplotlib.pyplot as plt

# get args
def get_args(parser):
    parser.add_argument("paths", nargs='+', default=[])
    #parser.add_argument('--output', '-o', type=str, default=None, help="output figure to file instead of showing")
    #parser.add_argument('--force', '-f', action='store_true', default=False, help="force overwriting of existing files")
    parser.add_argument('--best-only', action='store_true', default=False, help="Consider only best individuum instead of whole population")
    parser.add_argument('--trim', type=int, default=None, help="trim genotype to first n entries.")
    
    return parser

def load_genotyp_from_file(file, trim=None):
    with open(file) as f:
        content = f.read()
        genotyp = json.loads(content)
    genotyp[0] = genotyp[0][:trim]
    genotyp[1] = genotyp[1][:trim]
    return genotyp

def load_genotyps(path, best_only=False, trim=None):
    genotyps = []
    for folder in os.listdir(path):
        genotyp_folder = path+"/"+folder+"/lastGen/genotyps/"
        if os.path.exists(genotyp_folder):
            if best_only:
                genotyps.append(load_genotyp_from_file(genotyp_folder+"/0.json", trim))
            else:
                for json in [f for f in os.listdir(genotyp_folder) if f.endswith(".json")]:
                    genotyps.append(load_genotyp_from_file(genotyp_folder+"/"+json, trim))
    return genotyps

def process_stats(genotyps):
    stats = {
        #"pitches": [],
        #"mean_pitches": [],
        #"durations": [],
        #"lengths": [],
        #"intervals": [],
        "event_count": 0,
        "event_count_wo_rests": 0,
        
        "half_notes": 0,
        "quarter_notes": 0,
        "eighth_notes": 0,
        
        "half_notes_wo_rests": 0,
        "quarter_notes_wo_rests": 0,
        "eighth_notes_wo_rests": 0,
        
        "interval_count": 0,
        "tone_repetitions": 0,
        
        "rests":0,
    }
    for genotyp in genotyps:
        pitches = genotyp[0]
        durations = genotyp[1]
        
        stats["event_count"] += len(pitches)
        
        stats["half_notes"] += durations.count(4)
        stats["quarter_notes"] += durations.count(2)
        stats["eighth_notes"] += durations.count(1)
        
        stats["rests"] += pitches.count(-1)
        
        pitches_ignore_rests = [p for p in pitches if p >= 0]
        durations_ignore_rests = []
        for i in range(len(durations)):
            if pitches[i] > -1:
                durations_ignore_rests.append(durations[i])
               
        stats["event_count_wo_rests"] += len(pitches_ignore_rests)

        stats["half_notes_wo_rests"] += durations_ignore_rests.count(4)
        stats["quarter_notes_wo_rests"] += durations_ignore_rests.count(2)
        stats["eighth_notes_wo_rests"] += durations_ignore_rests.count(1)

#        stats["pitches"] += [mean(pitches_ignore_rests)] # remove rests
#        stats["durations"] += [mean(durations)]
#        stats["lengths"] += [sum(durations)]
        if len(pitches_ignore_rests) >= 2:
            intervals = [abs(pitches_ignore_rests[i] - pitches_ignore_rests[i+1]) for i in range(len(pitches_ignore_rests) - 1)]
            #stats["intervals"] += [mean(intervals)]
        
        stats["interval_count"] += len(intervals)
        stats["tone_repetitions"] += intervals.count(0)

    for key, value in stats.items():
        if not "count" in key:
            if key.endswith("_wo_rests"):
                stats[key] = round(value / stats["event_count_wo_rests"], 3)
            elif key == "tone_repetitions":
                stats[key] = round(value / stats["interval_count"], 3)
            else:
                stats[key] = round(value / stats["event_count"], 3)
    return stats

def read_prompt(path):
    for folder in os.listdir(path):
        prompt_file = path+"/"+folder+"/prompt.txt"
        if os.path.exists(prompt_file):
            with open(prompt_file) as f:
                content = f.read().strip()
            # returns the first prompt that is found
            # WARNING: will not check if all prompts in path are the same
            return content
    print(f"File {prompt_file} not found.")
    return None


if __name__ == "__main__": 

    if len(sys.argv) <= 1:
        print("Please provide 1 or more paths to experiment data folders.\nFor usage info use --help.")
        sys.exit(1)

    # process arguments
    args = get_args(argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)).parse_args()
    print("Processing data from paths:", '; '.join(args.paths))

    for path in args.paths:
        assert os.path.exists(path), f"Path does not exist: {path}"

    stats = {}
    for path in args.paths:
        genotyps = load_genotyps(path, best_only=args.best_only, trim=args.trim)
        stats[path] = process_stats(genotyps)

    print(yaml.dump(stats, allow_unicode=True, default_flow_style=False))

    #description = ""

    #if args.test:
    #    if len(args.paths) < 2:
    #        print("For testing, at least two paths are needed. Abort.")
    #        sys.exit(1) 
    #    z, p = wilcoxon(
    #        stats[args.paths[0]][args.property],
    #        stats[args.paths[1]][args.property],
    #    )
    #    #\nWilcoxon ({args.paths[0]}<{args.paths[1]}): "\
    #    description += f""\
    #        + f"z={z:.2f}, "\
    #        + f"p={p:.2g}"
    
    
       
    #if args.show_prompts:
    #    for path in args.paths:
    #        prompt_text = read_prompt(path)
    #        if not prompt_text is None:
    #            description += "\n" + path + ": \"" + prompt_text + "\""

    #if not args.description is None:
    #    description += "\n" + '\n'.join(args.description)

    #make_boxplot(
    #    stats,
    #    property=args.property,
    #    description=description,
    #)
                
