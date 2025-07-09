import os, sys, argparse
import json

from statistics import fmean as mean
import matplotlib.pyplot as plt

from scipy.stats import ranksums

# get args
def get_args(parser):
    parser.add_argument("paths", nargs='+', default=[])
    parser.add_argument('--output', '-o', type=str, default=None, help="output figure to file instead of showing")
    parser.add_argument('--force', '-f', action='store_true', default=False, help="force overwriting of existing files")
    parser.add_argument('--property', '-p', type=str, choices={"pitches", "durations", "lengths", "intervals"}, default="pitches", help="the statistical property to plot")
    parser.add_argument('--best-only', action='store_true', default=False, help="Consider only best individuum instead of whole population")
    parser.add_argument('--test', action='store_true', default=False, help="Test for significance (using one-sided Wilcoxon rank-sum test)")
    parser.add_argument('--show-prompts', action='store_true', default=False, help="Tries to read prompt.txt in experiment folder and puts them below the plot.")
    parser.add_argument('--description', type=str, nargs='+', default=None, help="Additional text that will be shown below the plot")
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
        "pitches": [],
        #"mean_pitches": [],
        "durations": [],
        "lengths": [],
        "intervals": [],
    }
    for genotyp in genotyps:
        pitches = genotyp[0]
        durations = genotyp[1]

        pitches_ignore_rests = [p for p in pitches if p >= 0]
        stats["pitches"] += [mean(pitches_ignore_rests)] # remove rests
        stats["durations"] += [mean(durations)]
        stats["lengths"] += [sum(durations)]
        if len(pitches_ignore_rests) >= 2:
            intervals = [abs(pitches_ignore_rests[i] - pitches_ignore_rests[i+1]) for i in range(len(pitches_ignore_rests) - 1)]
            stats["intervals"] += [mean(intervals)]
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

def make_boxplot(stats, property="pitches", description=None):
    fig, ax = plt.subplots(figsize=(len(stats),4))

    plt.boxplot(
        [v[property] for v in stats.values()],
        tick_labels=stats.keys(),
        showmeans=True, meanprops=dict(marker='o', markerfacecolor='green'),
        widths=0.7,
    )
    plt.ylabel(property.replace("_", " "))

    if property == "pitches":
        ax.set_yticks(list(range(0,15)))
        ax.yaxis.set_major_formatter(lambda x, pos: "cdefgab"[x%7] + '\''*(x//7+1))
    if property == "intervals":
        ax.set_yticks(list(range(0,6+1)))
        plt.ylim(bottom=0, top=6)
    elif property == "durations":
        ax.set_yticks([1,2,4])
        #ax.set_yticklabels(["Eighth note", "Quarter note", "Half note"])
        ax.set_yticklabels([1, 2, 4])
    elif property == "lengths":
        #max_bars = 15
        #plt.ylim(bottom=0, top=8*max_bars)
        #ax.set_yticks(list(range(0,8*max_bars,8)))
        ax.yaxis.set_major_formatter(lambda x, pos: "Bar " + str(int(x//8+1)))

    if not description is None and not description == "":
        ax.text(0.05,-0.1, description, size=10, ha="left", va="top",
         transform=ax.transAxes, wrap=True, linespacing=1.66)

    #fig.tight_layout()

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
        plt.savefig(args.output, bbox_inches='tight')
        print("Figure written to file:", args.output)

def wilcoxon(t1, t2, alt="less"):
    assert len(t1) == len(t2), "lengths do not match"
    statistic, pvalue = ranksums(list(t1), list(t2), alternative=alt)
    print(f"z={statistic}, p={pvalue}")
    return statistic, pvalue

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

    description = ""

    if args.test:
        if len(args.paths) < 2:
            print("For testing, at least two paths are needed. Abort.")
            sys.exit(1) 
        z, p = wilcoxon(
            stats[args.paths[0]][args.property],
            stats[args.paths[1]][args.property],
        )
        #\nWilcoxon ({args.paths[0]}<{args.paths[1]}): "\
        description += f""\
            + f"z={z:.2f}, "\
            + f"p={p:.2g}"
        
    if args.show_prompts:
        for path in args.paths:
            prompt_text = read_prompt(path)
            if not prompt_text is None:
                description += "\n" + path + ": \"" + prompt_text + "\""

    if not args.description is None:
        description += "\n" + '\n'.join(args.description)

    make_boxplot(
        stats,
        property=args.property,
        description=description,
    )
                
