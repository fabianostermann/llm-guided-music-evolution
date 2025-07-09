import os
import numpy as np
import json
import csv

def clear_folder_except_files(folder_path, files_to_keep):
    try:
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            raise ValueError(f"The folder '{folder_path}' does not exist.")

        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            if os.path.isfile(file_path) and file_name not in files_to_keep:
                os.remove(file_path)

    except Exception as e:
        print(f"Error: {e}")

def clear_folder(folder_path):
    try:
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            raise ValueError(f"The folder '{folder_path}' does not exist.")

        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)

    except Exception as e:
        print(f"Error: {e}")


def writeStringToFile(file_path, content):
    with open(file_path, 'w') as file:
        file.write(content)


def saveGenotyp(genotyp, path):
    genotyp_json = list(genotyp)
    with open(path+".json", "w") as datei:
        json.dump(genotyp_json, datei)

def loadGenotyp(path, generation = -1):
    if generation > -1:
        path = path + str(generation) + ".json"
    
    with open(path, "r") as json_datei:
        geladener_json = json.load(json_datei)
    
    genotyp = np.array(geladener_json)
    return genotyp


def getIntervalls(genotyp):
    intervalls = np.zeros(1)
    pitches = genotyp[0]
    for i in range(len(pitches)-1):
        #a
        a = pitches[i]
        if a > -1 :
            b = pitches[i+1]
            if b >-1:
                diff = int(np.abs(a-b))
                if diff>len(intervalls)-1:
                    intervalls = np.pad(intervalls,(0,diff-len(intervalls)+1),'constant',constant_values=0)
                intervalls[diff] += 1
    return intervalls
    
def writeIntervalls(individuum, path):
    intervalls = getIntervalls(individuum.genotyp)
    with open(path, 'a', newline='') as csv_datei:
        wrt = csv.writer(csv_datei)
        for i in range(len(intervalls)):
            wrt.writerow((i,intervalls[i]))

def addToCSV(individuum, stayCounter, gen, path):
    #filename
    with open(path, 'a', newline='') as csv_datei:
        wrt = csv.writer(csv_datei)
        wrt.writerow((gen,individuum.fitness,stayCounter))


def to_tuple(array):
    if isinstance(array, np.ndarray):
        return tuple(map(to_tuple, array))
    else:
        return array
