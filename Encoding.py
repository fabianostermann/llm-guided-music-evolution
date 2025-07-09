#import numpy as np
from enum import Enum
import sys
import json

Encoding_2_Oct_Chroma_Sharps = {
    -1 : 'z',
    0 : 'C',
    1 : '^C',
    2 : 'D',
    3 : '^D',
    4 : 'E',
    5 : 'F',
    6 : '^F',
    7 : 'G',
    8 : '^G',
    9 : 'A',
    10 : '^A',
    11 : 'B',
    12 : 'c',
    13 : '^c',
    14 : 'd',
    15 : '^d',
    16 : 'e',
    17 : 'f',
    18 : '^f',
    19 : 'g',
    20 : '^g',
    21 : 'a',
    22 : '^a',
    23 : 'b',
    24 : "c'"
}

Encoding_2_Oct_Chroma_Flats = {
    -1 : 'z',
    0 : 'C',
    1 : '_D',
    2 : 'D',
    3 : '_E',
    4 : 'E',
    5 : 'F',
    6 : '_G',
    7 : 'G',
    8 : '_A',
    9 : 'A',
    10 : '_B',
    11 : 'B',
    12 : 'c',
    13 : '_d',
    14 : 'd',
    15 : '_e',
    16 : 'e',
    17 : 'f',
    18 : '_g',
    19 : 'g',
    20 : '_a',
    21 : 'a',
    22 : '_b',
    23 : 'b',
    24 : "c'"
}

Encoding_2_Oct_Dia = {
    -1 : 'z',
    0 : 'C',
    1 : 'D',
    2 : 'E',
    3 : 'F',
    4 : 'G',
    5 : 'A',
    6 : 'B',
    7 : 'c',
    8 : 'd',
    9 : 'e',
    10 : 'f',
    11 : 'g',
    12 : 'a',
    13 : 'b',
    14 : "c'"
}

Encoding_note_lengths = {
    # valid for L:1/8
    1 : '', # Achtel
    2 : '2',# Viertel
    3 : '3',# punkt. Viertel
    4 : '4',# Halbe
}

defaultStorage = {
    'z' : '',
    'C' : '',
    'D' : '',
    'E' : '',
    'F' : '',
    'G' : '',
    'A' : '',
    'B' : '',
    'c' : '',
    'd' : '',
    'e' : '',
    'f' : '',
    'g' : '',
    'a' : '',
    'b' : '',
    "c'" : '',
}

class ABC_Generator:
    '''
    Class translating genotype into phenotype with valid ABC syntax
    '''

    pitchEncoder = None

    #init
    def __init__(self, encoder):
        self.pitchEncoder = encoder

    def getHeaderString(self):
        #bestimme speed string
        header = "M:4/4\nL:1/8\nK:C\n"
        return header

    def encode_voice(self, genotype):
        '''
        Transles genotype into phenotype with valid ABC syntax
        '''
        abc = self.getHeaderString()  

        bar_length = 8
        
        pitches = genotype[0]
        lengths = genotype[1]
        
        curr_bar_length = 0
        accidentals = defaultStorage.copy()

        for n in range(len(pitches)): 
            
            filler=""
            length1 = length2 = None
            
            pitch1 = self.pitchEncoder.get(pitches[n])
            pitch2 = ''

            if '^' in pitch1:
                if accidentals[pitch1[1:]] == '^':
                    pitch1 = pitch1[1:]
                else:
                    accidentals[pitch1[1:]] = '^'

            elif accidentals[pitch1] != '':
                accidentals[pitch1] = ''
                pitch1 = "="+pitch1
                
            if curr_bar_length + lengths[n] > bar_length: 
                length1 = bar_length-curr_bar_length
                length2 = lengths[n]-length1
                filler = "- | "
                if pitches[n] == -1 :
                    filler = " | "
                curr_bar_length = length2
                pitch2 = self.pitchEncoder.get(pitches[n])
                accidentals = defaultStorage.copy()
                if '^' in pitch2:
                    accidentals[pitch2[1:]]= '^'

            elif curr_bar_length< bar_length/2 and curr_bar_length + lengths[n] > bar_length/2 :
                length1 = bar_length/2-curr_bar_length
                length2 = lengths[n]-length1
                filler =  "- "
                if pitches[n] == -1 :
                    filler = " "
                curr_bar_length += length2+length1
                pitch2 = self.pitchEncoder.get(pitches[n])
                if '^' in pitch2 or '=' in pitch2:
                    pitch2 = pitch2[1:]
            else:
                length1 = lengths[n]
                curr_bar_length += length1
                if curr_bar_length == bar_length:
                    curr_bar_length = 0
                    filler = " |"
                    accidentals = defaultStorage.copy()
                      
            stringl1 = stringl2 = ''

            if length2 != None : 
                stringl2 = Encoding_note_lengths.get(length2)
                filler += pitch2 + stringl2 
                    
            stringl1 = Encoding_note_lengths.get(length1)
            
            abc+= pitch1+stringl1+filler+' '

        if curr_bar_length > 0 and curr_bar_length < bar_length:    
            if curr_bar_length == 1:
                abc += "z z2 z4"
            elif curr_bar_length == 2:
                abc += "z2 z4"
            elif curr_bar_length == 3:
                abc += "z z4"
            elif curr_bar_length == 4:
                abc += "z4"
            elif curr_bar_length == 5:
                abc += "z z2"
            elif curr_bar_length == 6:
                abc += "z2"
            elif curr_bar_length == 7:
                abc += "z"

            abc += ' |'
        return abc
        
if __name__ == "__main__":
    Encoding = Encoding_2_Oct_Dia
    print("Note encoding:", Encoding)
    abcgen = ABC_Generator(Encoding)
    
    genotype = [[0,1,2,3,4],[4,2,1,1,1]]
    if len(sys.argv) > 1:
        genotype = json.loads(sys.argv[1])
    print("Genotype:\n", genotype)
    
    abc = abcgen.encode_voice(genotype)
    print("ABC:\n",abc)
