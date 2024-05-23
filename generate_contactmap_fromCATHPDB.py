import argparse
import numpy as np
import os
import sys
import shutil
import yaml
from Bio.PDB import PDBParser
import glob


def encode_sequence(seq):
    alphabet = ["A", "C", "D", "E", "F", "G","H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
    char_to_int = dict((c,i) for i,c in enumerate(alphabet))
    integer_encoded = [char_to_int[char] for char in seq]  # convert the protein sequence to the number sequence
    one_hot_encode = [] 
        
    for value in integer_encoded:
      letter = [0 for _ in range(len(alphabet))]
      letter[value] = 1
      one_hot_encode.append(letter) # each row only has one "1", represnting the type of residue
        
        #print(one_hot_encode)    
    return np.array((one_hot_encode)) # create a 2-D array for one_hot array.

def pad_seq(seq,max_len): #  padding
    leng=len(seq)
    if leng<max_len:
        pad_len = max_len-leng
        return np.pad(list(seq), (0, pad_len), mode='constant', constant_values=0)
    else:
        return np.asarray(list(seq[:max_len]))
    

def pad_coordinates(x,max_len): 
    leng = len(x)
    if leng>=max_len:
        return np.asarray(x[:max_len]) # we trim the contact map 2D matrix if it's dimension > max_len
    else:
       pad_len=max_len-leng
       return np.pad(x, [(0, pad_len),(0,0)], mode='constant', constant_values=0)

def truncate_seq(seq,max_len): #  padding
    return seq[:max_len]
    

def truncate_coordinates(x,max_len): 
     return np.asarray(x[:max_len])

def calc_residue_dist(residue_one, residue_two) :
    """Returns the C-alpha distance between two residues"""
    diff_vector  = residue_one["CA"].coord - residue_two["CA"].coord
    return np.sqrt(np.sum(diff_vector * diff_vector))

def calc_dist_matrix(chain_one, chain_two) :
    """Returns a matrix of C-alpha distances between two chains"""
    answer = np.zeros((len(chain_one), len(chain_two)), float)
    for row, residue_one in enumerate(chain_one) :
        for col, residue_two in enumerate(chain_two) :
            answer[row, col] = calc_residue_dist(residue_one, residue_two)
    return answer

def truncate_concatmap(matrix,max_len): 
    
    return matrix[:max_len,:max_len] # we trim the contact map 2D matrix if it's dimension > max_len

def extract_coordinates(chain) :
    """Returns a list of C-alpha coordinates for a chain"""
    pos=[]
    for row,residue in enumerate(chain):
        pos.append(residue["CA"].coord)
    
    return pos

def pad_concatmap(matrix,max_len,pad_value=255): 
    leng = len(matrix)
    if leng>=max_len:
        return matrix[:max_len,:max_len] # we trim the contact map 2D matrix if it's dimension > max_len
    else:
       pad_len=max_len-leng
       return np.pad(matrix, [(0, pad_len), (0, pad_len)], mode='constant', constant_values=pad_value)


#input = open("/home/wangdu/CATH/seq_subfamily/selected_seq_subfamily_12_basedon_S40pdb.fa")
#input = open("/data/duolin/CATH/seq_subfamily/Nonredundant_subfamily_basedon_S40pdb.fa")
input = open("/data/duolin/CATH/seq_subfamily/Rep_subfamily_basedon_S40pdb.fa") #1553
from collections import OrderedDict
seqlens=[]
pdblist=OrderedDict({})
for line in input:
    #if line[1:].split("|")[0].startswith("3"): #remove 3
    #     line=next(input)
    #     continue
    
    pdbid=line.split("|")[3].split("/")[0].strip().strip()
    pdblist[pdbid]=line[1:].split("|")[0].strip()
    line=next(input)
    seqlens.append(len(line))
    if len(line)==22:
       print(pdbid) #3r4yA01


parser = argparse.ArgumentParser(description='Pretreat')
parser.add_argument('-data', metavar='DIR', default='./pdbsamples3000/folder-4/', help='path to dataset')
parser.add_argument('-max_len', default=512, type=int, help='max sequence len to consider.')
parser.add_argument('-outputfile', metavar='DIR', default='', help='outputfile')

args = parser.parse_args()
args.data = "/data/duolin/CATH/CATH_4_3_0_non-redundant/dompdb/"
#args.outputfile="/home/wangdu/CATH/CATH_4_3_0_non-redundant_contactmap_pad255.npz" args.max_len=500
#args.outputfile="/home/wangdu/CATH/CATH_4_3_0_non-redundant_contactmap_nopad.npz" #for this args.max_len=500
#args.outputfile="/data/duolin/CATH/CATH_4_3_0_non-redundant_more_contactmap_nopad.npz"
args.outputfile="/data/duolin/CATH/CATH_4_3_0_non-rep_superfamily_contactmap_nopad.npz"


file_list = glob.glob(args.data + "*.pdb")
data_path = []
#for path in os.listdir(args.data):
for path in pdblist:
    if path in pdblist:
        if os.path.isfile(os.path.join(args.data, path)):
           data_path.append(os.path.join(args.data, path))

print(data_path) #139 6198

"""for test chainids
chainids={}
for i in range(len(data_path)):
    parser = PDBParser()# This is from Bio.PDB
    structure = parser.get_structure('protein', data_path[i])
    model = structure[0]
    chains = model.get_chains()
    index=0
    for chain in chains:
        index+=1
        print(chain.id)
        chainids[chain.id]=1
    
    if index>1:
        break
"""

import warnings
from Bio import BiopythonWarning
from Bio.PDB.PDBExceptions import PDBConstructionWarning

warnings.simplefilter('ignore', BiopythonWarning)
warnings.simplefilter('ignore', PDBConstructionWarning)

#pad_contact=np.zeros((len(data_path),args.max_len,args.max_len),dtype=float)
pad_contact=[]
idlist=[]
seqlen=[]
error=0
sucnum=0 #136
for i in range(len(data_path)):
    parser = PDBParser()# This is from Bio.PDB
    structure = parser.get_structure('protein', data_path[i])
    model = structure[0]
    chainkey = model.get_chains()
    sequence=[]
    chains = model.get_chains()
    #matrix=[]
    for chain in chains:
        chainid=chain.id
    
    print(chainid)
    chain = model[chainid]
    try:
        matrix=calc_dist_matrix(chain,chain)
        reallen = len(matrix)
        reallen = min(reallen,args.max_len)
        #matrix=np.asarray(matrix)
        #pad_contact.append(pad_concatmap(matrix,args.max_len))
        pad_contact.append(matrix[:args.max_len,:args.max_len])
        sucnum+=1
        idlist.append(pdblist[data_path[i].split("/")[-1]])
        seqlen.append(reallen)
    except:
        continue
    


#pad_contact=np.asarray(pad_contact)
pad_contact = np.array(pad_contact, dtype=object)
np.savez(args.outputfile,idlist=idlist,contactmap=pad_contact,seqlen=seqlen)



