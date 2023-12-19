#! /usr/bin/env python
import subprocess
cmd = 'which gfn2nmr'
model_param = subprocess.getoutput(cmd)[:-7] + 'best_params'

import sys
import numpy as np
import math
import torch
from torch.nn import Linear
import os
from torch_geometric.nn import NNConv, BatchNorm
import torch.nn as nn
import linecache
import pandas as pd
import time
from xyz2mol import read_xyz_file, xyz2mol

class Arguments(): #The class used to store parameters
    charge = 0
    crest_arg = ' -gfnff '
    xyz = 'default.xyz'
    exp_data = ' '
    conf_search = True
    energy_cutoff = 3 #kcal/mol
    pre_opt = False
    device = torch.device('cpu')
    continue_mode = True
    draw2d = False


class Coord() : #the class used to describ atoms in input xyz files
    atom = ''
    x = 0.0
    y = 0.0
    z = 0.0
    def __init__(self, atom, x, y, z) :
        self.atom = atom
        self.x = x
        self.y = y
        self.z = z
    def ret_content(self):
        string = self.atom + ' ' + str('%.6f'%self.x) + ' ' + str('%.6f'%self.y) + ' ' + str('%.6f'%self.z)
        return string

class Structure(): #the class to describ structure
    coords = []
    energy = 0.0
    def __init__(self,coords,energy):
        self.coords = coords
        self.energy = energy
    def __lt__(self,other):
        return self.energy < other.energy

class str_feat(): #describ the graph feature of structure
    node_feats = []
    edge_index = []
    edge_attr = []
    y = []
    train_mask = []
    def __init__(self,node_feats,edge_index,edge_attr):
        self.node_feats = node_feats
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        

def read_xyz_ensemble(filename,energy_cut): #read an ensemble of xyz coordinates and return a list of structures
    structures = []
    energy = 0
    i = 0
    with open(filename,'r') as ensemble:
        for line_number, key_word in enumerate(ensemble.readlines()):
            if line_number == i:
                atom_num = int(key_word.strip())
                coords = []
            if line_number == 1 and energy_cut > 0:
                lowest_energy = float(key_word.strip())
            if line_number == i + 1 and energy_cut > 0:
                energy = float(key_word.strip())
                if energy - lowest_energy > energy_cut / 627.5:
                    break
            if line_number < i + atom_num + 1 and line_number > i + 1:
                coord = Coord(key_word.split()[0],float(key_word.split()[1]),float(key_word.split()[2]),float(key_word.split()[3]))
                coords.append(coord)
            if line_number == i + atom_num +1:
                coord = Coord(key_word.split()[0],float(key_word.split()[1]),float(key_word.split()[2]),float(key_word.split()[3]))
                coords.append(coord)
                structure = Structure(coords,energy)
                structures.append(structure)
                if energy_cut == 0:
                    break
                i = atom_num + i + 2
    
    return structures

def coord_to_xyzfile(coords,file_operator,title):#write xyz file according to coord list
    atom_num = len(coords)
    print(atom_num, file=file_operator)
    print(title, file=file_operator)
    for i in range(atom_num):
        print(coords[i].ret_content(), file=file_operator)

def read_node_feats(filename): # read node feat calculated by xtb
    with open(filename,'r') as f:
        for line_number, key_word in enumerate(f.readlines()):
            if "covCN" in key_word:
                start_line = line_number + 2
            if "Mol. C6AA /au·bohr" in key_word:
                end_line = line_number
        node_feats = []
        
        for i in range(start_line,end_line):
            feat = []
            line = linecache.getline(filename, i).split()
            
            feat.append(float(line[1]))
            for j in range(3,5):
                feat.append(float(line[j]))
            
            feat.append(feat[2]**2)
            feat.append(feat[2]*feat[0])
            node_feats.append(feat)
            linecache.clearcache()
    
    return node_feats

def distance_index(coord1,coord2): #return the distance index between two coordinates
    distance = (coord1.x-coord2.x)**2 + (coord1.y-coord2.y)**2 + (coord1.z-coord2.z)**2
    if distance == 0:
        distance_index = 4
    else:
        distance_index = 1 / distance
    return distance_index

def calc_mod(vector): #calculate the mod of a vector
    mod = 0
    for i in range(len(vector)):
        mod = vector[i]**2 + mod
    mod = mod ** 0.5
    return mod

def calc_angle(coord1,coord2,coord3): #calculate the angle between two vectors
    vector1 = [(coord1.x -coord2.x),(coord1.y -coord2.y),(coord1.z -coord2.z)]
    vector2 = [(coord3.x -coord2.x),(coord3.y -coord2.y),(coord3.z -coord2.z)]
    mod1 = calc_mod(vector1)
    mod2 = calc_mod(vector2) 
    cos_angle = (vector1[0]*vector2[0] + vector1[1]*vector2[1] + vector1[2]*vector2[2])/(mod1*mod2)
    if cos_angle > 1:
        cos_angle = 1
    if cos_angle < -1:
        cos_angle = -1
    return cos_angle

def calc_dihedral(coord1,coord2,coord3,coord4): #calculate the dihedral between three vectors
    #these two ifs are used to exclude that two vectors are on the same line
    #which will cause errors in the following calculation
    if (-0.9999> calc_angle(coord1,coord2,coord3)> -1.0001) or (1.0001> calc_angle(coord1,coord2,coord3)> 0.9999):
        return 0
    if (-0.9999> calc_angle(coord2,coord3,coord4)> -1.0001) or (1.0001> calc_angle(coord2,coord3,coord4)> 0.9999):
        return 0
    vector1 = np.array([(coord1.x -coord2.x),(coord1.y -coord2.y),(coord1.z -coord2.z)],dtype='float32')
    vector2 = np.array([(coord3.x -coord2.x),(coord3.y -coord2.y),(coord3.z -coord2.z)],dtype = 'float32')
    vector3 = np.array([(coord3.x -coord4.x),(coord3.y -coord4.y),(coord3.z -coord4.z)],dtype = 'float32')
    norm_v1 = np.cross(vector1,vector2) 
    norm_v2 = np.cross(vector2,vector3)
    mod1 = calc_mod(norm_v1)
    mod2 = calc_mod(norm_v2)
    cos_angle = (norm_v1[0]*norm_v2[0] + norm_v1[1]*norm_v2[1] + norm_v1[2]*norm_v2[2])/(mod1*mod2)
    
    if cos_angle > 1: #These two ifs to avoid the value more than 1 because of errors in float number calculation
        cos_angle = 1
    if cos_angle < -1:
        cos_angle = -1
        
    radian = math.acos(cos_angle) #obtain the radian value of angle
    
    sign2 = 1
    mod3 = calc_mod(norm_v1)
    mod4 = calc_mod(vector3) 
    cos_angle2 = (norm_v1[0]*vector3[0] + norm_v1[1]*vector3[1] + norm_v1[2]*vector3[2])/(mod3*mod4)#计算第四个原子和前三个原子的法向量夹角
    if cos_angle2 < 0: #determine the direction (or sign) of dihedral
        sign2 = -1
    return radian*sign2

def read_edge_feats(structure):#construct the edge_index and edge_features from the 3D structure

    edge1 = [] #the list to record edge index
    edge2 = []
    bond_order = [] 

    with open("wbo",'r') as wbo: #read Wiberg bond order from the wbo file
        for line_number, key_word in enumerate(wbo.readlines()):
            line = key_word.split()
            if float(line[2]) < 0.52:
                continue
            edge1.append(int(line[0])-1) #the atoms connect with wbo>0.52 were assigned with an edge
            edge2.append(int(line[1])-1)
            edge1.append(int(line[1])-1)
            edge2.append(int(line[0])-1)
            bond_order.append(float(line[2]))
            bond_order.append(float(line[2]))

    edge_attr = []
    for i in range(len(edge1)):
        attr = []
        attr.append(bond_order[i])
        distance = distance_index(structure.coords[edge1[i]],structure.coords[edge2[i]])
        attr.append(distance)
        attr = attr + [0.0,0.0] # add other positions to the edge_attr of bond edges
        edge_attr.append(attr)
    
    two_bonds_con_list = [] # search for the atoms separated by two bonds
    for i in range(len(edge1)):
        if edge_attr[i][0] < 0.52:
            continue
        two_bonds_con = [edge1[i],edge2[i]]
        for j in range(len(edge1)):
            if edge2[j] != edge1[i] and edge1[j] == edge2[i]:
                if edge_attr[j][0] < 0.52:
                    continue
                two_bonds_con.append(edge2[j])
                two_bonds_con_list.append(two_bonds_con)
                two_bonds_con = two_bonds_con[:-1]
    
    
    three_bonds_con_list = [] # search for the atoms separated by three bonds
    for i in range(len(two_bonds_con_list)):
        three_bonds_con = two_bonds_con_list[i]
        for j in range(len(edge1)):
            if edge2[j] != two_bonds_con_list[i][0] and edge2[j] != two_bonds_con_list[i][1] and edge1[j] == two_bonds_con_list[i][2]:
                if edge_attr[j][0] < 0.52:
                    continue
                three_bonds_con.append(edge2[j])
                three_bonds_con_list.append(three_bonds_con)
                three_bonds_con = three_bonds_con[:-1]
    
    
    angle_edge1 = [] # list to store the bond angle edges
    angle_edge2 = []
    angle_edge_attr = []
    for i in range(len(two_bonds_con_list)):
        angle_edge1.append(two_bonds_con_list[i][0])
        angle_edge2.append(two_bonds_con_list[i][2])
        angle_edge_attr.append([0.0, 
                              0.0,
                              math.acos(calc_angle(structure.coords[two_bonds_con_list[i][0]],structure.coords[two_bonds_con_list[i][1]],structure.coords[two_bonds_con_list[i][2]]))+1,
                              0.0])
    
    dihedral_edge1 = [] # list to store the dihedral angle edges
    dihedral_edge2 = []
    dihedral_edge_attr = []
    for i in range(len(three_bonds_con_list)):
        dihedral_edge1.append(three_bonds_con_list[i][0])
        dihedral_edge2.append(three_bonds_con_list[i][3])
        
        dihedral_edge_attr.append([0.0, 
                              0.0,
                              0.0,
                              calc_dihedral(structure.coords[three_bonds_con_list[i][0]],structure.coords[three_bonds_con_list[i][1]],structure.coords[three_bonds_con_list[i][2]],structure.coords[three_bonds_con_list[i][3]]),
                              ])
    
    edge_index1 = edge1 + angle_edge1 + dihedral_edge1 # sum all the edges
    edge_index2 = edge2 + angle_edge2 + dihedral_edge2
    edge_attr = edge_attr + angle_edge_attr + dihedral_edge_attr 
    
    edge_index = [edge_index1,edge_index2] # generate the edge_index

    return edge_index,edge_attr

def sum_feat_data(str_feats_list): #The function to sum feat data of multi conformers
    sum_node = np.zeros([len(str_feats_list[0].node_feats),len(str_feats_list[0].node_feats[0])])
    sum_edge_attr = [[0.0,0.0,0.0,0.0] for i in range(len(str_feats_list[0].edge_attr))]
    sum_edge_attr = np.array(sum_edge_attr)
    sum_dihedral_orient = [[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0] for i in range(len(str_feats_list[0].edge_attr))]
    # create lists to store the summarized features,
    for i in range(len(str_feats_list)):
        sum_node = np.array(str_feats_list[i].node_feats) + sum_node
        sum_edge_attr = np.array(str_feats_list[i].edge_attr) + sum_edge_attr

        #The following codes is to count the dihedral interval distribution
        for j in range(len(str_feats_list[0].edge_attr)):
            #sum_dihedral_orient has 12 positions, the first 6 positions used to store dihedral value, 
            #the last 6 positions used to count the number of times of distribution in the corresponding interval
            if str_feats_list[i].edge_attr[j][0] == 0 and str_feats_list[i].edge_attr[j][1] == 0 and str_feats_list[i].edge_attr[j][2] == 0:   
                if str_feats_list[i].edge_attr[j][3] >= -0.523599 and str_feats_list[i].edge_attr[j][3] <= 0.523599: 
                    sum_dihedral_orient[j][0] = sum_dihedral_orient[j][0] + str_feats_list[i].edge_attr[j][3]+0.523599
                    sum_dihedral_orient[j][6] += 1
                    
                if str_feats_list[i].edge_attr[j][3] > 0.523599 and str_feats_list[i].edge_attr[j][3] <= 1.570796:
                    sum_dihedral_orient[j][1] = sum_dihedral_orient[j][1] + str_feats_list[i].edge_attr[j][3]-0.523599
                    sum_dihedral_orient[j][7] += 1
                    
                if str_feats_list[i].edge_attr[j][3] > 1.570796 and str_feats_list[i].edge_attr[j][3] <= 2.617994:
                    sum_dihedral_orient[j][2] = sum_dihedral_orient[j][2] + str_feats_list[i].edge_attr[j][3]-1.570796
                    sum_dihedral_orient[j][8] += 1

                if str_feats_list[i].edge_attr[j][3] > 2.617994 and str_feats_list[i].edge_attr[j][3] <= 3.1416:
                    sum_dihedral_orient[j][3] = sum_dihedral_orient[j][3] + str_feats_list[i].edge_attr[j][3] - 2.617994
                    sum_dihedral_orient[j][9] += 1

                if str_feats_list[i].edge_attr[j][3] < -2.617994 and str_feats_list[i].edge_attr[j][3] >= -3.1416:
                    sum_dihedral_orient[j][3] = sum_dihedral_orient[j][3] + str_feats_list[i].edge_attr[j][3] + 3.141592689793 + 0.523599
                    sum_dihedral_orient[j][9] += 1

                if str_feats_list[i].edge_attr[j][3] >= -2.617994 and str_feats_list[i].edge_attr[j][3] <= -1.570796:
                    sum_dihedral_orient[j][4] = sum_dihedral_orient[j][4] + str_feats_list[i].edge_attr[j][3] + 2.617994
                    sum_dihedral_orient[j][10] += 1

                if str_feats_list[i].edge_attr[j][3] > -1.570796 and str_feats_list[i].edge_attr[j][3] < -0.523599:
                    sum_dihedral_orient[j][5] = sum_dihedral_orient[j][5] + str_feats_list[i].edge_attr[j][3] + 1.570796
                    sum_dihedral_orient[j][11] += 1

    average_node = sum_node / len(str_feats_list) #average node features
    average_edge_attr = sum_edge_attr /len(str_feats_list) #average edge features other than dihedral features

    for i in range(len(sum_dihedral_orient)): #average the dihedral values in the first 6 positions
        #the average values are added by 1 to distinguish the zero values of edges other than dihedral edges
        if sum_dihedral_orient[i][6] > 0:
            sum_dihedral_orient[i][0] = sum_dihedral_orient[i][0]/sum_dihedral_orient[i][6] + 1
        if sum_dihedral_orient[i][7] > 0:            
            sum_dihedral_orient[i][1] = sum_dihedral_orient[i][1]/sum_dihedral_orient[i][7] + 1
        if sum_dihedral_orient[i][8] > 0:
            sum_dihedral_orient[i][2] = sum_dihedral_orient[i][2]/sum_dihedral_orient[i][8] + 1
        if sum_dihedral_orient[i][9] > 0:
            sum_dihedral_orient[i][3] = sum_dihedral_orient[i][3]/sum_dihedral_orient[i][9] + 1
        if sum_dihedral_orient[i][10] > 0:
            sum_dihedral_orient[i][4] = sum_dihedral_orient[i][4]/sum_dihedral_orient[i][10] + 1
        if sum_dihedral_orient[i][11] > 0:
            sum_dihedral_orient[i][5] = sum_dihedral_orient[i][5]/sum_dihedral_orient[i][11] + 1


    average_node = average_node.tolist()
    average_edge_attr = average_edge_attr.tolist() 

    for i in range(len(sum_dihedral_orient)): # prepare all the features and edge_index to input to the model
        average_edge_attr[i] = average_edge_attr[i][:-1] + sum_dihedral_orient[i][:-6]
        average_edge_attr[i].append(average_edge_attr[i][1]*average_edge_attr[i][0])
        average_edge_attr[i].append(average_edge_attr[i][0]*average_edge_attr[i][0])


    averaged = str_feat(average_node,str_feats_list[0].edge_index,average_edge_attr)
                
    return averaged

def read_feats(filename,structure): #function to read feats from structure class
    node_feats = read_node_feats(filename) #node features are read from output files of xtb
    edge_index, edge_attr = read_edge_feats(structure)# edge features are read from 3D coordinates
    feat = str_feat(node_feats,edge_index,edge_attr)
    return feat

def read_exp_data(filename,atom_number): #read experimental data from a txt file
    label = []
    y = [0.0 for i in range(atom_number)]
    train_mask = [False for i in range(atom_number)]
    try:
        with open(filename,'r') as f:
            for line_num, key_word in enumerate(f.readlines()):
                line = key_word.split()
                if len(line) == 0:
                    break
                label.append(line)
    except (FileNotFoundError):
        print(f"\033[1;33mExptl data {filename} not found.\033[0m")
        return y,train_mask,1
    
    for i in range(len(label)):
        try:
            y[int(label[i][0])-1] = float(label[i][1])
        except (ValueError) :
            print("\033[1;33mWarning! Problem in reading exptl data file. It will be ignored.\033[0m")
            return y,train_mask,1

        train_mask[int(label[i][0])-1] = True

    return y,train_mask,0

def auto_mask(coords): #if the experimental data are not provided, generate mask to cover atoms other than carbon
    atom_number = len(coords)
    y = [' ' for i in range(atom_number)]
    train_mask = [False for i in range(atom_number)]
    for i in range(atom_number):
        if coords[i].atom == 'C' :
            train_mask[i] = True
    
    return y,train_mask

class nn_edge(torch.nn.Module): #the edge nn to extract edge features
    def __init__(self,input_n,output_n):
        super().__init__()
        self.nn1 = Linear(input_n,output_n,bias=True)
        self.norm1 = BatchNorm(output_n)
        self.nn2 = Linear(output_n,output_n,bias=True)
        self.norm2 = BatchNorm(output_n)
        self.nn3 = Linear(output_n,output_n,bias=True)
        self.norm3 = BatchNorm(output_n)
                        
        self.conv1 = nn.Conv2d(in_channels=2,out_channels=1,kernel_size=1,bias=False)
        self.conv2 = nn.Conv2d(in_channels=3,out_channels=1,kernel_size=1,bias=False)
        
        self.attention1 = edge_Attention(input_n,output_n,3)
        
    def forward(self,feat):
        x = self.nn1(feat)
        x = self.norm1(x)
        x = nn.ReLU(inplace=True)(x)
        x1 = x
        
        x = self.nn2(x)
        x = self.norm2(x)
        x = nn.ReLU(inplace=True)(x)
        x2 = x
        x = torch.stack((x1,x),0)
        x= self.conv1(x)[0]
        
        x = self.nn3(x)
        x = self.norm3(x)
        x = nn.ReLU(inplace=True)(x)
        x = torch.stack((x1,x2,x),0)
        x= self.conv2(x)[0]
        
        x = self.attention1(x,feat)
                       
        return x 
    
class Dens_Block5(torch.nn.Module): #Residual block 
    def __init__(self,input_n,edge_dim,edge_nn_dim):
        super().__init__()
        self.norm1 = BatchNorm(input_n)
        self.norm2 = BatchNorm(input_n)
        self.norm3 = BatchNorm(input_n)

        self.edge_nn1 = nn.Sequential(BatchNorm(edge_dim),nn_edge(edge_dim,edge_nn_dim),Linear(edge_nn_dim,input_n*input_n),
                                      BatchNorm(input_n*input_n))
        self.edge_nn2 = nn.Sequential(BatchNorm(edge_dim),nn_edge(edge_dim,edge_nn_dim),Linear(edge_nn_dim,input_n*input_n),
                                      BatchNorm(input_n*input_n))
        self.edge_nn3 = nn.Sequential(BatchNorm(edge_dim),nn_edge(edge_dim,edge_nn_dim),Linear(edge_nn_dim,input_n*input_n),
                                      BatchNorm(input_n*input_n))

        self.conv1 = NNConv(input_n,input_n,self.edge_nn1)
        self.conv2 = NNConv(input_n,input_n,self.edge_nn2)
        self.conv3 = NNConv(input_n,input_n,self.edge_nn3)

        
    def forward(self, x, edge_index,edge_attr):
        
        x = self.conv1(x,edge_index,edge_attr)
        x = self.norm1(x)
        x = nn.Hardtanh(min_val=-1,max_val=10,inplace=True)(x)
        x1 = x
        
        x = self.conv2(x,edge_index,edge_attr)
        x = self.norm2(x)
        x = nn.Hardtanh(min_val=-1,max_val=10,inplace=True)(x)
        
        x = self.conv3(x,edge_index,edge_attr)
        x = self.norm3(x)
        x = nn.Hardtanh(min_val=-1,max_val=10,inplace=True)(x)
        
        return x + x1   
    
class Cross_Attention(torch.nn.Module): #attention block to summarize the node features output from each stacked residual blocks
    def __init__(self,in_n,out_n,ratio):
        super().__init__()
        self.layer1 = Linear(in_n,out_n//ratio)
        self.bn = BatchNorm(out_n//ratio)
        self.layer2 = Linear(out_n//ratio,out_n,bias=False)
    def forward(self,x,feat):
        feat = self.layer1(feat)
        feat = self.bn(feat)
        feat = nn.ReLU(inplace=False)(feat)
        feat = self.layer2(feat)
        feat = nn.Hardtanh(min_val=0,max_val=0.9,inplace=False)(feat)
        
        return x * feat
    
class edge_Attention(torch.nn.Module): # attention block for nn_edge module
    def __init__(self,in_n,out_n,ratio):
        super().__init__()
        self.layer1 = Linear(in_n,out_n//ratio)
        self.bn = BatchNorm(out_n//ratio)
        self.layer2 = Linear(out_n//ratio,out_n,bias=False)
    def forward(self,x,feat):
        feat = self.layer1(feat)
        feat = self.bn(feat)
        feat = nn.ReLU(inplace=True)(feat)
        feat = self.layer2(feat)
        feat = nn.Hardsigmoid(inplace=True)(feat)
        
        return x * feat
        
class model(torch.nn.Module): # the main trunk of model
    def __init__(self):
        super().__init__()
        self.bn = BatchNorm(5)
        self.layer1 = Dens_Block5(5,11,18)
        self.layer2 = Dens_Block5(5,11,18)
        self.layer3 = Dens_Block5(5,11,18)
        
        self.layer4 = Dens_Block5(5,11,18)
        self.layer5 = Dens_Block5(5,11,18)
        self.layer6 = Dens_Block5(5,11,18)
        
        self.layer7 = Dens_Block5(5,11,18)
        self.layer8 = Dens_Block5(5,11,18)
        self.layer9 = Dens_Block5(5,11,18)
        
        self.layer10 = Dens_Block5(5,11,18)
        self.layer11 = Dens_Block5(5,11,18)
        self.layer12 = Dens_Block5(5,11,18)
       
        self.layer13 = Dens_Block5(5,11,18)
        self.layer14 = Dens_Block5(5,11,18)
        self.layer15 = Dens_Block5(5,11,18)   
        
        self.layer16 = Dens_Block5(5,11,18)
        self.layer17 = Dens_Block5(5,11,18)
        self.layer18 = Dens_Block5(5,11,18)
        
        self.attention1 = Cross_Attention(90,90,20)
        self.attention2 = Cross_Attention(90,90,20)
        
        self.layer19 = Linear(90,1)
        
    
    def forward(self, x, edge_index,edge_attr):
        x = self.bn(x)
              
        h1 = self.layer1(x,edge_index,edge_attr)
        h2 = h1
        h1 = self.layer2(h1,edge_index,edge_attr)
        x1 = h1
        h1 = self.layer3(h1,edge_index,edge_attr)
        y1 = h1
        h1 = h1 + h2 + x
        
        h1 = self.layer4(h1,edge_index,edge_attr)
        h3 = h1
        h1 = self.layer5(h1,edge_index,edge_attr)
        x2 = h1
        h1 = self.layer6(h1,edge_index,edge_attr)
        y2 = h1
        h1 = h1 + h2 + h3
        
        h1 = self.layer7(h1,edge_index,edge_attr)
        h4 = h1
        h1 = self.layer8(h1,edge_index,edge_attr)
        x3 = h1
        h1 = self.layer9(h1,edge_index,edge_attr)
        y3 = h1
        h1 = h1 + h3+ h4
        
        h1 = self.layer10(h1,edge_index,edge_attr)
        h5 = h1
        h1 = self.layer11(h1,edge_index,edge_attr)
        x4 = h1
        h1 = self.layer12(h1,edge_index,edge_attr)
        y4 = h1
        h1 = h1 + h4 + h5

        h1 = self.layer13(h1,edge_index,edge_attr)
        h6 = h1
        h1 = self.layer14(h1,edge_index,edge_attr)
        x5 = h1
        h1 = self.layer15(h1,edge_index,edge_attr)
        y5 = h1
        h1 = h1 + h5 + h6
        
        h1 = self.layer16(h1,edge_index,edge_attr)
        h7 = h1
        h1 = self.layer17(h1,edge_index,edge_attr)
        x6 = h1
        h1 = self.layer18(h1,edge_index,edge_attr)
        
                
        h1 = torch.cat((h1,h2,h3,h4,h5,h6,h7,
                       x1,x2,x3,x4,x5,x6,
                       y1,y2,y3,y4,y5),1)
        h1 = self.attention1(h1,h1) + self.attention2(h1,h1)
        
        h1 = self.layer19(h1)
        h1 = h1.squeeze(-1)
        return h1


def calculation(arg):
    try:
        with open(arg.xyz, 'r') as f: #read the number of atoms
            atomnumber = int(linecache.getline(arg.xyz, 1))
        
    except (FileNotFoundError):
        print(f"\033[1;31mError! no file named {arg.xyz} found.\033[0m")
        quit()


    current_path = os.getcwd()

    work_dir =  arg.xyz[:-4]



    if os.path.isdir(work_dir):
        if arg.continue_mode == False:
            command_line = 'rm -R ' + work_dir + '/*'
            os.system(command_line)  #clean work directory
    else:
        command_line = 'mkdir -p ' + work_dir  
        os.system(command_line)  #creat a work directory for all output files

    command_line = 'cp ' + arg.xyz + ' ' + work_dir + '/' + arg.xyz 
    os.system(command_line) #copy the input file into the work directory

    os.chdir(work_dir) #change current directory to work directory

    if arg.pre_opt :
        command_line = 'xtb ' + arg.xyz + ' --opt --gfn 1 --chrg ' + str(arg.charge) + ' > pre_opt.out'
        print("Pre-optimize the geometry with command ",command_line)
        os.system(command_line)
        command_line = 'cp xtbopt.xyz ' + arg.xyz
        os.system(command_line)

    feat_list = []

    time_cs = 0
    check_point1 = time.time()
    if arg.conf_search :
        
        if arg.continue_mode :
            if os.path.isfile('crest_ensemble.xyz'):
                print("\033[1;36mcrest_ensemble.xyz found! Please ensure it was optimized on gfn2-xtb level.\033[0m")
                structures = read_xyz_ensemble("crest_ensemble.xyz",arg.energy_cutoff) 
                check_point2 = time.time()
            elif os.path.isfile('crest_conformers.xyz'):
                print("\033[1;36mcrest_conformers.xyz found!\033[0m")
                command_line = 'crest -screen crest_conformers.xyz -gfn2 -niceprint -chrg ' + str(arg.charge)
                os.system(command_line) 
                structures = read_xyz_ensemble("crest_ensemble.xyz",arg.energy_cutoff)
                check_point2 = time.time()
            else:
                command_line = 'crest ' +  arg.xyz + arg.crest_arg + ' -chrg ' + str(arg.charge) + ' >' + arg.xyz[:-4] +'_crest.out'
                print("\033[1;36mSearching conformers by crest with arguments \033[0m", arg.crest_arg + ' -chrg ' + str(arg.charge))
                os.system(command_line) #call crest to perform conformational search

                print("\033[1;36mOptimize the conformers on gfn2 level to remove high energy conformers\033[0m")
                command_line = 'crest -screen crest_conformers.xyz -gfn2 -niceprint ' 
                os.system(command_line)

                structures = read_xyz_ensemble("crest_ensemble.xyz",arg.energy_cutoff)             
                check_point2 = time.time()
        else:
            command_line = 'crest ' +  arg.xyz + arg.crest_arg + ' -chrg ' + str(arg.charge) + ' >' + arg.xyz[:-4] +'_crest.out'
            print("\033[1;36mSearching conformers by crest with arguments \033[0m", arg.crest_arg + ' -chrg ' + str(arg.charge))
            os.system(command_line) #call crest to perform conformational search

            print("\033[1;36mOptimize the conformers on gfn2 level to remove high energy conformers\033[0m")
            command_line = 'crest -screen crest_conformers.xyz -gfn2 -niceprint ' 
            os.system(command_line)

            structures = read_xyz_ensemble("crest_ensemble.xyz",arg.energy_cutoff) 
            check_point2 = time.time()

        for i in range(len(structures)):
            print(f'Running {i+1}/{len(structures)}')
            with open(f"tmp{i}.xyz",'w') as inp:
                coord_to_xyzfile(structures[i].coords,inp,'  ')
            command = f'xtb tmp{i}.xyz --gfn 2  --opt --chrg {str(arg.charge)} > tmp{i}.out'
            if os.system(command) == 0:
 
                feat_list.append(read_feats(f'tmp{i}.out',structures[i]))
        

    else:
        check_point2 = time.time()
        command = 'xtb ' + arg.xyz + ' --gfn 2 --opt --chrg ' + str(arg.charge) + ' > tmp.out'
        if os.system(command) == 0 :
            structures = read_xyz_ensemble("xtbopt.xyz",0)
            feat_list.append(read_feats('tmp.out',structures[0]))
        
    averaged = sum_feat_data(feat_list) 

    os.chdir(current_path)
    if 'txt' in arg.exp_data :
        y,train_mask,errormark = read_exp_data(arg.exp_data,atomnumber)
        if errormark:
            y,train_mask = auto_mask(structures[0].coords)
    else :
        y,train_mask = auto_mask(structures[0].coords)

    averaged.y = y
    averaged.train_mask = train_mask

    m1 = model()

    m1.load_state_dict(torch.load(model_param,map_location=arg.device),strict=True)
    m1.eval()

    h = m1(
        torch.tensor(averaged.node_feats,dtype=torch.float32),
        torch.tensor(averaged.edge_index,dtype=torch.int64),
        torch.tensor(averaged.edge_attr,dtype=torch.float32)
        )
    pred = h[torch.tensor(averaged.train_mask,dtype=torch.bool)].tolist()

    atom_no = []
    exp = []
    for i in range(len(averaged.train_mask)):
        if averaged.train_mask[i] :
            atom_no.append(i+1)
            exp.append(averaged.y[i])

    check_point3 = time.time()

    print("\033[1;36mTotal number of calculated conformers: \033[0m", len(structures))
    print("\033[1;36mTime for dealing with conformers: \033[0m", format(check_point2-check_point1,"0.2f"),"s")
    print("\033[1;36mTime for 13C NMR calculation:  \033[0m", format(check_point3-check_point2,"0.2f"),"s")
    print("")
    print("\033[1;36mGenerating data files... \033[0m")
    outputfile = arg.xyz[:-4]+'.xlsx'

    df1 = pd.DataFrame({'Num':atom_no, 'exptl':exp,'pred':pred})
    df1.to_excel(outputfile, sheet_name='sheet1')

    d = ""
    if arg.draw2d :
        train_mask_noH = []
        for i in range(len(structures[0].coords)):
            if structures[0].coords[i].atom != 'H':
                train_mask_noH.append(train_mask[i])
        from rdkit.Chem import RemoveHs

        atoms, _, xyz_coordinates = read_xyz_file(arg.xyz)

        mol = xyz2mol(atoms, xyz_coordinates,charge=arg.charge)[0]
        mol = RemoveHs(mol)

        from rdkit.Chem.Draw import rdMolDraw2D
        opt =rdMolDraw2D.MolDrawOptions()

        j = 0
        for i in range(len(train_mask_noH)):
            if train_mask_noH[i]:
                opt.atomLabels[i] = str('%.2f'%pred[j])
                j = j+1

        opt.padding = 0.1
        opt.additionalAtomLabelPadding = 0.1
        opt.baseFontSize = 0.4
        size = atomnumber*30
        draw = rdMolDraw2D.MolDraw2DCairo(size,size)
        draw.SetDrawOptions(opt)
        rdMolDraw2D.PrepareAndDrawMolecule(draw,mol,kekulize = False)
        draw.FinishDrawing()
        draw.WriteDrawingText(arg.xyz[:-4]+'.png')
        d = "and "+ arg.xyz[:-4]+".png" 
    print(f"\033[1;36mNormal termination! Results saved as {outputfile} {d}\033[0m")
    return 0
