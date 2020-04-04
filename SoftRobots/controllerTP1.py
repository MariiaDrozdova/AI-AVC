#!/usr/bin/env python
# -*- coding: utf-8 -*-
import Sofa
from math import *
import numpy as np


class controller(Sofa.PythonScriptController):

    def initGraph(self, node):
        self.rootNode = node
        self.MechaObject = node.getObject('mObject')
        self.nbNodes= len(self.MechaObject.position)
        self.mesh = node.getObject('Mesh')
        self.k = 0.000000000000000000000000000000001
        self.m = 1000.1
        self.nodes = self.MechaObject.position
        self.edges = self.mesh.edges
         
        
        self.init_lengths = self.computeLength(self.nodes)
        self.gravity = np.array(self.rootNode.gravity).squeeze() * self.m
        
    def computeSpringForce(self):
        res = dict()
        for node in range(len(self.nodes)):
            res[node] = []
        
        for i, edge in enumerate(self.edges):
            node_0 = np.array(self.nodes[edge[0]])
            node_1 = np.array(self.nodes[edge[1]])
            
            cur_length = np.linalg.norm(node_1 - node_0)
            cur_direction = (node_1 - node_0) / cur_length

            force = np.abs(cur_length - self.init_lengths[i])  * self.k * cur_direction
            
            res[edge[0]].append(force) 
            res[edge[1]].append(-force) 
        return res
    
       
    def computeLength(self, nodes):
        lengths = []
        for edge in self.edges:
            node = edge[0]
            next_node = edge[1]
            
            length = np.linalg.norm(np.array(nodes[node]) - np.array(nodes[next_node]))
            lengths.append(length)
        return np.array(lengths)
     
    def onBeginAnimationStep(self,deltaTime):
        vecs = self.MechaObject.velocity
        spring_forces = self.computeSpringForce()
        for i in range(1, len(self.nodes)):
            if i != 0:
                total_force = -self.m * self.gravity 
                
                for force in spring_forces[i]:
                    total_force += np.array(force)
                acc = total_force / self.m
                self.nodes[i] += vecs[i] * deltaTime
                vecs[i] = np.array(vecs[i])
                vecs[i] += acc * deltaTime
                
                
                self.nodes[i] = self.nodes[i].tolist()
                vecs[i] = vecs[i].tolist()

        self.MechaObject.position = self.nodes
        self.MechaObject.velocity = vecs

        
        
        
        
