import os
import sys
import pickle
import json
from collections import Counter, defaultdict, deque, namedtuple
from itertools import combinations, chain, groupby, product
from seamr import sem
import numpy as np
import networkx as nx
import math

from datetime import datetime

from tqdm import tqdm

import seamr
from seamr.core import Event
from seamr.core cimport Event
from seamr.core import gen_active_sensors

from scipy import stats
from scipy.spatial import Delaunay

common_concepts = {}

from libc.math cimport sqrt as cSqrt
from libc.math cimport erf

from libc.math cimport tanh as cTanh

from libcpp.deque cimport deque as CPPDeque
cimport numpy as np

from scipy.misc import imresize, imsave

import time

cdef struct GDist:
    int row
    int col
    double cost

class Patch:
    def __init__(self, obj, ph, kind, objKey, kindKey, val=0.0):
        self.obj = obj
        self.phenomenon = ph
        self.kind = kind
        self.objKey = objKey
        self.kindKey = kindKey
        self.value = float(val)

    def __bool__(self):
        return self.value != 0

def findCut(cnts):
    s = sum(cnts.values())
    vals = sorted(cnts.items())
    
    cbelow = 0
    cabove = s

    maxEnt = 0
    bestCut = 0
    for i in range( len(vals)-1 ):
        c = vals[i][1]
        cbelow += c
        cabove -= c

        plow = cbelow / s
        phi = cabove / s

        iEnt = 0.0
        iEnt -= plow * math.log(plow)
        iEnt -= phi * math.log(phi)

        if iEnt > maxEnt:
            maxEnt = iEnt
            bestCut = i
    
    return (vals[bestCut][0] + vals[bestCut+1][0]) * 0.5

cdef double euclidean( int aX, int aY, int bX, int bY ):
    cdef double dx = aX - bX
    cdef double dy = aY - bY
    
    return cSqrt(dx ** 2 + dy ** 2)

cdef np.ndarray getDist(int gs, int srcY, int srcX, np.ndarray[np.float64_t, ndim=2] mask):
    '''
    Runs A* on the matrix to get the minimum distance between all points and a 
    specific source point.
    '''

    cdef np.ndarray[np.float64_t, ndim=2] dists = np.zeros_like(mask, dtype=np.float64)
    cdef np.ndarray probs
    cdef np.ndarray distMask

    cdef int inf = mask.size
    cdef CPPDeque[GDist] q = CPPDeque[GDist]()

    cdef np.ndarray[np.float64_t, ndim=1] xline
    cdef np.ndarray[np.float64_t, ndim=1] yline

    cdef GDist current

    cdef int width = mask.shape[1] 
    cdef int height = mask.shape[0]

    cdef int offY
    cdef int offX

    cdef int newX
    cdef int newY
    cdef int edgeX
    cdef int edgeY
    cdef int i

    cdef double eps = 1.0 / 10**6
    cdef double newCost
    cdef double maxDist = euclidean(0, 0, mask.shape[1], mask.shape[0])

    cdef list edges = []
    edges.extend([ (0, edgeX) for edgeX in range(mask.shape[1]) ])
    edges.extend([ (height, edgeX) for edgeX in range(mask.shape[1]) ])
    edges.extend([ (edgeY, 0) for edgeY in range(mask.shape[0]) ])
    edges.extend([ (edgeY, width) for edgeY in range(mask.shape[0]) ])

    dists.fill( maxDist )

    for (edgeY, edgeX) in edges:
        dEdge = euclidean(srcX, edgeX, srcY, edgeY)

        xline = np.linspace(srcX, edgeX, int(dEdge) * 10 )
        yline = np.linspace(srcY, edgeY, int(dEdge) * 10 )

        for i in range(xline.size):
            newX = int(xline[i])
            newY = int(yline[i])

            if 0 <= newX < width and 0 <= newY < height and mask[newY,newX] == 0.0:
                dists[newY, newX] = euclidean(srcX, srcY, newX, newY)
            else:
                break

    return dists / maxDist

def load_shapes(env):
    cose = sem.COSE()

    if not env.endswith('.json'):
        env += '.json'

    p = os.path.join(os.path.dirname(__file__), '../graphs/')
    with open(os.path.join(p, env)) as f:
        shapes = json.load(f)
        for n in shapes['nodes']:
            if n['sensor']:
                n['ref'] = cose.sensorType(n['sensor'])
        return shapes

def load_grid(env):
    '''
    Returns an array of positions dictionaries with keys:
        x
        y
        xmax
        ymax
        xn
        yn
        obs : 1 or 0
    '''

    if not env.endswith('.json'):
        env += '.json'

    p = os.path.join(os.path.dirname(__file__), '../grids/')
    with open(os.path.join(p, env)) as f:
        return json.load(f)

def get_shape_pos(env, frac = 2.0 / 3.0, norm=False, for_sensors=True):
    cose = sem.COSE()

    nodes = load_shapes(env)['nodes']
    
    if norm:
        xs = [ n['x'] for n in nodes ]
        ys = [ n['y'] for n in nodes ]

        xmin,xmax = min(xs), max(xs)
        ymin,ymax = min(ys), max(ys)

        for n in nodes:
            n['x'] = (n['x'] - xmin) / (xmax - xmin)
            n['y'] = (n['y'] - ymin) / (ymax - ymin)

    return [ n for n in nodes if cose.is_a(n['ref'], 'cose:Sensor') == for_sensors ]

def get_graph_dist(env, single_env):
    if not env.endswith('.json'):
        env += '.json'

    p = os.path.join(os.path.dirname(__file__), '../graphs/')
    g = nx.Graph()

    is_concept = {}
    s_kind = {}
    
    with open(os.path.join(p, env)) as f:
        gdef = json.load(f)
        
        for n in gdef['nodes']:
            is_concept[ n['id'] ] = n['sensor'] == ''
            s_kind[ n['id'] ] = n['ref']
            g.add_node(n['id'], attr_dict=n)

        g.add_edges_from(gdef['edges'])
    
    for u,v,d in g.edges(data=True):
        dx = g.node[u]['x'] - g.node[v]['x']
        dy = g.node[u]['y'] - g.node[v]['y']
        d['length'] = math.sqrt(dx **2 + dy ** 2)
    
    if single_env:
        concepts = get_env_concepts(env, False)
    else:
        concepts = get_concepts(False)

    concept_idx = dict([ (k,i) for i,k in enumerate(concepts) ])
    kernel = np.zeros((len(concepts), len(concepts)))
    s_dist = {}

    for u,v in combinations(g.nodes(), 2):

        try:
            uv_len = nx.shortest_path_length(g, source=u, target=v, weight='length')
        except nx.exception.NetworkXNoPath:
            uv_len = float('inf')
            
        if is_concept[u] and is_concept[v]:
            
            if s_kind[u] in concept_idx and s_kind[v] in concept_idx:
                ui = concept_idx[ s_kind[u] ]
                vi = concept_idx[ s_kind[v] ]
                kernel[ui,vi] = uv_len
                kernel[vi,ui] = uv_len
            
        elif is_concept[u] != is_concept[v]:
            
            s,c = (v,u) if is_concept[u] else (u,v)
            sen = '%s:%s' % (env.split('.')[0], g.node[s]['sensor'])
            con = g.node[c]['ref']
            
            if sen not in s_dist:
                s_dist[sen] = {}
                
            s_dist[ sen ][ con ] = min(uv_len, s_dist[sen].get(con, float('inf')))
    
    return (s_dist, concept_idx, kernel,)

def get_env_pos(env, for_sensors):
    cose = sem.COSE()

    if not env.endswith('.json'):
        env += '.json'

    p = os.path.join(os.path.dirname(__file__), '../graphs/')
    with open(os.path.join(p, env)) as f:
        return [ n for n in json.load(f)['nodes'] 
                        if for_sensors == cose.is_a(n['ref'], 'cose:Sensor') ]

def get_env_concepts(env, for_sensors):
    cose = sem.COSE()

    if not env.endswith('.json'):
        env += '.json'

    p = os.path.join(os.path.dirname(__file__), '../graphs/')
    with open(os.path.join(p, env)) as f:
        return sorted([ n['ref'] for n in json.load(f)['nodes'] 
                        if for_sensors == cose.is_a(n['ref'], 'cose:Sensor') ])

def get_concepts( for_sensors, frac = 2.0 / 3.0 ):
    global common_concepts

    if for_sensors not in common_concepts:
        cSet = set()
        p = os.path.join(os.path.dirname(__file__), '../graphs/')

        freq = Counter()
        n = 0
        for j in os.listdir(p):
            freq.update( get_env_concepts(j, for_sensors) )
            n += 1
        
        cut = frac * n
        for k,v in freq.items():
            if v >= cut:
                cSet.add(k)
        
        common_concepts[ for_sensors ] = sorted(cSet)
    
    return common_concepts[ for_sensors ]

#---------------------------------------------------------------------------

cdef class Pos:
    cdef public double x
    cdef public double y
    cdef public double ts
    cdef public double hr
    cdef public double secOfDay

    def __init__(self, double x, double y, double ts):
        self.x = x
        self.y = y
        self.ts = ts

        d = datetime.fromtimestamp(ts)

        self.hr = d.hour
        self.secOfDay = d.hour * 3600 + d.minute * 60 + d.second

class Box:
    def __init__(self, res, name, tp, space, left, top, right, bottom):
        self.reasoner = res
        self.name = name
        self.type = tp
        self.space = space
        self.left=left
        self.top=top
        self.right=right
        self.bottom = bottom
    
    @property
    def center(self):
        return ((self.left + self.right) * 0.5, (self.top + self.bottom) * 0.5,)
    
    def corners(self):
        return [ (self.left, self.top), (self.left, self.bottom),
                    (self.right, self.top), (self.right, self.bottom) ]

    def overlaps(self, other):
        return not any([
            self.top > other.bottom,
            self.left > other.right,
            self.bottom < other.top,
            self.right < other.left
        ])

    def dist(self, toOther):
        if self.overlaps( toOther ):
            return 0.0
        else:
            c = self.center
            pointDists = [ self.reasoner.getPathLength(c, corner,  adjust = False) for corner in toOther.corners() ]
            pointDists.append( self.reasoner.getPathLength(c, toOther.center, adjust = False) )
            pointLength = list(filter(lambda d: d != 0.0, pointDists))

            if len(pointLength) > 0:
                return 1.0 - min(pointLength)
            else:
                return float('inf')

class Reasoner:
    '''
    Use a bayesian approache to model probability of new event given
    different streams, highest stream wins, but at some cut-off or rule 
    a new stream is generated. Old streams die after some time.
    P(event | stream)
    '''
    def __init__(self, env, sensors, gridSize=10, minIntensity=0.05):

        cose = sem.COSE()
        
        # env could also be a dataset name
        self.env = env.split('.')[0]
        self.cose = cose

        self.minIntensity = minIntensity
        self.epsilon = 10.0 ** -5
        self.latest = 0.0

        self.gridSize = gridSize
        self.pathLengths = {}
        self.pos = {}
        self.shapes = load_shapes(self.env)

        self.pathLengths = {}
        self.pathSim = {}

        self.sensors = sensors
        self.sensorName = { s.id : s.name.replace(':','_') for s in sensors }

        assert gridSize > 0

        lookup = { s.name : s.id for s in sensors }

        self.stickySensors = {
            s.id for s in sensors
            if cose.is_a(s.kind, 'cose:DirectInteractionSensor')
        }
        
        self.sensorNodes = []
        self.objNodes = []
        self.objBoxes = []
        self.spaceNodes = {}
        self.nodeSpace = {}
        self.nodeType = {}

        g = nx.Graph()
        self.g = g

        ref = lambda u: g.node[u]['ref']

        grid = load_grid( self.env )

        ymax = grid[-1]['yn'] + 1
        xmax = grid[-1]['xn'] + 1
        for n in self.shapes['nodes']:
            if 'x' in n and 'y' in n:
                ymax = max(ymax, (n['y'] // gridSize) + 1 )
                xmax = max(xmax, (n['x'] // gridSize) + 1 )

        gridMatrix = np.zeros( ( int(ymax), int(xmax) ) )
        for obj in grid:
            gridMatrix[ obj['yn'], obj['xn'] ] = int(obj['obs'])

        self.gridMatrix = gridMatrix
        self.gridShape = gridMatrix.shape
        
        #-----------------------
        self.currentScores = np.zeros_like( gridMatrix )
        #-----------------------

        spaceNodes = defaultdict(set)
        boxIndex = {}

        typeAlias = smarterTypes( self.env )

        for node in self.shapes['nodes']:
            node['space'] = None
            if 'x' in node:
                self.g.add_node(node['id'], **node)
                self.pos[ node['id'] ] = [node['x'], node['y']]
                
                if node['sensor']:
                    if node['sensor'] in lookup:
                        self.sensorNodes.append( node )
                        self.pos[ lookup[node['sensor']] ] = [node['x'], node['y']]

                else:
                    l = int(node['left']//gridSize)
                    t = int(node['top']//gridSize)
                    r = int(node['right']//gridSize)
                    b = int(node['bottom']//gridSize)
                    
                    nodeType = self.typeAdjust( typeAlias.get(node['id'], node['ref']) )

                    node['space'] = cose.getObjSpace( nodeType )
                    spaceNodes[ node['space'] ].add( node['id'] )

                    self.objBoxes.append([ node['id'],
                                           self.typeAdjust(node['ref']),
                                           node['space'],
                                           l,
                                           t,
                                           max(l+1, r),
                                           max(t+1, b), ])
                    
                    boxIndex[ node['id'] ] = self.objBoxes[-1][3:]

                    self.objNodes.append( node )

        ox = [ n['x'] for n in self.objNodes ]
        oy = [ n['y'] for n in self.objNodes ]

        sx = [ n['x'] for n in self.sensorNodes ]
        sy = [ n['y'] for n in self.sensorNodes ]

        for u,v in self.shapes['edges']:
            if u in self.pos and v in self.pos:
                self.g.add_edge(u, v, length = self.dist(self.pos[u], self.pos[v]))

        for sp, sNodes in spaceNodes.items():
            spaceG = self.g.subgraph( sNodes )
            for i, areaG in enumerate(nx.connected_component_subgraphs(spaceG)):
                l = float('inf')
                r = -l
                t = float('inf')
                b = -t
                iSpaceID = 'space-%s-%s' % (sp, i)
                self.spaceNodes[iSpaceID] = set()

                for n in areaG.nodes():
                    nodeID = areaG.node[n]['id']
                    nl, nt, nr, nb = boxIndex[n]
                    l = min(l, nl)
                    r = max(r, nr)
                    t = min(t, nt)
                    b = max(b, nb)

                    self.nodeSpace[ nodeID ] = iSpaceID
                    self.spaceNodes[ iSpaceID ].add( nodeID )
                    self.nodeType[ iSpaceID ] = sp

                self.objBoxes.append([ iSpaceID, sp, sp, l, t, r, b ])

        rx = self.gridShape[1] - 1
        bx = self.gridShape[0] - 1

        self.objBoxes = [ (n, self.typeAdjust(c), sp,
                            int(max(0,l)),
                            int(max(0,t)),
                            int(min(r, rx)),
                            int(min(b, bx))) for (n,c,sp,l,t,r,b) in self.objBoxes ]

        # -------- Setup the position estimation types ------------
        self.typeEst = {}

        reservedObjs = set()
        availableObjs = []

        for (n,c,sp,l,t,r,b) in self.objBoxes:
            availableObjs.append( Box(self, n, c, sp, l, t, r, b) )

        reasoningModules = [ DoorReasoner,
                             PressureReasoner,
                             OpenObjectReasoner,
                             ProximityReasoner,
                             LightSwitchReasoner,
                             ContactReasoner,
                             WideMotionReasoner,
                             NarrowMotionReasoner,
                             TightMotionReasoner,
                             LightLevelReasoner,
                             TemperatureReasoner ]

        for Cls in reasoningModules:
            availableObjs = [ a for a in availableObjs if a.name not in reservedObjs ]
            self.typeEst[ Cls.typeOfSensor ] = Cls( sensors,
                                                    self,
                                                    reservedObjs,
                                                    availableObjs,
                                                    minIntensity = self.minIntensity )

        self.imageMask = self.setImageMask()

    def saveImages(self, toDir, scale=1):
        for sID, sName in self.sensorName.items():
            sImg = self.getImage([sID], scale=scale)
            imsave(os.path.join(toDir, '%s.png' % sName), sImg)

            for m in self.typeEst.values():
                if hasattr(m, 'mats') and sID in m.mats:
                    mask = m.mats[sID]
                    if scale > 1:
                        mask = imresize(mask, scale=scale)
                    
                    imsave(os.path.join(toDir, '%s_mask.png' % (sName,)), (mask * 255).astype(np.uint8) )

    @staticmethod
    def typeAdjust(t):
        '''
        This is used to manually adjust the level-of-detail used for some modeling
        '''
        if t in ['cose:SingleBed', 'cose:DoubleBed']:
            return 'cose:Bed'
        elif t in ['cose:Loveseat','cose:Sofa']:
            return 'cose:Sofa'
        elif t in ['cose:Shower','cose:Bathtub']:
            return 'cose:Shower'
        elif t in ['cose:Dresser','cose:Wardrobe','cose:Nightstand']:
            return 'cose:Dresser'
        elif t in ['cose:Freezer','cose:Refrigerator']:
            return 'cose:Refrigerator'
        elif t in ['cose:CookingRange','cose:Oven']:
            return 'cose:CookingRange'
        elif t in ['cose:ClothesWasher','cose:ClothesDryer']:
            return 'cose:LaundryAppliance'
        return t

    def getObjBoxes(self):
        return [ Box(self, *args) for args in self.objBoxes ]

    def getSensorBoxes(self):
        boxes = []
        
        rx = self.gridShape[1] - 1
        bx = self.gridShape[0] - 1

        clipX = lambda p: max(0, min(p // self.gridSize, rx))
        clipY = lambda p: max(0, min(p // self.gridSize, bx))

        for n in self.sensorNodes:
            env,name = n['sensor'].split(':')
            tp = self.cose.sensorType( name )
            boxes.append(
                Box( self, 
                    name,
                    tp,
                    None,
                    clipX(n['left']),
                    clipY(n['top']),
                    clipX(n['right']),
                    clipY(n['bottom']) ) 
            )

        return boxes

    def setImageMask(self, mag=255):
        z = np.ones( ( self.gridMatrix.shape[0], self.gridMatrix.shape[1], 3 ), dtype=np.uint8 )

        walls = (1 - self.gridMatrix).astype(np.uint8)
        z *= np.repeat(np.expand_dims(walls, -1), 3, axis=-1)
        
        mag = min(255, max(mag, 0))

        spaceColor = np.array([255,127,0])
        objColor = np.array([55,126,184])

        for _id, _ref, _sp, l,t,r,b in self.objBoxes:
            if _id.startswith('space'):
                clr = spaceColor
            else:
                clr = objColor
            z[ t:b, l, : ] = clr
            z[ t:b, r, : ] = clr
            z[ t, l:r, : ] = clr
            z[ b, l:r, : ] = clr

        return z

    def getImage(self, tSensors, scale=0, node=None):
        y,x = self.gridShape

        image = np.zeros( (y,x,3), dtype=np.uint8 )
        image.fill(255)

        if tSensors:
            pos = { T[0] : (T[3], T[4], T[5], T[6]) for T in self.objBoxes }

            for p in self.getScores(tSensors):
                l,t,r,b = pos[ p.obj ]
                scoreVal = 255 - np.uint8(p.value * 255.0)
                image[t:b, l:r, :] = scoreVal

        image = np.where(self.imageMask == 1, image, self.imageMask)

        if scale == 0:
            return image
        else:
            return imresize(image,size = float(scale))

    def sensorDist(self, a, b):
        return self.getPathLength( self.pos[a], self.pos[b] )

    def dist(self, a, b):
        return math.sqrt( (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 )

    def toCoord(self, x, y, adjust = True):
        scale = self.gridSize if adjust else 1

        x = min(self.gridMatrix.shape[1]-1, max(0, x//scale))
        y = min(self.gridMatrix.shape[0]-1, max(0, y//scale))
        
        return (int(x), int(y),)

    def getPathMat(self, key):
        if key not in self.pathLengths:
            self.pathLengths[key] = getDist(int(self.gridSize),
                                            key[1],
                                            key[0],
                                            self.gridMatrix)

        return self.pathLengths[key]
    
    def getPathSim(self, key, decay = None):
        if key not in self.pathSim:
            sim = np.power(1.0 - self.getPathMat(key), decay or 1.0)
            smax = sim.max()
            if smax > 0:
                sim /= smax
            self.pathSim[key] = sim
            
        return self.pathSim[key]

    def getPathLength(self, start, end, adjust = True):
        start = self.toCoord(*start, adjust = adjust)
        end = self.toCoord(*end, adjust = adjust)

        if start in self.pathLengths:
            refMat = self.pathLengths[start]
        elif end in self.pathLengths:
            refMat = self.pathLengths[end]
            end,start = start,end
        else:
            refMat = self.getPathMat(start)
            
        return refMat[end[1], end[0]]

    def getDefaultPatches(self):
        patches = []
        
        for M in self.typeEst.values():
            patches.extend( M.getDefaultPatches() )

        return patches

    def getScores(self, sensors, useKind=False):

        if useKind:

            bestScore = {}
            
            for M in self.typeEst.values():
                for p in M.getScores( sensors ):
                    
                    if p.kindKey not in bestScore:
                        bestScore[ p.kindKey ] = p
                    elif bestScore[ p.kindKey ].value < p.value:
                        bestScore[ p.kindKey ] = p

            return list( bestScore.values() )

        else:
            
            scores = []
            for M in self.typeEst.values():
                scores.extend( M.getScores( sensors ) )
            return scores

    def getKey(self, c, p):
        return ('%s-%s' % (Reasoner.typeAdjust(c), p)).replace('cose:','')

    def genPatches(self, stream, times, withSensors=False):

        patches = []
        
        for onSensors, t in gen_active_sensors(stream, times):
            patches = self.getScores(onSensors)
            
            if withSensors:
                yield patches, onSensors, t
            else:
                yield patches, t
            
    def genEvents(self, events, index, useObjKey = False):

        # ------- Setup keys --------------
        if useObjKey:
            key = lambda p: p.objKey
        else:
            key = lambda p: p.kindKey
        
        blank = 'FunctionalSpace-Temperature'
        index[blank] = len(index)
        blankID = index[blank]
        for p in sorted(self.getDefaultPatches(), key=key):
            pKey = key(p)
            if pKey not in index:
                index[pKey] = len(index)

        # ------- Now setup specialized types --------------
        isOn = set()

        temps = { s.id for s in self.sensors if self.cose.is_a(s.kind, 'cose:TemperatureSensor') }
        lights = { s.id for s in self.sensors if self.cose.is_a(s.kind, 'cose:LightLevelSensor') }

        lightCount = Counter()
        tempCount = Counter()

        sensorNames = {s.id : s.name for s in self.sensors}

        # ------- Now setup the cache | this keeps integer states vs patches --------------
        stateCache = {}

        def getScores( activeSet ):
            onHash = hash(frozenset(activeSet))
            if onHash not in stateCache:
                stateCache[onHash] = { index[key(p)] : p.value for p in self.getScores(activeSet) }
            return stateCache[onHash]

        for e in events():
            if e.sensorID in lights:
                lightCount[ int(e.state) ] += 1
            elif e.sensorID in temps:
                tempCount[ int(e.state) ] += 1

        if len(tempCount):
            tempCut = findCut(tempCount)
        else:
            tempCut = None

        if len(lightCount):
            lightCut = findCut(lightCount)
        else:
            lightCut = None

        prevState = { i : 0 for i in index.values() }

        for e in events():
            sID = e.sensorID

            if sID in temps:
                if e.state > tempCut:
                    isOn.add( sID )
                else:
                    isOn.discard( sID )

            elif sID in lights:
                if e.state > lightCut:
                    isOn.add( sID )
                else:
                    isOn.discard( sID )
                    
            else:
                if e.state == 1:
                    isOn.add( sID )
                else:
                    isOn.discard( sID )

            uVals = getScores(isOn)
            nYielded = 0
            for k,v in list(prevState.items()):
                uV = uVals.get(k, 0)
                if uV != v:
                    yield (e.srcLine, k, uV, e.timestamp)
                    nYielded += 1
                    prevState[k] = uV
            
            if nYielded == 0:
                yield (e.srcLine, blankID, 0, e.timestamp)

    def fit(self, events):
        ''' Fit type specific models '''
        pass
    
    def reset(self):
        for m in self.typeEst.values():
            m.reset()

# ----------------------------------------------------------------------

class TypedPositionModule:

    typeOfSensor = 'cose:Sensor'

    def __init__(self, sensors, parent, blockedObj, objBoxes, minIntensity = 0.001):
        cose = sem.COSE()
        self.cose = cose
        self.parent = parent
        self.minIntensity = minIntensity
        self.sensors = [ s for s in sensors if cose.is_a(s.kind, self.typeOfSensor) ]
        self.sensorIDs = { s.id for s in self.sensors }
        self.sensorName = { s.id : s.name for s in self.sensors }
        self.pos = { k : p for k,p in parent.pos.items() if k in self.sensorIDs }
        self.sensorShape = {}
        self.st = 0.0

    def getMats(self, list sensors):
        '''
        Return the probability of reaching the specified location
        at the specified time.

        For direct interaction sensors this is a max probability.
        '''
        return [ self.mats[sID] for sID in sensors if sID in self.mats ]

    def okEvent(self, e):
        return e.sensorID in self.sensorIDs

class RayTracedReasoner( TypedPositionModule ):
    
    phenomenon = 'cose:Movement'
    
    def __init__(self, sensors, parent, blockedObj, objBoxes, minIntensity=0.05):
        super().__init__(sensors, parent, blockedObj, objBoxes)
        self.mats = {}
        self.tooLow = 10.0 ** -2
        self.minIntensity = minIntensity

        self.boxes = list(filter(self.objIsOK, objBoxes))
        self._pKeys = { b.name : (parent.getKey(b.name, self.phenomenon),
                         parent.getKey(b.type, self.phenomenon)) for b in self.boxes }
        
        mats = {}
        for sID, (x,y) in self.pos.items():
            mats[ sID ] = self.parent.getPathSim( self.parent.toCoord(x,y), decay=self.distFactor )
        
        if len(mats):
            
            for sID, sMat in list(mats.items()):

                boxVec = []
                for b in self.boxes:
                    objKey, kindKey = self._pKeys[ b.name ]
                    sVal = sMat[b.top : b.bottom+1, b.left : b.right+1].mean()
                    boxVec.append( sVal )
                self.mats[sID] = np.array( boxVec )

    def getDefaultPatches(self):
        patches = []
        for b in self.boxes:
            objKey, kindKey = self._pKeys[ b.name ]
            patches.append( Patch(b.name, self.phenomenon, b.type, objKey, kindKey, 0.0) )
        return patches

    def objIsOK(self, obj):
        return True

    def getScores(self, sensors):
        scores = []
        sMats = [ self.mats[s] * w for s,w in sensors.items() if s in self.mats ]

        if len(sMats):
            m = np.stack(sMats).sum(axis=0)
            for sVal, b in zip(m, self.boxes):
                objKey, kindKey = self._pKeys[ b.name ]                
                if sVal > self.minIntensity:
                    scores.append( Patch(b.name, self.phenomenon, b.type, objKey, kindKey, sVal) )

        return scores

class WideMotionReasoner( RayTracedReasoner ):
    typeOfSensor = 'cose:WideAreaMotionDetector'
    distFactor = 5.0

class NarrowMotionReasoner( RayTracedReasoner ):
    typeOfSensor = 'cose:NarrowAreaMotionDetector'
    distFactor = 10.0

class TightMotionReasoner( RayTracedReasoner ):
    typeOfSensor = 'cose:TightAreaMotionDetector'
    distFactor = 15.0

# ----------------------------------------------------------------------

class DirectReasoner( TypedPositionModule ):

    blocking = True
    phenomenon = 'cose:Contact'
    typeOfSensor = 'cose:DirectInteractionSensor'


    def __init__(self, rawSensors, parent, blockedObj, objBoxes, minIntensity=0.05):
        super().__init__(rawSensors, parent, blockedObj, objBoxes, minIntensity=minIntensity)

        cose = sem.COSE()
        self.utilityMats = {}
        self.mats = {}
        self.patches = {}

        if len(self.sensors):
            boxIndex = { b.name : b for b in objBoxes }
            self.sensorAssoc = self.getSensorAssoc( self.sensors[0].name.split(':')[0] )

            nrows = self.parent.gridShape[0]
            ncols = self.parent.gridShape[1]
            gs = self.parent.gridSize

            for s in self.sensors:
                if s.name in self.sensorAssoc:
                    if getattr(self, 'blocking', False):
                        blockedObj.add( s.id )

                    sn = self.sensorAssoc[s.name]
                    
                    objKey = self.parent.getKey(sn['id'], self.phenomenon)
                    kindKey = self.parent.getKey(sn['ref'], self.phenomenon)

                    sPatch = (sn['id'], self.phenomenon, sn['ref'], objKey, kindKey,)
                    self.patches[ s.id ] = sPatch
    
    def _make_patch(self, s, w):
        sID, sPh, sRef, oKey, kKey = self.patches[s]
        return Patch(sID, sPh, sRef, oKey, kKey, w)

    def getDefaultPatches(self):
        return [ Patch(*T) for T in self.patches.values() ]

    def getScores(self, sensors):
        if len(self.patches):
            return [ self._make_patch(s,w) for s,w in sensors.items() if s in self.patches ]
        else:
            return []

    def reset(self):
        self.isOn = set()

    def objectIsOK(self, o):
        return True

    def getSensorAssoc(self, env):
        from seamr import sem
        
        def dist(a,b):
            return math.sqrt( (a['x'] - b['x'])**2 + (a['y'] - b['y'])**2 )

        areas = ['cose:KitchenObject',
                 'cose:BathroomObject',
                 'cose:BedroomObject',
                 'cose:OfficeObject']

        cose = sem.COSE()

        assoc = {}
        shapes = load_shapes( env )
        
        edges = set( map(tuple, shapes['edges']) )
        edges |= set([ (b,a) for a,b in edges ])

        objs = [ n for n in shapes['nodes'] if self.objectIsOK(n) and n['sensor'] == '' and 'x' in n and 'y' in n ]
        snode = [ n for n in shapes['nodes'] if n['sensor'] != '' and cose.is_a(n['ref'], self.typeOfSensor) ]

        for n in sorted(snode, key=lambda x: x['id']):

            ok_objs = sorted(filter(lambda o: (n['id'], o['id']) in edges, objs),
                                key=lambda o: dist(n,o))

            if len(ok_objs) > 0:

                # Rule 1: The nearest object must be a door
                if cose.is_a(ok_objs[0]['ref'], 'cose:InteriorDoor'):
                    gobjs = {}
                    for a in areas:
                        area_objs = [ x for x in ok_objs if cose.is_a(x['ref'], a) ]
                        if len(area_objs):
                            gobjs[a] = area_objs

                    # Find what kind of door
                    if len(gobjs):
                        nearest = min(gobjs.keys(), key=lambda k: np.min([ dist(n,x) for x in gobjs[k] ]))
                        ok_alt = {}
                        ok_alt.update( ok_objs[0] )
                        ok_alt['ref'] = nearest.replace('Object','Door')
                        assoc[ n['sensor'] ] = ok_alt

                else:
                    assoc[ n['sensor'] ] = ok_objs[0]

        return assoc

class ContactReasoner( DirectReasoner ):
    blocking     = True
    phenomenon   = 'cose:Contact'
    typeOfSensor = 'cose:ContactSensor'

    def objectIsOK(self, o):
        return self.cose.is_a(o['ref'], 'cose:AmbientObject')

class DoorReasoner( DirectReasoner ):
    blocking     = True
    phenomenon   = 'cose:Contact'
    typeOfSensor = 'cose:DoorSensor'

    def objectIsOK(self, o):
        return self.cose.is_a(o['ref'], 'cose:MayHaveDoorSensor')

class PressureReasoner( DirectReasoner ):
    blocking     = False
    phenomenon   = 'cose:Contact'
    typeOfSensor = 'cose:BinaryPressureSensor'

    def objectIsOK(self, o):
        return self.cose.is_a(o['ref'], 'cose:PlaceToSit')

class OpenObjectReasoner( DirectReasoner ):
    blocking     = False
    phenomenon   = 'cose:Contact'
    typeOfSensor = 'cose:BinaryPhotocell'

    def objectIsOK(self, o):
        return self.cose.is_a(o['ref'], 'cose:PlaceToSit')

class ProximityReasoner( DirectReasoner ):
    phenomenon   = 'cose:Contact'
    typeOfSensor = 'cose:ProximitySensor'

    def objectIsOK(self, o):
        return True

class EnvStateReasoner( RayTracedReasoner ):
    minLevel = 0.1

    def __init__(self, sensors, parent, blockedObj, objBoxes, minIntensity=0.05):
        super().__init__(sensors, parent, blockedObj, objBoxes, minIntensity = self.minLevel)

    def objIsOK(self, obj):
        return obj.type and self.cose.is_a(obj.type, 'cose:FunctionalSpace')

class LightLevelReasoner( EnvStateReasoner ):
    minLevel     = 10.0
    distFactor   = 15.0
    phenomenon   = 'cose:Light'
    typeOfSensor = 'cose:LightLevelSensor'

class LightSwitchReasoner( EnvStateReasoner ):
    minLevel     = 0.1
    distFactor   = 15.0
    phenomenon   = 'cose:Light'
    typeOfSensor = 'cose:LightSwitchSensor'

class TemperatureReasoner( EnvStateReasoner ):
    minLevel     = 25.0
    distFactor   = 10.0
    phenomenon   = 'cose:Temperature'
    typeOfSensor = 'cose:TemperatureSensor'

# ----------------------------------------------------------------------

def smarterTypes(env):
    from seamr import sem
    
    def dist(a,b):
        return math.sqrt( (a['x'] - b['x'])**2 + (a['y'] - b['y'])**2 )

    areas = ['cose:BathroomObject', 'cose:BedroomObject']

    cose = sem.COSE()

    assoc = {}
    shapes = load_shapes( env )
    
    edges = set( map(tuple, shapes['edges']) )
    edges |= set([ (b,a) for a,b in edges ])

    objs = [ n for n in shapes['nodes'] if n['sensor'] == '' and 'x' in n and 'y' in n ]
    
    for door in [o for o in objs if cose.is_a(o['ref'], 'cose:InteriorDoor')]:

        ok_objs = sorted(filter(lambda o: (door['id'], o['id']) in edges and not (door is o), objs), key=lambda o: dist(n,o))    

        gobjs = {}
        for a in areas:
            area_objs = [ x for x in ok_objs if cose.is_a(x['ref'], a) ]
            if len(area_objs):
                gobjs[a] = area_objs

        if len(gobjs):
            nearest = min(gobjs.keys(), key=lambda k: np.min([ dist(n,x) for x in gobjs[k] ]))
            assoc[door['id']] = nearest.replace('Object','Door')

    return assoc