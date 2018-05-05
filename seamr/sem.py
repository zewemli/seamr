import rdflib
import os, sys
import networkx as nx
import json
import random
from tqdm import tqdm
import math
from datetime import datetime

import numpy as np

from scipy.spatial import distance

from collections import defaultdict, Counter, deque
from itertools import chain, groupby, product, combinations
from urllib.parse import urlparse, urlunparse

from seamr import log

cose_g = None
cose_type_graph = None

def memoize(func):
    cache = {}
    def wrapper(self, *args):
        if args not in cache:
            ret = func(self, *args)
            cache[args] = ret
        return cache[args]
    return wrapper

class Prefixer(object):

    def __init__(self):

        self.term_pfx = "http://serc.wsu.edu/owl/terms.owl"

        self.pfx = {
            "http://www.w3.org/2000/01/rdf-schema": "rdfs",
            "http://www.w3.org/2002/07/owl": "owl",
            "http://www.w3.org/1999/02/22-rdf-syntax-ns": "rdf",
            "http://serc.wsu.edu/owl/cose.owl" : "cose",
            "http://serc.wsu.edu/owl/activity.owl" : "activity",
            self.term_pfx : "term"
        }

        self.exp = dict([ (v,k) for (k,v) in self.pfx.items() ])

    def namespaces(self):
        return dict([ (v, rdflib.Namespace(k+"#")) for (k,v) in self.pfx.items() if urlparse(k).netloc ])

    def __call__(self, url_str, debug=False):
        url_str = str(url_str)
        url = urlparse(url_str)

        if url.fragment:
            pfx = "{}://{}{}".format( url.scheme, url.netloc, url.path )
            obj = url.fragment
        else:
            pfx = self.term_pfx
            obj = url_str

        if pfx not in self.pfx:
            suffix = os.path.basename(pfx).split(".")[0]
            self.pfx[pfx] = suffix

        return "%s:%s" % (self.pfx[pfx], obj)
    
    def get(self, term, pfx = "cose"):
        if ":" in term:
            pfx,term = term.split(":")

        uri = "%s#%s" % (self.exp[pfx], term)
        return rdflib.term.URIRef( uri )

    def handle(self, items):
        return tuple(map(self, items))

class Activity:

    def __init__(self, name = None):
        self.name = name
        self.locs = set()
        self.objs = set()
        self.times = set()
    
    def setSpaces(self, objectConstraints):
        if len(self.locs) == 0:
            for o in self.objs:
                if o in objectConstraints:
                    self.locs |= objectConstraints[o]

    def addObj(self, obj):
        self.objs.add(obj)
    
    def addLoc(self, loc):
        self.locs.add(loc)

    def addTime(self, tm):
        self.times.add(tm)

    def concepts(self):
        return self.locs | self.objs | self.times

    def intersect(self, other):
        return ( self.locs & other.locs ), (self.objs & other.objs)

    def nCommon(self, other):
        locs,objs = self.intersect(other)
        return len( locs | objs )

    def __iter__(self):
        yield from self.objs
        yield from self.locs

class COSE(object):
    discLevels = {
        "cose:high": 1.0,
        "cose:normal": 0.7,
        "cose:low": 0.35
    }

    allPrefixes = {"cose": rdflib.Namespace("http://serc.wsu.edu/owl/cose.owl#"),
                    "act": rdflib.Namespace("http://serc.wsu.edu/owl/activity.owl#"),
                    "owl": rdflib.OWL,
                    "rdf": rdflib.RDF,
                    "dogont": rdflib.Namespace("http://elite.polito.it/ontologies/dogont.owl#"),
                    "ucum": rdflib.Namespace("http://purl.oclc.org/NET/muo/ucum#"),
                    "muo-vocab": rdflib.Namespace("http://purl.oclc.org/NET/muo/muo-vocab.owl#"),
                    "rdfs": rdflib.RDFS}

    def __init__(self):

        global cose_g
        global cose_type_graph

        activityOwlPath = os.path.join(os.path.dirname(__file__), "..", "ontology", "activity.ttl")
        coseOwlPath = os.path.join(os.path.dirname(__file__), "..", "ontology", "cose.ttl")

        self._is_a_cache = {}

        self._infer_cache = {}
        self._obs_cache = {}
        self._state_cache = {}

        if cose_g is None:
            cose_g = rdflib.Graph()
            cose_g.parse(coseOwlPath, format = "turtle")
            cose_g.parse(activityOwlPath, format = "turtle")
            
        self.g = cose_g
        self.gprefix = Prefixer()

        for (s,p,o) in self.g:
            self.gprefix(s)
            self.gprefix(p)
            self.gprefix(o)

        self.locs = set([ a for (a,) in self.query("""SELECT ?a
                            WHERE {
                                ?a rdfs:subClassOf* cose:FunctionalSpace .
                                }""") ])
                                
        self.items = set([ a for (a,) in self.query("""SELECT ?a
                                                        WHERE {
                                                            ?a rdfs:subClassOf* cose:HouseholdItem .
                                                       }""") ])
        
        self.prefixes = []
        
        m = [[["te"], "cose:BinaryTemperatureSensor"],
            [["ir"], "cose:TelevisionActivitySensor"],
            [["ma"], "cose:AreaMotionDetector"],
            [["m"], "cose:PassiveInfraredSensor"],
            [["so"], "cose:SonarMotionDetector"],
            [["ph"], "cose:BinaryPhotocell"],
            [["batv", "batp"], "cose:BatteryLevelSensor"],
            [["ls", "ll"], "cose:LightLevelSensor"],
            [["l"], "cose:LightSwitchSensor"],
            [["co","d"], "cose:DoorSensor"],
            [["di"], "cose:ProximitySensor"],
            [["fo","pr"], "cose:BinaryPressureSensor"],
            [["i"], "cose:ContactSensor"],
            [["ad1-b", "ad1-c", "a002", "a003"], "cose:FlowSensor"],
            [["ss", "sg"], "cose:ShakeSensor"],
            [["ad1-a", "a001"], "cose:ObjectTemperatureSensor"],
            [["t"], "cose:TemperatureSensor"],
            [["p"], "cose:PowerUsageSensor"],
            [["e"], "cose:DeviceActivitySensor"]]

        for (pfx, cls) in m:
            for p in pfx:
                self.prefixes.append([ p, cls ])
        self.prefixes.sort(reverse=True)

        self.spaceTypes = {}

        for (st,) in self.query("""SELECT ?a WHERE { ?a rdfs:subClassOf cose:SpaceObjects . }"""):
            self.spaceTypes[ st ] = st.replace("Object","")
        self.objSpace = {}

        # Finally setup the type graph
        if cose_type_graph is None:
            cose_type_graph = nx.DiGraph()
            q = """ SELECT ?c ?p WHERE { ?c rdfs:subClassOf ?p . } """
            cose_type_graph.add_edges_from( self.query(q) )

    def typeGraph(self):
        """
        Access here to make sure it was loaded
        """
        global cose_type_graph
        return cose_type_graph

    def getObjSpace(self, obj):
        if obj not in self.objSpace:
            for sType, space in self.spaceTypes.items():
                if self.is_a(obj, sType):
                    self.objSpace[obj] = space
                    break
        
        return self.objSpace.get(obj, None)

    def discLevel(self, v):
        assert 0 < v < 1
        if v > 0.70:
            return "cose:high"
        elif v < 0.35:
            return "cose:low"
        else:
            return "cose:normal"

    def sensorEntities(self, sensor):
        sType = self.sensorType(sensor)

        if sType == "cose:TelephoneActivitySensor":
            return "cose:Telephone" 
        elif sType == "cose:TelevisionActivitySensor":
            return "cose:Television"
        else:
            return None

    def isStateSensor(self, kind):
        return kind in set([ a for (a,) in self.query("""SELECT ?a WHERE {
            ?a rdfs:subClassOf* cose:EnvironmentStateSensor . }""") ])

    def sensorType(self, sensorName):
        sensorName = sensorName.split(":")[-1]
        
        for pfx, cls in self.prefixes:
            if sensorName.startswith(pfx):
                return cls
        
        return "cose:Sensor"

    def query(self, query):
        res = self.g.query(query, initNs=self.gprefix.namespaces())
        return list(map(self.gprefix.handle, res))

    @staticmethod
    def time_order():
        return {
            "cose:Night" : 0,
            "cose:Morning" : 1,
            "cose:MidMorning" : 2,
            "cose:Midday" : 3,
            "cose:Afternoon" : 4,
            "cose:Twilight" : 5,
            "cose:Evening" : 6
        }


    @staticmethod
    def time_label(sec_of_day):

        if sec_of_day > 86400:
            sec_of_day = sec_of_day % 86400

        if 0 <= sec_of_day <= (5 * 3600):
            return "cose:Night"
        elif (5 * 3600) <= sec_of_day <= (9 * 3600):
            return "cose:Morning"
        elif (9 * 3600) <= sec_of_day <= (12 * 3600):
            return "cose:MidMorning"
        elif (12*3600) <= sec_of_day <= (14 * 3600):
            return "cose:Midday"
        elif (14 * 3600) <= sec_of_day <= (17 * 3600):
            return "cose:Afternoon"
        elif (17*3600) <= sec_of_day <= (21 * 3600):
            return "cose:Twilight"
        elif (21*3600) <= sec_of_day:
            return "cose:Evening"
        else:
            return "cose:Night"

    @staticmethod
    def time_label_num(sec_of_day):

        if sec_of_day > 86400:
            sec_of_day = sec_of_day % 86400
    
        if 0 <= sec_of_day <= (5 * 3600):
            return 0
        elif (5 * 3600) <= sec_of_day <= (9 * 3600):
            return 1
        elif (9 * 3600) <= sec_of_day <= (12 * 3600):
            return 2
        elif (12*3600) <= sec_of_day <= (14 * 3600):
            return 3
        elif (14 * 3600) <= sec_of_day <= (17 * 3600):
            return 4
        elif (17*3600) <= sec_of_day <= (21 * 3600):
            return 5
        elif (21*3600) <= sec_of_day:
            return 6
        else:
            return 0

    @staticmethod
    def time_cond(db, time_name):

        h = lambda x: x * 3600

        if time_name.endswith("Night"):
            cond = (db.Sensor.sec_of_day >= 0) & (db.Sensor.sec_of_day <= h(6))
        elif time_name.endswith("Morning"):
            cond = (db.Sensor.sec_of_day >= h(6)) & (db.Sensor.sec_of_day <= h(11))
        elif time_name.endswith("Midday"):
            cond = (db.Sensor.sec_of_day >= h(11)) & (db.Sensor.sec_of_day <= h(14))
        elif time_name.endswith("Afternoon"):
            cond = (db.Sensor.sec_of_day >= h(14)) & (db.Sensor.sec_of_day <= h(17))
        elif time_name.endswith("Twilight"):
            cond = (db.Sensor.sec_of_day >= h(17)) & (db.Sensor.sec_of_day <= h(21))
        elif time_name.endswith("Evening"):
            cond = db.Sensor.sec_of_day >= h(21)

        return cond

    def collapseModels(self, models = None):
        cnt = Counter()
        co_cnt = Counter()
        concepts = set()

        if models is None:
            models = self.get_activity_models()

        for m in models.values():
            m_cons = m.objs | m.locs
            concepts |= m_cons
            cnt.update( m_cons )
            co_cnt.update( product(sorted(m_cons), sorted(m_cons)) )

        g = nx.Graph()
        
        for ((a,b), k_num) in co_cnt.items():
            if cnt[a] == cnt[b] and cnt[a] == k_num:
                g.add_edge(a, b, weight = k_num)
        
        map_to = {}

        subclass = self.gprefix.get("subClassOf", "rdfs")

        _type = self.gprefix.get("type","rdf")
        _class = self.gprefix.get("Class","owl")

        for i,subg in enumerate(nx.connected_component_subgraphs(g)):
            if subg.order() > 1:
                nodes = subg.nodes()
                
                new_node = "_or_".join([ n.split(":")[-1] for n in nodes])
                new_uri = self.gprefix.get(new_node, "cose")

                self.g.add( (new_uri, _type, _class) )

                for n in nodes:
                    map_to[n] = "cose:"+new_node
                    self.g.add( (new_uri, subclass, self.gprefix.get(n)) )
        
        new_models = {}
        for name,m in models.items():
            m_cons = set([ map_to.get(k,k) for k in m.objs | m.locs ])

            locs = set([ k for k in m_cons if self.is_a(k, "cose:FunctionalSpace") ])
            objs = set([ k for k in m_cons if self.is_a(k, "cose:HouseholdItem") ])

            act = Activity(name)
            act.locs |= locs
            act.objs |= objs
            new_models[name] = act

        return new_models

    def getSpaceConstraints(self):
        q = """SELECT ?o ?s WHERE { ?o cose:objectPossibleSpace ?s . }"""
        ocons = defaultdict(set)
        scons = defaultdict(set)

        for (o,s) in self.query(q):
            ocons[o].add(s)
            scons[s].add(o)

        return ocons, scons 

    def is_a(self, obj, query_type):
    
        if obj == query_type:
            return True
        else:
            key = (obj, query_type,)
            if key not in self._is_a_cache:
                try:
                    self._is_a_cache[ key ] = nx.has_path(cose_type_graph, obj, query_type)
                except:
                    self._is_a_cache[ key ] = False

            return self._is_a_cache[key] 

    def act_hierarchy(self):
        q = """SELECT ?p ?c WHERE {
            ?p rdfs:subClassOf* activity:Activity .
            ?c rdfs:subClassOf ?p .
            FILTER(?c != activity:Activity) .
        }"""

        pf = lambda w: str(w).split("#")[-1]

        g = nx.DiGraph()
        
        g.add_edges_from(map(lambda T: (pf(T[0]), pf(T[1])), self.g.query(q, initNs = self.gprefix.namespaces())) )

        return g
        
    def get_activity_models(self):
        obj_inherit = """SELECT ?s ?o WHERE {
                    ?s rdfs:subClassOf* ?p .
                    ?p rdfs:subClassOf* activity:Activity .
                    ?p activity:involvesObject ?o .
                }"""

        loc_inherit = """SELECT ?s ?o WHERE {
                    ?s rdfs:subClassOf* ?p .
                    ?p rdfs:subClassOf* activity:Activity .
                    ?p activity:occursIn ?o .
                }"""

        tm_inherit = """SELECT ?s ?o WHERE {
                    ?s rdfs:subClassOf* ?p .
                    ?p rdfs:subClassOf* activity:Activity .
                    ?p activity:normalTime ?o .
                }"""
        
        acts = defaultdict(Activity)

        for (s,o) in self.query( obj_inherit ):
            acts[s].addObj(o)
        
        for (s,l) in self.query( loc_inherit ):
            acts[s].addLoc(l)

        for (s,t) in self.query( tm_inherit ):
            acts[s].addTime(t)
        
        for (k,cls) in acts.items():
            cls.name = k

        for k,m in list(acts.items()):
            acts[ k.split(":")[-1].lower() ] = m

        return acts
    
    def get_noncomposable(self):
        roots = """SELECT ?s WHERE {
                        ?s rdfs:subClassOf cose:NonComposableInteraction .
                    }"""

        subclasses = """SELECT ?s WHERE {
                    ?s rdfs:subClassOf* %s .
                }"""
        
        rt = {}
        for (r,) in self.query( roots ):
            rt[r] = set([ s for (s,) in self.query(subclasses % r) ])
        
        return rt

    def sensorInfoType(self, sensor):
        stype = self.sensorType(sensor)
        query = """
        SELECT ?p WHERE {
            %s rdfs:subClassOf* ?p .
            ?p rdfs:subClassOf cose:SensorInformationType .
        }""" % stype

        for (t,) in self.query(query):
            return t

class LabelTransfer:

    def __init__(self, store):
        cose = COSE()
        interior = set()
        terminals = set([x.split(":")[-1] for x in set( activity_type_map.keys() )])
        
        self.g = cose.act_hierarchy()
        self.g.remove_node("Activity")
        
        for i,(lbl,act) in enumerate(sorted(activity_type_map.items())):
            act = act.split(":")[-1]
            assert act in self.g, act
            self.g.add_edge(act, lbl)

        topLevel = { s for s in self.g.nodes() if self.g.in_degree(s) == 0 }
        self.abstractActs = dict()
        try:
            for t in terminals:
                for r in topLevel:
                    if nx.has_path(self.g, r, t):
                        self.abstractActs[t] = r
                        break
        except:
            print(sorted(self.g.nodes()))
            raise
        # Edges are parent -> child
        for n in self.g.nodes():
            if self.g.out_degree(n) > 0:
                interior.add( n )

        self.actNames = sorted(interior)
        self.labelNames = sorted(terminals)

        self.actID   = {k:i for i,k in enumerate( self.actNames )}
        self.labelID = {k:i for i,k in enumerate( self.labelNames )}

        self._localizers = {}
        self._local_np = {}

        for ds in store.datasets():
            dsMat = np.zeros( (len(interior), len(terminals)), dtype=np.float32 )
            lbls = store.get_labels(ds, days=range(30))
            localActs = set([ l.activity for l in lbls.labels ])

            g = self.g.subgraph( interior | (terminals & localActs) )

            pathLens = dict( nx.all_pairs_shortest_path_length(g) )

            mat = np.zeros( (len(interior), len(terminals)), dtype=np.float32 )

            for src, lendict in pathLens.items():
                if src in self.actID:
                    sID = self.actID[ src ]
                    for dest,dLen in lendict.items():
                        if dest in terminals and dLen > 0:
                            tID = self.labelID[ dest ]
                            mat[sID, tID] = 1.0 / dLen

            mat /= mat.sum(axis=0, keepdims=True) + 0.000000001

            self._local_np[ ds ] = (mat > 0).astype( np.int32 )
    
    def remap(self, y):
        if y is None:
            return None
        else:
            return [self.abstractActs.get(x,x) for x in y]

# ==============================================================================

class EnvKnowledge:
    """
    WORK HERE : TODO

    Make a PMI model of sensor x label

    Walk through the dataset and attach a sensor event to the current
    activity whose label is most closely related to the sensor. 

    Now that we have pairs of (events, activity) use the activity models
    to build up an association between sensors x concepts.

    Now use the sensor x concepts weighted vectors to weight the
    events instead of the layouts used in the spatial reasoner.

    Compare clustering and classification preformance.

    Also, cluster the sensors based on their concept affinity
    and learn a segmenting model over that. Use the segmenting groups
    as a clustering label (group id = cluster label) then compare
    that to KMeans over a longer matrix.
    """

    def __init__(self, store, dataset, days, times):
        from seamr import core

        cose = COSE()
        activity_models = cose.collapseModels( cose.get_activity_models() )
        a_concepts = {}

        n = len(times)

        objs = set()
        locs = set()

        for act in activity_models.values():
            objs |= act.objs
            locs |= act.locs

        concepts = sorted( objs | locs )
        conceptIndex = { c:i for i,c in enumerate( concepts ) }

        modelVec = {}
        for lbl,mod in activity_models.items():
            z = np.zeros(len(concepts))
            z[ [ conceptIndex[c] for c in (mod.objs | mod.locs) ] ] = 1.0
            modelVec[lbl] = z

        sensor_assoc = np.zeros( (len(sensors), len(concepts)) )

        for batch,ts in core.Features.gen_batches(events(), times):
            lbl = activity_type_map.get(labels.strLabel(ts), None)
            sFreq = self.prob([e.sensorID for e in batch])

            if lbl in modelVec:
                mod = modelVec[ lbl ]

                for s,p in sFreq.items():
                    sensor_assoc[ s, : ] += mod * p
        
    @staticmethod
    def prob(ids):
        c = Counter(ids)
        return { k : float(n) / len(ids) for k,n in c.items() }
    
# ==============================================================================

def load_models(db):
    cose = COSE()
    models = cose.get_activity_models()
    return dict([ (a.id, models.get(activity_type_map[ a.name ], Activity(name = a.name)), ) for a in db.query( db.Activity )])

sensorTypeInfers = {
    "cose:TelephoneActivitySensor" : "cose:Telephone",
    "cose:TelevisionActivitySensor" : "cose:Television"
}

activity_type_map = dict([ (k, "activity:%s" % v) for k,v in {
  "bathe": "Bathing",
  "bathing": "Bathing",
  "bed_toilet_transition": "Bed_To_Toilet",
  "bed_to_toilet": "Bed_To_Toilet",
  "breakfast": "Breakfast",
  "brushing_teeth": "Personal_Hygiene",
  "caregiver": "Having_Guest",
  "changing_clothes": "Changing_Clothes",
  "chores": "Housekeeping",
  "clean": "Housekeeping",
  "cleaning": "Housekeeping",
  "cook_breakfast": "Cooking",
  "cook": "Cooking",
  "cook_dinner": "Cooking",
  "cooking": "Cooking",
  "cook_lunch": "Cooking",
  "desk_activity": "Work",
  "dining_rm_activity": "Other",
  "dinner": "Dinner",
  "dishes": "Wash_Dishes",
  "door_ajar": "Other",
  "dress": "Changing_Clothes",
  "drink": "Other",
  "drug_management": "Take_Medicine",
  "eat_breakfast": "Eating",
  "eat_dinner": "Eating",
  "eat": "Eating",
  "eating": "Eating",
  "eat_lunch": "Eating",
  "enter_home": "Enter_Home",
  "entertain_guests": "Having_Guest",
  "eve_meds": "Take_Medicine",
  "evening_meds": "Take_Medicine",
  "exercise": "Exercise",
  "going_out": "Out_Of_Home",
  "groceries": "Other",
  "groom": "Personal_Hygiene",
  "grooming": "Personal_Hygiene",
  "group_meeting": "Having_Guest",
  "guest_bathroom": "Toileting",
  "having_breakfast": "Eating",
  "having_conversation": "Having_Guest",
  "having_dinner": "Eating",
  "having_guest": "Having_Guest",
  "having_lunch": "Eating",
  "having_shower": "Bathing",
  "having_snack": "Eating",
  "housekeeping": "Housekeeping",
  "inspection": "Other",
  "kitchen_activity": "Other",
  "laundry": "Housekeeping",
  "leave_home": "Leave_Home",
  "listening": "Relax",
  "listening_to_music": "Relax",
  "loose_connection": "Other",
  "lunch": "Lunch",
  "maintenance": "Other",
  "make_bed": "Wake",
  "master_bathroom": "Toileting",
  "master_bedroom_activity": "Other",
  "meal_preparation": "Cooking",
  "meditate": "Relax",
  "morning_meds": "Take_Medicine",
  "movers": "Other",
  "napping": "Napping",
  "night_wandering": "Wandering",
  "other": "Other",
  "paramedics": "Other",
  "personal_hygiene": "Personal_Hygiene",
  "phone": "Talking_On_The_Phone",
  "piano": "Other",
  "preparing_breakfast": "Cooking",
  "preparing_dinner": "Cooking",
  "preparing_lunch": "Cooking",
  "reading_book": "Read",
  "reading": "Read",
  "read": "Read",
  "relax": "Relax",
  "resperate": "Other",
  "shaving": "Personal_Hygiene",
  "shower": "Bathing",
  "sleep": "Sleeping",
  "sleeping": "Sleeping",
  "sleeping_in_bed": "Sleeping",
  "sleeping_not_in_bed" : "Napping",
  "sleep_out_of_bed": "Napping",
  "snack": "Eating",
  "step_out": "Leave_Home",
  "studying": "Study",
  "study": "Study",
  "system_technicians": "Other",
  "take_medicine": "Take_Medicine",
  "talking_on_the_phone": "Talking_On_The_Phone",
  "toileting": "Toileting",
  "toilet": "Toileting",
  "using_internet": "Work",
  "wake": "Wake",
  "wakeup": "Wake",
  "wash_bathtub": "Housekeeping",
  "wash_breakfast_dishes": "Wash_Dishes",
  "wash_dinner_dishes": "Wash_Dishes",
  "wash_dishes": "Wash_Dishes",
  "washing_dishes": "Wash_Dishes",
  "wash_lunch_dishes": "Wash_Dishes",
  "wandering_in_room" : "Wandering",
  "watching_tv": "Watch_TV",
  "watch_tv": "Watch_TV",
  "work_at_table": "Work",
  "work_bedroom_1": "Work",
  "work_bedroom_2": "Work",
  "work_in_office": "Work",
  "work_livingrm": "Work",
  "work_on_computer": "Work",
  "work_table": "Work",
  "work": "Work",
  "yoga": "Exercise",
}.items() ])

null_act = activity_type_map["other"]

act_space = {
    "bathing": [ "bathroom" ],
    "bed_to_toilet": [ "bathroom", "bedroom" ],
    "breakfast": [ "kitchen","diningroom","livingroom" ],
    "changing_clothes": [ "bedroom", "bathroom" ],
    "cooking": [ "kitchen" ],
    "dinner": [ "kitchen","diningroom","livingroom" ],
    "eating": [ "diningroom", "livingroom", "kitchen" ],
    "enter_home": [ "foyer" ],
    "exercise": [ "livingroom", "bedroom" ],
    "having_guest": [ "livingroom", "kitchen", "bedroom", "bathroom" ],
    "housekeeping": [ "livingroom", "bathroom", "bedroom", "foyer" ],
    "leave_home": [ "foyer" ],
    "lunch": [ "kitchen","diningroom","livingroom" ],
    "napping": [ "livingroom","bedroom" ],
    "out_of_home": [None],
    "personal_hygiene": [ "bathroom", "bedroom" ],
    "read": [ "livingroom", "bedroom" ],
    "relax": ["livingroom", "bedroom"],
    "sleeping": [ "bedroom" ],
    "study": [ "livingroom", "bedroom"],
    "take_medicine": [ "kitchen","bathroom" ],
    "talking_on_the_phone": ["livingroom","kitchen","bedroom"],
    "toileting": ["bathroom"],
    "wake": [ "bedroom" ],
    "wandering": ["bedroom", "kitchen" ],
    "wash_dishes": [ "kitchen" ],
    "watch_tv": [ "livingroom", "bedroom" ],
    "work": [ "livingroom", "bedroom", "kitchen", "office" ]
}


act_object = {
    "bathing": [
        "bathroom"
    ],
    "bed_to_toilet": [
        "bathroom"
    ],
    "breakfast": [
        "kitchen","diningroom","livingroom"
    ],
    "changing_clothes": [
        "bedroom"
    ],
    "cooking": [ "kitchen" ],
    "dinner": [ "kitchen","diningroom","livingroom" ],
    "eating": [ "diningroom", "livingroom" ],
    "enter_home": [ "foyer" ],
    "exercise": [ "livingroom", "bedroom" ],
    "having_guest": [ "livingroom" ],
    "housekeeping": [ "livingroom" ],
    "leave_home": [ "foyer" ],
    "lunch": [
        "kitchen","diningroom","livingroom"
    ],
    "napping": [
        "livingroom","bedroom"
    ],
    "out_of_home": [None],
    "personal_hygiene": [ "bathroom" ],
    "read": [ "livingroom", "bedroom" ],
    "relax": ["livingroom", "bedroom"],
    "sleeping": [
        "bedroom"
    ],
    "study": [
        "livingroom", "bedroom"
    ],
    "take_medicine": [ "kitchen","bathroom" ],
    "talking_on_the_phone": [
        "livingroom","kitchen","bedroom"
    ],
    "toileting": [
        "bathroom"
    ],
    "wake": [
        "bedroom"
    ],
    "wandering": [
        "bedroom","kitchen"
    ],
    "wash_dishes": [
        "kitchen"
    ],
    "watch_tv": [
        "livingroom"
    ],
    "work": [ "livingroom","bedroom","diningroom" ]
}
