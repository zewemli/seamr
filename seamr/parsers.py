import os, sys
from datetime import date, datetime, timedelta
from time import monotonic
import logging
import re
import gzip

class FileStream(object):

    def __init__(self, msg, files):
        self.files = list(files)
        self.fiter = iter(self.files)

        self.env = sorted(set([ os.path.basename(os.path.dirname(l)) for l in files ]))[0]

        self.current_file = None
        self.ws = re.compile(r"\s+")
        self.lps = 100
        self.n = 0
        self.cnt = 0
        
        self.current_file = self.next_file()

        self.first_date = None
        self.last_date = None

    def __iter__(self):
        self.n = 0
        self.started = monotonic()
        return self

    def next_file(self):
        fname = next(self.fiter)
        if self.current_file:
            self.current_file.close()

        try:
            return open(fname, "rb")

        except FileNotFoundError:
            if os.path.exists( fname + ".gz" ):
                return gzip.open(fname + ".gz", "rb")
            else:
                raise

    def make_one(self):
        self.n += 1
        line = str(next(self.current_file),'utf-8').lower()
        return self.ws.split( line.strip() )

    def pos(self):
        return "%s:%s" % (os.path.basename(self.current_file.name), self.n)

    def handle_next(self):
        while True:
            try:
                val = self.make_one()
                if len(val) > 1:
                    return val
            except (IOError, StopIteration):
                raise StopIteration()
            except Exception as e:
                logging.warn("%s | %s | Check %s", e, self.env, self.pos())

    def parse_time(self, sdate, stime):

        try:
            dot = stime.index(".")
            ms = int( stime[dot+1:] )
        except ValueError:
            dot = len(stime)
            ms = 0

        return datetime(int(sdate[:4]),
                        int(sdate[5:7]),
                        int(sdate[8:10]),
                        int(stime[:2]),
                        int(stime[3:5]),
                        int(stime[6:dot]),
                        ms )


    def __next__(self):

        try:
            return self.handle_next()
        except StopIteration:
            self.current_file = self.next_file()

            return self.handle_next()

class TimedFileStream(FileStream):

    def make_one(self):
        val = super(TimedFileStream,self).make_one()
        assert len(val) >= 4, (self.n, "Line not long enough", val, self.pos())
        latest = self.parse_time(val[0], val[1])

        if not self.first_date:
            self.first_date = latest
        self.last_date = latest

        return (self.n, latest, val,)

class ARASFileStream(FileStream):

    def __init__(self, *args, **kw):
        super(ARASFileStream, self).__init__(*args, **kw)
        self.first_date = datetime(2012,1,1)

    def make_one(self):
        vals = list(map(int, super(ARASFileStream,self).make_one()))
        latest = self.first_date + timedelta(seconds=self.n)

        return (self.n, latest, vals,)

class DataParser(object):
    state_int = {
        "off":0, "on": 1,
        "present": 0, "absent": 1,
        "open": 1, "close": 0,
        "still": 0, "moved": 1
    }

    re_res = re.compile(r"^r\d", flags=re.IGNORECASE)

    ws = re.compile(r"\s+")

    def __init__(self, rootdir):
        self.basedir = rootdir
        self.pstate = {}

    @classmethod
    def can_parse(cls, fname):
        base = os.path.basename(fname)
        return any([ r.match(base) for r in cls._parsable ])

    def state_val(self, state):

        f = self.state_int.get(state, None)
        if f is None:
            return float(state)
        else:
            return f


# --------------- Generic CASAS Parser ----------------
class CASASParser(DataParser):

    _parsable = list(map(re.compile, [r"aruba.*",
                            r"cairo.*",
                            r"kyoto.*",
                            r"milan.*",
                            r"paris.*",
                            r"tulum.*"]))

    def get_file_stream(self):
        
        for opt in ["annotated","data","data.txt"]:
            pt = os.path.join(self.basedir, opt)
            if os.path.exists(pt):
                return TimedFileStream("Reading %s " % self.basedir, [pt])
        
        raise ValueError("Could not find a file")

    def get_label(self, l):

        if l[-1].endswith("_begin") or l[-1].endswith("_end"):
            lp = l[-1].split("_")
            lbl = "_".join(lp[:-1])
            l[-1] = lbl
            l.append(lp[-1])

        if len(l[4]) == 2 and self.re_res.match(l[4]):
            # Matches: R1 sleep start
            return l[4].lower(), "_".join(l[5:-1])

        elif self.re_res.search(l[4]):
            # Matches R1_sleep start
            lbl = l[4]
            i = lbl.index("_")
            return lbl[:i].lower(), lbl[i+1:]
        
        else:
            # Matches: Sleep start
            return "r1", "_".join(l[4:-1])

    def gen_events(self):
        for (i, time, l) in self.get_file_stream():
            if not l[2].startswith("bat"):
                sensor = l[2]
                
                try:
                    state = self.state_val(l[3].lower())
                except ValueError:
                    state = abs(self.pstate.get(sensor, 1.0) - 1)
                
                self.pstate[sensor] = state
                yield (i, sensor, state, time)
                    
    def gen_labels(self):

        current_label = None
        label_start = {}

        stream = self.get_file_stream()

        for (lnum, time, l) in stream:

            if len(l) > 4:
                if "begin" in l[-1]:
                    res_label = self.get_label(l)

                    if res_label in label_start:
                        logging.warn( "%s : LINE: %s | %s | start %s already in %s" % (stream.current_file.name, lnum, l, res_label, label_start) )
                        st = label_start.pop(res_label)
                        yield (lnum, res_label[0], res_label[1], st, time)

                    label_start[res_label] = time

                elif "end" in l[-1]:

                    res_label = self.get_label(l)
                    if res_label in label_start:
                        st = label_start.pop(res_label)
                        yield (lnum, res_label[0], res_label[1], st, time )
                    else:
                        logging.error( "%s : LINE: %s | %s | end of %s not in %s" % (stream.current_file.name, lnum, l, res_label, label_start) )

class HorizonHouse(CASASParser):

    _parsable = list(map(re.compile, [r"hh\d+"]))

    def get_file_stream(self):
        return TimedFileStream("Reading %s " % self.basedir, [
            os.path.join(self.basedir, "ann.txt")
        ])


    def get_label(self, lbl):

        if "." in lbl:
            return tuple(lbl.split("."))
        else:
            return None, lbl

    def gen_labels(self):

        current_label = None
        
        label_start = {}
        label_start_line = {}
        active_labels = set()
        label_res = {}
        label_latest = {}

        max_delta = timedelta(seconds=60)

        most_recent = 0
        stream = self.get_file_stream()

        for (lnum, time, l) in stream:

            if len(l) > 4:
                most_recent = time
                l[-1] = l[-1].lower()
                if "=" in l[-1]:
                    lbl, lbl_state = l[-1].replace('"',"").split("=")
                    res, lbl = self.get_label(lbl)

                    if "begin" in lbl_state:

                        if lbl not in label_start:
                            label_start[lbl] = time
                        
                        active_labels.add(lbl) 
                        label_start_line[lbl] = lnum

                        if res:
                            label_res[lbl] = res
                    
                    elif lbl in active_labels:
                        # Start event-based tracking, also handles "end" events
                        active_labels.discard(lbl)
                        label_latest[lbl] = time

                    else:
                        if l[3] == "on":
                            # Assume that this is really a start label
                            label_start[lbl] = time

                            active_labels.add(lbl) 
                            label_start_line[lbl] = lnum

                            if res:
                                label_res[lbl] = res

                        else:
                            logging.error("LINE: %s %s | %s | %s :: bad label or end of activity with no start ::" % (stream.env, stream.pos(), l, time))

                else:
                    res, lbl = self.get_label(l[-1])

                    if res:
                        label_res[lbl] = res
                    
                    if lbl not in label_start:
                        label_start[lbl] = time
                        label_start_line[lbl] = lnum

                    label_latest[lbl] = time

            else:
                # No Label
                pass
            
            if most_recent and (time - most_recent) > max_delta:

                for (k, e) in list(label_latest.items()):
                    if (time - e) > max_delta and k not in active_labels:
                        try:
                            s = label_start.pop(k)
                            e = label_latest.pop(k)
                            line = label_start_line.pop(k)

                            if k in label_res:
                                l_res = label_res.pop(k)
                            else:
                                l_res = "r1"

                        except:
                            logging.error("[] Problem with %s @ %s : %s %s " % (stream.env, stream.pos(), k, time, l, self.basedir))
                            raise
                        
                        yield (line, l_res, k, s, e)

        # -- Now clean up
        for (k, e) in list(label_latest.items()):
            s = label_start.pop(k)
            e = label_latest.pop(k)
            line = label_start_line.pop(k)

            if k in label_res:
                l_res = label_res.pop(k)
            else:
                l_res = "r1"
            
            yield (line, l_res, k, s, e)


# -------------------- ARAS --------------------

class ArasParser(DataParser):

    _parsable = list(map(re.compile, [r"aras.*"]))

    def __init__(self, *args, **kw):
        super(ArasParser,self).__init__(*args, **kw)

        self.sensors, self.activities = self.parse_readme(self.basedir)

    def get_file_stream(self):
        re_d = re.compile(r"\d+")
        day_int = lambda p: int(re_d.search(p).group(0))

        files = sorted([f for f in os.listdir(self.basedir) if f.startswith("DAY_")], key=day_int)
        files = [ os.path.join(self.basedir, f) for f in files ]

        return ARASFileStream("Reading %s" % self.basedir, files)

    def parse_readme(self, in_dir):
        sensors = []
        activities = []

        with open(os.path.join(in_dir,"README")) as f:
            in_sensors = False
            in_acts = False

            for l in f:
                lp = self.ws.split(l.strip())

                if len(lp) <= 1:
                    in_sensors = False
                    in_acts = False

                if in_sensors:
                    sensors.append(lp[1])
                elif in_acts:
                    activities.append( "_".join(lp[1:]) )

                if lp[0] == "Column" and lp[1] == "Sensor":
                    in_sensors = True
                elif lp[0] == "ID" and lp[1] == "ACTIVITY":
                    in_acts = True

        sensors    = [s.lower() for s in sensors]
        activities = [a.lower() for a in activities]

        return sensors, activities

    def gen_labels(self):
        """
        Yields (resident, activity, start_time, end_time)
        """
        src = iter( self.get_file_stream() )
        i, start_time, current_states = next(src)

        lbl_range = [("r1",-2), ("r2",-1)]

        started = [ start_time ] * len(current_states)

        second = timedelta(seconds=1)

        for (lnum, time, l) in src:

            for res,i in lbl_range:
                if l[i] != current_states[i]:
                    yield (lnum, res, self.activities[ current_states[i] - 1 ], started[i], time - second)
                    started[i] = time

            current_states = l


    def gen_events(self):
        """
        Yields (time, sensor, state)
        """
        src = iter( self.get_file_stream() )

        start_line, start_time, current_states = next(src)

        for (sensor,s) in zip(self.sensors, current_states):
            if s:
                yield (start_line, sensor, 1, start_time)

        for lnum, time, l in src:

            for i, sensor in enumerate(self.sensors):
                if l[i] != current_states[i]:
                    yield (lnum, sensor, l[i], time)

            current_states = l
