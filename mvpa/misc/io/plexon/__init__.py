#!/usr/bin/python
#coding:utf-8
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   Copyright (c) 2008 Scott Gorlin <gorlins@mit.edu>
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

"""This module provides an interface to Plexon datafiles (and potentially the MAP
server, though this is unimplemented to date.

For low(er)-level code, see header.py

See www.plexoninc.com for hardware or API information
"""
import ctypes
import header
import numpy as N
import copy

class Event(object):
    """Represents an event with metadata, stored at a given time
    
    Allows for comparison of the time, so it's easy to sort and divide, etc
    """
    def __init__(self, timestamp):
        self.time=timestamp
    def __float__(self):
        return self.time
    def __cmp__(self, t):
        return cmp(self.time,float(t))
    
    def __add__(self, t):
        o = copy.copy(self)
        o.time+=float(t)
        return o
    def __sub__(self, t):
        o = copy.copy(self)
        o.time-=float(t)
        return o
class Word(Event):
    """Stores a strobed word"""
    def __init__(self, timestamp, word):
        Event.__init__(self, timestamp)
        self.word=word
class Plexon(object):
    """Class which wraps all the nice header objects"""
    def __init__(self, filename=None, **readargs):
        self.FileHeader = header.PL_FileHeader()
        self.ChanHeader = []
        self.EventHeader = []
        self.SlowChannelHeader = []
        self.DataBlockHeader = []
        self.DataBlock = []
        self.continuousAD={}
        self.strobes = []
        self.spikes={}
        if filename:
            self.read(filename, **readargs)
    def read(self, filename, onlyReadHeader=False, readWaves=False, readContinuous=False):
        """Reads all the good stuff from a plx file
        
        onlyReadHeader: just reads the file, chan, event, and slowchan headers.
        Otherwise will continue to read the bulk of the file
        
        readWaves: reads waveform data stored after each spike
        If false, still reads the spike event info, just doesn't store the shape
        of the waveforms
        
        readContinuous: reads all analog data"""

        data =  open(filename, 'rb')
        # Reads all the headers
        self.FileHeader._unpack(data)
        
        self.ChanHeader = [header.PL_ChanHeader()._unpack(data) for i in range(self.FileHeader.NumDSPChannels)]
        self.EventHeader = [header.PL_EventHeader()._unpack(data) for i in range(self.FileHeader.NumEventChannels)]
        self.SlowChannelHeader = [header.PL_SlowChannelHeader()._unpack(data) for i in range(self.FileHeader.NumSlowChannels)]
            
        if not onlyReadHeader:
            # Prep struct
            self.DataBlockHeader = []
            self.spikes = {}
            self.waves = {}
            self.events={}
            self.strobes=[]
            self.continuousAD={}
            for c in self.ChanHeader:
                self.spikes[c.Channel] = {}
                self.waves[c.Channel] = {}
                for i in range(c.NUnits):
                    self.spikes[c.Channel][i] = []
                    self.waves[c.Channel][i] = []
            
            cgain = {}
            for c in self.ChanHeader:
                cgain[c.Channel] = c.Gain
                self.spikes[c.Channel]={}
            for c in self.SlowChannelHeader:
                self.continuousAD[c.Channel]=[]
            if self.FileHeader.Version >= 105:
                def vConv(x,c):
                    return 2.*x*self.FileHeader.SpikeMaxMagnitudeMV/((2.**self.FileHeader.BitsPerSpikeSample)*cgain[c]*self.FileHeader.SpikePreAmpGain)
            elif self.FileHeader.Version >= 103:
                def vConv(x,c):
                    return 2.*x*self.FileHeader.SpikeMaxMagnitudeMV/((2.**self.FileHeader.BitsPerSpikeSample)*cgain[c]*1000.)
            else:
                def vConv(x,c):
                    return x*3000./(2048.*cgain[c]*1000.)
            # Read
            self.__hdr=header.PL_DataBlockHeader()
            self.__readContinuous = readContinuous
            self.__readWaves = readWaves
            self.__shortsize = ctypes.sizeof(ctypes.c_short)
            try:
                while True:
                    self.__readEvent(data)
            except EOFError:
                pass
        data.close()
    
    def __readEvent(self, data):
        """Read a PL_Event from a datastream"""
        # Gets data header
        hdr = self.__hdr
        hdr._unpack(data)
        
        # convert timestamp to secs
        s = float((hdr.UpperByteOf5ByteTimestamp<<32)+hdr.TimeStamp) / self.FileHeader.ADFrequency
        
        # Parses the header, acts according to type
        if hdr.Type == header.PL_SingleWFType:
            try:
                self.spikes[hdr.Channel][hdr.Unit].append(s)
            except KeyError:
                self.spikes[hdr.Channel][hdr.Unit] = [s]
          
            if hdr.NumberOfWaveforms:
                if self.__readWaves:
                    wave=N.fromfile(data, dtype=ctypes.c_short, count=hdr.NumberOfWordsInWaveform)
                    self.waves[hdr.Channel][hdr.Unit].append(vConv(wave, hdr.Channel))
                else:
                    junkwv = data.read(hdr.NumberOfWordsInWaveform*self.__shortsize)
                    #data.seek(hdr.NumberOfWordsInWaveform*self.__shortsize,1)
                                  
        elif hdr.Type == header.PL_ExtEventType:
            if hdr.Channel==header.PL_StrobedExtChannel: #Should be 257
                self.strobes.append(Word(s, hdr.Unit))
            else:
                try:
                    self.events[hdr.Channel].append(s)
                except KeyError:
                    self.events[hdr.Channel] = [s]
        elif hdr.Type == header.PL_ADDataType:
            if self.__readContinuous:
                ad = N.fromfile(data, dtype=ctypes.c_short, count=hdr.NumberOfWordsInWaveform)
                self.continuousAD[hdr.Channel].append((s,ad))
            else:
                junkad = data.read(hdr.NumberOfWordsInWaveform*self.__shortsize)
                #data.seek(hdr.NumberOfWordsInWaveform*self.__shortsize, 1)
            
        elif hdr.Type == header.PL_StereotrodeWFType:
            raise NotImplementedError()
        elif hdr.Type == header.PL_TetrodeWFType:
            raise NotImplementedError()        
        else:
            raise RuntimeError('Unrecognized data type %d'%hdr.Type)
        
    def sliceByWord(self, startWord=-0.5, endWord=2, zeroWord=None, limAsTime=True, ignore0=True):
        """Slices spikes by strobed words
        if limAsTime, zeroWord must be given, and startWord and endWord are 
        offsets from zeroWord in seconds
        
        otherwise, startWord and endWord can be strobed words (and zeroWord is 
        optional).  In this mode, it takes care of mismatched numbers of strobes
        in each condition by matching each end strobe to the immediately 
        preceeding start word (allowing for false starts)
        
        Generates a list of time tuples ans sends to sliceByTime, returns the 
        output
        """
        if limAsTime:
            realZeros = [s for s in self.strobes if s.word==zeroWord]
            realStarts = [z+startWord for z in realZeros]
            ends = [z+endWord for z in realZeros]
        else:
            starts = [s for s in self.strobes if s.word==startWord]
            ends = [s for s in self.strobes if s.word==endWord]
            if zeroWord:
                zeros = [s for s in self.strobes if s.word==zeroWord]
            else:
                zeros = starts
                
            # Take care of false starts
            realStarts = []
            realZeros = []
            for end in ends:
                allStarts = [s for s in starts if s < end]
                allZeros = [s for s in zeros if s < end]
                realStarts.append(allStarts[-1])
                realZeros.append(allZeros[-1])
        times = zip(realStarts, ends, realZeros)
        return self.sliceByTime(times, ignore0=ignore0)

    def sliceByTime(self, times=None, ignore0=True):
        """Calls the module function, returns a dict w/ every channel and unit
        containing spikes
        
        requires times to be a list of tuples[(start, end),(...),...]
        (or (start, end, zerotime))
        
        returns (spikes, events, strobes) 
        spikes={ channelNum:{unitNum:[[spks in time0], [spks in time1], ...]}}
        events={eventChannel:[[evtTimes in time1],...]}
        strobes=[[words in time1], [words in time2]...]
        
        ignore0: ignores spikes in unit 0 (unsorted), ie doesn't return them
        """
        outspks = {}
        for c in self.spikes.keys():
            for u in self.spikes[c].keys():
                if self.spikes[c][u]:
                    if ignore0 and u==0:
                        continue
                    if not c in outspks:
                        outspks[c] = {}
                    
                    s = sliceSpikes(self.spikes[c][u], times)
                    # Sanity check - make sure we have spikes!! (should check with file)
                    if reduce(lambda t, trial: t or len(trial)>0,s, False):
                        outspks[c][u] = s
            if (c in outspks.keys()) and len(outspks[c].keys())==0:
                del outspks[c]
        outevts = {}
        for e in self.events.keys():
            outevts[e] = sliceSpikes(self.events[e], times)
        return (outspks, outevts, sliceSpikes(self.strobes, times))
         
def __spksort(t):
    while __spksort.cbeg < __spksort.N and __spksort.spikes[__spksort.cbeg] < t[0]:
        __spksort.cbeg+=1
    cend=__spksort.cbeg
    while cend < __spksort.N and __spksort.spikes[cend] <= t[1]:
        cend+=1
    out = __spksort.spikes[__spksort.cbeg:cend]
    __spksort.cbeg=cend
    if len(t)==3:
        zero = float(t[2])
    else:
        zero = float(t[0])
    if len(out) and out[-1]-zero > 2:
        pass
    return [o-zero for o in out]   
def sliceSpikes(spikes, times):
    """Slice spikes into trials
    
    spikes - list of spike times
    
    times - ordered, nonoverlappping list of (start, end) or (start, end, zero) 
    (if the argument zero is provided, the timestamp is zeroed to that time 
    instead of start)
    
    returns [[spike1, spike2...] for every tuple in times]
    """
    # Sets locals for __spksort, maps it for every tuple in times
    __spksort.N = len(spikes)
    __spksort.cbeg = 0
    __spksort.spikes=spikes
    return map(__spksort, times)
try:
    if not __debug__:
        import psyco
        psyco.bind(__spksort)
        psyco.bind(sliceSpikes)
        psyco.bind(Plexon)
    pass
except Exception:
    print "Psyco binding failed, no biggie"
def trainToHist(trainlist, bins=N.arange(-.5, 2, .001), sigma=0):
    """Convert a list of spike trains into a pdf"""
    bins = N.asarray(bins)
    if sigma > 0:
        dtype=float
    else:
        dtype =int

    h = N.asarray([N.histogram(N.asarray(x), bins=bins, new=True)[0] for x in trainlist], dtype=dtype)
    if sigma > 0:
        import scipy.signal
        g = scipy.signal.gaussian(N.ceil(1+6.5*sigma), sigma)
        g /= g.sum()
        for i in range(h.shape[0]):
            h[i,:]=scipy.signal.convolve(h[i,:], g, mode='same')
        
       
    return h
    
def testLoad():
    """Loads a test plx, plots some rasters"""
    import pylab
    import time
    t1 = time.time()
    p = loadTestPlx()
    t2 = time.time()
    print 'Load time: %f'%(t2-t1)
    (outspks, events, strobes) = p.sliceByTime([(a, a+1, a+3) for a in range(1000)], ignore0=False)
    t3 = time.time()
    print 'Slice time: %f'%(t3-t2)
    bins = N.arange(-.5, 2, .001)
    for c in outspks.keys():
        for u in outspks[c].keys():
            pylab.figure()
            pylab.hold(True)
            for (t, x) in enumerate(outspks[c][u]):
                if len(x) > 0:
                    pylab.scatter(x, [t]*len(x), s=0.5)
            pylab.hold(False)
            pylab.title('Channel %i Unit %i'%(c, u))
    pylab.show()
    return p

def loadTestPlx(**loadargs):
    """Load the provided test1.plx or test2.plx from Plexon
    
    plx=1 or 2, loadargs sent to Plexon(...)
    """
    import os
    (folder, junk) = os.path.split(os.path.abspath(__file__))
    fn = os.path.join(folder, 'test.plx')
    return Plexon(filename=fn, **loadargs)

if __name__ == '__main__':
    #import cProfile
    #cProfile.run('testLoad()')
    testLoad()
    pass
