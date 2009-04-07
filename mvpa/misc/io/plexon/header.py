#!/usr/bin/python
#coding:utf-8
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   Copyright (c) 2008 Scott Gorlin <gorlins@mit.edu>
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
from ctypes import *

"""This module implements Plexon.h in python, allowing the user to read plx files"""

# Wraps Structure for some cool functionality
class StructureWrapper(Structure):
    """A couple additions to the ctypes.Structure class to make things easier
    
    Like a normal ctypes.Structure, subclass this.  Then define your class like:
    
    class MyClass(Structure):
        _fields_ = [('Var1', c_char), 
                    ('VarArray', 2*(3*c_int))] # == int VarArray[2][3]
        
    (multiple parentheses gives you arrays of different dims and sizes)
    
    use any ctypes type for variables
    """
    #__metaclass__=MetaStruct
    def _unpack(self, s):
        """Set this struct from a binary array exposing the read() method
        (ie file object, or StringIO buffer)
        
        Automatically calculates the size to read, from the current position
        
        Must contain as many bytes as this structure, else EOFError is raised
        
        Will overwrite current data.  Use Structure()._unpack(...) to create a
        new instance (since it's faster to overwrite current data if you don't
        need a new instance)
        
        Returns self 
        """
        try:
            size = self.__class__.__size
        except AttributeError:
            self.__class__.__size = sizeof(self.__class__)
            size = self.__size
        ss = s.read(size)
        if len(ss) < size:
            raise EOFError()
        memmove(addressof(self), ss, size)
        return self
    
    def _pack(self):
        """Return a string representation of the binary data"""
        return buffer(self)[:]
    

PL_SingleWFType = 1
PL_StereotrodeWFType = 2  
PL_TetrodeWFType = 3
PL_ExtEventType = 4
PL_ADDataType = 5
PL_StrobedExtChannel = 257
PL_StartExtChannel = 258
PL_StopExtChannel  = 259
PL_Pause = 260
PL_Resume = 261

MAX_WF_LENGTH = 56
MAX_WF_LENGTH_LONG = 120

class PL_FileHeader(StructureWrapper):
    _fields_ = [
        ('MagicNumber', c_uint), # = 0x58454c50;
        ('Version', c_int),  # Version of the data format; determines which data items are valid
        ('Comment', 128*c_char),       # User-supplied comment 
        ('ADFrequency', c_int),        # Timestamp frequency in hertz
        ('NumDSPChannels', c_int),   # Number of DSP channel headers in the file
        ('NumEventChannels', c_int),   # Number of Event channel headers in the file
        ('NumSlowChannels', c_int),    # Number of A/D channel headers in the file
        ('NumPointsWave', c_int), # Number of data points in waveform
        ('NumPointsPreThr', c_int), # Number of data points before crossing the threshold
        
        ('Year', c_int),      # Time/date when the data was acquired
        ('Month', c_int), 
        ('Day', c_int),  
        ('Hour', c_int),
        ('Minute', c_int), 
        ('Second', c_int), 
        
        ('FastRead', c_int),           # reserved
        ('WaveformFreq', c_int),       # waveform sampling rate; ADFrequency above is timestamp freq 
        ('LastTimestamp', c_double),   # duration of the experimental session, in ticks
        
        # The following 6 items are only valid if Version >= 103
        ('Trodalness', c_byte),       # 1 for single, 2 for stereotrode, 4 for tetrode
        ('DataTrodalness', c_byte),   # trodalness of the data representation
        ('BitsPerSpikeSample', c_byte),         # ADC resolution for spike waveforms in bits (usually 12)
        ('BitsPerSlowSample', c_byte),   # ADC resolution for slow-channel data in bits (usually 12)
        ('SpikeMaxMagnitudeMV', c_ushort), # the zero-to-peak voltage in mV for spike waveform adc values (usually 3000)
        ('SlowMaxMagnitudeMV', c_ushort), # the zero-to-peak voltage in mV for slow-channel waveform adc values (usually 5000)
        
        # Only valid if Version >= 105
        ('SpikePreAmpGain', c_ushort),       # usually either 1000 or 500
        
        ('Padding', 46*c_char),      # so that this part of the header is 256 bytes
    
    
        # Counters for the number of timestamps and waveforms in each channel and unit.
        # Note that these only record the counts for the first 4 units in each channel.
        # channel numbers are 1-based - array entry at [0] is unused
        ('TSCounts', 130*(5*c_int)), # number of timestamps[channel][unit]
        ('WFCounts', 130*(5*c_int)), # number of waveforms[channel][unit]
        
        # Starting at index 300, this array also records the number of samples for the 
        # continuous channels.  Note that since EVCounts has only 512 entries, continuous 
        # channels above channel 211 do not have sample counts.
        ('EVCounts', 512*c_int),   # number of timestamps[event_number]
    ]

class PL_DataBlockHeader(StructureWrapper):
    _fields_ = [
        ("Type", c_short),# Data type; 1=spike, 4=Event, 5=continuous
        ("UpperByteOf5ByteTimestamp", c_ushort),# Upper 8 bits of the 40 bit timestamp
        ("TimeStamp", c_ulong),               # Lower 32 bits of the 40 bit timestamp
        ('Channel', c_short),                    # Channel number
        ('Unit', c_short),                       # Sorted unit number; 0=unsorted
        ('NumberOfWaveforms', c_short),          # Number of waveforms in the data to folow, usually 0 or 1
        ('NumberOfWordsInWaveform', c_short),# Number of samples per waveform in the data to follow
    ]
    # 16 bytes
    
    
class PL_EventHeader(StructureWrapper): 
    _fields_=[
        ('Name', 32*c_char), # name given to this event
        ('Channel', c_int),   # event number, 1-based
        ('Comment', 128*c_char),
        ('Padding', 33*c_int),
    ]
    
class PL_ChanHeader(StructureWrapper):
    _fields_ = [
        ('Name', 32*c_char),    # Name given to the DSP channel
        ('SIGName', 32*c_char),   # Name given to the corresponding SIG channel
        ('Channel', c_int),        # DSP channel number, 1-based
        ('WFRate', c_int),    # When MAP is doing waveform rate limiting, this is limit w/f per sec divided by 10
        ('SIG', c_int),            # SIG channel associated with this DSP channel 1 - based
        ('Ref', c_int),            # SIG channel used as a Reference signal, 1- based
        ('Gain', c_int),           # actual gain divided by SpikePreAmpGain. For pre version 105, actual gain divided by 1000. 
        ('Filter', c_int),         # 0 or 1
        ('Threshold', c_int),      # Threshold for spike detection in a/d values
        ('Method', c_int),         # Method used for sorting units, 1 - boxes, 2 - templates
        ('NUnits', c_int),         # number of sorted units
        ('Template', 5*(64*c_short)),# Templates used for template sorting, in a/d values
        ('Fit', 5*c_int),         # Template fit 
        ('SortWidth', c_int),      # how many points to use in template sorting (template only)
        ('Boxes', 5*(2*(4*c_short))), # the boxes used in boxes sorting
        ('SortBeg', c_int),       # beginning of the sorting window to use in template sorting (width defined by SortWidth)
        ('Comment', 128*c_char),
        ('Padding', 11*c_int),
    ]
    
class PL_SlowChannelHeader(StructureWrapper):
    _fields_=[
        ('Name', 32*c_char),     # name given to this channel
        ('Channel', c_int),       # channel number, 0-based
        ('ADFreq', c_int),         # digitization frequency
        ('Gain', c_int),           # gain at the adc card
        ('Enabled', c_int),       # whether this channel is enabled for taking data, 0 or 1
        ('PreAmpGain', c_int),     # gain at the preamp
        
        # As of Version 104, this indicates the spike channel (PL_ChanHeader.Channel) of
        # a spike channel corresponding to this continuous data channel. 
        # <=0 means no associated spike channel.
        ('SpikeChannel', c_int),
        
        ('Comment', 128*c_char),
        ('Padding', 28*c_int),
    ]

# These other structs don't seem to be used in the files, but may be useful


#
# PL_Event is used in PL_GetTimestampStructures(...)
#
class PL_Event(StructureWrapper): # 16 bytes
    _fields_=[
        ('Type', c_char), # PL_SingleWFType, PL_ExtEventType or PL_ADDataType
        ('NumberOfBlocksInRecord', c_char),     # reserved   
        ('BlockNumberInRecord', c_char),        # reserved 
        ('UpperTS', c_ubyte),           # really uchar, but ctypes has none Upper 8 bits of the 40-bit timestamp
        ('TimeStamp', c_ulong),         # Lower 32 bits of the 40-bit timestamp
        ('Channel', c_short),                    # Channel that this came from, or Event number
        ('Unit', c_short),                       # Unit classification, or Event strobe value
        ('DataType', c_char),                   # reserved
        ('NumberOfBlocksPerWaveform', c_char),  # reserved
        ('BlockNumberForWaveform', c_char),     # reserved
        ('NumberOfDataWords', c_char),          # number of shorts (2-byte integers) that follow this header 
    ]


#
# The same as PL_Event above, but with Waveform added
#
class PL_Wave(StructureWrapper):# size should be 128
    _fields_=[
        ('Type', c_char), # PL_SingleWFType, PL_ExtEventType or PL_ADDataType
        ('NumberOfBlocksInRecord', c_char),     # reserved   
        ('BlockNumberInRecord', c_char),        # reserved 
        ('UpperTS', c_ubyte),           # really uchar, but ctypes has none Upper 8 bits of the 40-bit timestamp
        ('TimeStamp', c_ulong),         # Lower 32 bits of the 40-bit timestamp
        ('Channel', c_short),                    # Channel that this came from, or Event number
        ('Unit', c_short),                       # Unit classification, or Event strobe value
        ('DataType', c_char),                   # reserved
        ('NumberOfBlocksPerWaveform', c_char),  # reserved
        ('BlockNumberForWaveform', c_char),     # reserved
        ('NumberOfDataWords', c_char),          # number of shorts (2-byte integers) that follow this header 
        ('WaveForm', MAX_WF_LENGTH*c_short),    # The actual waveform data
    ]
    
#
# An extended version of PL_Wave for longer waveforms
#
class PL_WaveLong(StructureWrapper):
    _fields_=[
        ('Type', c_char), # PL_SingleWFType, PL_ExtEventType or PL_ADDataType
        ('NumberOfBlocksInRecord', c_char),     # reserved   
        ('BlockNumberInRecord', c_char),        # reserved 
        ('UpperTS', c_ubyte),           # really uchar, but ctypes has none Upper 8 bits of the 40-bit timestamp
        ('TimeStamp', c_ulong),         # Lower 32 bits of the 40-bit timestamp
        ('Channel', c_short),                    # Channel that this came from, or Event number
        ('Unit', c_short),                       # Unit classification, or Event strobe value
        ('DataType', c_char),                   # reserved
        ('NumberOfBlocksPerWaveform', c_char),  # reserved
        ('BlockNumberForWaveform', c_char),     # reserved
        ('NumberOfDataWords', c_char),          # number of shorts (2-byte integers) that follow this header 
        ('Waveform', MAX_WF_LENGTH_LONG*c_short), #
    ]
    
    


#######################################/
# Plexon continuous data file (.DDT) File Structure Definitions
#######################################/

LATEST_DDT_FILE_VERSION=103

class DigFileHeader(StructureWrapper):
    _fields_=[
        ('Version', c_int),        # Version of the data format; determines which data items are valid
        ('DataOffset',c_int),     # Offset into the file where the data starts
        ('Freq', c_double),           # Digitization frequency
        ('NChannels', c_int),      # Number of recorded channels; for version 100-101, this will always
        # be the same as the highest channel number recorded; for versions >= 102,
        # NChannels is the same as the number of enabled channels, i.e. channels
        # whose entry in the ChannelGain array is not 255.
        
        ('Year', c_int),      # Time/date when the data was acquired
        ('Month', c_int), 
        ('Day', c_int),  
        ('Hour', c_int),
        ('Minute', c_int), 
        ('Second', c_int), 
        
        ('Gain', c_int),           # As of Version 102, this is the *preamp* gain, not ADC gain
        ('Comment', 128*c_char),   # User-supplied comment 
        ('BitsPerSample', c_ubyte),    # ADC resolution, usually either 12 or 16. Added for ddt Version 101    
        ('ChannelGain', 64*c_ubyte),  # Gains for each channel; 255 means channel was disabled (not recorded). 
        # The gain for Channel n is located at ChannelGain[n-1]
        # Added for ddt Version 102 
        ('Unused', c_ubyte),           # padding to restore alignment 
        ('MaxMagnitudeMV', c_short),   # ADC max input voltage in millivolts: 5000 for NI, 2500 for ADS64
        # Added for ddt version 103
        ('Padding', 188*c_ubyte),
    ]




