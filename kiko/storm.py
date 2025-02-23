import datetime
import enum
from dataclasses import dataclass
from functools import cached_property

import pytz
import numpy as np

from kiko.bdeck import BDeckFile
from kiko.utils import datetime_to_mjd

TROPICAL_TYPE = set(['TD', 'TS', 'TY', 'HU', 'ST'])
NORTHERN_HEMISPHERE_BASIN = set(['WP', 'EP', 'CP', 'AL', 'IO'])

@dataclass
class BasinACE:
    wpac: float = 0.
    epac: float = 0.
    nio: float = 0.
    shem: float = 0.
    atl: float = 0.

    @cached_property
    def total(self):
        return self.wpac + self.epac + self.nio + self.shem + self.atl

class Basin(enum.Enum):
    WPAC = 0
    EPAC = 1
    NIO = 2
    SHEM = 3
    ATL = 4

def ensure_utc(time: datetime.datetime):
    if not time.tzinfo:
        return pytz.utc.localize(time)
    if time.tzinfo == pytz.utc:
        return time
    return time.astimezone(pytz.utc)

def is_synoptic(time: datetime.datetime):
    return time.hour % 6 == 0

def is_tropical(stype: str):
    return stype in TROPICAL_TYPE

def get_basin(longitude: float, latitude: float) -> Basin:
    if latitude < 0:
        return Basin.SHEM
    if longitude < 100:
        if latitude < 40:
            return Basin.NIO
        else:
            if longitude < 70:
                return Basin.ATL
            else:
                return Basin.WPAC
    elif longitude < 180:
        return Basin.WPAC
    else:
        if longitude < 240:
            return Basin.EPAC
        elif longitude > 300:
            return Basin.ATL
        else:
            # Complex boundary between EPAC and NATL, return EPAC for now.
            return Basin.EPAC

class Storm(object):

    def __init__(self, atcfid, time, longitude, latitude, wind,
                 pressure=None, storm_type=None, name=None):
        self.metadata = dict()
        self.time = [ensure_utc(i) for i in time]
        self.longitude = np.array(longitude)
        self.latitude = np.array(latitude)
        self.wind = np.array(wind)
        if pressure is None:
            self.pressure = None
        else:
            self.pressure = np.array(pressure)
        if storm_type is None:
            self.storm_type = None
            self._tropical_flag = None
        else:
            self.storm_type = np.array(storm_type)
            self._tropical_flag = [is_tropical(i) for i in self.storm_type]

        self.atcf_id = atcfid # e.g. WP01
        if name:
            self.metadata['name'] = name
        self.mjd = np.array([datetime_to_mjd(i) for i in self.time])
        self._synoptic_flag = [is_synoptic(i) for i in self.time]

    @classmethod
    def from_bdeck(cls, bdeck_path):
        bdeck = BDeckFile(bdeck_path)
        bdeck.open()
        data = bdeck.read_all(formal_advisory=False)
        storm = cls(bdeck.metadata['fullcode'], data['time'], data['lon'],
                    data['lat'], data['wind'], data.get('pres', None), data.get('raw_category', None),
                    bdeck.metadata.get('name', None))
        bdeck.close()
        return storm

    @property
    def start_time(self):
        return self.time[0]

    @property
    def end_time(self):
        return self.time[-1]

    @cached_property
    def start_time_tropical(self):
        if self.storm_type is None:
            return self.start_time
        for index, stype in enumerate(self.storm_type):
            if stype in TROPICAL_TYPE:
                return self.time[index]
        return None

    @cached_property
    def end_time_tropical(self):
        if self.storm_type is None:
            return self.end_time
        for index in range(len(self.storm_type) - 1, -1, -1):
            if self.storm_type[index] in TROPICAL_TYPE:
                return self.time[index]
        return None

    @cached_property
    def max_wind(self):
        return max(self.wind)

    @property
    def atcf_basin(self):
        return self.atcf_id[:2]
    
    @property
    def atcf_number(self):
        return int(self.atcf_id[2:])

    @cached_property
    def daily_ace(self):
        # Group ace by day, then by basin
        data: dict[int, BasinACE] = dict()
        # Use MJD as key
        valid_flag = np.logical_and(self._tropical_flag, self._synoptic_flag)
        ace_flag = np.logical_and(valid_flag, self.wind >= 35)
        wind = self.wind[ace_flag]
        lon = self.longitude[ace_flag]
        lat = self.latitude[ace_flag]
        mjd_filtered = self.mjd[ace_flag]
        ace = (wind ** 2) / 10000
        for _ace, _lon, _lat, _mjd in zip(ace, lon, lat, mjd_filtered):
            basin = get_basin(_lon, _lat)
            mjd = int(_mjd)
            if mjd not in data:
                data[mjd] = BasinACE()
            match basin:
                case Basin.WPAC:
                    data[mjd].wpac += _ace
                case Basin.EPAC:
                    data[mjd].epac += _ace
                case Basin.NIO:
                    data[mjd].nio += _ace
                case Basin.SHEM:
                    data[mjd].shem += _ace
                case Basin.ATL:
                    data[mjd].atl += _ace
        return data

    @cached_property
    def total_ace(self):
        return sum([i.total for i in self.daily_ace.values()])
    
    @cached_property
    def atcf_season(self):
        start_year = self.start_time.year
        end_year = self.end_time.year
        if start_year == end_year:
            if self.atcf_basin in NORTHERN_HEMISPHERE_BASIN:
                return start_year
            if self.start_time.month >= 7:
                return start_year + 1
            return start_year
        # Start not eq end, year crossover
        if self.atcf_number < 3:
            # Named after year crossover
            # Assume at most 2 crossover storms per basin
            return end_year
        return start_year
        
    @property
    def full_atcf_id(self):
        '''Long-style ATCF ID (e.g.) WP012025'''
        return f'{self.atcf_id}{self.atcf_season}'
    
    @cached_property
    def tropical_interval(self):
        pass

    def get_interval(self):
        pass