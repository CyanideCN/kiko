import datetime
import enum
from dataclasses import dataclass
from functools import cached_property

import pytz
import numpy as np
from shapely.geometry import LineString, Polygon, Point
from scipy.interpolate import interp1d

from kiko.bdeck import BDeckFile
from kiko.utils import datetime_to_mjd, movement

TROPICAL_TYPE = set(['TD', 'TS', 'TY', 'HU', 'ST'])
NORTHERN_HEMISPHERE_BASIN = set(['WP', 'EP', 'CP', 'AL', 'IO'])

class Basin(enum.IntEnum):
    WPAC = 0
    EPAC = 1
    NIO = 2
    SHEM = 3
    ATL = 4

@dataclass
class BasinACE:
    wpac: float = 0.
    epac: float = 0.
    nio: float = 0.
    shem: float = 0.
    atl: float = 0.

    @property
    def total(self):
        return self.wpac + self.epac + self.nio + self.shem + self.atl

    def set(self, basin: Basin, value):
        match basin:
            case Basin.WPAC:
                self.wpac += value
            case Basin.EPAC:
                self.epac += value
            case Basin.NIO:
                self.nio += value
            case Basin.SHEM:
                self.shem += value
            case Basin.ATL:
                self.atl += value
            case _:
                raise ValueError('Invalid basin')

    def get(self, basin: Basin):
        return [self.wpac, self.epac, self.nio, self.shem, self.atl][basin]

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
            self._tropical_flag = np.ones_like(self.longitude, dtype=bool) # Assume all tropical
        else:
            self.storm_type = np.array(storm_type)
            self._tropical_flag = [is_tropical(i) for i in self.storm_type]

        self.atcf_id = atcfid # e.g. WP01
        if name:
            self.metadata['name'] = name
        self.mjd = np.array([datetime_to_mjd(i) for i in self.time])
        self._synoptic_flag = [is_synoptic(i) for i in self.time]
        self.heading, self.speed = movement(self.longitude, self.latitude, self.mjd)

        self.flags = {'continuous': True, 'interpolated': False, 'subset': False}

    @classmethod
    def from_bdeck(cls, bdeck_path):
        bdeck = BDeckFile(bdeck_path)
        bdeck.open()
        data = bdeck.read_all(formal_advisory=False)
        stype = np.array(data.get('raw_category'))
        if stype is not None:
            stype = stype[stype != '']
            if stype.size == 0:
                stype = None
        storm = cls(bdeck.metadata['fullcode'], data['time'], data['lon'],
                    data['lat'], data['wind'], data.get('pres', None), stype,
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
            data[mjd].set(basin, _ace)
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
        if self.atcf_number % 60 < 3: # For extra ATCF numbering starting from 60
            # Named after year crossover
            # Assume at most 2 year crossover storms per basin
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

    @cached_property
    def _geom(self):
        return LineString(np.array([self.longitude, self.latitude]))

    def sel_by_bbox(self, bbox: Polygon):
        cont_flag = True
        sel_idx = []
        bbox_buffered = bbox.buffer(0.01)
        for i, (x, y) in enumerate(zip(self.longitude, self.latitude)):
            if bbox_buffered.contains(Point(x, y)):
                if cont_flag and len(sel_idx) != 0:
                    if (i - sel_idx[-1]) > 1:
                        cont_flag = False
                sel_idx.append(i)
        if not sel_idx:
            return None
        sel_time = [self.time[i] for i in sel_idx]
        sel_lon = self.longitude[sel_idx]
        sel_lat = self.latitude[sel_idx]
        sel_wind = self.wind[sel_idx]
        sel_pressure = self.pressure[sel_idx] if self.pressure is not None else None
        sel_storm_type = self.storm_type[sel_idx] if self.storm_type is not None else None
        s = Storm(self.atcf_id, sel_time, sel_lon, sel_lat, sel_wind, sel_pressure,
                  sel_storm_type, self.metadata.get('name', None))
        s.flags['continuous'] = cont_flag
        if len(sel_idx) != len(self.longitude):
            s.flags['subset'] = True
        return s

    def interp(self, hour_interval):
        if not self.flags['continuous']:
            raise ValueError('Cannot interpolate discontinuous data')
        if hour_interval <= 0:
            raise ValueError("hour_interval must be positive")

        new_times = np.arange(self.mjd[0], self.mjd[-1], hour_interval / 24.0)

        interp_lon = interp1d(self.mjd, self.longitude, kind='linear', fill_value="extrapolate")
        interp_lat = interp1d(self.mjd, self.latitude, kind='linear', fill_value="extrapolate")
        interp_wind = interp1d(self.mjd, self.wind, kind='linear', fill_value="extrapolate")

        new_longitude = interp_lon(new_times)
        new_latitude = interp_lat(new_times)
        new_wind = interp_wind(new_times)

        if self.pressure is not None:
            interp_pressure = interp1d(self.mjd, self.pressure, kind='linear', fill_value="extrapolate")
            new_pressure = interp_pressure(new_times)
        else:
            new_pressure = None

        if self.storm_type is not None:
            interp_storm_type = interp1d(self.mjd, self.storm_type, kind='nearest', fill_value="extrapolate")
            new_storm_type = interp_storm_type(new_times)
        else:
            new_storm_type = None

        new_time = [self.time[0] + datetime.timedelta(days=(mjd - self.mjd[0])) for mjd in new_times]

        s = Storm(self.atcf_id, new_time, new_longitude, new_latitude, new_wind, new_pressure, new_storm_type, self.metadata.get('name', None))
        s.flags['interpolated'] = True
        return s
