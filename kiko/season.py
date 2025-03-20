from calendar import isleap
import datetime

import numpy as np

from kiko.storm import Storm
from kiko.utils import datetime_to_mjd, find_overlaps

class SeasonDataset(object):

    def __init__(self, storms: list[Storm]):
        self.season_dict: dict[int, list[Storm]] = dict()
        self.storm_dict: dict[str, Storm] = dict()
        for s in storms:
            if s.atcf_season not in self.season_dict:
                self.season_dict[s.atcf_season] = list()
            self.season_dict[s.atcf_season].append(s)
            self.storm_dict[s.full_atcf_id] = s

    @classmethod
    def from_bdeck(cls, file_list: list[str]):
        storms = [Storm.from_bdeck(i) for i in file_list]
        return cls(storms)

    def daily_ace(self, year: int, push_leap_day: bool = False, basin=None):
        # Check year itself, year before and after
        for _yr in [year - 1, year, year + 1]:
            if _yr in self.season_dict:
                break
        else:
            return ValueError(f'Year {year} not in dataset')
        # TODO: Find a better way to handle missing years
        data_size = 366 if isleap(year) else 365
        ace = np.zeros(data_size)
        for _yr in [year - 1, year, year + 1]:
            if _yr not in self.season_dict:
                continue
            for storm in self.season_dict[_yr]:
                for mjd, basin_ace in storm.daily_ace.items():
                    day_of_year = int(mjd - datetime_to_mjd(datetime.datetime(year, 1, 1)))
                    if 0 <= day_of_year < data_size:
                        if basin is None:
                            ace[day_of_year] += basin_ace.total
                        else:
                            ace[day_of_year] += basin_ace.get(basin)
        if push_leap_day and isleap(year):
            ace[59] += ace[60]  # Add 2/29 data to 3/1
            ace = np.delete(ace, 60)  # Remove 2/29
        return ace

    def cumulative_ace(self, year: int, push_leap_day: bool = False, basin=None):
        return np.cumsum(self.daily_ace(year, push_leap_day=push_leap_day, basin=basin))

    def overlapping_storm(self, year: int, tropical=True, basin=None):
        storm_list = self.season_dict[year]
        if tropical:
            valid_storm_list = list()
            storm_periods = list()
            for i in storm_list:
                if i.start_time_tropical and i.end_time_tropical:
                    storm_periods.append((i.start_time_tropical, i.end_time_tropical))
                    valid_storm_list.append(i)
        else:
            storm_periods = [(i.start_time, i.end_time) for i in storm_list]
            valid_storm_list = storm_list
        overlaps = find_overlaps(storm_periods)
        if len(overlaps) == 0:
            return []
        # Convert index to storm atcf code
        storm_overlap = list()
        for p in overlaps:
            storm_idx = p[2]
            storm_atcf = [valid_storm_list[i].full_atcf_id for i in storm_idx]
            storm_overlap.append((p[0], p[1], storm_atcf))
        return storm_overlap

    def get_storm(self, atcf_id: str):
        return self.storm_dict.get(atcf_id)