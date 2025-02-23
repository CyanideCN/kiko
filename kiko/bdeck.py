#https://gist.github.com/crazyapril/8c2303f539aa5e1b71d09da8744eada0

import datetime


class AttrDict(dict):

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def _safe_int(s):
    try:
        return int(s)
    except ValueError:
        return -999


class BDeckFile:
    """Reader for BDeck format files.
    You could access data step by step or once for all. For data in a single
    timepoint, access `idata` attribute with `iter_data` method. For data in
    the entire file, access `data` attribute with `read_all` method. It also
    includes an attribute named `metadata` for some extra information. If all
    data is read into memory (you could check the `_fullyread` attribute),
    use `export_dataframe` method to get a `DataFrame` instance for further
    exploitation.
    Note BDeck format has two types: short type (used before 2005) and long
    type (used after 2005). You could check `self.idata._long_format` to find
    the exact type while reading. The short type has less information. The
    reader supports following entries for short type:
    - time, a `datatime.datetime` instance
    - timestr, a string of time in the format of 'YYmmddHH'
    - basin, the basin code for the storm
    - number, the number assigned by the agency
    - technum, see JTWC website (listed below) for more information about
        technum, techcode and tau
    - techcode
    - tau
    - lat, latitude of the storm
    - lon, longitude of the storm
    - wind, wind speed in knots
    - category, best category of the storm, see docs in `_get_category` method
        for more information
    - raw_category, raw category of the storm in the file, empty for short type
    And long type has more entries:
    - pres, pressure in millibars
    - r34, radii (nm) of wind above 34kt. None or a tuple of numbers, in the
        order of NE, SE, SW and NW quadrants
    - r50, radii (nm) of wind above 50kt
    - r64, radii (nm) of wind above 64kt
    - lci, least closed isobar, pressure in millibars
    - lci_radius, radius (nm) of least closed isobar
    - rmw, radius (nm) of maximum wind
    - name, name of the storm
    - depth, system depth, see JTWC website for details
    You could access these entries easily in `data` and `idata` attribute.
    Get more information about BDeck format and meanings of entries in JTWC
    website: https://www.metoc.navy.mil/jtwc/jtwc.html?western-pacific.
    `BDeckFile` also has a `metadata` attribute to provide extra information.
    It includes following entries:
    - maxwind, maximum wind speed in the lifetime
    - minpres, minimum air pressure in the lifetime, invalid for short type
    - peaktime, timepoints when wind speed of system is at its peak
    - fullcode, a long code for storm
    - name, name of storm when system is at its peak, invalid for short type
    It should be noted that `metadata` would only be meaningful after all the
    data is read.
    Example:
    ```
    # Recommended way to open a BDeck file
    with BDeckFile(filename) as f:
        for time in f.iter_data():
            print(time)
            print(f.idata.wind)
            print(f.idata['wind']) # both are acceptable
            print(f.idata.r34) # Only valid for long type
        # now all data is read
        print(f.data.wind) # Get a list of wind
    # Another way to open a BDeck file
    f = BDeckFile(filename2)
    f.open() # required!
    # Read all data in once, exclude non-tropical data
    f.read_all(tropical_nature=True)
    print(f.data['wind'])
    print(f.metadata['peaktime'])
    # Use pandas for further use, you need read all data first
    df = f.export_dataframe()
    # select data when the storm is a C4 or C5
    segment = df.loc[df['category'].isin(('C4', 'C5'))]
    # select data when the storm is located above 15N
    segment = df.loc[df['lat'] >= 15]
    # calculate ACE
    ace = (df[df.wind >= 35].wind ** 2).sum() / 1e4
    f.close() # don't forget!
    ```
    """

    def __init__(self, filename):
        """Initiating a BDeckFile instance by feeding the filename of BDeck
        file.
        Parameters
        ----------
        filename : string
            Full filename of BDeck file.
        """
        self.filename = filename
        self.file = None
        self._opened = False
        self._fullyread = False
        self.data = None
        self.idata = None
        self.metadata = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, trace):
        self.close()

    def open(self):
        """Open the given BDeck file."""
        if not self._opened:
            self.file = open(self.filename)
        self._opened = True
        self.data = AttrDict()
        self.idata = AttrDict()
        self.metadata = AttrDict()
        self.metadata.update({
            'maxwind': 0,
            'minpres': 9999,
            'peaktime': [],
            'fullcode': '',
            'name': ''
        })

    def reset(self):
        """Reset the file handle for further use."""
        if self._opened:
            self.file.seek(0, 0)

    def clear(self):
        """Clear data already read and then reset."""
        self.data = AttrDict()
        self.idata = AttrDict()
        self.metadata = AttrDict()
        self.reset()
        self._fullyread = False

    def close(self):
        """Close the given BDeck file."""
        if self._opened:
            self.file.close()
        self._opened = False

    def _get_category(self, wind, category=None):
        """Get best category classification from wind and raw category
        info in the file. If no raw category is given, category is
        assigned by SSHWS classification. Otherwise, categories will be
        aligned with raw info except for TY/HU/ST/MH categories, which
        would be reclassified using SSHWS.
        Example:
        ``` pyshell
        >>> self._get_category(45)
        >>> 'TS'
        >>> self._get_category(85)
        >>> 'C2'
        >>> self._get_category(85, 'EX')
        >>> 'EX'
        >>> self._get_category(85, 'TY')
        >>> 'C2'
        ```
        Parameters
        ----------
        wind : int
            Wind speed in knots.
        category : string, optional
            Raw category information in the file, by default None
        """
        if category not in (None, 'TY', 'HU', 'ST', 'MH'):
            return category
        if wind > 137:
            cat = 'C5'
        elif wind > 114:
            cat = 'C4'
        elif wind > 96:
            cat = 'C3'
        elif wind > 83:
            cat = 'C2'
        elif wind > 64:
            cat = 'C1'
        elif wind > 34:
            cat = 'TS'
        else:
            cat = 'TD'
        return cat

    def read_all(self, formal_advisory=True, tropical_nature=False):
        """Read all data in the file. You can access all the information in
        `self.data` and also `self.metadata`.

        Parameters
        ----------
        formal_advisory : bool, optional
            Only reads data from formal advisories, i.e. 00z, 06z, 12z and
            18z, by default True
        tropical_nature : bool, optional
            Only reads data when the cyclone has tropical nature, dismisses
            subtropical and extratropical data, by default False
        """
        self.read_data(formal_advisory=formal_advisory,
                tropical_nature=tropical_nature)
        return self.data

    def read_data(self, formal_advisory=True, tropical_nature=False):
        """Get next slice of data in the BDeck file by iteration. After
        iterating one step, you can access data in the `idata` attribute.
        
        Parameters
        ----------
        formal_advisory : bool, optional
            Only reads data from formal advisories, i.e. 00z, 06z, 12z and
            18z, by default True
        tropical_nature : bool, optional
            Only reads data when the cyclone has tropical nature, dismisses
            subtropical and extratropical data, by default False
        """
        if not self._opened:
            raise IOError('The file has not been opened!')
        lines = self.file.readlines()
        count = len(lines)
        i = 0
        while i < count:
            line = lines[i]
            linesegs = line.split(',')
            _long_format = len(linesegs) > 20
            # check if it is formal advisories
            if formal_advisory and linesegs[2][-2:] not in ('00', '06', '12', '18'):
                i += 1
                continue
            # check if it has tropical nature
            if _long_format and tropical_nature and \
                    linesegs[10].strip() in ('SS', 'SD', 'EX'):
                i += 1
                continue
            self.idata['_long_format'] = _long_format
            self.idata['basin'] = linesegs[0]
            self.idata['number'] = _safe_int(linesegs[1])
            timestr = linesegs[2].strip()
            self.idata['timestr'] = timestr
            self.idata['time'] = datetime.datetime.strptime(timestr, '%Y%m%d%H')
            self.idata['technum'] = linesegs[3].strip()
            self.idata['techcode'] = linesegs[4].strip()
            self.idata['tau'] = _safe_int(linesegs[5])
            lat = _safe_int(linesegs[6][:-1]) / 10
            if linesegs[6][-1] == 'S':
                lat = -lat
            self.idata['lat'] = lat
            lon = _safe_int(linesegs[7][:-1]) / 10
            if linesegs[7][-1] == 'W':
                lon = -lon
            self.idata['lon'] = lon
            wind = _safe_int(linesegs[8])
            self.idata['wind'] = wind
            self.idata['category'] = self._get_category(wind)
            self.idata['raw_category'] = ''
            if len(linesegs) > 9:
                self.idata['pres'] = _safe_int(linesegs[9])
                tmp_str = linesegs[10].strip()
                self.idata['category'] = self._get_category(wind, tmp_str)
                self.idata['raw_category'] = tmp_str
            if _long_format:
                self.idata['r34'] = None
                self.idata['r50'] = None
                self.idata['r64'] = None
                self.idata['lci'] = _safe_int(linesegs[17]) # last closed isobar
                self.idata['lci_radius'] = _safe_int(linesegs[18])
                self.idata['rmw'] = _safe_int(linesegs[19])
                if len(linesegs) > 28:
                    self.idata['name'] = linesegs[27].strip()
                    self.idata['depth'] = linesegs[28].strip()
                if wind > 34:
                    self.idata['r34'] = tuple(map(int, linesegs[13:17]))
                if wind >= 50:
                    i += 1
                    if i == count:
                        break
                    line = lines[i]
                    linesegs = line.split(',')
                    if linesegs[2].strip() != timestr:
                        continue
                    self.idata['r50'] = tuple(map(int, linesegs[13:17]))
                if wind > 64:
                    i += 1
                    if i == count:
                        break
                    line = lines[i]
                    linesegs = line.split(',')
                    if linesegs[2].strip() != timestr:
                        continue
                    self.idata['r64'] = tuple(map(int, linesegs[13:17]))
            self._record_all()
            self._record_meta()
            i += 1
            #print(line)
        self._fullyread = True

    def _record_all(self):
        """Write single time information into `self.data`."""
        if self._fullyread:
            return
        for key, value in self.idata.items():
            if key not in self.data:
                self.data[key] = [value]
            else:
                self.data[key].append(value)

    def _record_meta(self):
        """Write metadata into `self.metadata`. Note: metadata would not
        be meaningful if data is not fully read. Currently, we record five
        entries of metadata: `maxwind`, `minpres`, `peaktime`, `fullcode`,
        and `name`.
        """
        if self._fullyread:
            return
        if self.idata['wind'] > self.metadata['maxwind']:
            self.metadata['maxwind'] = self.idata['wind']
            self.metadata['peaktime'] = [self.idata['time']]
            if 'name' in self.idata:
                self.metadata['name'] = self.idata['name']
        elif self.idata['wind'] == self.metadata['maxwind']:
            self.metadata['peaktime'].append(self.idata['time'])
        if 'pres' in self.idata and self.idata['pres'] < self.metadata['minpres']:
            self.metadata['minpres'] = self.idata['pres']
        self.metadata['fullcode'] = '{}{:02d}'.format(self.idata['basin'],
            self.idata['number'])