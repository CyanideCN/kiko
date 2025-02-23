from datetime import datetime

def datetime_to_mjd(dt: datetime) -> float:
    """
    Convert a datetime object to Modified Julian Date (MJD).
    MJD is defined as JD - 2400000.5
    
    Args:
        dt (datetime): Python datetime object
        
    Returns:
        float: Modified Julian Date
    """
    # First calculate the Julian Date
    year = dt.year
    month = dt.month
    day = dt.day
    
    # If January or February, treat as 13th or 14th month of previous year
    if month <= 2:
        year -= 1
        month += 12
    
    # Julian Date calculation
    a = year // 100
    b = 2 - a + (a // 4)
    
    jd = (int(365.25 * (year + 4716)) +
          int(30.6001 * (month + 1)) +
          day + b - 1524.5)
    
    # Add time component
    jd += (dt.hour + dt.minute/60.0 + dt.second/3600.0 + dt.microsecond/3600000000.0) / 24.0
    
    # Convert to MJD
    return jd - 2400000.5

def mjd_to_datetime(mjd: float) -> datetime:
    """
    Convert Modified Julian Date (MJD) to datetime object.
    
    Args:
        mjd (float): Modified Julian Date
        
    Returns:
        datetime: Python datetime object
    """
    # Convert MJD to JD
    jd = mjd + 2400000.5
    
    # Separate the integer and decimal parts
    jd_int = int(jd)
    frac = jd - jd_int + 0.5
    
    if frac >= 1.0:
        jd_int += 1
        frac -= 1.0
    
    # Convert fractional day to hours, minutes, seconds, microseconds
    hours = int(frac * 24)
    frac = frac * 24 - hours
    minutes = int(frac * 60)
    frac = frac * 60 - minutes
    seconds = int(frac * 60)
    frac = frac * 60 - seconds
    microseconds = int(frac * 1000000)
    
    # Convert Julian Date to calendar date
    l = jd_int + 68569
    n = (4 * l) // 146097
    l = l - (146097 * n + 3) // 4
    i = (4000 * (l + 1)) // 1461001
    l = l - (1461 * i) // 4 + 31
    j = (80 * l) // 2447
    day = l - (2447 * j) // 80
    l = j // 11
    month = j + 2 - 12 * l
    year = 100 * (n - 49) + i + l
    
    return datetime(year, month, day, hours, minutes, seconds, microseconds)


def find_overlaps(intervals: list[tuple[datetime, datetime]]) -> list[tuple[datetime, datetime, list[int]]]:
    # Create events: each event is a tuple (time, event_type, index)
    events = []
    for i, (start, end) in enumerate(intervals):
        events.append((start, 'start', i))
        events.append((end, 'end', i))
    
    # Sort events by time; if times are equal, start events come first.
    events.sort(key=lambda x: (x[0], 0 if x[1] == 'start' else 1))
    
    active = set()  # active set of interval indices
    overlaps = []   # to store output: (overlap_start, overlap_end, [indices])
    prev_time = None

    for time, event_type, idx in events:
        # If we have moved time and at least two intervals are active,
        # record the overlapping segment.
        if prev_time is not None and time > prev_time and len(active) >= 2:
            overlaps.append((prev_time, time, sorted(active)))
        
        # Process the current event
        if event_type == 'start':
            active.add(idx)
        else:  # event_type == 'end'
            active.remove(idx)
        
        prev_time = time

    return overlaps
