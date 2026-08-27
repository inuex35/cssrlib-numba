"""GNSS time scales.

gtime_t and the conversions between GPS, Galileo, BeiDou and UTC, plus the
latitude interpolation the tropospheric mapping needs."""

from math import floor
import numpy as np
from datetime import datetime, timezone

from cssrlib.domain.enums import *  # noqa: F401,F403


class gtime_t():
    """ class to define the time """

    def __init__(self, time=0, sec=0.0):
        self.time = time
        self.sec = sec

    def __gt__(self, other):
        return self.time > other.time or \
            (self.time == other.time and self.sec > other.sec)

def epoch2time(ep):
    """ calculate time from epoch """
    doy = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    time = gtime_t()
    year = int(ep[0])
    mon = int(ep[1])
    day = int(ep[2])

    if year < 1970 or year > 2099 or mon < 1 or mon > 12:
        return time
    days = (year-1970)*365+(year-1969)//4+doy[mon-1]+day-2
    if year % 4 == 0 and mon >= 3:
        days += 1
    sec = int(ep[5])
    time.time = days*86400+int(ep[3])*3600+int(ep[4])*60+sec
    time.sec = ep[5]-sec
    return time


leaps_ = [[2017, 1, 1, 0, 0, 0, -18],
          [2015, 7, 1, 0, 0, 0, -17],
          [2012, 7, 1, 0, 0, 0, -16],
          [2009, 1, 1, 0, 0, 0, -15],
          [2006, 1, 1, 0, 0, 0, -14],
          [1999, 1, 1, 0, 0, 0, -13],
          [1997, 7, 1, 0, 0, 0, -12],
          [1996, 1, 1, 0, 0, 0, -11],
          [1994, 7, 1, 0, 0, 0, -10],
          [1993, 7, 1, 0, 0, 0, -9],
          [1992, 7, 1, 0, 0, 0, -8],
          [1991, 1, 1, 0, 0, 0, -7],
          [1990, 1, 1, 0, 0, 0, -6],
          [1988, 1, 1, 0, 0, 0, -5],
          [1985, 7, 1, 0, 0, 0, -4],
          [1983, 7, 1, 0, 0, 0, -3],
          [1982, 7, 1, 0, 0, 0, -2],
          [1981, 7, 1, 0, 0, 0, -1]]


def timeget():
    """ return current time in UTC """
    now = datetime.now(timezone.utc)
    ep = np.array([now.year, now.month, now.day, now.hour, now.minute,
                   now.second])
    return epoch2time(ep)


def gpst2utc(t: gtime_t):
    for i in range(len(leaps_)):
        tu = timeadd(t, leaps_[i][6])
        if timediff(tu, epoch2time(leaps_[i])) >= 0.0:
            return tu
    return t


def utc2gpst(t: gtime_t):
    for i in range(len(leaps_)):
        if timediff(t, epoch2time(leaps_[i])) >= 0.0:
            return timeadd(t, -leaps_[i][6])
    return t


def timeadd(t: gtime_t, sec: float):
    """ return time added with sec """
    new_sec = t.sec + sec
    tt = floor(new_sec)
    return gtime_t(t.time + int(tt), new_sec - tt)


def timediff(t1: gtime_t, t2: gtime_t):
    """ return time difference """
    dt = t1.time-t2.time
    dt += t1.sec-t2.sec
    return dt


def gpst2time(week, tow):
    """ convert to time from gps-time """
    t = epoch2time(gpst0)
    if tow < -1e9 or tow > 1e9:
        tow = 0.0
    t.time += 86400*7*week+int(tow)
    t.sec = tow-int(tow)
    return t


def time2gpst(t: gtime_t):
    """ convert to gps-time from time """
    t0 = epoch2time(gpst0)
    sec = t.time-t0.time
    week = int(sec/(86400*7))
    tow = sec-week*86400*7+t.sec
    return week, tow


def gst2time(week, tow):
    """ convert to time from galileo system time """
    t = epoch2time(gst0)
    if tow < -1e9 or tow > 1e9:
        tow = 0.0
    t.time += 86400*7*week+int(tow)
    t.sec = tow-int(tow)
    return t


def time2gst(t: gtime_t):
    """ convert to galileo system time from time """
    t0 = epoch2time(gst0)
    sec = t.time-t0.time
    week = int(sec/(86400*7))
    tow = sec-week*86400*7+t.sec
    return week, tow


def bdt2time(week, tow):
    """ convert to time from BeiDou system time """
    t = epoch2time(bdt0)
    if tow < -1e9 or tow > 1e9:
        tow = 0.0
    t.time += 86400*7*week+int(tow)
    t.sec = tow-int(tow)
    return t


def bdt2gpst(t: gtime_t):
    """ convert from BeiDou system time to GPS time  """
    return timeadd(t, 14.0)


def gpst2bdt(t: gtime_t):
    """ convert to GPS time from BeiDou system time """
    return timeadd(t, -14.0)


def time2bdt(t: gtime_t):
    """ convert to BeiDou system time from time """
    t0 = epoch2time(bdt0)
    sec = t.time-t0.time
    week = int(sec/(86400*7))
    tow = sec-week*86400*7+t.sec
    return week, tow


def time2epoch(t):
    """ convert time to epoch """
    mday = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 31, 28, 31, 30, 31,
            30, 31, 31, 30, 31, 30, 31, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31,
            30, 31, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    days = int(t.time/86400)
    sec = int(t.time-days*86400)
    day = days % 1461
    for mon in range(48):
        if day >= mday[mon]:
            day -= mday[mon]
        else:
            break
    ep = [0, 0, 0, 0, 0, 0]
    ep[0] = 1970+days//1461*4+mon//12
    ep[1] = mon % 12+1
    ep[2] = day+1
    ep[3] = sec//3600
    ep[4] = sec % 3600//60
    ep[5] = sec % 60+t.sec
    return ep


def time2doy(t):
    """ convert time to day of year (as float value!) """
    ep = time2epoch(t)
    ep[1] = ep[2] = 1.0
    ep[3] = ep[4] = ep[5] = 0.0
    return timediff(t, epoch2time(ep))/86400+1


def interpc(coef, lat):
    """ linear interpolation (lat step=15) """
    i = int(lat/15.0)
    m = coef.shape[1]-1
    if i < 1:
        return coef[:, 0]
    if i > m:
        return coef[:, m]
    d = lat/15.0-i
    return coef[:, i-1]*(1.0-d)+coef[:, i]*d


def str2time(s, i, n):
    """ string to time conversion """
    if i < 0 or len(s) < i:
        return -1
    ep = np.array([float(x) for x in s[i:i+n].split()])
    if len(ep) < 6:
        return -1
    if ep[0] < 100.0:
        ep[0] += 2000.0 if ep[0] < 80.0 else 1900.0
    return epoch2time(ep)


def time2str(t):
    """ time to string conversion """
    e = time2epoch(t)
    return "{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}"\
        .format(e[0], e[1], e[2], e[3], e[4], int(e[5]))


def adjtime(t: gtime_t, tref: gtime_t, dt=rCST.WEEK_SEC):
    """ adjust time for week (day) rollover """
    tt = timediff(t, tref)
    if tt < -dt/2.0:
        return timeadd(t, dt)
    if tt > dt/2.0:
        return timeadd(t, -dt)
    return t
