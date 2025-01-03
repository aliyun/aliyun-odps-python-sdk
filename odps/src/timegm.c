#include <stdint.h>
#include <time.h>
// Algorithm: http://howardhinnant.github.io/date_algorithms.html

int32_t days_from_epoch(int y, int m, int d)
{
    y -= m <= 2;
    int era = y / 400;
    int yoe = y - era * 400;                                   // [0, 399]
    int doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;  // [0, 365]
    int doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;           // [0, 146096]
    return era * 146097 + doe - 719468;
}

// It does not modify broken-down time
time_t timegm(struct tm const* t)
{
    int year = t->tm_year + 1900;
    int month = t->tm_mon;          // 0-11
    if (month > 11)
    {
        year += month / 12;
        month %= 12;
    }
    else if (month < 0)
    {
        int years_diff = (11 - month) / 12;
        year -= years_diff;
        month += 12 * years_diff;
    }
    int32_t days_since_epoch = days_from_epoch(year, month + 1, t->tm_mday);

    return 60 * (60 * (24L * days_since_epoch + t->tm_hour) + t->tm_min) + t->tm_sec;
}

struct tm* gmtime_safe(const time_t* timer, struct tm* buf)
{
#if defined(_WIN32)
    struct tm* ptr;
    // gmtime is guaranteed as threadsafe in Windows
    ptr = gmtime(timer);
    *buf = *ptr;
    return ptr;
#else
    return gmtime_r(timer, buf);
#endif
}

struct tm* localtime_safe(const time_t* timer, struct tm* buf)
{
#if defined(_WIN32)
    struct tm* ptr;
    // localtime is guaranteed as threadsafe in Windows
    ptr = localtime(timer);
    *buf = *ptr;
    return ptr;
#else
    return localtime_r(timer, buf);
#endif
}
