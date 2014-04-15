#include "mtime.h"

int64_t GetTimeMs64() {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        uint64_t ret = tv.tv_usec;

        /* Convert from micro seconds (10^-6) to milliseconds (10^-3) */
        ret /= 1000;

        /* Adds the seconds (10^0) after converting them to milliseconds (10^-3) */
        ret += (tv.tv_sec * 1000);

        return ret;
}

int64_t GetTimeMsFrom(int64_t start) {
        return GetTimeMs64() - start;
}

int64_t GetTimeMius64() {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        uint64_t ret = tv.tv_usec;

        /* Adds the seconds (10^0) after converting them to milliseconds (10^-3) */
        ret += (tv.tv_sec * 1000000);

        return ret;
}

int64_t GetTimeMiusFrom(int64_t start) {
        return GetTimeMius64() - start;
}
