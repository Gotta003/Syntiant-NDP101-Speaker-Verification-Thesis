#ifndef SYNTIANT_FIX_H
#define SYNTIANT_FIX_H

#if defined(ARDUINO_SAMD_MKRZERO)
#undef Serial
#define Serial SERIAL_PORT_MONITOR
#endif

#endif
