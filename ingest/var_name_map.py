# ingest/var_name_map.py
# small mapping to normalize common Argo variable names to our canonical names

VAR_MAP = {
    "PLATFORM_NUMBER": "id",
    "platform_number": "id",
    "PLATFORM": "id",
    "WMO_PLATFORM_NUMBER": "id",

    "JULD": "date",
    "JULD_LOCATION": "date",
    "time": "date",
    "TIME": "date",

    "LATITUDE": "lat",
    "latitude": "lat",
    "LAT": "lat",

    "LONGITUDE": "lon",
    "longitude": "lon",
    "LON": "lon",

    "PRES": "depth",
    "PRES_RELATIVE": "depth",
    "pressure": "depth",
    "PRESSURE": "depth",

    "TEMP": "temperature",
    "TEMP_ADJUSTED": "temperature",
    "temp": "temperature",
    "temperature": "temperature",

    "PSAL": "salinity",
    "PSAL_ADJUSTED": "salinity",
    "salinity": "salinity",
}
