# Sirius ClickBench Results

**Machine:** NVIDIA GH200 480GB (96GB HBM3, aarch64)  
**Dataset:** ClickBench hits.parquet — 99,997,497 rows  
**Build:** cudf-25.12-optimization branch, cuDF 25.12.0, CUDA sm_90  
**Command:** `bash run_official.sh` (GPU_CACHING=80GB, GPU_PROCESSING=40GB)  
**Note:** Cold run = first run per query (includes GPU buffer load). Warm = subsequent runs.  

**Summary:** 43 queries — GPU: 43, CPU fallback: 0  
**Sum of best warm times:** 1.155s  

| Q | Cold (s) | Warm1 (s) | Warm2 (s) | Best Warm (s) | Status | Query |
|---|----------|-----------|-----------|---------------|--------|-------|
| 0 | 0.012 | 0.000 | 0.001 | **0.001** | 🟢 GPU | `SELECT COUNT(*) FROM hits;` |
| 1 | 0.337 | 0.004 | 0.004 | **0.004** | 🟢 GPU | `SELECT COUNT(*) FROM hits WHERE AdvEngineID <> 0;` |
| 2 | 0.693 | 0.004 | 0.004 | **0.004** | 🟢 GPU | `SELECT SUM(AdvEngineID), COUNT(*), AVG(ResolutionWidth) FROM hits;` |
| 3 | 0.726 | 0.004 | 0.003 | **0.003** | 🟢 GPU | `SELECT AVG(UserID) FROM hits;` |
| 4 | 0.458 | 0.007 | 0.006 | **0.006** | 🟢 GPU | `SELECT COUNT(DISTINCT UserID) FROM hits;` |
| 5 | 1.935 | 0.011 | 0.009 | **0.009** | 🟢 GPU | `SELECT COUNT(DISTINCT SearchPhrase) FROM hits;` |
| 6 | 0.251 | 0.004 | 0.003 | **0.003** | 🟢 GPU | `SELECT MIN(EventDate), MAX(EventDate) FROM hits;` |
| 7 | 0.237 | 0.005 | 0.004 | **0.004** | 🟢 GPU | `SELECT AdvEngineID, COUNT(*) FROM hits WHERE AdvEngineID <> 0 GROUP BY AdvEngineID ORDER B` |
| 8 | 0.826 | 0.013 | 0.012 | **0.012** | 🟢 GPU | `SELECT RegionID, COUNT(DISTINCT UserID) AS u FROM hits GROUP BY RegionID ORDER BY u DESC L` |
| 9 | 1.189 | 0.096 | 0.095 | **0.095** | 🟢 GPU | `SELECT RegionID, SUM(AdvEngineID), COUNT(*) AS c, AVG(ResolutionWidth), COUNT(DISTINCT Use` |
| 10 | 1.506 | 0.006 | 0.006 | **0.006** | 🟢 GPU | `SELECT MobilePhoneModel, COUNT(DISTINCT UserID) AS u FROM hits WHERE MobilePhoneModel <> '` |
| 11 | 1.471 | 0.006 | 0.006 | **0.006** | 🟢 GPU | `SELECT MobilePhone, MobilePhoneModel, COUNT(DISTINCT UserID) AS u FROM hits WHERE MobilePh` |
| 12 | 1.621 | 0.015 | 0.015 | **0.015** | 🟢 GPU | `SELECT SearchPhrase, COUNT(*) AS c FROM hits WHERE SearchPhrase <> '' GROUP BY SearchPhras` |
| 13 | 2.032 | 0.022 | 0.020 | **0.020** | 🟢 GPU | `SELECT SearchPhrase, COUNT(DISTINCT UserID) AS u FROM hits WHERE SearchPhrase <> '' GROUP ` |
| 14 | 1.855 | 0.017 | 0.018 | **0.017** | 🟢 GPU | `SELECT SearchEngineID, SearchPhrase, COUNT(*) AS c FROM hits WHERE SearchPhrase <> '' GROU` |
| 15 | 0.484 | 0.013 | 0.012 | **0.012** | 🟢 GPU | `SELECT UserID, COUNT(*) FROM hits GROUP BY UserID ORDER BY COUNT(*) DESC LIMIT 10;` |
| 16 | 2.015 | 0.028 | 0.027 | **0.027** | 🟢 GPU | `SELECT UserID, SearchPhrase, COUNT(*) FROM hits GROUP BY UserID, SearchPhrase ORDER BY COU` |
| 17 | 2.016 | 0.028 | 0.025 | **0.025** | 🟢 GPU | `SELECT UserID, SearchPhrase, COUNT(*) FROM hits GROUP BY UserID, SearchPhrase LIMIT 10;` |
| 18 | 3.084 | 0.046 | 0.045 | **0.045** | 🟢 GPU | `SELECT UserID, extract(minute FROM EventTime) AS m, SearchPhrase, COUNT(*) FROM hits GROUP` |
| 19 | 0.432 | 0.004 | 0.004 | **0.004** | 🟢 GPU | `SELECT UserID FROM hits WHERE UserID = 435090932899640449;` |
| 20 | 9.083 | 0.041 | 0.041 | **0.041** | 🟢 GPU | `SELECT COUNT(*) FROM hits WHERE URL LIKE '%google%';` |
| 21 | 7.348 | 0.014 | 0.014 | **0.014** | 🟢 GPU | `SELECT SearchPhrase, MIN(URL), COUNT(*) AS c FROM hits WHERE URL LIKE '%google%' AND Searc` |
| 22 | 14.773 | 0.025 | 0.025 | **0.025** | 🟢 GPU | `SELECT SearchPhrase, MIN(URL), MIN(Title), COUNT(*) AS c, COUNT(DISTINCT UserID) FROM hits` |
| 23 | 61.883 | 0.065 | 0.066 | **0.065** | 🟢 GPU | `SELECT * FROM hits WHERE URL LIKE '%google%' ORDER BY EventTime LIMIT 10;` |
| 24 | 2.921 | 0.011 | 0.012 | **0.011** | 🟢 GPU | `SELECT SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY EventTime LIMIT 10;` |
| 25 | 1.566 | 0.010 | 0.010 | **0.010** | 🟢 GPU | `SELECT SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY SearchPhrase LIMIT 10;` |
| 26 | 2.092 | 0.009 | 0.009 | **0.009** | 🟢 GPU | `SELECT SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY EventTime, SearchPhrase LI` |
| 27 | 10.269 | 0.134 | 0.135 | **0.134** | 🟢 GPU | `SELECT CounterID, AVG(STRLEN(URL)) AS l, COUNT(*) AS c FROM hits WHERE URL <> '' GROUP BY ` |
| 28 | 6.816 | 0.242 | 0.240 | **0.240** | 🟢 GPU | `SELECT REGEXP_REPLACE(Referer, '^https?://(?:www\.)?([^/]+)/.*$', '\1') AS k, AVG(STRLEN(R` |
| 29 | 0.519 | 0.034 | 0.033 | **0.033** | 🟢 GPU | `SELECT SUM(ResolutionWidth), SUM(ResolutionWidth + 1), SUM(ResolutionWidth + 2), SUM(Resol` |
| 30 | 2.801 | 0.010 | 0.009 | **0.009** | 🟢 GPU | `SELECT SearchEngineID, ClientIP, COUNT(*) AS c, SUM(IsRefresh), AVG(ResolutionWidth) FROM ` |
| 31 | 3.423 | 0.015 | 0.015 | **0.015** | 🟢 GPU | `SELECT WatchID, ClientIP, COUNT(*) AS c, SUM(IsRefresh), AVG(ResolutionWidth) FROM hits WH` |
| 32 | 1.413 | 0.064 | 0.063 | **0.063** | 🟢 GPU | `SELECT WatchID, ClientIP, COUNT(*) AS c, SUM(IsRefresh), AVG(ResolutionWidth) FROM hits GR` |
| 33 | 5.742 | 0.055 | 0.054 | **0.054** | 🟢 GPU | `SELECT URL, COUNT(*) AS c FROM hits GROUP BY URL ORDER BY c DESC LIMIT 10;` |
| 34 | 5.725 | 0.057 | 0.057 | **0.057** | 🟢 GPU | `SELECT 1, URL, COUNT(*) AS c FROM hits GROUP BY 1, URL ORDER BY c DESC LIMIT 10;` |
| 35 | 0.425 | 0.020 | 0.019 | **0.019** | 🟢 GPU | `SELECT ClientIP, ClientIP - 1, ClientIP - 2, ClientIP - 3, COUNT(*) AS c FROM hits GROUP B` |
| 36 | 6.153 | 0.007 | 0.007 | **0.007** | 🟢 GPU | `SELECT URL, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07` |
| 37 | 7.340 | 0.006 | 0.006 | **0.006** | 🟢 GPU | `SELECT Title, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-` |
| 38 | 6.169 | 0.005 | 0.004 | **0.004** | 🟢 GPU | `SELECT URL, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07` |
| 39 | 13.892 | 0.009 | 0.009 | **0.009** | 🟢 GPU | `SELECT TraficSourceID, SearchEngineID, AdvEngineID, CASE WHEN (SearchEngineID = 0 AND AdvE` |
| 40 | 2.997 | 0.005 | 0.006 | **0.005** | 🟢 GPU | `SELECT URLHash, EventDate, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventD` |
| 41 | 1.757 | 0.004 | 0.005 | **0.004** | 🟢 GPU | `SELECT WindowClientWidth, WindowClientHeight, COUNT(*) AS PageViews FROM hits WHERE Counte` |
| 42 | 1.382 | 0.007 | 0.004 | **0.004** | 🟢 GPU | `SELECT DATE_TRUNC('minute', EventTime) AS M, COUNT(*) AS PageViews FROM hits WHERE Counter` |

_Generated: GH200 run, cudf-25.12-optimization branch_
