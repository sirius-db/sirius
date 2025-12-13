# =============================================================================
# Copyright 2025, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

import duckdb
import os
import sys

if __name__ == "__main__":
    con = duckdb.connect('clickbench_test.duckdb', config={"allow_unsigned_extensions": "true"})
    extension_path = os.path.join(
      os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
      'build/release/extension/sirius/sirius.duckdb_extension')
    con.execute("load '{}'".format(extension_path))

    print("Setting up test data directory...")
    os.system("mkdir -p test_datasets")

    # Check if data already exists
    if not os.path.exists("test_datasets/test_hits.tsv"):
        print("Downloading ClickBench test data...")
        os.system("cd test_datasets && wget https://pages.cs.wisc.edu/~yxy/sirius-datasets/test_hits.tsv.gz && gzip -d test_hits.tsv.gz")
    else:
        print("Test data already exists, skipping download...")

    print("Creating hits table...")
    con.execute('DROP TABLE IF EXISTS hits;')

    con.execute('''
    CREATE TABLE hits
    (
        WatchID BIGINT NOT NULL,
        JavaEnable SMALLINT NOT NULL,
        Title TEXT,
        GoodEvent SMALLINT NOT NULL,
        EventTime TIMESTAMP NOT NULL,
        EventDate Date NOT NULL,
        CounterID INTEGER NOT NULL,
        ClientIP INTEGER NOT NULL,
        RegionID INTEGER NOT NULL,
        UserID BIGINT NOT NULL,
        CounterClass SMALLINT NOT NULL,
        OS SMALLINT NOT NULL,
        UserAgent SMALLINT NOT NULL,
        URL TEXT,
        Referer TEXT,
        IsRefresh SMALLINT NOT NULL,
        RefererCategoryID SMALLINT NOT NULL,
        RefererRegionID INTEGER NOT NULL,
        URLCategoryID SMALLINT NOT NULL,
        URLRegionID INTEGER NOT NULL,
        ResolutionWidth SMALLINT NOT NULL,
        ResolutionHeight SMALLINT NOT NULL,
        ResolutionDepth SMALLINT NOT NULL,
        FlashMajor SMALLINT NOT NULL,
        FlashMinor SMALLINT NOT NULL,
        FlashMinor2 TEXT,
        NetMajor SMALLINT NOT NULL,
        NetMinor SMALLINT NOT NULL,
        UserAgentMajor SMALLINT NOT NULL,
        UserAgentMinor VARCHAR(255) NOT NULL,
        CookieEnable SMALLINT NOT NULL,
        JavascriptEnable SMALLINT NOT NULL,
        IsMobile SMALLINT NOT NULL,
        MobilePhone SMALLINT NOT NULL,
        MobilePhoneModel TEXT,
        Params TEXT,
        IPNetworkID INTEGER NOT NULL,
        TraficSourceID SMALLINT NOT NULL,
        SearchEngineID SMALLINT NOT NULL,
        SearchPhrase TEXT,
        AdvEngineID SMALLINT NOT NULL,
        IsArtifical SMALLINT NOT NULL,
        WindowClientWidth SMALLINT NOT NULL,
        WindowClientHeight SMALLINT NOT NULL,
        ClientTimeZone SMALLINT NOT NULL,
        ClientEventTime TIMESTAMP NOT NULL,
        SilverlightVersion1 SMALLINT NOT NULL,
        SilverlightVersion2 SMALLINT NOT NULL,
        SilverlightVersion3 INTEGER NOT NULL,
        SilverlightVersion4 SMALLINT NOT NULL,
        PageCharset TEXT,
        CodeVersion INTEGER NOT NULL,
        IsLink SMALLINT NOT NULL,
        IsDownload SMALLINT NOT NULL,
        IsNotBounce SMALLINT NOT NULL,
        FUniqID BIGINT NOT NULL,
        OriginalURL TEXT,
        HID INTEGER NOT NULL,
        IsOldCounter SMALLINT NOT NULL,
        IsEvent SMALLINT NOT NULL,
        IsParameter SMALLINT NOT NULL,
        DontCountHits SMALLINT NOT NULL,
        WithHash SMALLINT NOT NULL,
        HitColor CHAR NOT NULL,
        LocalEventTime TIMESTAMP NOT NULL,
        Age SMALLINT NOT NULL,
        Sex SMALLINT NOT NULL,
        Income SMALLINT NOT NULL,
        Interests SMALLINT NOT NULL,
        Robotness SMALLINT NOT NULL,
        RemoteIP INTEGER NOT NULL,
        WindowName INTEGER NOT NULL,
        OpenerName INTEGER NOT NULL,
        HistoryLength SMALLINT NOT NULL,
        BrowserLanguage TEXT,
        BrowserCountry TEXT,
        SocialNetwork TEXT,
        SocialAction TEXT,
        HTTPError SMALLINT NOT NULL,
        SendTiming INTEGER NOT NULL,
        DNSTiming INTEGER NOT NULL,
        ConnectTiming INTEGER NOT NULL,
        ResponseStartTiming INTEGER NOT NULL,
        ResponseEndTiming INTEGER NOT NULL,
        FetchTiming INTEGER NOT NULL,
        SocialSourceNetworkID SMALLINT NOT NULL,
        SocialSourcePage TEXT,
        ParamPrice BIGINT NOT NULL,
        ParamOrderID TEXT,
        ParamCurrency TEXT,
        ParamCurrencyID SMALLINT NOT NULL,
        OpenstatServiceName TEXT,
        OpenstatCampaignID TEXT,
        OpenstatAdID TEXT,
        OpenstatSourceID TEXT,
        UTMSource TEXT,
        UTMMedium TEXT,
        UTMCampaign TEXT,
        UTMContent TEXT,
        UTMTerm TEXT,
        FromTag TEXT,
        HasGCLID SMALLINT NOT NULL,
        RefererHash BIGINT NOT NULL,
        URLHash BIGINT NOT NULL,
        CLID INTEGER NOT NULL
    );
    ''')

    print("Loading data into hits table...")
    con.execute('''
    COPY hits FROM 'test_datasets/test_hits.tsv' (QUOTE '');
    ''')

    row_count = con.execute("SELECT COUNT(*) FROM hits").fetchone()[0]
    print(f"Data loaded successfully. Total rows: {row_count}")

    con.close()
