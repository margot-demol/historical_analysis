import string

labels = [
    #"gps_Jason-3_2020",
    "argos_Jason-3_2020",
    "gps_SARAL_2020",
    "argos_SARAL_2020",
    "gps_Cryosat-2_2020",
    "argos_Cryosat-2_2020",
    "gps_Sentinel-3_A_2020",
    "argos_Sentinel-3_A_2020",
    "gps_Sentinel-3_B_2020",
    "argos_Sentinel-3_B_2020",
    #"gps_Jason-3_2019",
    "argos_Jason-3_2019",
    "gps_SARAL_2019",
    "argos_SARAL_2019",
    "gps_Cryosat-2_2019",
    "argos_Cryosat-2_2019",
    "gps_Sentinel-3_A_2019",#,too big, to do after
    "argos_Sentinel-3_A_2019",
    "gps_Sentinel-3_B_2019",
    "argos_Sentinel-3_B_2019",
    #"gps_Jason-3_2018",
    "argos_Jason-3_2018",
    #"gps_Jason-2_2018",
    #"argos_Jason-2_2018",
    "gps_SARAL_2018",
    "argos_SARAL_2018",
    "gps_Cryosat-2_2018",
    "argos_Cryosat-2_2018",
    "gps_Sentinel-3_A_2018",
    "argos_Sentinel-3_A_2018",
    "gps_Sentinel-3_B_2018",
    "argos_Sentinel-3_B_2018",
    #"gps_Jason-3_2017",
    "argos_Jason-3_2017",
    #"gps_Jason-2_2017",
    "argos_Jason-2_2017",
    "gps_SARAL_2017",
    "argos_SARAL_2017",
    "gps_Cryosat-2_2017",
    "argos_Cryosat-2_2017",
    "gps_Sentinel-3_A_2017",
    "argos_Sentinel-3_A_2017",
    #"gps_Jason-3_2016",
    "argos_Jason-3_2016",
    #"gps_Jason-2_2016",
    "argos_Jason-2_2016",
    "gps_SARAL_2016",
    "argos_SARAL_2016",
    "gps_Cryosat-2_2016",
    "argos_Cryosat-2_2016",
    "gps_Sentinel-3_A_2016",
    "argos_Sentinel-3_A_2016",
    #"gps_Jason-2_2015",
    "argos_Jason-2_2015",
    "gps_SARAL_2015",
    "argos_SARAL_2015",
    "gps_Cryosat-2_2015",
    "argos_Cryosat-2_2015",
    #"gps_Jason-2_2014",
    "argos_Jason-2_2014",
    "gps_SARAL_2014",
    "argos_SARAL_2014",
    "gps_Cryosat-2_2014",
    "argos_Cryosat-2_2014", #PB aviso load
    #"gps_Jason-2_2013",
    "argos_Jason-2_2013",
    "gps_SARAL_2013",
    "argos_SARAL_2013",
    "gps_Cryosat-2_2013",
    "argos_Cryosat-2_2013",
    #"gps_Jason-2_2012",
    "argos_Jason-2_2012",
    "gps_Cryosat-2_2012",
    "argos_Cryosat-2_2012",
    #"gps_Jason-2_2011",
    "argos_Jason-2_2011", #pb aviso
    "gps_Cryosat-2_2011",
    "argos_Cryosat-2_2011",#pb aviso
    #"gps_Jason-2_2010",
    "argos_Jason-2_2010",
    "gps_Cryosat-2_2010",
    "argos_Cryosat-2_2010",
    #"gps_Jason-2_2009",
    "argos_Jason-2_2009",
    #"gps_Jason-2_2008",
    "argos_Jason-2_2008",
]

zarr_dir = "/home/datawork-lops-osi/aponte/margot/historical_coloc"
images_dir = "/home1/datawork/mdemol/historical_analysis/images"
result_dir = "/home1/datawork/mdemol/historical_analysis/results"

import histlib.box as box

nc_files = {l: box.load_collocalisations(int(l.split('_')[-1]), drifter=l.split('_')[0], product_type=l.split('_')[1], satellite=l.split('_')[2], ) for l in labels}


lon_180_to_360 = lambda lon: lon % 360
lon_360_to_180 = lambda lon: (lon + 180) % 360 - 180

lettres = ["(" + l + ")" for l in list(string.ascii_lowercase)]
