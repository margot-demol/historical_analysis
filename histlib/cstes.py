import string

labels = [
    "gps_Jason-3_2020",#
    "argos_Jason-3_2020",#
    "gps_SARAL_2020",
    "argos_SARAL_2020",
    "gps_Cryosat-2_2020",
    "argos_Cryosat-2_2020",
    "gps_Sentinel-3_A_2020",
    "argos_Sentinel-3_A_2020",
    "gps_Sentinel-3_B_2020",
    "argos_Sentinel-3_B_2020",
    "gps_Jason-3_2019",#
    "argos_Jason-3_2019",
    "gps_SARAL_2019",
    "argos_SARAL_2019",
    "gps_Cryosat-2_2019",
    "argos_Cryosat-2_2019",
    "gps_Sentinel-3_A_2019",#,too big, to do with append dims
    "argos_Sentinel-3_A_2019",
    "gps_Sentinel-3_B_2019",
    "argos_Sentinel-3_B_2019",
    "gps_Jason-3_2018",#
    "argos_Jason-3_2018",
    "gps_SARAL_2018",
    "argos_SARAL_2018",
    "gps_Cryosat-2_2018",
    "argos_Cryosat-2_2018",
    "gps_Sentinel-3_A_2018",
    "argos_Sentinel-3_A_2018",
    "gps_Sentinel-3_B_2018",
    "argos_Sentinel-3_B_2018",
    "gps_Jason-3_2017",#
    "argos_Jason-3_2017",
    "gps_Jason-2_2017",#
    "argos_Jason-2_2017",
    "gps_SARAL_2017",
    "argos_SARAL_2017",
    "gps_Cryosat-2_2017",
    "argos_Cryosat-2_2017",
    "gps_Sentinel-3_A_2017",
    "argos_Sentinel-3_A_2017",
    "gps_Jason-3_2016",#
    "argos_Jason-3_2016",
    "gps_Jason-2_2016",#
    "argos_Jason-2_2016",
    "gps_SARAL_2016",
    "argos_SARAL_2016",
    "gps_Cryosat-2_2016",
    "argos_Cryosat-2_2016",
    "gps_Sentinel-3_A_2016",
    "argos_Sentinel-3_A_2016",
    "gps_Jason-2_2015",#
    "argos_Jason-2_2015",
    "gps_SARAL_2015",
    "argos_SARAL_2015",
    "gps_Cryosat-2_2015",
    "argos_Cryosat-2_2015",
    "gps_Jason-2_2014",#
    "argos_Jason-2_2014",
    "gps_SARAL_2014",
    "argos_SARAL_2014",
    "gps_Cryosat-2_2014",
    "argos_Cryosat-2_2014", 
    "gps_Jason-2_2013",#
    "argos_Jason-2_2013",
    "gps_SARAL_2013",
    "argos_SARAL_2013",
    "gps_Cryosat-2_2013",
    "argos_Cryosat-2_2013",
    "gps_Jason-2_2012",#
    "argos_Jason-2_2012",
    "gps_Cryosat-2_2012",
    "argos_Cryosat-2_2012",
    "gps_Jason-2_2011",#
    "argos_Jason-2_2011", 
    "gps_Cryosat-2_2011",
    "argos_Cryosat-2_2011",
    "gps_Jason-2_2010",#
    "argos_Jason-2_2010",
    "gps_Cryosat-2_2010",
    "argos_Cryosat-2_2010",
    #"gps_Jason-2_2009",# no erastar
    #"argos_Jason-2_2009",
    #"gps_Jason-2_2008",#
    #"argos_Jason-2_2008",
]

zarr_dir = "/home/datawork-lops-osi/aponte/margot/historical_coloc"
matchup_dir = "/home/datawork-lops-osi/aponte/margot/historical_coloc_ok/matchup"
images_dir = "/home1/datawork/mdemol/historical_analysis/images"
result_dir = "/home1/datawork/mdemol/historical_analysis/results"

import histlib.box as box

#nc_files = {l: box.load_collocalisations(int(l.split('_')[-1]), drifter=l.split('_')[0], product_type=l.split('_')[1], satellite=l.split('_')[2], ) for l in labels}


lon_180_to_360 = lambda lon: lon % 360
lon_360_to_180 = lambda lon: (lon + 180) % 360 - 180

lettres = ["(" + l + ")" for l in list(string.ascii_lowercase)]


var = [
    'drifter_acc_x_0',
    'drifter_acc_y_0',
    'drifter_coriolis_x_0',
    'drifter_coriolis_y_0',
    'e5_cstrio_z0_alti_wd_x',
    'e5_cstrio_z0_drifter_wd_x',
    'e5_cstrio_z15_alti_wd_x',
    'e5_cstrio_z15_drifter_wd_x',
    'es_cstrio_z0_alti_wd_x',
    'es_cstrio_z0_drifter_wd_x',
    'es_cstrio_z15_alti_wd_x',
    'es_cstrio_z15_drifter_wd_x',
    'e5_cstrio_z0_alti_wd_y',
    'e5_cstrio_z0_drifter_wd_y',
    'e5_cstrio_z15_alti_wd_y',
    'e5_cstrio_z15_drifter_wd_y',
    'es_cstrio_z0_alti_wd_y',
    'es_cstrio_z0_drifter_wd_y',
    'es_cstrio_z15_alti_wd_y',
    'es_cstrio_z15_drifter_wd_y',
    'alti_ggx_adt_filtered',
    'alti_ggx_adt_unfiltered',
    'alti_ggx_adt_unfiltered_denoised',
    'alti_ggx_sla_filtered',
    'alti_ggx_sla_unfiltered',
    'alti_ggx_sla_unfiltered_denoised',
    'alti_ggx_adt_filtered_ocean_tide',
    'alti_ggx_adt_filtered_ocean_tide_internal_tide',
    'alti_ggx_adt_filtered_ocean_tide_internal_tide_dac',
    'aviso_alti_ggx_adt',
    'aviso_alti_ggx_sla',
    'aviso_drifter_ggx_adt',
    'aviso_drifter_ggx_sla',
    'aviso_alti_ggy_adt',
    'aviso_alti_ggy_sla',
    'aviso_drifter_ggy_adt',
    'aviso_drifter_ggy_sla',
]


var2 =['nb_coloc_bin',
 'exc_acc_co__0__adt_filtered__es_cstrio_z15__drifter_x',
 'exc_coriolis_co__0__adt_filtered__es_cstrio_z15__drifter_x',
 'exc_ggrad_co__0__adt_filtered__es_cstrio_z15__drifter_x',
 'exc_wind_co__0__adt_filtered__es_cstrio_z15__drifter_x',
 'sum_co__0__adt_filtered__es_cstrio_z15__drifter_x',
 'exc_acc_aviso__0__adt__es_cstrio_z15__alti_x',
 'exc_coriolis_aviso__0__adt__es_cstrio_z15__alti_x',
 'exc_ggrad_aviso__0__adt__es_cstrio_z15__alti_x',
 'exc_wind_aviso__0__adt__es_cstrio_z15__alti_x',
 'sum_aviso__0__adt__es_cstrio_z15__alti_x',
 'exc_acc_aviso__0__adt__es_cstrio_z15__drifter_x',
 'exc_coriolis_aviso__0__adt__es_cstrio_z15__drifter_x',
 'exc_ggrad_aviso__0__adt__es_cstrio_z15__drifter_x',
 'exc_wind_aviso__0__adt__es_cstrio_z15__drifter_x',
 'sum_aviso__0__adt__es_cstrio_z15__drifter_x',
 'drifter_acc_x_0',
 'drifter_coriolis_x_0',
 'alti_ggx_adt_filtered',
 'es_cstrio_z15_drifter_wd_x',
 'drifter_acc_x_0',
 'drifter_coriolis_x_0',
 'aviso_alti_ggx_adt',
'aviso_drifter_ggx_adt',
 'es_cstrio_z15_alti_wd_x']

id_co_dic =  {'acc':'drifter_acc_x_0','coriolis':'drifter_coriolis_x_0','ggrad':'alti_ggx_adt_filtered','wind':'es_cstrio_z15_drifter_wd_x'}
id_aviso_dic =  {'acc':'drifter_acc_x_0','coriolis':'drifter_coriolis_x_0','ggrad':'aviso_alti_ggx_adt','wind':'es_cstrio_z15_alti_wd_x'}

term_color ={'acc':'r', 'coriolis':'g','ggrad':'c', 'wind':'mediumvioletred'}