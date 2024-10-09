from collections import defaultdict
import pdb

class Categorizing:

    def __init__(self, dataset):
        self.birdstrikes = {'"dam_eng1"': 1, '"dam_eng2"': 2, '"dam_windshld"': 3, '"dam_wing_rot"': 4, '"damage"': 5, '"dam_eng3"': 6, '"dam_tail"': 7, '"dam_nose"': 8, '"dam_lghts"': 9, '"dam_lg"': 10, '"dam_fuse"': 11, '"dam_eng4"': 12, '"dam_other"': 13, '"cost_repairs"': 14, '"incident_date"': 15, '"time_of_day"': 16, '"faaregion"': 17, '"location"': 18, '"latitude (generated)"': 19, '"longitude (generated)"': 20, '"state"': 21, '"distance"': 22, '"ac_mass"': 23, '"ac_class"': 24, '"speed"': 25, '"height"': 26, '"phase_of_flt"': 27, '"precip"': 28, '"sky"': 29, '"birds_struck"': 30, '"birds_seen"': 31, '"size"': 32}
        self.weather = {'tmax_f': 1, 'tmin_f': 2, 'tmax': 3, 'tmin': 4, 'latitude (generated)': 5, 'longitude (generated)': 6, 'lat': 7, 'lng': 8, 'state': 9, 'name': 10, 'number of records': 11, 'date': 12, 'icepellets': 13, 'freezingrain': 14, 'blowingsnow': 15, 'blowingspray': 16, 'drizzle': 17, 'freezingdrizzle': 18, 'prcp': 19, 'rain': 20, 'snow': 21, 'snowgeneral': 22, 'snwd': 23, 'hail': 24, 'glaze': 25, 'heavyfog': 26, 'groundfog': 27, 'icefog': 28, 'fog': 29, 'mist': 30, 'thunder': 31, 'tornado': 32, 'dust': 33, 'highwinds': 34, 'smoke': 35}
        self.faa = {'"arrdelay"': 1, '"depdelay"': 2, '"airtime"': 3, '"securitydelay"': 4, '"uniquecarrier"': 5, '"flightdate"': 6, '"distance"': 7, '"origin"': 8, '"dest"': 9, '"latitude (generated)"': 10, '"longitude (generated)"': 11, '"origincityname"': 12, '"destcityname"': 13, '"cancelled"': 14, '"diverted"': 15, '"cancellationcode"': 16}


    
