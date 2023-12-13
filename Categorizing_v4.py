from collections import defaultdict
import pdb

class Categorizing:

    def __init__(self, dataset):
        self.all_attrs = None
        self.categorized_attrs = None
        # self.check = set()
        if dataset == 'birdstrikes1':
            self.states = {"Damage":0, "Incident":1, "Aircraft":2, "Environment":3, "Wildlife":4, "Misc":5}
            # self.states = {"Damage":0, "Incident":1, "Aircraft":2, "Environment":3}
        elif dataset == 'weather1':
            self.states = {"Temperature":0, "Location":1, "Metadata":2, "CommonPhenomena":3, "Fog":4, "Extreme":5, "Misc":6}
    def birdstrikes1(self, category):
        if category in ['"dam_eng1"', '"dam_eng2"', '"dam_windshld"', '"dam_wing_rot"', '"damage"', '"dam_eng3"', '"dam_tail"', '"dam_nose"', '"dam_lghts"', '"dam_lg"', '"dam_fuse"', '"dam_eng4"', '"dam_other"', '"cost_repairs"']:
             return "Damage" #State (Damage of aircraft)
        elif category in ['"incident_date"', '"time_of_day"', '"faaregion"', '"location"', '"latitude (generated)"', '"longitude (generated)"', '"state"', '"distance"']:
             return "Incident" #State (Incident details: Location and Time)
        elif category in ['"ac_mass"', '"ac_class"', '"speed"', '"height"', '"phase_of_flt"']:
             return "Aircraft" #State (Aircraft related information)
        elif category in ['"precip"','"sky"']:
             return "Environment" #State (Aircraft Environment)
        elif category in ['"birds_struck"', '"birds_seen"', '"size"']:
             return "Wildlife" #State (Information on the wildlife involved)
        else:
            return "Misc"    

    def weather1(self, category):
        if category in ["tmax_f","tmin_f","tmax","tmin"]:
            return "Temperature" # Temperature information 
        elif category in ["latitude (generated)","longitude (generated)","lat","lng","state","name"]:
            return "Location" # Location of the Global Historical Climatology Network Station
        elif category in ["number of records","date"]:
            return "Metadata" # Date of the incidents
        elif category in ["icepellets","freezingrain","blowingsnow","blowingspray","drizzle","freezingdrizzle","prcp","rain","snow","snowgeneral","snwd","hail", "glaze"]:
            return "CommonPhenomena" # Attribute related to Snow & Rain including Precipitation  
        elif category in ["heavyfog","groundfog","icefog","fog","mist"]:
            return "Fog" # Fog events
        elif category in ["thunder","tornado"]:
            return "Extreme" # Extreme weather conditions
        elif category in ["dust","highwinds","smoke"]:
            return "Misc" # miscellaneous 


    def faa1(self):
        self.all_attrs = [('"calculation(percent delta)"', 'carrier'), ('"destcityname"', 'dest'), ('"calculation(arrival y/n)"', 'delay'), ('"longitude (generated)"', 'aggregate'),
                          ('"deststate"', 'dest'), ('"weatherdelay"', 'delay'), ('"uniquecarrier"', 'carrier'), ('"crsdeptime"', 'time'), ('"deptime"', 'time'), ('"distance"', 'distance'),
                          ('"depdelay"', 'delay'), ('"arrdelay"', 'delay'), ('"calculation(delayed y/n)"', 'delay'), ('"calculation(total delays)"', 'delay'),
                          ('"flightdate"', 'time'), ('"calculation(arrdelayed)"', 'delay'), ('"carrierdelay"', 'delay'), ('"calculation([arrdelay]+[depdelay])"', 'delay'),
                          ('"latitude (generated)"', 'aggregate'), ('"airtime"', 'time'), ('"arrtime"', 'time'), ('"calculation(is delta flight)"', 'carrier'), ('"crselapsedtime"', 'time'), ('"taxiin"', 'taxi'),
                          ('"crsarrtime"', 'time'), ('"originstate"', 'origin'), ('"taxiout"', 'taxi'), ('"diverted"', 'diverted'), ('"lateaircraftdelay"', 'delay'), ('"calculation(delay?)"', 'delay'),
                          ('"origincityname"', 'origin'), ('"securitydelay"', 'delay'), ('"cancellationcode"', 'cancellation'), ('"origin"', 'origin'), ('"calculation([dest]+[origin])"', 'dest'),
                          ('"nasdelay"', 'delay'), ('"calculation(depdelayed)"', 'delay'), ('"number of records"', 'aggregate'), ('"cancelled"', 'cancellation'),
                          ('"dest"', 'dest'), ('"actualelapsedtime"', 'time')]

        self.categorized_attrs = defaultdict()
        test = set()
        for attrs, category in self.all_attrs:
            self.categorized_attrs[attrs] = category
            test.add(category)
        # pdb.set_trace()
        # self.show(test)
        return test

    def get_category(self, cur_attrs, dataset):
        ret = set()
        if dataset == 'birdstrikes1':
            for attr in cur_attrs:        
                ret.add(self.birdstrikes1(attr))
        ret = list(ret)
        return ret

    def show(self, test):
        # print(self.check)
        print(test)
        for t in test:
            print(t, end=' : ')
            for attrs, category in self.all_attrs:
                if t == category:
                    print(attrs, end=' ')
            print()
        print(len(self.all_attrs))


if __name__ == '__main__':
    c = Categorizing()
    c.birdstrikes1()
