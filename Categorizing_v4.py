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
        else: # 'FAA1'
            self.states = {"Performance":0, "Airline":1, "Location":2, "Status":3}
    
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


    def faa1(self, category):
        if category in [ '"arrdelay"', '"depdelay"', '"airtime"', '"securitydelay"']:
            return "Performance" #Performance of an airline using delay
        elif category in ['"uniquecarrier"', '"flightdate"']:
            return "Airline" # Airline and Incident information of uniquely identifying each incident. 
        elif category in ['"distance"', '"origin"', '"dest"', '"latitude (generated)"', '"longitude (generated)"', '"origincityname"',  '"destcityname"']:
            return "Location" # Logistic information related to the Source and Destination of a flight
        elif category in ['"cancelled"', '"diverted"', '"cancellationcode"']:
            return "Status" # Flight status, whether it was cancelled / diverted


    def get_category(self, cur_attrs, dataset):
        ret = set()
        if dataset == 'birdstrikes1':
            for attr in cur_attrs:        
                ret.add(self.birdstrikes1(attr))
        elif dataset == 'faa1':
            for attr in cur_attrs:        
                ret.add(self.faa1(attr))
        else: #Dataset is Weather1 
            for attr in cur_attrs:        
                ret.add(self.weather1(attr))
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
