from collections import defaultdict
import pdb

class Categorizing:

    def __init__(self):
        self.all_attrs = None
        self.categorized_attrs = None
        self.check = set()

    def birdstrikes1(self):
        self.all_attrs = [('"dam_eng1"', 'damaged_parts'), ('"dam_eng2"', 'damaged_parts') , ('"dam_windshld"', 'damaged_parts'), ('"dam_wing_rot"', 'damaged_parts'), ('"number of records"', 'aggregation'), ('"damage"', 'damaged_parts'),
                     ('"ac_class"', 'airplane'), ('"incident_date"', 'event_time'), ('"precip"', 'weather'), ('"sky"', 'weather'), ('"phase_of_flt"', 'flight_related'), ('"operator"', 'location'), ('"ac_mass"', 'airplane'),
                     ('"state"', 'location'), ('"size"', 'birds'), ('"birds_struck"', 'birds'), ('"time_of_day"', 'event_time'), ('"type_eng"', 'airplane'), ('"birds_seen"', 'birds'), ('"distance"', 'location'),
                     ('"height"', 'flight_related'), ('"dam_eng3"', 'damaged_parts'), ('"indicated_damage"', 'damaged_parts'), ('"dam_tail"', 'damaged_parts'), ('"dam_nose"', 'damaged_parts'), ('"dam_lghts"', 'damaged_parts'),
                     ('"dam_lg"', 'damaged_parts'), ('"dam_fuse"', 'damaged_parts'), ('"dam_eng4"', 'damaged_parts'), ('"dam_other"', 'damaged_parts'), ('"reported_date"', 'event_time'), ('"warned"', 'flight_related'), ('"dam_prop"', 'damaged_parts'),
                     ('"dam_rad"', 'damaged_parts'), ('"index_nr"', 'flight_related'), ('"speed"', 'flight_related'), ('"incident_month"', 'event_time'), ('"faaregion"', 'location'), ('"location"', 'location'),
                     ('"airport_id"', 'location'), ('"atype"', 'airplane'), ('"airport"', 'location'), ('"incident_year"', 'event_time'), ('"longitude (generated)"', 'location'), ('"latitude (generated)"', 'location'),
                     ('"calculation(phase of flt dedup)"', 'flight_related'), ('"time"', 'event_time')]

        self.categorized_attrs = defaultdict()
        test = set()

        for attrs, category in self.all_attrs:
            self.categorized_attrs[attrs] = category
            test.add(category)
        # self.show(test)
        # print(self.categorized_attrs.keys())
        return test

    def weather1(self):
        self.all_attrs = [('"heavyfog"', 'fog'), ('"number of records"', 'aggregation'), ('"calculation(heavy fog (is null))"', 'fog'), ('"date"', 'time'), ('"tmax_f"', 'temperature'),
                          ('"tmin_f"', 'temperature'), ('"latitude (generated)"', 'location'), ('"longitude (generated)"', 'location'), ('"lat"', 'location'), ('"lng"', 'location'),
                          ('"state"', 'location'), ('"freezingrain"', 'rain'), ('"blowingsnow"', 'snow'), ('"blowingspray"', 'snow'), ('"drizzle"', 'rain'), ('"dust"', 'windy'),
                          ('"fog"', 'fog'), ('"mist"', 'fog'), ('"groundfog"', 'fog'), ('"freezingdrizzle"','rain'), ('"glaze"', 'snow'), ('"hail"', 'rain'), ('"highwinds"', 'windy'),
                          ('"icefog"', 'fog'), ('"icepellets"', 'snow'), ('"prcp"', 'rain'), ('"rain"', 'rain'), ('"smoke"', 'smoke'), ('"tmax"', 'temperature'), ('"tmin"', 'temperature'),
                          ('"snow"', 'snow'), ('"snowgeneral"', 'snow'), ('"snwd"', 'snow'), ('"thunder"', 'rain'), ('"tornado"', 'tornado')]

        self.categorized_attrs = defaultdict()
        test = set()
        for attrs, category in self.all_attrs:
            self.categorized_attrs[attrs] = category
            test.add(category)
        test = list(test)
        # self.show(test)
        return test

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

    def get_category(self, cur_attrs):
        ret = set()
        for attr in cur_attrs:
            if attr in self.categorized_attrs:
                ret.add(self.categorized_attrs[attr])
            else:
                self.check.add(attr)
        ret = list(ret)
        # pdb.set_trace()
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
