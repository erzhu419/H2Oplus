class Passenger(object):
    __slots__ = ('appear_time', 'boarding_time', 'arrive_time',
                 'appear_station', 'destination_station',
                 'travel_bus', 'boarded', 'arrived')

    def __init__(self, t, boarding_station, destination_station):
        self.appear_time = t
        self.boarding_time = None
        self.arrive_time = None

        self.appear_station = boarding_station
        self.destination_station = destination_station

        self.travel_bus = None

        self.boarded = False
        self.arrived = False

    @property
    def travel_time(self):
        return self.arrive_time - self.boarding_time if self.arrived else -1

    @property
    def waiting_time(self):
        return self.boarding_time - self.appear_time if self.boarded else -1

