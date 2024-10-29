class Klimatyzacja:
    def __init__(self):
        self.external_temperature = 0.0
        self.internal_temperature = 0.0
        self.fan_speed = 0
        self.target_temperature = 0.0

    def set_temperatures(self, external_temp, internal_temp):
        self.external_temperature = external_temp
        self.internal_temperature = internal_temp
        self.adjust_climate()

    def adjust_climate(self):
        temperature_difference = self.external_temperature - self.internal_temperature
        
        if temperature_difference < -5:
            self.fan_speed = 100
            self.target_temperature = self.internal_temperature + 5
        elif -5 <= temperature_difference < 0:
            self.fan_speed = 70
            self.target_temperature = self.internal_temperature + 3
        elif 0 <= temperature_difference <= 5:
            self.fan_speed = 30
            self.target_temperature = self.internal_temperature - 3
        else:
            self.fan_speed = 0
            self.target_temperature = self.internal_temperature - 5

    def get_status(self):
        return {
            "external_temperature": self.external_temperature,
            "internal_temperature": self.internal_temperature,
            "fan_speed": self.fan_speed,
            "target_temperature": self.target_temperature
        }






climate_control = Klimatyzacja()
climate_control.set_temperatures(30, 200)
status = climate_control.get_status()
print(status)