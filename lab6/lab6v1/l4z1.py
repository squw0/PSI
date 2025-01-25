#3.2

import numpy as np
import matplotlib.pyplot as plt

class SugenoController:
    def __init__(self):
        pass

    def evaluate(self, light_intensivity, hour):
        brightness1 = min(light_intensivity, hour)
        brightness2 = max(light_intensivity, hour)
        brightness3 = 0.7 * light_intensivity + 0.3 * hour

        brightness = (brightness1 + brightness2 + brightness3) / 3

        return brightness
    
    def plot_membership_functions(self):
        light_intensivity_range = np.linspace(0,10,100)
        hour_range = np.linspace(0,23,24)

        membership_light_intensivity = np.minimum(light_intensivity_range, 10 - light_intensivity_range) #the higher the intensivity the less light needed
        membership_hour = 1 - hour_range/24 #from 0 - 6 night(need a lot of light) #from 6 - 17 day(doesn't need lot) from 17-24 night

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(light_intensivity_range, membership_light_intensivity, label='Light intensivity')
        plt.title('Membership function - Light intesivity')
        plt.xlabel('Light intesivity')
        plt.ylabel('Membership')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(hour_range, membership_hour, label='Time of day in hours')
        plt.title('Membership function - Time of day in hours')
        plt.xlabel('Time of day in hours')
        plt.ylabel('PMembership')
        plt.legend()

        plt.tight_layout()
        plt.show()
    
if __name__ == "__main__":
    sugeno_controller = SugenoController()

    sugeno_controller.plot_membership_functions()

    light_intensivity_input = 24
    hour_input = 13

    brightness = sugeno_controller.evaluate(light_intensivity_input, hour_input)
    print(f"Light intensivity: {light_intensivity_input} %")
    print(f"Hour: {hour_input}")
    print(f"Recommended light brightness: {round(brightness)} %")