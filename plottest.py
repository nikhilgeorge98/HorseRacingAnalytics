import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('Both angles.csv')

figure, axis = plt.subplots(2, 1)

axis[0].plot(data['frame'], data['front angle'])
axis[0].set_title("Hip range of motion(forward)")
axis[0].set_ylabel("Flexion/Extension(in deg)")

axis[1].plot(data['frame'], data['side angle'])
axis[1].set_title("Hip range of motion(sideways)")
axis[1].set_ylabel("Flexion/Extension(in deg)")

plt.show()