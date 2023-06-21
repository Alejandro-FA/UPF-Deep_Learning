import matplotlib.pyplot as plt
import scipy

gan_log_file = "Results/logs/GAN_training.log"

generator_loss = []
discriminator_loss = []

with open(gan_log_file) as file:
    for line in file:
        splitted_line = line.split(",")
        generator_loss.append(float(splitted_line[2].split(":")[1]))
        discriminator_loss.append(float(splitted_line[3].split(":")[1]))


figure2 = plt.figure(figsize=(5, 5))
plt.title("GAN loss evolution", fontsize=14, fontweight="bold")
smoothed_disc_loss = scipy.signal.savgol_filter(discriminator_loss, 100, 5)
smoothed_gen_loss = scipy.signal.savgol_filter(generator_loss, 100, 5)
plt.plot(discriminator_loss, label="Discriminator loss", color="blue")
plt.plot(generator_loss, label="Generator loss", color="red")
plt.plot(smoothed_disc_loss, label="Smoothed Discriminator loss", color="#9398FF")
plt.plot(smoothed_gen_loss, label="Smoothed Generator loss", color="#B73D3D")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid()
plt.legend()

plt.savefig("Results/gan_loss_evolution_smoothed.png", dpi=300)