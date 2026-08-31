from interflib.simulation import synthesize_interferogram, SceneGeometry, SensorModel
import matplotlib.pyplot as plt

result = synthesize_interferogram(
    geometry=SceneGeometry.cs505mu(),   # real 2448x2048 sensor geometry
    sensor=SensorModel.cs505mu(),       # 12-bit, full well >=10 ke-, read noise <2.5 e-
)
image = result.image                    # simulated frame
truth = result.fraction_truth           # exact fringe fraction that produced it

print(f"Simulated fringe fraction: {truth:.6f}")
plt.imshow(image, cmap="gray", vmin=0, vmax=4095)
plt.show()