import matplotlib.pyplot as plt
import numpy as np
import mujoco
from stable_baselines3 import PPO
from env import InvertedPendulumEnv

if __name__ == "__main__":
    print("Loading Viewer...")
    env = InvertedPendulumEnv(render_mode="human")

    print("Loading trained model...")
    model = PPO.load("ppo_inverted_pendulum")

    obs, info = env.reset()

    # --- Resolve hinge2 dof index once ---
    slider_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "slider")
    slider_dof_adr = env.model.jnt_dofadr[slider_id]
    hip_hinge_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "hip_hinge")
    hip_dof_adr = env.model.jnt_dofadr[hip_hinge_id]

    # --- DATA ARRAYS ---
    t = []
    slider_force =[]
    policy_cmd = []
    slider_pos = []
    pole_angle = []

    print("Running simulation and recording data...")

    while True:
        action, _states = model.predict(obs, deterministic=True)

        # Save PPO output (policy command)
        policy_cmd.append(float(action[0]))

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        # --- Log real MuJoCo signals AFTER stepping ---
        t.append(float(env.data.time))
        slider_force.append(float(env.data.qfrc_actuator[slider_dof_adr]))

        slider_pos.append(float(env.data.qpos[0]))
        pole_angle.append(float(np.degrees(env.data.qpos[1])))

        if terminated or truncated or env.data.time > 40.0:  # Safety stop after 20 seconds
            print(f"Episode finished at {env.data.time:.2f} seconds.")
            break

    env.close()

    #Plotting Section
    print("Generating Graph...")

    #plt.figure(figsize=(11,5))
    fig, (plt1, plt2) = plt.subplots(2, 1, figsize=(10,8), sharex=True)

    #plt.plot(t, policy_cmd, label="Policy Output (action)", linewidth=3.0)
    #plt.plot(t, actuator_force, label="Actuator Force", linewidth=1.0)
    plt1.plot(t, pole_angle, label="Pole Angle (deg)",color = 'green', linewidth=1.5)
    plt1.plot(t, slider_pos, label="Slider Position (m)",color = 'blue', linewidth=1.5)
    #plt.plot(t, angle1, label="Hinge2 Angle (rad)", linewidth=1.0)
    plt1.axhline(0,color ='black', linestyle="--", linewidth=1)
    #plt1.set_xlabel("Time (seconds)")
    plt1.set_ylabel("Angle (Degrees)")
    plt1.set_title("Joint Angles vs Time(s)")
    plt1.legend()
    plt1.grid(True, alpha=0.3)

    plt2.plot(t, slider_force, label="Slider Force (Physics)", color='green', linewidth=1.5)
    plt2.axhline(0, color='black', linestyle="--", linewidth=1)
    plt2.set_xlabel("Time (seconds)")
    plt2.set_ylabel("Force (N)")
    plt2.set_title("Force Applied to Slider vs Time(s)")
    plt2.legend()
    plt2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()