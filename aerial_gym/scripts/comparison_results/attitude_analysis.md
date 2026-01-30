# Attitude Stability Analysis: Why did the Baseline become smoother?

The user observed a counter-intuitive result:
> **"When the Baseline (PID/APF) was switched from Ground Truth sensors to Depth Camera (to match RL), its attitude fluctuations DECREASED, becoming smoother than the RL agent."**

Previously, with Ground Truth, the Baseline had *higher* volatility than RL. Now it is the opposite.

## core Reason: "Ignorance is Bliss" (Limited FOV)

The primary reason for this improved stability is the **Limited Field of View (FOV)** of the depth camera compared to the omniscient Ground Truth sensor.

### 1. Ground Truth (Old Baseline) = High Reactivity
*   **360° Awareness**: The Ground Truth implementation likely calculated repulsion from *all* obstacles within a certain radius (`d_obs`), regardless of whether they were in front, behind, or to the side.
*   **Constant Jitter**: As the drone flew, obstacles passing by on the side or behind would constantly exert changing repulsion forces. Even if the path ahead was clear, the drone was being "pushed" slightly by everything around it.
*   **Result**: The potential field gradient was highly dynamic, causing the velocity command to fluctuate constantly, leading to attitude jitter (roll/pitch changes).

### 2. Depth Camera (New Baseline) = Focused Stability
*   **Tunnel Vision**: The Depth Camera has a limited FOV (e.g., 87° horizontal). It **only** sees what is directly in front of validation.
*   **Filtered Inputs**: Obstacles to the side or behind are **invisible** to the repulson logic. If they aren't in the image, they don't exist.
*   **Smoother Gradients**: If the path directly ahead is clear, the repulsion vector is effectively **zero**. The drone flies straight without being "bothered" by passing obstacles. It only reacts when it is directly facing a threat.
*   **Result**: Fewer repulsion vectors sum up to a more stable total gradient. The drone makes fewer corrections, leading to a significantly smoother flight (lower attitude standard deviation).

### 3. Point Cloud Averaging
*   **Vector Sum** The new implementation sums the repulsion vectors from up to 500 individual points on the object's surface.
*   This acts as a statistical **low-pass filter**. Small noisy variations in depth pixel values generally average out when summing hundreds of vectors, producing a stable "center of mass" repulsion direction.
*   In contrast, Ground Truth might have been sensitive to the exact distance of single point-source obstacle centers.

### Summary
The Baseline became "smoother" because it became "blinder." By removing the distraction of obstacles it can't see (outside FOV), we unintentionally stabilized its control loop. It no longer twitches to avoid things that aren't in its immediate path.

**RL vs Baseline**:
*   **RL** (PPO) might still be learning a high-frequency control policy (micro-corrections) to maximize agility or handle the noisy inputs it *does* see.
*   **Baseline** (PID) is a deterministic mathematical function. When given cleaner/fewer inputs (due to limited FOV), it outputs a smoother signal.
